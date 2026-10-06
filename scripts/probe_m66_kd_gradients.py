"""M66b: train-only, zero-update GT/preservation/Vehicle-KD gradient diagnosis."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import re
import statistics
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import m64_teacher_qualification as q
import m65_a2_preservation_control as c
import m66_r0_a2_feature_pilot as p

REVISION = "M66B-ZERO-UPDATE-KD-GRADIENTS-2026-10-06-r1"
MANIFEST_SHA = "68c2247a9ab28603e10f7f9748fc17a43e5bdefeedc57eebe87e2c62f430d968"
IMPLEMENTATION_SHA = "60f8527b55ba8e646027da5b590ed60d8ca5b0b1fd12c74e3e8e0cc429708ec8"
GATE_SHA = "e4d61aeb1cdea202e0b0e414b7492dc6207a1631eca7bc6a786af6ed8df3056d"
SMOKE_SHA = "8591e5f926c1662b766bacc6e0f043f433a80f95c5ee16e7fe5879039024213e"
ENDPOINTS = dict(control="70f5ee945504ead3d4cbbdce0b932b1bfdd6964893ad5fdcd615495df6773316",
                 kd="31a3fb8e873a27de938a1d532d9999a3bebd598a231957afc78204377f34a6f0")
ROLES = ("original", "control", "kd")
TERMS = ("gt", "preservation", "kd")
BATCHES, BATCH_SIZE, SEED = 16, 4, 444
NOTES = [
    "Norms and dot products use weighted, un-clipped gradients on the same trainable coordinates.",
    "Preservation and KD include their actual M66 coefficients (0.10 each); GT includes native loss weights.",
    "Negative cosine means locally opposing directions, not proof of a harmful AP change.",
    "KD reaches the final depth head's hidden linear, not its output linear or earlier decoder depth heads.",
    "At original A2, preservation gradients can be zero because the frozen anchor is identical.",
    "This is a prospective endpoint probe, not a replay or causal explanation of 928 historical AdamW updates.",
    "No optimizer, clipping, parameter update, validation inference, coefficient sweep or model promotion occurs.",
]


def parameter_group(name):
    match = re.fullmatch(r"depth_embed\.([0-2])\.layers\.([01])\.(weight|bias)", name)
    if not match:
        raise ValueError(f"Unexpected M66 trainable parameter: {name}")
    return f"depth_head_{match[1]}/{'hidden' if match[2] == '0' else 'output'}"


def cpu_gradients(loss, parameters):
    import torch
    if not loss.requires_grad:
        return [None] * len(parameters)
    values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
    if any(value is not None and not bool(torch.isfinite(value).all()) for value in values):
        raise RuntimeError("Non-finite diagnostic gradient; no update taken")
    return [value.detach().cpu() if value is not None else None for value in values]


def gradient_measurements(names, components):
    """Vector norms/cosines, not an average of tensor-wise cosine values."""
    import torch
    if not names or len(set(names)) != len(names) or set(components) != set(TERMS):
        raise ValueError("Unique trainable names and all three gradient terms are required")
    if any(len(components[key]) != len(names) for key in TERMS):
        raise ValueError("Gradient/name lengths differ")
    groups = {}
    for index, name in enumerate(names):
        values = [components[key][index] for key in TERMS]
        shapes = [value.shape for value in values if value is not None]
        if shapes and any(shape != shapes[0] for shape in shapes):
            raise ValueError("Component gradient shapes differ")
        if any(value is not None and not bool(torch.isfinite(value).all()) for value in values):
            raise RuntimeError("Non-finite component gradient")
        # None means disconnected; an allocated all-zero tensor means connected but zero.
        vectors = [value.double() if value is not None else None for value in values]
        for group in ("all", parameter_group(name)):
            record = groups.setdefault(group, dict(
                sq=[0.] * 3, cross=[0.] * 3,
                reach={key: dict(connected_tensors=0, nonzero_tensors=0, nonzero_elements=0) for key in TERMS}))
            for j, (key, value) in enumerate(zip(TERMS, vectors)):
                if value is not None:
                    record["sq"][j] += float(value.square().sum())
                    record["reach"][key]["connected_tensors"] += 1
                    nonzero = int(torch.count_nonzero(value))
                    record["reach"][key]["nonzero_tensors"] += int(nonzero > 0)
                    record["reach"][key]["nonzero_elements"] += nonzero
            for j, (a, b) in enumerate(((0, 1), (0, 2), (1, 2))):
                if vectors[a] is not None and vectors[b] is not None:
                    record["cross"][j] += float((vectors[a] * vectors[b]).sum())
    result = {}
    for group, record in groups.items():
        g2, s2, k2 = record["sq"]
        gs, gk, sk = record["cross"]
        control2 = max(0., g2 + s2 + 2 * gs)
        combined2 = max(0., control2 + k2 + 2 * (gk + sk))
        if not all(math.isfinite(value) for value in (*record["sq"], *record["cross"], control2, combined2)):
            raise RuntimeError("Non-finite aggregate gradient statistic")
        gn, sn, kn, cn, tn = map(math.sqrt, (g2, s2, k2, control2, combined2))
        def cosine(dot, a, b):
            return max(-1., min(1., dot / (a * b))) if a > 0 and b > 0 else None
        deflection = cosine(control2 + gk + sk, cn, tn)
        result[group] = dict(gt_norm=gn, preservation_norm=sn, kd_norm=kn,
            preservation_to_gt_norm_ratio=sn / gn if gn > 0 else None,
            kd_to_gt_norm_ratio=kn / gn if gn > 0 else None,
            kd_to_control_norm_ratio=kn / cn if cn > 0 else None,
            kd_gt_cosine=cosine(gk, kn, gn), kd_preservation_cosine=cosine(sk, kn, sn),
            kd_control_cosine=cosine(gk + sk, kn, cn), control_total_norm=cn,
            total_with_kd_norm=tn,
            total_deflection_degrees=math.degrees(math.acos(deflection)) if deflection is not None else None,
            reach=record["reach"])
    return result


def signed(value):
    result = copy.deepcopy(value)
    result["report_sha256"] = q.signature(result)
    return result


def validate_signed(value):
    signature = value.get("report_sha256")
    if signature != q.signature({k: v for k, v in value.items() if k != "report_sha256"}):
        raise RuntimeError("Diagnostic report signature differs")
    return value


def write_once(path, value):
    if path.exists():
        if q.read_json(path) == value:
            return
        raise RuntimeError(f"Preserve {path}; choose a new diagnostic RUN_ID")
    q.write_json(path, value)


def reviewed_inputs(manifest_path):
    """Only read historical files. Never recover/rewrite their receipts."""
    m = q.read_json(manifest_path)
    if (m.get("manifest_sha256") != MANIFEST_SHA
            or q.signature({k: v for k, v in m.items() if k != "manifest_sha256"}) != MANIFEST_SHA
            or m.get("revision") != p.REVISION or m.get("recipe") != p.RECIPE
            or m.get("implementation_sha256") != IMPLEMENTATION_SHA
            or p.implementation_hash() != IMPLEMENTATION_SHA):
        raise RuntimeError("Use the reviewed, unchanged M66 manifest and implementation")
    root = Path(m["output"])
    if manifest_path.resolve() != (root / "m66_manifest.json").resolve():
        raise RuntimeError("Manifest must remain in its original M66 output directory")
    paths = dict(manifest=manifest_path, paired_gate=root / "m66_paired_gate.json",
        smoke=root / "m66_training_smoke.json", original=Path(m["models"]["a2"]["checkpoint"]),
        teacher=Path(m["r0"]["checkpoint"]), **{role: p.checkpoint_path(m, role) for role in ENDPOINTS})
    expected = dict(paired_gate=GATE_SHA, smoke=SMOKE_SHA, original=q.A2_SHA, teacher=p.R0_SHA, **ENDPOINTS)
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Required reviewed M66 input is missing: {path}. The results ZIP does not contain weights.")
        if name in expected and q.sha(path) != expected[name]:
            raise RuntimeError(f"Reviewed M66 input bytes differ: {name}")
    gate = q.read_json(paths["paired_gate"])
    if not gate.get("complete") or gate.get("manifest_sha256") != MANIFEST_SHA or gate.get("pilot_passed") is not False:
        raise RuntimeError("Wrong completed M66 review")
    p.validated_smoke(m)
    summaries = {}
    for role in ENDPOINTS:
        receipt_path = paths[role].with_suffix(".json")
        summary_path = root / role / "training_summary.json"
        paths[role + "_receipt"], paths[role + "_summary"] = receipt_path, summary_path
        receipt = q.read_json(receipt_path)
        expected_receipt = dict(complete=True, manifest_sha256=MANIFEST_SHA,
            checkpoint_sha256=ENDPOINTS[role], completed_control_epoch=1, role=role)
        if receipt != expected_receipt:
            raise RuntimeError(f"Wrong completed checkpoint receipt: {role}")
        summaries[role] = q.read_json(summary_path)
        p.validate_summary(m, role, summaries[role])
    if summaries["control"]["batch_input_sha256"] != summaries["kd"]["batch_input_sha256"]:
        raise RuntimeError("Historical paired training inputs differ")
    return m, paths, summaries


def verify_inputs(args):
    m, paths, summaries = reviewed_inputs(args.manifest)
    repo = args.repo.resolve()
    if not repo.name.startswith("MonoDETR_M66B"):
        raise RuntimeError("Use a fresh MonoDETR_M66B checkout; do not alter historical sources")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    origin = subprocess.check_output(["git", "remote", "get-url", "origin"], cwd=repo, text=True).strip()
    if head != q.A2_COMMIT or origin != "https://github.com/ZrrSkywalker/MonoDETR.git":
        raise RuntimeError("Wrong prospective MonoDETR checkout identity")
    entry = m["models"]["a2"]
    source = q.source_hash(repo)
    if source != entry["patched_source_sha256"]:
        raise RuntimeError("Fresh patched A2 source differs from reviewed M66")
    build_path = repo / "lib/models/monodetr/ops/m64_build_receipt.json"
    build = q.read_json(build_path)
    binary = Path(build["binary"]).resolve()
    if (not build.get("import_verified") or binary.parent != build_path.parent.resolve()
            or q.sha(binary) != build["binary_sha256"]):
        raise RuntimeError("Fresh local attention binary is missing or changed")
    if c.data_identity(args.dataset_root) != m["data_identity"]:
        raise RuntimeError("KITTI splits, labels or calibration differ")
    env = q.environment()
    if env["gpu"] is None or env["cuda"] != "13.0" or env["packages"]["torch"] != "2.10.0+cu130":
        raise RuntimeError("Use the prospective private Torch 2.10/CUDA13 GPU runtime")
    if build["identity"]["torch"] != env["packages"]["torch"] or build["identity"]["python"] != sys.version:
        raise RuntimeError("Native extension was built for another interpreter")
    runtime = q.read_json(args.runtime_receipt)
    if runtime.get("complete") is not True or Path(runtime["python"]).absolute() != Path(sys.executable).absolute():
        raise RuntimeError("Run using the new diagnostic's private-runtime interpreter")
    view = copy.deepcopy(m)
    view["models"]["a2"].update(repo=str(repo), attention_build_receipt=str(build_path),
        attention_build_sha256=q.sha(build_path), attention_binary=str(binary), attention_binary_sha256=q.sha(binary))
    view["dataset_root"] = str(args.dataset_root.resolve())
    for cfg in (view["models"]["a2"]["config"], view["training_config"], view["r0"]["config"]):
        cfg["dataset"]["root_dir"] = view["dataset_root"]
    identity = dict(revision=REVISION, historical_manifest_sha256=MANIFEST_SHA,
        probe_script_sha256=q.sha(Path(__file__)), implementation_sha256=IMPLEMENTATION_SHA,
        inputs={name: dict(path=str(path), sha256=q.sha(path)) for name, path in paths.items()},
        environment=env, original_environment=m["environment"], historical_runtime_equivalence_claimed=False,
        source_sha256=source, build_receipt_sha256=q.sha(build_path), attention_binary_sha256=q.sha(binary),
        runtime_receipt_sha256=q.sha(args.runtime_receipt),
        dataset_identity=m["data_identity"], seed=SEED, batches=BATCHES, batch_size=BATCH_SIZE,
        sample_ids=q.split_ids(args.dataset_root / "ImageSets/train.txt", "train")[:64],
        sample_selection="First 64 train IDs in saved order; no validation-error selection",
        augmentation="Unchanged M66 online augmentation; seed reset after all model construction",
        coefficients=dict(preservation=p.RECIPE["preservation_weight"], kd=p.RECIPE["feature_kd_weight"]))
    return m, view, identity, summaries


def measure_batch(model, anchor, teacher, criterion, batch, taps, names, parameters):
    import torch
    images, calibs, raw, info = batch
    fingerprint = q.tensor_hash(dict(image=images, calib=calibs,
        **{key: value for key, value in raw.items() if isinstance(value, torch.Tensor)}))
    total, gt, keep, kd, keep_counts, kd_counts = p.losses(model, anchor, teacher, criterion, batch, *taps, "kd")
    weighted = dict(gt=gt, preservation=keep["total"] * p.RECIPE["preservation_weight"],
                    kd=kd * p.RECIPE["feature_kd_weight"])
    components = {key: cpu_gradients(value, parameters) for key, value in weighted.items()}
    statistics_by_group = gradient_measurements(names, components)
    targets = c.gt_targets(raw)
    gt_counts = {label: sum(int((target["labels"] == index).sum()) for target in targets)
                 for index, label in ((0, "Pedestrian"), (1, "Vehicle"))}
    return dict(sample_ids=[f"{int(i):06d}" for i in info["img_id"]], input_sha256=fingerprint,
        weighted_losses={key: float(value.detach()) for key, value in weighted.items()},
        raw_losses=dict(gt=float(gt.detach()), preservation=float(keep["total"].detach()), feature_kd=float(kd.detach())),
        gt_counts=gt_counts, preservation_pairs=keep_counts, kd_pairs=kd_counts, gradients=statistics_by_group)


def run_probe(args):
    original, view, identity, summaries = verify_inputs(args)
    if args.report.exists():
        old = validate_signed(q.read_json(args.report))
        validate_probe(old, args.role)
        if old["identity"] != identity:
            raise RuntimeError("Diagnostic identity changed; keep reports and choose a fresh RUN_ID")
        print(f"Verified complete zero-update report reused: {args.report}", flush=True)
        return
    import torch
    model, anchor, teacher, criterion, dataset, names = p.training_runtime(view)
    if args.role != "original":
        payload = q.safe_payload(p.checkpoint_path(original, args.role))
        if (payload.get("m66_manifest_sha256") != MANIFEST_SHA or payload.get("epoch") != 1
                or payload.get("training_summary") != summaries[args.role]):
            raise RuntimeError("Endpoint embedded lineage differs from reviewed summary")
        model.load_state_dict(payload["model_state"], strict=True)
        del payload
    parameters = [value for name, value in model.named_parameters() if value.requires_grad]
    if names != [name for name, value in model.named_parameters() if value.requires_grad]:
        raise RuntimeError("Trainable names and parameter order differ")
    for name in names:
        parameter_group(name)
    if any(value.grad is not None for value in model.parameters()):
        raise RuntimeError("Unexpected pre-existing gradients")
    states = {key: q.tensor_hash(value.state_dict()) for key, value in
              (("model", model), ("anchor", anchor), ("teacher", teacher))}
    c.seed_all(SEED)
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, range(BATCHES * BATCH_SIZE)),
                                       batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    taps = p.DepthFeatureTap(model), p.DepthFeatureTap(teacher)
    rows = []
    try:
        for index, batch in enumerate(loader, 1):
            print(f"{REVISION} {args.role}: gradients {index}/{BATCHES}; optimizer steps=0", flush=True)
            row = measure_batch(model, anchor, teacher, criterion, batch, taps, names, parameters)
            rows.append(dict(batch=index, **row))
            print(f"  Vehicle pairs={row['kd_pairs']['Vehicle']}; weighted norms={row['gradients']['all']}", flush=True)
    finally:
        for tap in taps:
            tap.close()
    checks = {key + "_parameters_and_buffers_unchanged": before == q.tensor_hash(value.state_dict())
              for (key, before), value in zip(states.items(), (model, anchor, teacher))}
    checks.update(parameter_grad_buffers_empty=all(value.grad is None for value in model.parameters()),
                  historical_files_unchanged=all(q.sha(entry["path"]) == entry["sha256"] for entry in identity["inputs"].values()),
                  no_pedestrian_external_targets=all(row["kd_pairs"]["Pedestrian"] == 0 for row in rows))
    if not all(checks.values()):
        raise RuntimeError(f"Zero-update invariant failed: {checks}")
    report = signed(dict(schema_version=1, complete=True, role=args.role, identity=identity,
        checked_train_images=BATCHES * BATCH_SIZE, rows=rows, optimizer_steps=0,
        trainable_parameter_names=names, checks=checks, interpretation=NOTES,
        training_authorized=False, checkpoint_promotion_authorized=False, deployment_authorized=False))
    validate_probe(report, args.role)
    write_once(args.report, report)
    print(f"Complete: {args.report}. Stop for review; A2 remains selected.", flush=True)


def validate_probe(report, role):
    validate_signed(report)
    if (report.get("complete") is not True or report.get("role") != role or report.get("optimizer_steps") != 0
            or report.get("checked_train_images") != BATCHES * BATCH_SIZE
            or report.get("identity", {}).get("revision") != REVISION
            or any(report.get(key) is not False for key in
                   ("training_authorized", "checkpoint_promotion_authorized", "deployment_authorized"))):
        raise RuntimeError("Incomplete or wrong-scope zero-update report")
    required = {"model_parameters_and_buffers_unchanged", "anchor_parameters_and_buffers_unchanged",
                "teacher_parameters_and_buffers_unchanged", "parameter_grad_buffers_empty",
                "historical_files_unchanged", "no_pedestrian_external_targets"}
    if set(report.get("checks", {})) != required or not all(report["checks"].values()):
        raise RuntimeError("Missing zero-update state checks")
    rows = report.get("rows", [])
    if len(rows) != BATCHES or [row["batch"] for row in rows] != list(range(1, BATCHES + 1)):
        raise RuntimeError("Expected 16 complete diagnostic batches")
    ids = [sample for row in rows for sample in row["sample_ids"]]
    if len(ids) != 64 or len(set(ids)) != 64 or any(len(row["sample_ids"]) != 4 for row in rows):
        raise RuntimeError("Wrong train sample count or duplicate images")
    if ids != report["identity"].get("sample_ids"):
        raise RuntimeError("Diagnostic did not use the first 64 frozen train IDs in order")
    names = report.get("trainable_parameter_names", [])
    if len(names) != 12 or len(set(names)) != 12:
        raise RuntimeError("Expected all twelve native depth-head parameter tensors")
    groups = {"all", *(parameter_group(name) for name in names)}
    for row in rows:
        if not re.fullmatch(r"[a-f0-9]{64}", row["input_sha256"]) or row["kd_pairs"]["Pedestrian"] != 0:
            raise RuntimeError("Invalid input fingerprint or forbidden KD class")
        if set(row["gradients"]) != groups or set(row["weighted_losses"]) != set(TERMS):
            raise RuntimeError("Missing component or head measurements")
        counts = row["kd_pairs"]
        if (set(counts) != {"Vehicle", "Pedestrian", "vehicle_near", "vehicle_far"}
                or any(not isinstance(value, int) or value < 0 for value in counts.values())
                or counts["Vehicle"] != counts["vehicle_near"] + counts["vehicle_far"]
                or counts["Vehicle"] > row["gt_counts"]["Vehicle"]):
            raise RuntimeError("Invalid KD eligibility coverage counts")
    return report


def summarize_reports(reports):
    if set(reports) != set(ROLES):
        raise RuntimeError("All three endpoint reports are required")
    for role, report in reports.items():
        validate_probe(report, role)
    first = reports["original"]
    fingerprints = [(row["sample_ids"], row["input_sha256"]) for row in first["rows"]]
    for report in reports.values():
        if (report["identity"] != first["identity"] or report["trainable_parameter_names"] != first["trainable_parameter_names"]
                or [(row["sample_ids"], row["input_sha256"]) for row in report["rows"]] != fingerprints):
            raise RuntimeError("Prospective roles used different identities, parameters or augmented inputs")
    rows = {}
    keys = ("kd_to_gt_norm_ratio", "kd_to_control_norm_ratio", "kd_gt_cosine",
            "kd_preservation_cosine", "kd_control_cosine", "total_deflection_degrees")
    for role, report in reports.items():
        active = [row for row in report["rows"] if row["kd_pairs"]["Vehicle"] > 0]
        def stats(key):
            values = [row["gradients"]["all"][key] for row in active if row["gradients"]["all"][key] is not None]
            return dict(count=len(values), median=statistics.median(values) if values else None,
                        minimum=min(values) if values else None, maximum=max(values) if values else None)
        cosines = [row["gradients"]["all"]["kd_control_cosine"] for row in active]
        defined = [value for value in cosines if value is not None]
        pairs = sum(row["kd_pairs"]["Vehicle"] for row in active)
        reach = {group: {key: dict(
            connected_tensors_max=max(row["gradients"][group]["reach"][key]["connected_tensors"] for row in report["rows"]),
            nonzero_tensors_max=max(row["gradients"][group]["reach"][key]["nonzero_tensors"] for row in report["rows"]))
            for key in TERMS} for group in report["rows"][0]["gradients"]}
        vehicle_gt = sum(row["gt_counts"]["Vehicle"] for row in report["rows"])
        rows[role] = dict(eligible_vehicle_pairs=pairs, active_batches=len(active), total_batches=BATCHES,
            transformed_vehicle_gt=vehicle_gt,
            eligible_fraction_of_all_transformed_vehicle_gt=pairs / vehicle_gt if vehicle_gt else None,
            eligible_near_pairs=sum(row["kd_pairs"]["vehicle_near"] for row in report["rows"]),
            eligible_far_pairs=sum(row["kd_pairs"]["vehicle_far"] for row in report["rows"]),
            sufficient_diagnostic_coverage=pairs >= 8,
            active_batch_statistics={key: stats(key) for key in keys},
            active_batch_kd_control_opposing_rate=sum(value < 0 for value in defined) / len(defined) if defined else None,
            weighted_loss_means={key: statistics.mean(row["weighted_losses"][key] for row in report["rows"]) for key in TERMS},
            gradient_reach=reach)
    return signed(dict(schema_version=1, complete=True, revision=REVISION, identity=first["identity"],
        paired_augmented_inputs_identical=True, report_signatures={key: value["report_sha256"] for key, value in reports.items()},
        roles=rows, optimizer_steps=0, interpretation=NOTES,
        training_authorized=False, checkpoint_promotion_authorized=False, deployment_authorized=False,
        next_action="Review measured gradient strength, conflict, coverage and reach before proposing another bounded experiment"))


def bundle(output, manifest):
    """Include small reports/logs, never weights or prediction caches; support partial failure."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / "m66b_gradient_results.zip"
    temporary = path.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(output.rglob("*")):
            if source.is_file() and source.suffix in (".json", ".log"):
                archive.write(source, str(source.relative_to(output)))
        if manifest.is_file():
            archive.write(manifest, "historical_inputs/m66_manifest.json")
            m = q.read_json(manifest)
            for relative in ("m66_paired_gate.json", "m66_training_smoke.json", "control/training_summary.json", "kd/training_summary.json"):
                source = Path(m["output"]) / relative
                if source.is_file():
                    archive.write(source, "historical_inputs/" + relative)
    temporary.replace(path)
    print(f"Return: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest="stage", required=True)
    probe = stages.add_parser("probe")
    for name in ("manifest", "repo", "dataset-root", "report", "runtime-receipt"):
        probe.add_argument("--" + name, required=True, type=Path)
    probe.add_argument("--role", required=True, choices=ROLES)
    summary = stages.add_parser("summarize")
    summary.add_argument("--output", required=True, type=Path)
    pack = stages.add_parser("bundle")
    pack.add_argument("--output", required=True, type=Path)
    pack.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    if args.stage == "probe":
        run_probe(args)
    elif args.stage == "summarize":
        reports = {role: q.read_json(args.output / f"{role}_gradients.json") for role in ROLES}
        result = summarize_reports(reports)
        write_once(args.output / "m66b_gradient_comparison.json", result)
        print(q.signature(result), result["roles"], flush=True)
    else:
        bundle(args.output, args.manifest)


if __name__ == "__main__":
    main()
