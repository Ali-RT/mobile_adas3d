"""Train-only, zero-update GT/geometry-KD gradient balance diagnostic.

Two fixed states, same first32 reviewed TRAIN frames. Weighted, unclipped
L2 norms/cosines on identical trainable coordinates; no optimizer, training,
validation inference, coefficient sweep or automatic promotion.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import torch
import recover_a2_lite_geometry as geometry
import stage_a2_lite_data as staging
from models.a2_lite_recovery_loss import gt_loss, GT_WEIGHTS

engine = geometry.engine
REVISION = "A2-LITE-GEOMETRY-GRADIENTS-2026-10-14-r1"
REVIEWED = {
    "geometry_manifest.json": "c96d55eb413a89e94cdb191a6ab916409275632b067b41f6c3dc6b5b42b374a3",
    "geometry_comparison.json": "0a7d6cd32f2f01e18d9f765bdf05507b641020f492b5aa3a85653d38e2671448",
    "kd/training_complete.json": "d2ce2f3b8a663b42ec03960ac37bd882e09d2d8817efd015a0c576398f80846f",
    "teacher_cache_complete.json": "4e6db19ce2b1f179c72872f934899453d1eec9af77aedb2fcbed10778111ca3c",
}
GEOMETRY_SIGNATURE = "d4ddaffd796f858507f1024de2b3bca35b01d1fe19db91adb4e948f60cd0cde5"
ENDPOINT_SHA = "9778f032e7ad949ad3b816bb266217d42ec4c4bb38311e0b607e33c8f13fcf18"
ROLES = ("initial", "geometry_epoch5")
GEOMETRY_TERMS = ("dimensions", "depth", "angle")
TERMS = ("gt_detection", "gt_geometry", *[f"{term}/{cls}" for term in GEOMETRY_TERMS for cls in ("Pedestrian", "Vehicle")])
DETECTION_TERMS = ("loss_ce", "loss_center", "loss_bbox", "loss_giou")
RECIPE = dict(frames=32, pedestrian_containing_frames=16, vehicle_only_frames=16,
              sample_selection="first32 reviewed TRAIN pilot IDs in unchanged order",
              roles=list(ROLES), kd_weight=.25, optimizer_steps=0,
              gradient_clipping_applied=False, adamw_preconditioning_applied=False,
              diagnostic_only=True, validation_used=False, automatic_pilot_authorization=False)


def signed(value, field="report_sha256"):
    if engine.signature({k: v for k, v in value.items() if k != field}) != value.get(field):
        raise RuntimeError("Changed diagnostic/historical record")
    return value


def reviewed(source):
    records = {}
    for name, digest in REVIEWED.items():
        raw = engine.read_record(source, name)
        if engine.accuracy.sha_bytes(raw) != digest:
            raise RuntimeError(f"Use the exact reviewed completed geometry run: {name}")
        records[name] = signed(json.loads(raw), "signature_sha256" if name == "geometry_manifest.json" else "report_sha256")
    m = records["geometry_manifest.json"]
    if (m["signature_sha256"] != GEOMETRY_SIGNATURE or m["implementation"] != geometry.implementation()
            or m["recipe"] != geometry.RECIPE or records["geometry_comparison.json"]["geometry_benefit_guardrail_passed"] is not False
            or records["kd/training_complete.json"]["checkpoint_sha256"] != ENDPOINT_SHA):
        raise RuntimeError("Reviewed geometry recipe/source/endpoint differs")
    for name, value in records.items():
        if name != "geometry_manifest.json" and value["run_signature"] != GEOMETRY_SIGNATURE:
            raise RuntimeError("Historical run identities differ")
    return m, records


def teacher_row(source, historical, coverage, sample):
    raw = engine.read_record(source, f"teacher_cache/{sample}.json")
    if engine.accuracy.sha_bytes(raw) != coverage["cache_sha256"][sample]:
        raise RuntimeError("Reviewed teacher cache bytes changed")
    row = signed(json.loads(raw), "record_sha256")
    if (row["sample_id"] != sample or row["run_signature"] != GEOMETRY_SIGNATURE
            or row["source_sha256"] != historical["data_file_sha256"][sample]
            or row["reused_from_run_signature"] != geometry.reference.PILOT_SIGNATURE):
        raise RuntimeError("Wrong original teacher guidance provenance")
    return raw, row


def selected_ids(historical, rows):
    ids = historical["pilot_ids"][:RECIPE["frames"]]
    if len(ids) != 32 or len(set(ids)) != 32 or not set(ids) <= set(historical["train_ids"]) or set(ids) & set(historical["val_ids"]):
        raise RuntimeError("Diagnostic must use the fixed32 reviewed TRAIN frames only")
    ped = sum(0 in rows[s]["target_labels"] for s in ids)
    vehicle_only = sum(0 not in rows[s]["target_labels"] and 1 in rows[s]["target_labels"] for s in ids)
    if ped != 16 or vehicle_only != 16:
        raise RuntimeError("Expected16 pedestrian-containing and16 vehicle-only encoded frames")
    return list(ids)


def implementation():
    return {**geometry.implementation(), "scripts/audit_a2_lite_gradient_balance.py": engine.sha256(__file__),
            "scripts/stage_a2_lite_data.py": engine.sha256(ROOT / "scripts/stage_a2_lite_data.py")}


def prepare(a):
    historical, records = reviewed(a.geometry)
    geometry.require_reference_environment(historical["cpu_environment"], engine.runtime_environment())
    if a.geometry.is_dir() and a.output.is_relative_to(a.geometry):
        raise RuntimeError("Use a separate diagnostic output; completed geometry run is read-only")
    if a.weights_dir.is_relative_to(Path("/content/drive")):
        raise RuntimeError("Copy diagnostic weight inputs onto local disk, not Google Drive")
    expected_weights = {"initial": engine.resolution.CHECKPOINT_SHA, "geometry_epoch5": ENDPOINT_SHA}
    sources = {"initial": a.initialization, "geometry_epoch5": a.checkpoint}
    for role, path in sources.items():
        if not path.is_file() or engine.sha256(path) != expected_weights[role]:
            raise RuntimeError(f"Missing/changed {role} checkpoint. Results ZIPs do not contain weights: {path}")
    coverage = records["teacher_cache_complete.json"]
    rows, raw_rows = {}, {}
    for sample in historical["pilot_ids"][:32]:
        raw_rows[sample], rows[sample] = teacher_row(a.geometry, historical, coverage, sample)
    ids = selected_ids(historical, rows)
    engine.base.prepare_repo(a.repo)
    if engine.base.q.source_hash(a.repo) != historical["source_sha256"]:
        raise RuntimeError("Fresh portable model source differs from reviewed student")
    expected_data = {s: historical["data_file_sha256"][s] for s in ids}
    staging.stage(a.dataset, a.candidate, ids, expected_data, workers=4)
    engine.accuracy.immutable_bytes(a.dataset / "ImageSets/train.txt", ("\n".join(ids) + "\n").encode())
    weights = {}
    a.weights_dir.mkdir(parents=True, exist_ok=True)
    for role, path in sources.items():
        destination = a.weights_dir / f"{role}.pth"
        if destination.exists():
            if engine.sha256(destination) != expected_weights[role]: raise RuntimeError("Preserve conflicting local weights")
        else:
            raw = staging.retry_read(path)
            if engine.accuracy.sha_bytes(raw) != expected_weights[role]: raise RuntimeError("Weight source changed while copying")
            staging.atomic_copy(destination, raw)
        weights[role] = dict(path=str(destination), sha256=expected_weights[role])
    for name in REVIEWED:
        engine.accuracy.immutable_bytes(a.output / "reviewed_geometry" / name, engine.read_record(a.geometry, name))
    for sample in ids:
        engine.accuracy.immutable_bytes(a.output / "teacher_subset" / f"{sample}.json", raw_rows[sample])
    view = dict(config=copy.deepcopy(historical["config"]), source=str(a.repo), dataset=str(a.dataset),
                initialization=weights["initial"]["path"], data_file_sha256=expected_data,
                train_ids=ids, val_ids=[], pilot_ids=ids)
    m = dict(revision=REVISION, recipe=RECIPE, implementation=implementation(),
             cpu_environment=engine.runtime_environment(), historical_run_signature=GEOMETRY_SIGNATURE,
             historical_source_sha256=historical["source_sha256"], view=view, weights=weights,
             source_weight_paths={k: str(v) for k, v in sources.items()}, output=str(a.output),
             subset_sha256=engine.signature(ids), teacher_record_sha256={s: engine.accuracy.sha_bytes(raw_rows[s]) for s in ids},
             staged_train_frames=32, staged_validation_frames=0, training_authorized=False)
    m["signature_sha256"] = engine.signature(m)
    engine.accuracy.immutable_json(a.output / "gradient_manifest.json", m)
    print("Prepared32 TRAIN frames and two read-only weight states. No training or validation started.", flush=True)


def load(output):
    m = signed(json.loads((output / "gradient_manifest.json").read_text()), "signature_sha256")
    if (m["revision"] != REVISION or m["recipe"] != RECIPE or m["implementation"] != implementation()
            or m["output"] != str(output) or m["cpu_environment"] != engine.runtime_environment()):
        raise RuntimeError("Diagnostic code/packages/paths changed; preserve output and use a new diagnostic RUN_ID")
    historical, records = reviewed(output / "reviewed_geometry")
    if engine.base.q.source_hash(Path(m["view"]["source"])) != m["historical_source_sha256"]:
        raise RuntimeError("Diagnostic portable source changed")
    rows = {}
    for sample in m["view"]["pilot_ids"]:
        raw = (output / "teacher_subset" / f"{sample}.json").read_bytes()
        if engine.accuracy.sha_bytes(raw) != m["teacher_record_sha256"][sample]: raise RuntimeError("Diagnostic teacher input changed")
        rows[sample] = signed(json.loads(raw), "record_sha256")
        if rows[sample]["run_signature"] != GEOMETRY_SIGNATURE or rows[sample]["source_sha256"] != m["view"]["data_file_sha256"][sample]:
            raise RuntimeError("Diagnostic teacher/source identity differs")
    if selected_ids(historical, rows) != m["view"]["pilot_ids"]:
        raise RuntimeError("Diagnostic training subset changed")
    for value in m["weights"].values():
        if engine.sha256(value["path"]) != value["sha256"]: raise RuntimeError("Diagnostic weights changed")
    return m


def parameter_groups(model):
    owners = {}
    for attr, group in (("class_embed", "classification_head"), ("bbox_embed", "box_head"),
                        ("dim_embed_3d", "dimensions_head"), ("depth_embed", "depth_head"), ("angle_embed", "angle_head")):
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters(): owners[id(p)] = group
    names, parameters, groups = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        if id(p) in owners: group = owners[id(p)]
        elif name.startswith("backbone."): group = "shared_backbone"
        elif name.startswith("depth_predictor."): group = "shared_depth_predictor"
        elif name.startswith("depthaware_transformer."): group = "shared_transformer"
        else: group = "shared_other"
        names.append(name); parameters.append(p); groups.append(group)
    if not names: raise RuntimeError("No trainable diagnostic coordinates")
    return names, parameters, groups


def losses(outputs, teacher, target, mapping):
    total, parts, _ = gt_loss(outputs, target)
    detection = sum(parts[k] * GT_WEIGHTS[k] for k in DETECTION_TERMS)
    geometry_gt = sum(parts[k] * GT_WEIGHTS[k] for k in GT_WEIGHTS if k not in DETECTION_TERMS)
    _, combined, counts = geometry.geometry_kd_loss(outputs, teacher, target, mapping)
    values = dict(gt_detection=detection, gt_geometry=geometry_gt)
    for cls, label in (("Pedestrian", 0), ("Vehicle", 1)):
        filtered = mapping.clone(); filtered[target["labels"] != label] = -1
        _, per_class, _ = geometry.geometry_kd_loss(outputs, teacher, target, filtered)
        for term in GEOMETRY_TERMS:
            present = sum(n > 0 for n in counts[term].values())
            values[f"{term}/{cls}"] = per_class[term] * (RECIPE["kd_weight"] / max(present, 1))
    for term in GEOMETRY_TERMS:
        reconstructed = sum(values[f"{term}/{cls}"] for cls in ("Pedestrian", "Vehicle"))
        if not torch.allclose(reconstructed, combined[term] * RECIPE["kd_weight"], atol=1e-6, rtol=1e-5):
            raise RuntimeError("Class decomposition changed the historical class-balanced teacher loss")
    if not torch.allclose(detection + geometry_gt, total, atol=1e-5, rtol=1e-5):
        raise RuntimeError("GT decomposition changed the training loss")
    return values, counts


def gram_by_group(groups, gradients):
    """Full-vector dot products on identical coordinates, never mean cosines."""
    if set(gradients) != set(TERMS) or any(len(v) != len(groups) for v in gradients.values()):
        raise ValueError("All eight weighted gradient terms and coordinate groups required")
    matrices = {}
    for index, group in enumerate(groups):
        values = [gradients[key][index] for key in TERMS]
        connected = [v for v in values if v is not None]
        if not connected: continue
        if any(v.shape != connected[0].shape or not torch.isfinite(v).all() for v in connected):
            raise RuntimeError("Nonfinite or inconsistent gradient coordinates")
        stacked = torch.stack([v.detach().reshape(-1).double() if v is not None else torch.zeros_like(connected[0]).reshape(-1).double() for v in values])
        gram = stacked @ stacked.T
        tags = ("all", group, "shared") if group.startswith("shared_") else ("all", group)
        for tag in tags:
            if tag not in matrices: matrices[tag] = torch.zeros_like(gram)
            matrices[tag] += gram
    return {k: v.cpu().tolist() for k, v in matrices.items()}


def measurements(matrix):
    n = len(TERMS)
    if len(matrix) != n or any(len(row) != n for row in matrix) or any(not math.isfinite(v) for row in matrix for v in row):
        raise ValueError("Invalid gradient Gram matrix")
    vectors = {"gt": [0, 1], "gt_detection": [0], "gt_geometry": [1], "kd_all": list(range(2, n))}
    for term in GEOMETRY_TERMS:
        vectors[term] = [i for i, name in enumerate(TERMS) if name.startswith(term + "/")]
        for cls in ("Pedestrian", "Vehicle"):
            vectors[f"{term}/{cls}"] = [TERMS.index(f"{term}/{cls}")]
    def dot(a, b): return sum(matrix[i][j] for i in vectors[a] for j in vectors[b])
    norms = {k: math.sqrt(max(0., dot(k, k))) for k in vectors}
    def cosine(a, b):
        return max(-1., min(1., dot(a, b) / (norms[a] * norms[b]))) if norms[a] > 0 and norms[b] > 0 else None
    result = dict(norms=norms, teacher={})
    for name in ("kd_all", *GEOMETRY_TERMS, *[f"{term}/{cls}" for term in GEOMETRY_TERMS for cls in ("Pedestrian", "Vehicle")]):
        result["teacher"][name] = dict(l2_norm=norms[name], kd_to_gt_norm_ratio=norms[name] / norms["gt"] if norms["gt"] > 0 else None,
                                         gt_cosine=cosine(name, "gt"), detection_gt_cosine=cosine(name, "gt_detection"))
    total2 = norms["gt"] ** 2 + norms["kd_all"] ** 2 + 2 * dot("gt", "kd_all")
    result["gt_plus_kd_norm"] = math.sqrt(max(0., total2))
    result["hypothetical_global_clip_multiplier"] = min(1., engine.RECIPE["clip_gradient_norm"] / max(result["gt_plus_kd_norm"], 1e-12))
    return result


def model_hash(model):
    return engine.signature({name: engine.sha_bytes_tensor(value) for name, value in model.state_dict().items()})


def summarize(frames):
    result = []
    groups = sorted({g for frame in frames for g in frame["gradients"]})
    for group in groups:
        for term in ("kd_all", *GEOMETRY_TERMS, *[f"{t}/{cls}" for t in GEOMETRY_TERMS for cls in ("Pedestrian", "Vehicle")]):
            rows = [f["gradients"][group]["teacher"][term] for f in frames if group in f["gradients"]]
            active = [r for r in rows if r["l2_norm"] > 0]
            ratios = [r["kd_to_gt_norm_ratio"] for r in active if r["kd_to_gt_norm_ratio"] is not None]
            cos = [r["gt_cosine"] for r in active if r["gt_cosine"] is not None]
            detection = [r["detection_gt_cosine"] for r in active if r["detection_gt_cosine"] is not None]
            result.append(dict(parameter_group=group, term=term, measured_frames=len(rows), active_frames=len(active),
                               median_kd_to_gt_norm_ratio=statistics.median(ratios) if ratios else None,
                               median_gt_cosine=statistics.median(cos) if cos else None,
                               negative_gt_cosine_fraction=sum(v < -.1 for v in cos) / len(cos) if cos else None,
                               median_detection_gt_cosine=statistics.median(detection) if detection else None,
                               negative_detection_cosine_fraction=sum(v < -.1 for v in detection) / len(detection) if detection else None,
                               dominance_signal_fraction=sum(v > 1 for v in ratios) / len(ratios) if ratios else None))
    return result


def probe(a, m):
    engine.setup_device(a.device); view = m["view"]
    dataset = engine.native_dataset(view, view["pilot_ids"])
    model = engine.build_model(view, a.device, True)
    if a.role == "geometry_epoch5":
        saved = torch.load(m["weights"][a.role]["path"], map_location="cpu", weights_only=True)
        if saved.get("run_signature") != GEOMETRY_SIGNATURE or saved.get("role") != "kd" or saved.get("step") != 2560:
            raise RuntimeError("Wrong completed geometry endpoint payload")
        model.load_state_dict(saved["model_state"], strict=True)
        del saved
    before = model_hash(model); names, parameters, groups = parameter_groups(model); frames = []
    for index, sample in enumerate(view["pilot_ids"]):
        engine.verify_files(view, sample)
        row = json.loads((a.output / "teacher_subset" / f"{sample}.json").read_text())
        inputs, target, _ = engine.item(dataset, index, a.device)
        if row["target_signature"] != engine.target_signature(target): raise RuntimeError("Native GT encoding differs from reviewed teacher record")
        destination = a.output / "frames" / a.role / f"{sample}.json"
        if destination.exists():
            report = engine.checked_report(destination, m)
            if report["sample_id"] != sample or report["role"] != a.role or report["model_state_sha256"] != before:
                raise RuntimeError("Changed resumed diagnostic frame")
        else:
            teacher = engine.native_arrays(row["outputs"], a.device)
            mapping = torch.tensor(row["object_to_teacher_query"], device=a.device)
            values, counts = losses(engine.forward(model, inputs), teacher, target, mapping)
            if not all(torch.isfinite(v) for v in values.values()): raise RuntimeError("Nonfinite diagnostic loss")
            gradients = {name: torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
                         if loss.requires_grad else [None] * len(parameters) for name, loss in values.items()}
            gram = gram_by_group(groups, gradients)
            if "all" not in gram or measurements(gram["all"])["norms"]["gt"] <= 0:
                raise RuntimeError("Ground-truth gradient is missing/zero; diagnostic cannot compare teacher balance")
            report = dict(complete=True, run_signature=m["signature_sha256"], sample_id=sample, role=a.role,
                          model_state_sha256=before, weighted_losses={k: float(v.detach()) for k, v in values.items()},
                          active_counts=counts, gt_counts={cls: int((target["labels"] == label).sum()) for cls, label in (("Pedestrian", 0), ("Vehicle", 1))},
                          gradients={group: measurements(matrix) for group, matrix in gram.items()}, optimizer_steps=0)
            engine.save_report(destination, report)
            del gradients, values
        frames.append(report)
        print(f"{a.role}: measured/verified {index + 1}/32 TRAIN frames; zero updates", flush=True)
    after = model_hash(model)
    if before != after or any(p.grad is not None for p in parameters):
        raise RuntimeError("Model state or .grad fields changed during zero-update diagnostic")
    report = dict(complete=True, run_signature=m["signature_sha256"], revision=REVISION, role=a.role,
                  frames=32, optimizer_steps=0, weights_and_buffers_unchanged=True,
                  state_sha256=before, trainable_parameters=sum(p.numel() for p in parameters),
                  parameter_names_sha256=engine.signature(names), summary=summarize(frames),
                  frame_sha256={s: engine.sha256(a.output / "frames" / a.role / f"{s}.json") for s in view["pilot_ids"]},
                  limitations=["Weighted unclipped L2 gradients, not AdamW-preconditioned updates or a historical replay.",
                               "Class rows are actual class-balanced KD contributions; GT reference remains all targets in that frame.",
                               "Opposing/dominant gradients are local diagnostic signals, not causal proof of AP/recall changes.",
                               "No validation inference, weight selection, new pilot or model promotion is authorized."])
    engine.save_report(a.output / f"{a.role}_gradient_report.json", report)
    print(json.dumps([r for r in report["summary"] if r["parameter_group"] in ("all", "shared") and r["term"] in GEOMETRY_TERMS], indent=2), flush=True)


def bundle(a):
    path = a.output / "a2_lite_gradient_results.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(a.output.rglob("*")):
            if source.is_file() and source.suffix in (".json", ".log", ".txt"):
                archive.write(source, source.relative_to(a.output))
    print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "probe", "bundle"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--role", choices=ROLES)
    for name in ("geometry", "initialization", "checkpoint", "repo", "dataset", "weights-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    a = parser.parse_args()
    for key in ("output", "geometry", "initialization", "checkpoint", "repo", "dataset", "weights_dir"):
        if getattr(a, key) is not None: setattr(a, key, getattr(a, key).resolve())
    a.candidate = [p.resolve() for p in a.candidate]; a.output.mkdir(parents=True, exist_ok=True)
    if a.stage == "prepare" and any(getattr(a, k) is None for k in ("geometry", "initialization", "checkpoint", "repo", "dataset", "weights_dir")):
        parser.error("prepare requires reviewed geometry results, both weight states, source and local paths")
    if a.stage == "probe" and a.role is None: parser.error("probe requires --role")
    try:
        if a.stage == "prepare": prepare(a)
        elif a.stage == "bundle": bundle(a)
        else: probe(a, load(a.output))
    except Exception as error:
        import traceback
        engine.write_json(a.output / "failures" / f"{a.stage}_{time.time_ns()}.json", dict(complete=False, error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__": main()
