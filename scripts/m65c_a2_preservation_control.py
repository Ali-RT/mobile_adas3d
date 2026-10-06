"""M65c: one scale-normalized, train-calibrated A2 preservation control."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
import zipfile

# Absolute script launches must resolve repository packages before importing them.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import m65_a2_preservation_control as c
import m64_teacher_qualification as q
from third_party.monodetr.m65c_preservation import RULES, SCALES, WEIGHTS, preservation

REVISION = "M65C-A2-SCALED-PRESERVATION-2026-10-06-r1"
REVIEW = ROOT / "artifacts/m65b_gradient_review_20261006.json"
REVIEW_SHA = "a4a2f78793e292e793bd249380aaf4ae997ce8374bc259d3d38499bb0edc3c45"
RECIPE = dict(c.RECIPE, preservation_weight="training-only calibrated scalar")
CALIBRATION = dict(samples=64, pedestrian_images=32, vehicle_only_images=32,
                   batch_size=4, seed=444, target_gradient_ratio=0.25,
                   minimum_coefficient=0.01, maximum_coefficient=100.0,
                   minimum_pairs_per_class=8, optimizer_steps=0)
INITIAL_LOSS_TOLERANCE = 1e-3


def implementation_hash():
    paths = [Path(__file__), ROOT / "third_party/monodetr/m65c_preservation.py"]
    return q.signature(dict(files={str(p.relative_to(ROOT)): q.sha(p) for p in paths},
                            reused_frozen_m65_implementation=c.implementation_hash()))


def reviewed_inputs(manifest, gradient_output):
    if q.sha(REVIEW) != REVIEW_SHA:
        raise RuntimeError("Reviewed M65b evidence changed")
    review = q.read_json(REVIEW)
    old = q.read_json(manifest)
    signature = old.get("manifest_sha256")
    if (q.signature({k: v for k, v in old.items() if k != "manifest_sha256"}) != signature
            or signature != review["identity"]["historical_manifest_sha256"]
            or old["implementation_sha256"] != c.implementation_hash()
            or old["implementation_sha256"] != review["identity"]["implementation_sha256"]
            or old["recipe"] != c.RECIPE or old["kd_authorized"] is not False):
        raise RuntimeError("Use the exact reviewed, unchanged M65 manifest and implementation")
    for name, digest in review["archive_files"].items():
        if q.sha(gradient_output / name) != digest:
            raise RuntimeError(f"Reviewed M65b file missing or changed: {name}")
    if old["data_identity"] != review["identity"]["dataset_identity"]:
        raise RuntimeError("Reviewed M65 data identity differs")
    endpoint = Path(old["output"]) / "control/checkpoint_epoch_1.pth"
    if q.sha(endpoint) != review["identity"]["control_checkpoint_sha256"]:
        raise RuntimeError("Reviewed M65 calibration endpoint missing or changed")
    payload = q.safe_payload(endpoint)
    if payload.get("m65_manifest_sha256") != signature or payload.get("epoch") != 1:
        raise RuntimeError("M65 calibration checkpoint lineage differs")
    c.validate_training_summary(old, payload.get("training_summary", {}))
    return old, review, endpoint


def calibration_ids(train_ids, label_directory):
    """Select via raw training labels only; not model errors or validation labels."""
    pedestrian, vehicle_only = [], []
    if len(set(train_ids)) != len(train_ids):
        raise RuntimeError("Duplicate train IDs")
    for sample in train_ids:
        classes = {line.split()[0] for line in (label_directory / f"{sample}.txt").read_text().splitlines()
                   if line.strip()}
        if classes & {"Pedestrian", "Person_sitting"}:
            pedestrian.append(sample)
        elif classes & {"Car", "Van", "Truck", "Tram"}:
            vehicle_only.append(sample)
    rng = random.Random(CALIBRATION["seed"])
    rng.shuffle(pedestrian)
    rng.shuffle(vehicle_only)
    if len(pedestrian) < 32 or len(vehicle_only) < 32:
        raise RuntimeError("Need 32 Pedestrian-containing and 32 Vehicle-only train images")
    selected = []
    for offset in range(0, 32, 2):
        selected += pedestrian[offset:offset + 2] + vehicle_only[offset:offset + 2]
    return selected


def prepare(a):
    old, review, endpoint = reviewed_inputs(a.m65_manifest, a.m65b_output)
    output, repo, dataset = a.output.resolve(), a.a2_repo.resolve(), a.dataset_root.resolve()
    if not repo.name.startswith("MonoDETR_M65C"):
        raise RuntimeError("Use a fresh MonoDETR_M65C checkout; preserve historical sources")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != q.A2_COMMIT:
        raise RuntimeError("Wrong A2 upstream pin")
    entry = copy.deepcopy(old["models"]["a2"])
    if entry["config"]["dataset"].get("meanshape") is not False:
        raise RuntimeError("Expected physical H/W/L targets")
    if q.sha(entry["checkpoint"]) != q.A2_SHA or q.source_hash(repo) != entry["patched_source_sha256"]:
        raise RuntimeError("Original A2 weights or repaired inference source changed")
    build_path = repo / "lib/models/monodetr/ops/m64_build_receipt.json"
    build = q.read_json(build_path)
    if not build.get("import_verified") or q.sha(build["binary"]) != build["binary_sha256"]:
        raise RuntimeError("Missing or changed locally built attention binary")
    identity = c.data_identity(dataset)
    if identity != old["data_identity"]:
        raise RuntimeError("KITTI labels/calibration/splits differ from reviewed M65")
    env = q.environment()
    if env["gpu"] is None or env["cuda"] != "13.0":
        raise RuntimeError("Use the isolated CUDA13 runtime with an allocated GPU")
    entry.update(repo=str(repo), attention_build_receipt=str(build_path),
                 attention_build_sha256=q.sha(build_path), attention_binary=build["binary"],
                 attention_binary_sha256=build["binary_sha256"])
    entry["config"]["dataset"]["root_dir"] = str(dataset)
    training = copy.deepcopy(old["training_config"])
    training["dataset"]["root_dir"] = str(dataset)
    ids = calibration_ids(q.split_ids(dataset / "ImageSets/train.txt", "train"),
                          dataset / "training/label_2")
    m = dict(schema_version=1, revision=REVISION, output=str(output), dataset_root=str(dataset),
             models=dict(a2=entry), training_config=training, data_identity=identity,
             environment=env, implementation_sha256=implementation_hash(), recipe=RECIPE,
             preservation_rules=RULES, preservation_scales=SCALES, preservation_weights=WEIGHTS,
             limits=c.CONTROL_LIMITS, calibration_recipe=CALIBRATION, calibration_ids=ids,
             m65_manifest=str(a.m65_manifest.resolve()), m65_manifest_file_sha256=q.sha(a.m65_manifest),
             m65b_output=str(a.m65b_output.resolve()), reviewed_m65b_sha256=REVIEW_SHA,
             calibration_endpoint=str(endpoint), calibration_endpoint_sha256=q.sha(endpoint),
             reviewed_a2_metrics=old["reviewed_a2_metrics"], reviewed_a2_nearby=old["reviewed_a2_nearby"],
             product_score_threshold=0.001, topk=50, kd_authorized=False,
             external_teacher_selected=False, full_training_authorized=False,
             deployment_authorized=False, baseline_checkpoint_overwritten=False)
    m["manifest_sha256"] = q.signature(m)
    path = output / "m65c_manifest.json"
    if path.exists() and q.read_json(path) != m:
        raise RuntimeError("M65c identity changed; preserve this run and choose a new RUN_ID")
    q.write_json(path, m)
    print(json.dumps(m, indent=2), flush=True)


def load(path):
    m = q.read_json(path)
    if (m.get("revision") != REVISION
            or q.signature({k: v for k, v in m.items() if k != "manifest_sha256"}) != m["manifest_sha256"]):
        raise RuntimeError("M65c manifest changed")
    expected = dict(recipe=RECIPE, preservation_rules=RULES, preservation_scales=SCALES,
                    preservation_weights=WEIGHTS, limits=c.CONTROL_LIMITS,
                    calibration_recipe=CALIBRATION, kd_authorized=False)
    if any(m.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Fixed M65c recipe changed or external KD enabled")
    if m["implementation_sha256"] != implementation_hash() or m["environment"] != q.environment():
        raise RuntimeError("Source/runtime changed; use a new RUN_ID, do not rewrite evidence")
    if q.sha(m["m65_manifest"]) != m["m65_manifest_file_sha256"]:
        raise RuntimeError("Historical M65 manifest changed")
    old, _, endpoint = reviewed_inputs(Path(m["m65_manifest"]), Path(m["m65b_output"]))
    if str(endpoint) != m["calibration_endpoint"] or q.sha(endpoint) != m["calibration_endpoint_sha256"]:
        raise RuntimeError("Calibration endpoint changed")
    entry = m["models"]["a2"]
    if q.sha(entry["checkpoint"]) != q.A2_SHA or q.source_hash(Path(entry["repo"])) != entry["patched_source_sha256"]:
        raise RuntimeError("Original A2 identity changed")
    if q.sha(entry["attention_build_receipt"]) != entry["attention_build_sha256"] or q.sha(entry["attention_binary"]) != entry["attention_binary_sha256"]:
        raise RuntimeError("Attention build changed")
    dataset = Path(m["dataset_root"])
    if c.data_identity(dataset) != old["data_identity"] or m["data_identity"] != old["data_identity"]:
        raise RuntimeError("KITTI identity changed")
    ids = calibration_ids(q.split_ids(dataset / "ImageSets/train.txt", "train"), dataset / "training/label_2")
    if m["calibration_ids"] != ids:
        raise RuntimeError("Train-only calibration selection changed")
    return m


def losses(model, anchor, criterion, batch, coefficient):
    import torch
    images, calibs, raw, _ = batch
    images, calibs, sizes = images.cuda(), calibs.cuda(), raw["img_size"].cuda()
    targets = c.gt_targets(raw)
    c.training_mode(model)
    parts = criterion(model(images, calibs, targets, sizes, dn_args=None), targets, None)
    gt = sum(value * criterion.weight_dict[key] for key, value in parts.items() if key in criterion.weight_dict)
    with torch.no_grad():
        reference = anchor(images, calibs, None, sizes, dn_args=0)
    model.eval()
    outputs = model(images, calibs, None, sizes, dn_args=0)
    keep, counts = preservation(outputs, reference, targets, criterion.matcher)
    c.training_mode(model)
    total = gt + coefficient * keep["total"]
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Non-finite GT/preservation loss; no update taken")
    return total, gt, keep, counts, outputs, reference, targets


def baseline(m):
    result = c.validated_metrics(m, "baseline")
    checks = {key: abs(result["product"][key] - value) <= 0.15 for key, value in m["reviewed_a2_metrics"].items()}
    checks.update({key + "_nearby": abs(result["nearby"][key] - value) <= 0.01
                   for key, value in m["reviewed_a2_nearby"].items()})
    report = dict(complete=True, manifest_sha256=m["manifest_sha256"], checks=checks,
                   passed=all(checks.values()), optimizer_steps=0)
    q.write_json(Path(m["output"]) / "m65c_baseline_gate.json", report)
    if not report["passed"]:
        raise RuntimeError("Original A2 does not reproduce; stop before calibration/training")
    return result


def evaluate(m, role):
    view = c.evaluation_view(m, role)
    predictions = q.validate_predictions(view, "a2") / "product"
    output = Path(view["output"]) / "metrics"
    shared = ["--profile", "colab_drive", "--dataset-root", m["dataset_root"],
              "--split-dir", Path(m["dataset_root"]) / "ImageSets", "--prediction-dir", predictions,
              "--split", "val", "--source-name", f"M65c_{role}"]
    q.run([sys.executable, "-u", ROOT / "scripts/evaluate_kitti_prediction_dir.py",
           "--config", ROOT / "configs/kitti_mobileadas3d_s1.yaml", *shared,
           "--classes", "Vehicle", "Pedestrian", "--output-dir", output / "product"])
    q.run([sys.executable, "-u", ROOT / "scripts/audit_product_prediction_geometry.py",
           "--dataset-root", m["dataset_root"], "--split-file", Path(m["dataset_root"]) / "ImageSets/val.txt",
           "--prediction-dir", predictions, "--output-dir", output / "nearby",
           "--checkpoint", view["models"]["a2"]["checkpoint"], "--expected-images", "3769",
           "--score-threshold", "0.001", "--match-iou-threshold", "0.5"])
    summary = dict(complete=True, manifest_sha256=m["manifest_sha256"], role=role,
                   evaluation_identity=view["manifest_sha256"],
                   product=q.metric_rows(q.read_json(output / "product/kitti_r40_summary.json")),
                   nearby={k: v["near_recall"] for k, v in q.read_json(output / "nearby/nearby_geometry_summary.json")["classes"].items()})
    q.write_json(output / "summary.json", summary)


def gradient_norm(grads):
    import torch
    squared = 0.0
    for gradient in grads:
        if gradient is not None:
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("Non-finite calibration gradient")
            squared += float(gradient.detach().double().square().sum())
    return math.sqrt(squared)


def choose_coefficient(rows, counts):
    if len(rows) != CALIBRATION["samples"] // CALIBRATION["batch_size"]:
        raise RuntimeError("Need all16 calibration batches")
    for component in ("classification", "depth", "uncertainty"):
        for cls in ("0", "1"):
            if counts[component][cls] < CALIBRATION["minimum_pairs_per_class"]:
                raise RuntimeError(f"Insufficient {component} calibration coverage for class{cls}")
    ratios = []
    for row in rows:
        gt, keep = row["gt_norm"], row["preservation_norm"]
        if not (math.isfinite(gt) and math.isfinite(keep) and gt > 0 and keep > 0):
            raise RuntimeError("Invalid/zero calibration gradient; no automatic fallback")
        ratios.append(keep / gt)
    median = statistics.median(ratios)
    coefficient = CALIBRATION["target_gradient_ratio"] / median
    if not (CALIBRATION["minimum_coefficient"] <= coefficient <= CALIBRATION["maximum_coefficient"]):
        raise RuntimeError(f"Calibrated coefficient {coefficient} outside fixed safe range; stop")
    return coefficient, median


def validated_calibration(m):
    report = q.read_json(Path(m["output"]) / "m65c_calibration.json")
    expected = dict(complete=True, manifest_sha256=m["manifest_sha256"], optimizer_steps=0,
                    parameters_unchanged=True, buffers_unchanged=True, anchor_unchanged=True,
                    calibration_ids=m["calibration_ids"], calibration_endpoint_sha256=m["calibration_endpoint_sha256"])
    if any(report.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Missing or changed calibration certification")
    if q.signature({k: v for k, v in report.items() if k != "report_sha256"}) != report["report_sha256"]:
        raise RuntimeError("Calibration receipt changed")
    coefficient, median = choose_coefficient(report["rows"], report["eligible_pairs"])
    if report["coefficient"] != coefficient or report["median_raw_gradient_ratio"] != median:
        raise RuntimeError("Calibration scalar changed")
    recorded_ids = [sample for row in report["rows"] for sample in row["image_ids"]]
    if recorded_ids != m["calibration_ids"]:
        raise RuntimeError("Calibration rows differ from frozen selection")
    return report


def calibrate(m):
    import torch
    baseline(m)
    if (Path(m["output"]) / "m65c_calibration.json").exists():
        report = validated_calibration(m)
        print(f"Calibration already certified, coefficient={report['coefficient']}; zero updates.", flush=True)
        return
    c.seed_all(RECIPE["seed"])
    model, anchor, criterion, dataset = c.training_runtime(m)
    # This fixed endpoint is used ONLY to measure a nonzero preservation signal.
    model.load_state_dict(q.safe_payload(m["calibration_endpoint"])["model_state"], strict=True)
    before, buffers, anchor_before = (q.tensor_hash(model.state_dict()),
                                      q.tensor_hash(dict(model.named_buffers())), q.tensor_hash(anchor.state_dict()))
    train_ids = q.split_ids(Path(m["dataset_root"]) / "ImageSets/train.txt", "train")
    subset = torch.utils.data.Subset(dataset, [train_ids.index(x) for x in m["calibration_ids"]])
    loader = torch.utils.data.DataLoader(subset, batch_size=4, num_workers=0, shuffle=False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    rows, counts = [], {key: {"0": 0, "1": 0} for key in WEIGHTS}
    for batch in loader:
        total, gt, keep, pairs, outputs, reference, targets = losses(model, anchor, criterion, batch, 1.0)
        gt_norm = gradient_norm(torch.autograd.grad(gt, parameters, retain_graph=True, allow_unused=True))
        keep_norm = gradient_norm(torch.autograd.grad(keep["total"], parameters, allow_unused=True))
        ids = [f"{int(x):06d}" for x in batch[3]["img_id"]]
        rows.append(dict(image_ids=ids, gt_norm=gt_norm, preservation_norm=keep_norm,
                         gt_loss=float(gt.detach()), preservation_loss=float(keep["total"].detach())))
        for key in counts:
            for cls in (0, 1):
                counts[key][str(cls)] += pairs[key][cls]
        print(f"M65c calibration batch={len(rows):02d}/16 gt_norm={gt_norm:.6f} keep_norm={keep_norm:.6f} pairs={pairs}", flush=True)
        del total, gt, keep, outputs, reference, targets
    coefficient, median = choose_coefficient(rows, counts)
    report = dict(complete=True, manifest_sha256=m["manifest_sha256"], optimizer_steps=0,
                   calibration_ids=m["calibration_ids"], calibration_endpoint_sha256=m["calibration_endpoint_sha256"],
                   rows=rows, eligible_pairs=counts, coefficient=coefficient,
                   median_raw_gradient_ratio=median, target_gradient_ratio=CALIBRATION["target_gradient_ratio"],
                   parameters_unchanged=before == q.tensor_hash(model.state_dict()),
                   buffers_unchanged=buffers == q.tensor_hash(dict(model.named_buffers())),
                   anchor_unchanged=anchor_before == q.tensor_hash(anchor.state_dict()),
                   method="0.25 / median(per-batch preservation_norm / gt_norm); no clipping or weight search")
    if not all(report[k] for k in ("parameters_unchanged", "buffers_unchanged", "anchor_unchanged")):
        raise RuntimeError("Calibration modified model/anchor state; stop")
    report["report_sha256"] = q.signature(report)
    q.write_json(Path(m["output"]) / "m65c_calibration.json", report)
    validated_calibration(m)
    print(json.dumps(report, indent=2), flush=True)


def smoke(m):
    import torch
    baseline(m)
    calibration = validated_calibration(m)
    c.seed_all(RECIPE["seed"])
    model, anchor, criterion, dataset = c.training_runtime(m)
    before, buffers, anchor_before = (q.tensor_hash(model.state_dict()),
                                      q.tensor_hash(dict(model.named_buffers())), q.tensor_hash(anchor.state_dict()))
    ids = q.split_ids(Path(m["dataset_root"]) / "ImageSets/train.txt", "train")
    subset = torch.utils.data.Subset(dataset, [ids.index(x) for x in m["calibration_ids"]])
    loader = torch.utils.data.DataLoader(subset, batch_size=1, num_workers=0, shuffle=False)
    for batch in loader:
        total, gt, keep, counts, outputs, reference, targets = losses(model, anchor, criterion, batch, calibration["coefficient"])
        if sum(counts["uncertainty"].values()):
            break
        del total, gt, keep, outputs, reference, targets
    else:
        raise RuntimeError("No reliable depth/uncertainty pair in calibration train IDs")
    initial = float(keep["total"].detach())
    if initial > INITIAL_LOSS_TOLERANCE:
        raise RuntimeError("Original A2 copies disagree before updates")
    shift = torch.zeros_like(outputs["pred_depth"])
    shift[..., 1] = SCALES["log_uncertainty"]
    probe, _ = preservation(dict(outputs, pred_depth=outputs["pred_depth"] + shift), reference, targets, criterion.matcher)
    parameters = [p for p in model.parameters() if p.requires_grad]
    grads = torch.autograd.grad(probe["uncertainty"], parameters, retain_graph=True, allow_unused=True)
    uncertainty_ok = gradient_norm(grads) > 0
    total.backward()
    gt_ok = any(p.grad is not None for p in parameters) and all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in parameters)
    checks = dict(finite_nonzero_uncertainty_gradients=uncertainty_ok, finite_gt_gradients=gt_ok,
                  parameters_unchanged=before == q.tensor_hash(model.state_dict()),
                  buffers_unchanged=buffers == q.tensor_hash(dict(model.named_buffers())),
                  anchor_unchanged=anchor_before == q.tensor_hash(anchor.state_dict()))
    report = dict(complete=True, manifest_sha256=m["manifest_sha256"],
                   calibration_sha256=calibration["report_sha256"], optimizer_steps=0,
                   initial_preservation_loss=initial, checks=checks, passed=all(checks.values()),
                   external_kd_enabled=False, paired_counts=counts)
    q.write_json(Path(m["output"]) / "m65c_training_smoke.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("M65c CUDA smoke failed; stop before training")


def validate_summary(m, summary, calibration):
    c.validate_training_summary(m, summary)
    if (summary.get("calibration_sha256") != calibration["report_sha256"]
            or summary.get("preservation_coefficient") != calibration["coefficient"]):
        raise RuntimeError("Completed checkpoint calibration differs")


def train(m):
    import torch
    baseline(m)
    calibration = validated_calibration(m)
    report = q.read_json(Path(m["output"]) / "m65c_training_smoke.json")
    required_checks = {"finite_nonzero_uncertainty_gradients", "finite_gt_gradients",
                       "parameters_unchanged", "buffers_unchanged", "anchor_unchanged"}
    if (report.get("passed") is not True or report.get("complete") is not True
            or report.get("optimizer_steps") != 0 or report.get("external_kd_enabled") is not False
            or set(report.get("checks", {})) != required_checks
            or not all(report.get("checks", {}).values())
            or report.get("manifest_sha256") != m["manifest_sha256"]
            or report.get("calibration_sha256") != calibration["report_sha256"]):
        raise RuntimeError("Matching successful zero-update CUDA smoke required")
    path = c.checkpoint_path(m)
    if path.exists():
        payload = q.safe_payload(path)
        if payload.get("m65c_manifest_sha256") != m["manifest_sha256"] or payload.get("epoch") != 1:
            raise RuntimeError("Existing checkpoint is not this completed M65c control")
        validate_summary(m, payload.get("training_summary", {}), calibration)
        receipt = dict(complete=True, manifest_sha256=m["manifest_sha256"],
                       checkpoint_sha256=q.sha(path), completed_control_epoch=1)
        sidecar = path.with_suffix(".json")
        if sidecar.exists() and q.read_json(sidecar) != receipt:
            raise RuntimeError("Completed checkpoint sidecar changed; preserve evidence")
        if not sidecar.exists():
            q.write_json(sidecar, receipt)
        q.write_json(Path(m["output"]) / "m65c_training_summary.json", payload["training_summary"])
        print("M65c epoch already complete; no additional optimizer updates.", flush=True)
        return
    c.seed_all(RECIPE["seed"])
    model, anchor, criterion, dataset = c.training_runtime(m)
    # A fresh original-A2 process, NOT the old calibration endpoint.
    buffers, anchor_before = q.tensor_hash(dict(model.named_buffers())), q.tensor_hash(anchor.state_dict())
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=RECIPE["learning_rate"], weight_decay=0.0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True,
        generator=torch.Generator().manual_seed(RECIPE["seed"]), num_workers=0, drop_last=False)
    sums, counts = dict(gt=0.0, preservation=0.0, total=0.0), {key: {0: 0, 1: 0} for key in WEIGHTS}
    for step, batch in enumerate(loader, 1):
        optimizer.zero_grad(set_to_none=True)
        total, gt, keep, pairs, outputs, reference, targets = losses(model, anchor, criterion, batch, calibration["coefficient"])
        total.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, RECIPE["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        values = dict(gt=float(gt.detach()), preservation=float(keep["total"].detach()), total=float(total.detach()))
        for key in sums:
            sums[key] += values[key]
        for key in counts:
            for cls in (0, 1):
                counts[key][cls] += pairs[key][cls]
        if step == 1 or step % 20 == 0 or step == len(loader):
            print(f"M65c epoch=001/001 batch={step:04d}/{len(loader):04d} lambda={calibration['coefficient']:.6f} "
                  f"gt={values['gt']:.6f} keep={values['preservation']:.6f} total={values['total']:.6f} "
                  f"grad_norm={float(norm):.6f} pairs={pairs}", flush=True)
        del total, gt, keep, outputs, reference, targets
    if len(loader) != 928 or not all(counts["classification"].values()):
        raise RuntimeError("Wrong update count or a class has no preservation pairs")
    unchanged_buffers = buffers == q.tensor_hash(dict(model.named_buffers()))
    unchanged_anchor = anchor_before == q.tensor_hash(anchor.state_dict())
    if not unchanged_buffers or not unchanged_anchor:
        raise RuntimeError("Frozen buffers or original-A2 anchor changed")
    summary = dict(complete=True, manifest_sha256=m["manifest_sha256"], completed_epochs=1,
                    optimizer_steps=len(loader), loss_means={k: v / len(loader) for k, v in sums.items()},
                    preservation_pairs=counts, preservation_coefficient=calibration["coefficient"],
                    calibration_sha256=calibration["report_sha256"], running_buffers_unchanged=True,
                    anchor_unchanged=True, external_kd_enabled=False, original_checkpoint_sha256=q.A2_SHA)
    validate_summary(m, summary, calibration)
    c.atomic_checkpoint(path, dict(epoch=1, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                        m65c_manifest_sha256=m["manifest_sha256"], training_summary=summary), m)
    q.write_json(Path(m["output"]) / "m65c_training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def review(m):
    source, candidate = baseline(m), c.validated_metrics(m, "control")
    validate_summary(m, q.read_json(Path(m["output"]) / "m65c_training_summary.json"), validated_calibration(m))
    checks = c.control_checks(source, candidate)
    historical = dict(vehicle_3d_moderate=15.8713, pedestrian_3d_moderate=5.1493,
                      vehicle_bev_moderate=21.3134, pedestrian_bev_moderate=5.9365)
    gates = {k: candidate["product"][k] >= value for k, value in historical.items()}
    gates["mean_3d_moderate"] = (candidate["product"]["vehicle_3d_moderate"] + candidate["product"]["pedestrian_3d_moderate"]) / 2 >= 10.5103
    result = dict(complete=True, revision=REVISION, manifest_sha256=m["manifest_sha256"],
                  baseline=source, control=candidate, preservation_checks=checks,
                  control_stable=all(checks.values()), historical_accuracy_gates=gates,
                  nearby_product_gates={"Vehicle": candidate["nearby"]["Vehicle"] >= 0.85,
                                        "Pedestrian": candidate["nearby"]["Pedestrian"] >= 0.80},
                  candidate_minus_baseline={k: candidate["product"][k] - v for k, v in source["product"].items()},
                  optimizer_steps=928, improved_student_selected=False, kd_authorized=False,
                  full_training_authorized=False, deployment_authorized=False,
                  next_step="Review this single control. Failure stops this recipe; success does not select a teacher or authorize KD.")
    q.write_json(Path(m["output"]) / "m65c_control_gate.json", result)
    print(json.dumps(result, indent=2), flush=True)
    print("STOP FOR REVIEW. Keep original A2 selected; no further epochs or automatic KD.", flush=True)


def bundle(output):
    path = output / "m65c_results.zip"
    files = [p for p in output.rglob("*") if p.is_file() and p.suffix in {".json", ".log", ".csv", ".yaml"}
             and "predictions" not in p.relative_to(output).parts]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, p.relative_to(output))
    print(f"Return {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "metrics", "baseline", "calibrate", "smoke", "train", "review", "bundle"))
    for name in ("manifest", "output", "m65-manifest", "m65b-output", "a2-repo", "dataset-root"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--model", choices=("baseline", "control"))
    args = parser.parse_args()
    required = ("output",) if args.stage == "bundle" else (
        "output", "m65_manifest", "m65b_output", "a2_repo", "dataset_root") if args.stage == "prepare" else ("manifest",)
    for name in required:
        if getattr(args, name) is None:
            parser.error(f"--{name.replace('_', '-')} is required")
    if args.stage == "bundle":
        bundle(args.output)
        return
    if args.stage == "prepare":
        prepare(args)
        return
    m = load(args.manifest)
    if args.stage in ("infer", "metrics") and not args.model:
        parser.error("--model baseline|control is required")
    if args.stage == "infer":
        q.infer(c.evaluation_view(m, args.model), "a2")
    elif args.stage == "metrics":
        evaluate(m, args.model)
    else:
        globals()[args.stage](m)


if __name__ == "__main__":
    main()
