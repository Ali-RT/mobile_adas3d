"""Read-only, class-preserving feature-KD gradients at three frozen states.

Stage only the first32 reviewed TRAIN frames and their existing teacher
features. No teacher forward, optimizer, validation inference, weight search,
old CUDA restoration, or changes to completed experiment files.
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
from torch.nn import functional as F
import recover_a2_lite_features as feature
import stage_a2_lite_data as staging
from models import a2_lite_recovery_loss as loss

engine = feature.engine
REVISION = "A2-LITE-FEATURE-ENDPOINT-GRADIENTS-2026-10-15-r1"
RUN_SIGNATURE = "12bae93a627ba26fd7fc42d2515a40f0da6893ffd364b2c61b18e7e07aa2e8b0"
REVIEWED = {
    "feature_manifest.json": "e71324ab9192b7f9a0066cbf47ce518d7f994c3d0be7d45121d893022e016807",
    "feature_comparison.json": "4b30559da11a9697612304b55130003e924e0c09aaa0a4d4d6bf8166ca4431d8",
    "feature_kd_smoke.json": "a526df835f45afeff59f44f16a9c036498d0f5b799563baeff6a80737ff280c6",
    "feature_cache_complete.json": "fe6268f4cb244d2611d0ebb9a4717d6eec95268b0ad73e01f7a9f4efd7790cd2",
    "control/training_complete.json": "aba5d92e14264fe2e456500db9680ab958ac6bcac333883ca8504893756ea818",
    "kd/training_complete.json": "e451388f5668d25d5ee7bdc4d7e7fb63dbf27f2f0fdbe69a4b14ca46c5f6eace",
}
WEIGHTS = {
    "initial": engine.resolution.CHECKPOINT_SHA,
    "control_epoch5": "95946687be2c46e9133c94b766b916b13fcb88b28716e4d2953366cbca087c5b",
    "feature_epoch5": "6cc836b0729ad59e3d09432f3a528cf027bb79633e45887c5f4c3ab1b8ace8dc",
}
ROLES = tuple(WEIGHTS)
CLASSES = (("Pedestrian", 0), ("Vehicle", 1))
TERMS = ("gt/Pedestrian", "gt/Vehicle", "gt/unused_channel", "kd/Pedestrian", "kd/Vehicle")
RECIPE = dict(frames=32, roles=list(ROLES), optimizer_steps=0, validation_used=False,
              frame_selection="first32 original TRAIN calibration IDs; no VAL selection",
              coefficient="same fully ramped feature coefficient at all three states",
              gt_partition="original all-target assignment/global normalization/dimension compensation; class logit channels include negatives; unused channel separate",
              kd_partition="exact original per-frame present-class shares, not class-rematched or reweighted",
              coordinates="trainable visual encoder and late backbone; same coordinates for every loss",
              gradient_clipping_applied=False, adamw_preconditioning_applied=False,
              training_authorized=False, deployment_qualified=False)
LIMITATIONS = [
    "32 TRAIN frames at fixed states, not a historical trajectory or generalization test.",
    "GT class rows include all negative logits in that class channel; they are not positive-object-only losses.",
    "The fully ramped coefficient at initialization is a counterfactual, not the first warmup update.",
    "Gradients are unclipped/unpreconditioned. Scalar loss size is not gradient dominance.",
    "Opposing gradients are local signals, not causal proof of Pedestrian AP regression.",
    "No coefficient sweep, training, validation inference or endpoint promotion is authorized.",
]


def signed(value, field="report_sha256"):
    if engine.signature({k: v for k, v in value.items() if k != field}) != value.get(field):
        raise RuntimeError("Changed audit or historical record")
    return value


def reviewed(source):
    records = {}
    for name, expected in REVIEWED.items():
        raw = engine.read_record(source, name)
        if engine.accuracy.sha_bytes(raw) != expected:
            raise RuntimeError(f"Use the exact reviewed completed feature run: {name}")
        records[name] = signed(json.loads(raw), "signature_sha256" if name == "feature_manifest.json" else "report_sha256")
    m = records["feature_manifest.json"]
    if (m["signature_sha256"] != RUN_SIGNATURE or m["recipe"] != feature.RECIPE
            or m["implementation"] != feature.implementation() or m["guardrails"] != feature.GUARDRAILS
            or not records["feature_comparison.json"]["complete"]
            or records["feature_comparison.json"]["benefit_guardrail_passed"] is not False):
        raise RuntimeError("Reviewed feature recipe, implementation or completed comparison differs")
    for name, value in records.items():
        if name != "feature_manifest.json" and value["run_signature"] != RUN_SIGNATURE:
            raise RuntimeError("Reviewed feature records have mixed run identities")
    for role, historical_role in (("control_epoch5", "control"), ("feature_epoch5", "kd")):
        t = records[f"{historical_role}/training_complete.json"]
        if (not t["complete"] or t["optimizer_steps"] != 2560 or t["fixed_endpoint_epoch"] != 5
                or t["checkpoint_sha256"] != WEIGHTS[role] or t["tiny_checkpoint_used"]
                or t["same_initialization_sha256"] != WEIGHTS["initial"]):
            raise RuntimeError("Both exact completed feature-pilot endpoints are required")
    calibration = records["feature_kd_smoke.json"]
    if (not calibration["passed"] or calibration["optimizer_steps"] != 0
            or feature.calibrate_feature_weight(calibration["checks"]) != calibration["calibration"]):
        raise RuntimeError("Frozen TRAIN calibration differs")
    return m, records


def runtime(device):
    engine.setup_device(device)
    return dict(cpu_environment=engine.runtime_environment(), execution_identity=feature.device_identity(device))


def implementation():
    return {**feature.implementation(), "scripts/audit_a2_lite_feature_gradients.py": engine.sha256(__file__),
            "scripts/stage_a2_lite_data.py": engine.sha256(ROOT / "scripts/stage_a2_lite_data.py")}


def subset(historical):
    ids = historical["pilot_ids"][:32]
    if (len(ids) != 32 or len(set(ids)) != 32 or not set(ids) <= set(historical["train_ids"])
            or set(ids) & set(historical["val_ids"])):
        raise RuntimeError("Use the exact first32 reviewed TRAIN calibration frames only")
    return ids


def local_copy(source, destination, expected):
    if destination.exists():
        if destination.is_symlink() or engine.sha256(destination) != expected:
            raise RuntimeError(f"Preserve changed local diagnostic input: {destination}")
        return
    raw = staging.retry_read(source)
    if engine.accuracy.sha_bytes(raw) != expected:
        raise RuntimeError(f"Diagnostic source bytes changed: {source}")
    staging.atomic_copy(destination, raw)


def prepare(a):
    historical, records = reviewed(a.feature_run)
    protected = [Path(historical["output"]), a.initialization.parent, a.control_checkpoint.parent, a.kd_checkpoint.parent]
    if a.feature_run.is_dir(): protected.append(a.feature_run)
    for destination in (a.output, a.repo, a.dataset, a.weights_dir):
        if any(destination.is_relative_to(p.resolve()) for p in protected):
            raise RuntimeError("Use separate audit paths; completed experiments and checkpoint directories stay read-only")
    for destination in (a.repo, a.dataset, a.weights_dir):
        if destination.is_symlink() or destination.is_relative_to(Path('/content/drive')):
            raise RuntimeError("Use private local source/data/weights paths, not Drive or symlinks")
    current = runtime(a.device)
    sources = dict(initial=a.initialization, control_epoch5=a.control_checkpoint, feature_epoch5=a.kd_checkpoint)
    for role, path in sources.items():
        if not path.is_file() or engine.sha256(path) != WEIGHTS[role]:
            raise RuntimeError(f"Missing/changed {role} weights: {path}. Results ZIPs contain no checkpoints.")
    existing = a.output / "feature_gradient_manifest.json"
    if existing.exists():
        old = signed(json.loads(existing.read_text()), "signature_sha256")
        if (old["runtime"] != current or old["implementation"] != implementation() or old["recipe"] != RECIPE
                or any(Path(old["view"][key]) != path for key, path in (("source", a.repo), ("dataset", a.dataset)))
                or any(Path(old["weights"][role]["path"]) != a.weights_dir / f"{role}.pth" for role in ROLES)):
            raise RuntimeError("Audit runtime/code/paths changed; preserve it and choose a new audit RUN_ID")
    a.output_verified = True
    ids = subset(historical)
    coverage = records["feature_cache_complete.json"]
    if not coverage["complete"] or not coverage["kd_pilot_authorized"]:
        raise RuntimeError("Completed teacher feature cache required; do not regenerate it")
    features = a.weights_dir / "teacher_features"
    rows = {}
    for sample in ids:
        destination = features / f"{sample}.pt"; expected = coverage["cache_sha256"][sample]
        candidates = [root / destination.name for root in (*a.feature_candidate, a.features_dir)]
        if not destination.exists():
            source = next((p for p in candidates if p.is_file()), None)
            if source is None: raise FileNotFoundError(f"Saved teacher feature missing: {sample}; use the completed run's teacher_features folder")
            local_copy(source, destination, expected)
        rows[sample] = feature.read_feature(destination, historical, sample, expected_sha=expected)
    ped = sum(bool((r["labels"] == 0).any()) for r in rows.values())
    vehicle_only = sum(not bool((r["labels"] == 0).any()) and bool((r["labels"] == 1).any()) for r in rows.values())
    if ped != 16 or vehicle_only != 16:
        raise RuntimeError("Expected16 Pedestrian-containing and16 Vehicle-only encoded TRAIN frames")
    engine.base.prepare_repo(a.repo)
    if engine.base.q.source_hash(a.repo) != historical["source_sha256"]:
        raise RuntimeError("Fresh portable source differs from the reviewed student")
    data = {s: historical["data_file_sha256"][s] for s in ids}
    staging.stage(a.dataset, a.candidate, ids, data, workers=4)
    engine.accuracy.immutable_bytes(a.dataset / "ImageSets/train.txt", ("\n".join(ids) + "\n").encode())
    weights = {}
    for role, source in sources.items():
        destination = a.weights_dir / f"{role}.pth"
        local_copy(source, destination, WEIGHTS[role])
        weights[role] = dict(path=str(destination), sha256=WEIGHTS[role])
    for name in REVIEWED:
        engine.accuracy.immutable_bytes(a.output / "reviewed_feature" / name, engine.read_record(a.feature_run, name))
    view = dict(config=copy.deepcopy(historical["config"]), source=str(a.repo), dataset=str(a.dataset),
                initialization=weights["initial"]["path"], train_ids=ids, val_ids=[], pilot_ids=ids, data_file_sha256=data)
    m = dict(revision=REVISION, recipe=RECIPE, implementation=implementation(), output=str(a.output),
             runtime=current, historical_runtime=dict(cpu_environment=historical["cpu_environment"], execution_identity=historical["execution_identity"]),
             historical_run_signature=RUN_SIGNATURE, historical_source_sha256=historical["source_sha256"],
             view=view, weights=weights, feature_root=str(features),
             feature_sha256={s: coverage["cache_sha256"][s] for s in ids},
             final_feature_coefficient=records["feature_kd_smoke.json"]["calibration"]["loss_scale"] * feature.RECIPE["kd_max_weight"],
             staged_train_frames=32, staged_validation_frames=0, training_authorized=False)
    m["signature_sha256"] = engine.signature(m)
    engine.accuracy.immutable_json(existing, m)
    print("Prepared only32 TRAIN frames, three read-only weight states and cached teacher features. Zero training.", flush=True)


def load(output, device):
    m = signed(json.loads((output / "feature_gradient_manifest.json").read_text()), "signature_sha256")
    if (m["revision"] != REVISION or m["recipe"] != RECIPE or m["implementation"] != implementation()
            or m["runtime"] != runtime(device) or m["output"] != str(output)):
        raise RuntimeError("Audit identity/runtime changed; preserve output and choose a new audit RUN_ID")
    historical, records = reviewed(output / "reviewed_feature")
    if (m["view"]["pilot_ids"] != subset(historical) or m["view"]["val_ids"]
            or m["view"]["config"] != historical["config"]
            or engine.base.q.source_hash(Path(m["view"]["source"])) != historical["source_sha256"]):
        raise RuntimeError("Audit source, TRAIN-only subset or model config changed")
    for role, value in m["weights"].items():
        if value["sha256"] != WEIGHTS[role] or engine.sha256(value["path"]) != value["sha256"]:
            raise RuntimeError("Read-only audit checkpoint changed")
    expected_scale = records["feature_kd_smoke.json"]["calibration"]["loss_scale"] * feature.RECIPE["kd_max_weight"]
    if m["final_feature_coefficient"] != expected_scale:
        raise RuntimeError("Frozen fully ramped feature coefficient changed")
    for sample in subset(historical):
        if (m["feature_sha256"][sample] != records["feature_cache_complete.json"]["cache_sha256"][sample]
                or m["view"]["data_file_sha256"][sample] != historical["data_file_sha256"][sample]):
            raise RuntimeError("Frozen feature/data subset provenance changed")
        feature.read_feature(Path(m["feature_root"]) / f"{sample}.pt", historical, sample, expected_sha=m["feature_sha256"][sample])
    return m, historical


def gt_partition(outputs, target):
    """Exact class shares of the ORIGINAL GT loss; never rematch by class."""
    i, j = loss.assignment(outputs, target); n = max(len(j), 1)
    labels = target["labels"][j].long(); logits = outputs["pred_logits"][0]
    desired = torch.zeros_like(logits); desired[i, labels] = 1
    prob = logits.sigmoid(); pt = prob * desired + (1 - prob) * (1 - desired)
    focal = F.binary_cross_entropy_with_logits(logits, desired, reduction="none") * (1 - pt).square()
    focal = focal * (.25 * desired + .75 * (1 - desired)) * loss.GT_WEIGHTS["loss_ce"] / n
    values = {f"gt/{name}": focal[:, label].sum() for name, label in CLASSES}
    values["gt/unused_channel"] = focal[:, 2].sum()
    if len(j):
        boxes, gt = outputs["pred_boxes"][0, i], target["boxes_3d"][j]
        depth, actual = outputs["pred_depth"][0, i], target["depth"][j, 0]
        dimensions, actual_dim = outputs["pred_3d_dim"][0, i], target["size_3d"][j]
        relative = (dimensions - actual_dim).abs() / actual_dim
        compensation = ((dimensions - actual_dim).abs().mean() / relative.mean().clamp_min(1e-8)).detach()
        angle, bins = outputs["pred_angle"][0, i], target["heading_bin"][j, 0].long()
        residual = angle[:, 12:].gather(1, bins[:, None]).squeeze(1)
        per_object = {
            "loss_center": (boxes[:, :2] - gt[:, :2]).abs().sum(-1),
            "loss_bbox": (boxes[:, 2:] - gt[:, 2:]).abs().sum(-1),
            "loss_giou": 1 - loss.overlaps(boxes, gt, True).diag(),
            "loss_depth": 1.4142 * torch.exp(-depth[:, 1]) * (depth[:, 0] - actual).abs() + depth[:, 1],
            "loss_dim": (relative * compensation).sum(-1),
            "loss_angle": F.cross_entropy(angle[:, :12], bins, reduction="none") + (residual - target["heading_res"][j, 0]).abs(),
        }
        weighted = sum(v * loss.GT_WEIGHTS[k] for k, v in per_object.items()) / n
        for name, label in CLASSES: values[f"gt/{name}"] += weighted[labels == label].sum()
    original, _, _ = engine.gt_loss(outputs, target)
    if not torch.allclose(sum(values.values()), original, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Diagnostic GT partition changed the original training objective")
    return values


def losses(outputs, student_features, row, target, coefficient):
    values = gt_partition(outputs, target)
    labels = target["labels"]; teacher = row["features"].to(student_features.device)
    eligible = row["eligible"].to(labels.device)
    original, counts = feature.feature_kd_loss(student_features, teacher, labels, eligible)
    present = max(sum(v > 0 for v in counts.values()), 1)
    for name, label in CLASSES:
        mask = eligible & (labels == label)
        part, _ = feature.feature_kd_loss(student_features, teacher, labels, mask)
        values[f"kd/{name}"] = part * (coefficient / present)
    if not torch.allclose(values["kd/Pedestrian"] + values["kd/Vehicle"], original * coefficient, atol=1e-5, rtol=1e-5):
        raise RuntimeError("Diagnostic KD partition changed the original class-balanced objective")
    return values, counts


def coordinates(model):
    selected = [(n, p, "visual_encoder" if n.startswith("depthaware_transformer.encoder.") else "late_backbone")
                for n, p in model.named_parameters() if p.requires_grad and n.startswith(("depthaware_transformer.encoder.", "backbone."))]
    if not selected or not any(g == "visual_encoder" for _, _, g in selected):
        raise RuntimeError("Missing trainable visual-encoder audit coordinates")
    return list(zip(*selected))


def gram(groups, gradients):
    if set(gradients) != set(TERMS) or any(len(v) != len(groups) for v in gradients.values()):
        raise ValueError("All five GT/KD class contributions must share identical coordinates")
    matrices = {}
    for index, group in enumerate(groups):
        entries = [gradients[k][index] for k in TERMS]; connected = [v for v in entries if v is not None]
        if not connected: continue
        if any(v.shape != connected[0].shape or not torch.isfinite(v).all() for v in connected):
            raise RuntimeError("Nonfinite/misaligned diagnostic gradients")
        stack = torch.stack([(v.detach() if v is not None else torch.zeros_like(connected[0])).reshape(-1).double() for v in entries])
        product = stack @ stack.T
        for tag in ("shared", group):
            if tag not in matrices: matrices[tag] = torch.zeros_like(product)
            matrices[tag] += product
    return {k: v.cpu().tolist() for k, v in matrices.items()}


def measurements(matrix):
    if (len(matrix) != len(TERMS) or any(len(row) != len(TERMS) for row in matrix)
            or any(not math.isfinite(v) for row in matrix for v in row)):
        raise ValueError("Invalid gradient Gram matrix")
    indices = {name: [i] for i, name in enumerate(TERMS)}
    indices.update(gt_all=[0, 1, 2], kd_all=[3, 4])
    dot = lambda a, b: sum(matrix[i][j] for i in indices[a] for j in indices[b])
    norms = {k: math.sqrt(max(0., dot(k, k))) for k in indices}
    rows = {}
    for teacher in ("kd_all", "kd/Pedestrian", "kd/Vehicle"):
        for reference in ("gt_all", "gt/Pedestrian", "gt/Vehicle"):
            product = norms[teacher] * norms[reference]
            rows[f"{teacher} vs {reference}"] = dict(
                kd_norm=norms[teacher], gt_norm=norms[reference],
                norm_ratio=norms[teacher] / norms[reference] if norms[reference] > 0 else None,
                cosine=max(-1., min(1., dot(teacher, reference) / product)) if product > 0 else None)
    return dict(norms=norms, comparisons=rows)


def summarize(frames):
    result = []
    strata = {"all": frames,
              "Pedestrian_present": [f for f in frames if f.get("gt_counts", {}).get("Pedestrian", 0) > 0],
              "Vehicle_only": [f for f in frames if f.get("gt_counts", {}).get("Pedestrian", 0) == 0 and f.get("gt_counts", {}).get("Vehicle", 0) > 0]}
    for stratum, selected in strata.items():
        for group in sorted({g for f in selected for g in f["gradients"]}):
            names = sorted({k for f in selected if group in f["gradients"] for k in f["gradients"][group]["comparisons"]})
            for name in names:
                rows = [f["gradients"][group]["comparisons"][name] for f in selected if group in f["gradients"]]
                active = [r for r in rows if r["kd_norm"] > 0 and r["gt_norm"] > 0]
                ratios = [r["norm_ratio"] for r in active]; cosines = [r["cosine"] for r in active]
                result.append(dict(parameter_group=group, frame_stratum=stratum, comparison=name, measured_frames=len(rows), active_frames=len(active),
                    median_norm_ratio=statistics.median(ratios) if ratios else None,
                    median_cosine=statistics.median(cosines) if cosines else None,
                    negative_cosine_fraction=sum(c < 0 for c in cosines) / len(active) if active else None,
                    cosine_below_minus_0_1_fraction=sum(c < -.1 for c in cosines) / len(active) if active else None,
                    norm_ratio_above_one_fraction=sum(r > 1 for r in ratios) / len(active) if active else None))
    return result


def probe(a, m, historical):
    view = m["view"]; engine.setup_device(a.device)
    model = engine.build_model(view, a.device, True)
    if a.role != "initial":
        saved = torch.load(m["weights"][a.role]["path"], map_location="cpu", weights_only=True)
        historical_role = "control" if a.role == "control_epoch5" else "kd"
        if (saved.get("run_signature") != RUN_SIGNATURE or saved.get("role") != historical_role
                or saved.get("step") != 2560 or saved.get("feature_loss_scale") != m["final_feature_coefficient"] / feature.RECIPE["kd_max_weight"]):
            raise RuntimeError("Wrong completed endpoint checkpoint payload")
        model.load_state_dict(saved["model_state"], strict=True); del saved
    before = engine.resolution.state_signature(model)
    names, parameters, groups = coordinates(model)
    dataset = engine.native_dataset(view, view["pilot_ids"]); frames = []
    with feature.EncoderFeatureTap(model) as tap:
        for index, sample in enumerate(view["pilot_ids"]):
            engine.verify_files(view, sample)
            inputs, target, _ = engine.item(dataset, index, a.device)
            row = feature.read_feature(Path(m["feature_root"]) / f"{sample}.pt", historical, sample, target, m["feature_sha256"][sample])
            path = a.output / "frames" / a.role / f"{sample}.json"
            if path.exists():
                report = engine.checked_report(path, m)
                if report["sample_id"] != sample or report["role"] != a.role or report["model_state_sha256"] != before:
                    raise RuntimeError("Resumed audit frame identity changed")
            else:
                outputs = engine.forward(model, inputs)
                features = feature.object_features(tap.take(), target, feature.STUDENT_SHAPES)
                values, counts = losses(outputs, features, row, target, m["final_feature_coefficient"])
                gradients = {key: torch.autograd.grad(values[key], parameters, retain_graph=True, allow_unused=True) for key in TERMS}
                matrices = gram(groups, gradients)
                if measurements(matrices["visual_encoder"])["norms"]["gt_all"] <= 0:
                    raise RuntimeError("Visual-encoder GT gradient is zero; cannot compare teacher balance")
                report = dict(complete=True, run_signature=m["signature_sha256"], role=a.role, sample_id=sample,
                    model_state_sha256=before, weighted_losses={k: float(v.detach()) for k, v in values.items()},
                    teacher_counts=counts, gt_counts={n: int((target["labels"] == i).sum()) for n, i in CLASSES},
                    gradients={g: measurements(v) for g, v in matrices.items()}, optimizer_steps=0)
                engine.save_report(path, report); del gradients, values, outputs, features
            frames.append(report)
            print(f"{a.role}: measured/verified {index + 1}/32 TRAIN frames; zero updates", flush=True)
    if before != engine.resolution.state_signature(model) or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Weights, buffers or .grad fields changed during read-only audit")
    report = dict(complete=True, revision=REVISION, run_signature=m["signature_sha256"], role=a.role,
        frames=len(frames), optimizer_steps=0, weights_and_buffers_unchanged=True, model_state_sha256=before,
        coordinate_names_sha256=engine.signature(names), coordinate_parameters=sum(p.numel() for p in parameters),
        fully_ramped_coefficient=m["final_feature_coefficient"], summary=summarize(frames), limitations=LIMITATIONS,
        frame_sha256={s: engine.sha256(a.output / "frames" / a.role / f"{s}.json") for s in view["pilot_ids"]})
    engine.save_report(a.output / f"{a.role}_gradient_report.json", report)


def compare(a, m):
    reports = {r: engine.checked_report(a.output / f"{r}_gradient_report.json", m) for r in ROLES}
    for role, value in reports.items():
        if (not value["complete"] or value["role"] != role or value["frames"] != 32
                or value["optimizer_steps"] != 0 or not value["weights_and_buffers_unchanged"]
                or value["fully_ramped_coefficient"] != m["final_feature_coefficient"]
                or value["frame_sha256"] != {s: engine.sha256(a.output / "frames" / role / f"{s}.json") for s in m["view"]["pilot_ids"]}):
            raise RuntimeError("Three complete, unchanged-weight audit states required")
    if len({r["coordinate_names_sha256"] for r in reports.values()}) != 1:
        raise RuntimeError("States used different trainable coordinates")
    table = [{**row, "role": role} for role, report in reports.items() for row in report["summary"]]
    value = dict(complete=True, revision=REVISION, run_signature=m["signature_sha256"],
                 roles=list(ROLES), optimizer_steps=0, staged_train_frames=32, validation_inference_performed=False,
                 weighted_gradient_comparison=table, limitations=LIMITATIONS, training_authorized=False,
                 deployment_qualified=False, next_step="Review class-specific gradient growth/conflict; no automatic coefficient change or new training")
    engine.save_report(a.output / "feature_gradient_comparison.json", value)
    print(json.dumps([v for v in table if v["parameter_group"] == "visual_encoder" and v["frame_stratum"] == "all" and v["comparison"].startswith("kd_all")], indent=2), flush=True)


def bundle(a):
    path = a.output / "a2_lite_feature_gradient_results.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(a.output.rglob("*")):
            if source.is_file() and source.suffix in (".json", ".log", ".txt"):
                archive.write(source, source.relative_to(a.output))
    print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "probe", "compare", "bundle"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--role", choices=ROLES)
    for name in ("feature-run", "features-dir", "initialization", "control-checkpoint", "kd-checkpoint", "repo", "dataset", "weights-dir"):
        parser.add_argument("--" + name, type=Path)
    for name in ("candidate", "feature-candidate"):
        parser.add_argument("--" + name, type=Path, action="append", default=[])
    a = parser.parse_args()
    for key, value in vars(a).items():
        if isinstance(value, Path): setattr(a, key, value.resolve())
    a.candidate = [p.resolve() for p in a.candidate]; a.feature_candidate = [p.resolve() for p in a.feature_candidate]
    required = ("feature_run", "features_dir", "initialization", "control_checkpoint", "kd_checkpoint", "repo", "dataset", "weights_dir")
    if a.stage == "prepare" and any(getattr(a, k) is None for k in required): parser.error("prepare requires reviewed feature run, cached features, all three weights and private local paths")
    if a.stage == "probe" and a.role is None: parser.error("probe requires --role")
    try:
        if a.stage == "prepare": prepare(a)
        elif a.stage == "bundle": bundle(a)
        else:
            m, historical = load(a.output, a.device)
            if a.stage == "probe": probe(a, m, historical)
            else: compare(a, m)
    except Exception as error:
        import traceback
        # Never write diagnostics into the completed training run on a bad path.
        if getattr(a, "output_verified", False) or (a.output / "feature_gradient_manifest.json").is_file():
            engine.write_json(a.output / "failures" / f"{a.stage}_{time.time_ns()}.json", dict(complete=False, error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
