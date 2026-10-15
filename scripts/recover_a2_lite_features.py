"""Fresh matched GT / box-focused feature-KD pilot; old runs stay read-only.

Same A2-Lite E1/D1, 640x192, original pruned initialization, exact reviewed
512 TRAIN frames, five fixed epochs. A2 E3/D3 at 1280x384 is a frozen
training-only teacher. No output KD, adapters, graph edits or CUDA build.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import torch
import recover_a2_lite_640 as engine
import recover_a2_lite_full as reference
from models.a2_lite_feature_kd import (EncoderFeatureTap, object_features, feature_kd_loss,
                                     STUDENT_SHAPES, TEACHER_SHAPES, CHANNELS, ROI_SIZE)

REVISION = "A2-LITE-512-FOREGROUND-FEATURE-KD-2026-10-14-r1"
RECIPE = copy.deepcopy(engine.RECIPE)
RECIPE.update(kd_family="box-focused final-visual-encoder features",
              prediction_kd_components=[], feature_channels=CHANNELS, roi_size=ROI_SIZE,
              teacher_encoder_layers=3, student_encoder_layers=1,
              teacher_input_hw=[384, 1280], feature_alignment="same normalized GT-box 3x3 grid per feature level; align_corners=False",
              feature_loss="cosine distance; equal point/level/object means, then equal present-class means",
              feature_teacher_eligibility="correct class, confidence>=0.1, 2D IoU>=0.5, positive geometry; no per-head error gate",
              feature_adapter=False, background_kd=False, gt_loss_changed=False,
              feature_gradient_target_ratio=.05, feature_calibration_max_scale=1000.,
              feature_calibration="fixed TRAIN-only median weighted-KD/GT encoder gradient ratio; no optimizer updates",
              fresh_matched_control=True, historical_results_are_context_only=True)
for _key in ("temperature", "teacher_error_ratio", "minimum_teacher_improvement"):
    RECIPE.pop(_key)
GUARDRAILS = copy.deepcopy(reference.GUARDRAILS)


@contextmanager
def recipe_context():
    old = engine.RECIPE, engine.REVISION
    engine.RECIPE, engine.REVISION = RECIPE, REVISION
    try:
        yield
    finally:
        engine.RECIPE, engine.REVISION = old


def implementation():
    return {**reference.implementation(), "scripts/recover_a2_lite_features.py": engine.sha256(__file__),
            "models/a2_lite_feature_kd.py": engine.sha256(ROOT / "models/a2_lite_feature_kd.py")}


def prepare(a):
    old, comparison, raw_m, raw_c = reference.reviewed_pilot(a.pilot)
    protected = [Path(old["output"]), a.initialization.parent, a.teacher_checkpoint.parent]
    protected += [p for p in (a.pilot, a.compact, a.teacher_export) if p.is_dir()]
    if any(a.output.is_relative_to(p) for p in protected):
        raise RuntimeError("Historical runs/checkpoints are read-only; use a separate feature output")
    if a.repo.is_relative_to(Path(old["source"])) or a.dataset.is_relative_to(Path(old["dataset"])):
        raise RuntimeError("Use new disposable source/data paths; previous inputs are staging candidates only")
    if not a.initialization.is_file() or engine.sha256(a.initialization) != engine.resolution.CHECKPOINT_SHA:
        raise RuntimeError("Use the original untrained pruned initialization, not any recovered endpoint")
    if not a.teacher_checkpoint.is_file() or engine.sha256(a.teacher_checkpoint) != engine.base.q.A2_SHA:
        raise RuntimeError("Need the exact original A2 epoch130 .pth checkpoint for internal features")
    a.output_verified = True
    a.output.mkdir(parents=True, exist_ok=True)
    engine.base.prepare_repo(a.repo)
    if engine.base.q.source_hash(a.repo) != old["source_sha256"]:
        raise RuntimeError("Fresh portable source differs from the reviewed model")
    train = engine.base.q.split_ids(a.split_dir / "train.txt", "train")
    val = engine.base.q.split_ids(a.split_dir / "val.txt", "val")
    reference.all_train_ids(train, val)
    if train != old["train_ids"] or val != old["val_ids"]:
        raise RuntimeError("Use the original Chen split and order")
    selected = list(old["pilot_ids"])
    if len(selected) != 512 or len(set(selected)) != 512 or not set(selected) <= set(train):
        raise RuntimeError("Expected exact reviewed512 TRAIN IDs")
    files = engine.accuracy.stage_data(a.dataset, a.candidate, train + val)
    if files != old["data_file_sha256"] or engine.choose_ids(train, a.dataset / "training/label_2") != selected:
        raise RuntimeError("Data bytes or TRAIN-only subset changed")
    for split, ids in (("train", train), ("val", val)):
        engine.accuracy.immutable_bytes(a.dataset / "ImageSets" / f"{split}.txt", ("\n".join(ids) + "\n").encode())
    student_root, _ = engine.accuracy.checked_package(a.compact, "compact_640", a.output / "assets/student")
    teacher_root = engine.materialize_export(a.teacher_export, a.output / "assets/teacher")
    teacher = engine.A2Teacher(teacher_root)
    if (teacher.package["model"]["sha256"] != old["teacher_model_sha256"]
            or [f["sample_id"] for f in teacher.package["fixtures"]] != engine.resolution.IDS):
        raise RuntimeError("A2 ONNX reference changed")
    for name, raw in (("recovery_manifest.json", raw_m), ("recovery_comparison.json", raw_c)):
        engine.accuracy.immutable_bytes(a.output / "reviewed_pilot" / name, raw)
    m = dict(revision=REVISION, recipe=RECIPE, guardrails=GUARDRAILS, implementation=implementation(),
             cpu_environment=engine.runtime_environment(), execution_identity=device_identity(a.device),
             source=str(a.repo), source_sha256=old["source_sha256"],
             config=copy.deepcopy(old["config"]), initialization=str(a.initialization),
             initialization_sha256=engine.resolution.CHECKPOINT_SHA, student_export=str(student_root),
             teacher_checkpoint=str(a.teacher_checkpoint), teacher_checkpoint_sha256=engine.base.q.A2_SHA,
             teacher_export=str(teacher_root), teacher_model_sha256=old["teacher_model_sha256"],
             teacher_package_signature=engine.signature(teacher.package),
             dataset=str(a.dataset), local_feature_cache=str(a.dataset / "teacher_feature_cache"),
             data_file_sha256=files, split_dir=str(a.split_dir), train_ids=train, val_ids=val, pilot_ids=selected,
             output=str(a.output), reference_run_signature=old["signature_sha256"],
             reference_results=comparison["results"], historical_environment=old["cpu_environment"],
             fresh_control_required=True, historical_environment_match_required=False,
             target_selection_train_only=True, architecture_changed=False, deployment_qualified=False,
             full_training_authorized=False, phone_retest_required=True)
    m["signature_sha256"] = engine.signature(m)
    engine.accuracy.immutable_json(a.output / "feature_manifest.json", m)
    print("Prepared fresh paired512/five-epoch GT and feature-KD arms; current runtime, no old CUDA restore.", flush=True)


def load(output):
    m = json.loads((output / "feature_manifest.json").read_text())
    if (engine.signature({k: v for k, v in m.items() if k != "signature_sha256"}) != m["signature_sha256"]
            or m["revision"] != REVISION or m["recipe"] != RECIPE or m["guardrails"] != GUARDRAILS
            or m["implementation"] != implementation() or m["cpu_environment"] != engine.runtime_environment()
            or m["output"] != str(output)
            or m["execution_identity"] != device_identity(m["execution_identity"]["device"])):
        raise RuntimeError("Feature run code/packages/recipe changed; preserve it and use a new RUN_ID")
    old, _, _, _ = reference.reviewed_pilot(output / "reviewed_pilot")
    if (m["pilot_ids"] != old["pilot_ids"] or m["data_file_sha256"] != old["data_file_sha256"]
            or m["config"] != old["config"] or m["source_sha256"] != old["source_sha256"]
            or engine.sha256(m["initialization"]) != engine.resolution.CHECKPOINT_SHA
            or engine.sha256(m["teacher_checkpoint"]) != engine.base.q.A2_SHA
            or engine.base.q.source_hash(Path(m["source"])) != m["source_sha256"]):
        raise RuntimeError("Frozen teacher/student/input identity changed")
    for split, key in (("train", "train_ids"), ("val", "val_ids")):
        if engine.base.q.split_ids(Path(m["split_dir"]) / f"{split}.txt", split) != m[key]:
            raise RuntimeError("Frozen split changed")
    return m


def build_teacher(m, device):
    if "MultiScaleDeformableAttention" in sys.modules:
        raise RuntimeError("Use a fresh portable-only stage process")
    if not any(isinstance(v, engine.PortableOnly) for v in sys.meta_path):
        sys.meta_path.insert(0, engine.PortableOnly())
    model = engine.base.inference_model(Path(m["source"]), m["config"]["model"])
    saved = torch.load(m["teacher_checkpoint"], map_location="cpu", weights_only=True)
    if saved.get("epoch") != 130:
        raise RuntimeError("Expected original teacher epoch130")
    model.load_state_dict(saved["model_state"], strict=True)
    modules = [v for v in model.modules() if hasattr(v, "coreml_spatial_shapes")]
    if (len(modules) != 6 or model.depthaware_transformer.encoder.num_layers != 3
            or model.depthaware_transformer.decoder.num_layers != 3
            or any(tuple(v.coreml_spatial_shapes) != TEACHER_SHAPES or not v.coreml_export for v in modules)):
        raise RuntimeError("Expected full E3/D3 portable A2 at1280x384")
    return model.to(device).eval().requires_grad_(False)


def feature_identity(m, sample):
    if sample not in m["pilot_ids"]:
        raise RuntimeError("Teacher features are TRAIN-only; validation access is forbidden")
    return dict(run_signature=m["signature_sha256"], sample_id=sample,
                source_sha256=m["data_file_sha256"][sample], teacher_checkpoint_sha256=m["teacher_checkpoint_sha256"])


def read_feature(path, m, sample, target=None, expected_sha=None):
    identity = feature_identity(m, sample)
    if expected_sha is not None and engine.sha256(path) != expected_sha:
        raise RuntimeError(f"Teacher feature bytes changed: {sample}")
    row = torch.load(path, map_location="cpu", weights_only=True)
    if any(row.get(k) != v for k, v in identity.items()):
        raise RuntimeError("Teacher feature identity changed")
    n = len(row["labels"])
    feature_kd_loss(row["features"], row["features"], row["labels"], row["eligible"])
    mapping = row["object_to_teacher_query"]
    if (n > 50 or row["features"].requires_grad or row["labels"].dtype != torch.long
            or mapping.shape != (n,) or mapping.dtype != torch.long
            or not ((mapping >= -1) & (mapping < 50)).all()
            or not torch.equal(row["eligible"], mapping >= 0)):
        raise RuntimeError("Invalid detached object feature cache")
    if target is not None and (row["target_signature"] != engine.target_signature(target)
                               or not torch.equal(row["labels"], target["labels"].cpu())):
        raise RuntimeError("Teacher features and native GT encoding differ")
    return row


def cache(a, m):
    engine.tiny_gate(m)
    engine.setup_device(a.device)
    model = build_teacher(m, a.device)
    before = engine.resolution.state_signature(model)
    teacher = engine.A2Teacher(m["teacher_export"])
    if engine.signature(teacher.package) != m["teacher_package_signature"]:
        raise RuntimeError("Signed teacher fixtures changed after preparation")
    parity = []
    # Exact signed inputs, not reconstructed pixels. This verifies the new
    # native full teacher before any internal features become training targets.
    with torch.no_grad():
        for fixture in teacher.package["fixtures"]:
            inputs = {n: engine.read_f32(teacher.root, fixture["inputs"][n]) for n in engine.INPUT_NAMES}
            expected = {n: engine.read_f32(teacher.root, fixture["cpu_reference_outputs"][n]) for n in engine.OUTPUT_NAMES}
            result = engine.compare_outputs(engine.tensor_arrays(engine.forward(model, inputs)), expected)
            parity.append(dict(sample_id=fixture["sample_id"], comparison=result))
            if not result["passed"]:
                engine.save_report(a.output / "teacher_fixture_failure.json", dict(complete=False,
                                   run_signature=m["signature_sha256"], parity=parity, optimizer_steps=0))
                raise RuntimeError("Native A2 teacher fixture parity failed; do not cache or train")
    dataset = engine.native_dataset(m, m["pilot_ids"])
    root = a.output / "teacher_features"
    counts, encoded, hashes = Counter(), Counter(), {}
    with EncoderFeatureTap(model) as tap, torch.no_grad():
        for index, sample in enumerate(m["pilot_ids"]):
            engine.verify_files(m, sample)
            _, target, original = engine.item(dataset, index, a.device)
            path = root / f"{sample}.pt"
            if path.exists():
                row = read_feature(path, m, sample, target)
            else:
                rebuilt = engine.teacher_inputs(Path(m["dataset"]) / "training/image_2" / f"{sample}.png", original["calibration"][0])
                engine.preprocessing_comparison(original, rebuilt)
                values = engine.forward(model, original)
                features = object_features(tap.take(), target, TEACHER_SHAPES)
                mapping = engine.teacher_matches(values, target)
                row = dict(**feature_identity(m, sample), target_signature=engine.target_signature(target),
                           labels=target["labels"].cpu(), eligible=(mapping >= 0).cpu(),
                           object_to_teacher_query=mapping.cpu(), features=features.detach().cpu())
                engine.atomic_checkpoint(path, row)
                read_feature(path, m, sample, target)
            hashes[sample] = engine.sha256(path)
            for label, active in zip(row["labels"].tolist(), row["eligible"].tolist()):
                encoded[label] += 1
                counts[label] += int(active)
            if (index + 1) % 32 == 0 or index + 1 == len(dataset):
                print(f"{a.device.upper()} A2 TRAIN feature cache {index + 1}/{len(dataset)}; no teacher optimizer", flush=True)
    unchanged = before == engine.resolution.state_signature(model) and all(p.grad is None for p in model.parameters())
    allowed = unchanged and min(counts[0], counts[1]) >= RECIPE["minimum_teacher_matches_per_class"]
    report = dict(complete=True, run_signature=m["signature_sha256"], images=len(dataset),
                  encoded_gt_counts=dict(Pedestrian=encoded[0], Vehicle=encoded[1]),
                  qualified_teacher_matches=dict(Pedestrian=counts[0], Vehicle=counts[1]),
                  kd_pilot_authorized=bool(allowed), cache_sha256=hashes, selection_train_only=True,
                  teacher_fixture_parity=parity, teacher_weights_buffers_unchanged=unchanged,
                  optimizer_steps=0, execution_identity=device_identity(a.device), output_kd_used=False,
                  feature_shape_per_object=[4, CHANNELS, ROI_SIZE, ROI_SIZE])
    engine.save_report(a.output / "feature_cache_complete.json", report)
    if not allowed:
        raise RuntimeError("Feature coverage/frozen-teacher check failed; return evidence before training")
    print("Teacher feature coverage passed for both classes; original teacher unchanged.", flush=True)


def local_cache(m):
    report = engine.checked_report(Path(m["output"]) / "feature_cache_complete.json", m)
    if (not report["complete"] or not report["kd_pilot_authorized"] or report["images"] != len(m["pilot_ids"])
            or set(report["cache_sha256"]) != set(m["pilot_ids"])):
        raise RuntimeError("Complete verified TRAIN-only feature cache required")
    root = Path(m["local_feature_cache"])
    for sample, expected in report["cache_sha256"].items():
        path = root / f"{sample}.pt"
        if not path.exists():
            raw = (Path(m["output"]) / "teacher_features" / path.name).read_bytes()
            if engine.accuracy.sha_bytes(raw) != expected:
                raise RuntimeError("Durable teacher feature changed")
            engine.accuracy.immutable_bytes(path, raw)
        read_feature(path, m, sample, expected_sha=expected)
    return root, report


def feature_smoke(a, m):
    path = a.output / "feature_kd_smoke.json"
    if path.exists():
        checked_calibration(m)
        return
    engine.tiny_gate(m)
    root, report = local_cache(m)
    engine.setup_device(a.device)
    model = engine.build_model(m, a.device, True)
    before = engine.resolution.state_signature(model)
    parameters = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith("depthaware_transformer.encoder.")]
    if not parameters:
        raise RuntimeError("Student visual encoder is not trainable")
    totals, checks = Counter(), []
    dataset = engine.native_dataset(m, m["pilot_ids"][:32])
    with EncoderFeatureTap(model) as tap:
        for index, sample in enumerate(dataset.idx_list):
            engine.verify_files(m, sample)
            inputs, target, _ = engine.item(dataset, index, a.device)
            row = read_feature(root / f"{sample}.pt", m, sample, target, report["cache_sha256"][sample])
            outputs = engine.forward(model, inputs)
            features = object_features(tap.take(), target, STUDENT_SHAPES)
            kd, counts = feature_kd_loss(features, row["features"].to(a.device), target["labels"], row["eligible"].to(a.device))
            gt, _, _ = engine.gt_loss(outputs, target)
            totals.update(counts)
            if sum(counts.values()) == 0:
                continue
            gg = torch.autograd.grad(gt, parameters, retain_graph=True, allow_unused=True)
            kg = torch.autograd.grad(kd, parameters, allow_unused=True)
            pairs = [(g, k) for g, k in zip(gg, kg) if g is not None and k is not None]
            if not pairs or not all(torch.isfinite(g).all() and torch.isfinite(k).all() for g, k in pairs):
                raise RuntimeError("Missing/nonfinite feature or GT gradients in visual encoder")
            gn = sum(float(g.square().sum()) for g, _ in pairs) ** .5
            kn = sum(float(k.square().sum()) for _, k in pairs) ** .5
            dot = sum(float((g * k).sum()) for g, k in pairs)
            if not gn > 0 or not kn > 0:
                raise RuntimeError("Zero encoder GT/feature learning signal")
            checks.append(dict(sample_id=sample, gt=float(gt.detach()), kd=float(kd.detach()), counts=counts,
                               encoder_gt_gradient_norm=gn, encoder_kd_gradient_norm=kn,
                               max_weight_kd_to_gt_gradient_ratio=RECIPE["kd_max_weight"] * kn / gn,
                               encoder_gradient_cosine=dot / (gn * kn)))
    unchanged = engine.resolution.state_signature(model) == before
    calibration = calibrate_feature_weight(checks)
    passed = unchanged and min(totals["Pedestrian"], totals["Vehicle"]) > 0 and bool(checks)
    passed = passed and calibration["within_scale_limit"]
    engine.save_report(path, dict(complete=True, passed=bool(passed), run_signature=m["signature_sha256"],
                       active_counts=dict(totals), checks=checks, calibration=calibration, optimizer_steps=0,
                       weights_buffers_unchanged=unchanged, prediction_kd_components=[],
                       gradient_statistics_are_diagnostic_not_an_accuracy_guarantee=True))
    if not passed:
        raise RuntimeError("Feature-KD gradient smoke failed; no paired training authorized")
    print("Feature KD has finite nonzero visual-encoder gradients; both classes covered; zero optimizer steps.", flush=True)


def calibrate_feature_weight(checks):
    """Units calibration only, not validation-driven recipe search or PCGrad."""
    ratios = [c["max_weight_kd_to_gt_gradient_ratio"] for c in checks]
    if not ratios or any(not math.isfinite(v) or v <= 0 for v in ratios):
        raise RuntimeError("Cannot calibrate missing/nonfinite TRAIN feature gradients")
    median = statistics.median(ratios)
    scale = RECIPE["feature_gradient_target_ratio"] / median
    return dict(loss_scale=scale, raw_median_weighted_gradient_ratio=median,
                target_median_weighted_gradient_ratio=RECIPE["feature_gradient_target_ratio"],
                calibrated_median_weighted_gradient_ratio=median * scale,
                within_scale_limit=bool(0 < scale <= RECIPE["feature_calibration_max_scale"]),
                scope="up to32 fixed TRAIN frames at original initialization; visual encoder only",
                fixed_for_entire_pilot=True, validation_used=False,
                not_a_guarantee_on_AdamW_updates_or_class_preservation=True)


def checked_calibration(m):
    report = engine.checked_report(Path(m["output"]) / "feature_kd_smoke.json", m)
    if (not report["passed"] or not report.get("complete") or report.get("optimizer_steps") != 0
            or not report.get("weights_buffers_unchanged") or report.get("prediction_kd_components") != []
            or min(report.get("active_counts", {}).get(c, 0) for c in ("Pedestrian", "Vehicle")) <= 0):
        raise RuntimeError("Passed feature smoke and frozen TRAIN calibration required")
    expected = calibrate_feature_weight(report["checks"])
    if report["calibration"] != expected or not expected["within_scale_limit"]:
        raise RuntimeError("Feature loss calibration changed; preserve evidence")
    return expected["loss_scale"]


def device_identity(device):
    return dict(device=device, cuda_build=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if device == "cuda" else None)


def run_steps(m, model, opt, dataset, total_steps, path, role, device, cache_root=None, cache_report=None, loss_scale=1.):
    """Atomic deterministic resume; no teacher model/adapter in the optimizer."""
    step, history, counts = 0, [], Counter()
    before = engine.buffers_signature(model)
    environment = device_identity(device)
    if not math.isfinite(loss_scale) or not 0 < loss_scale <= RECIPE["feature_calibration_max_scale"]:
        raise RuntimeError("Invalid frozen feature loss scale")
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (saved["run_signature"] != m["signature_sha256"] or saved["role"] != role
                or saved["device_identity"] != environment or saved["feature_loss_scale"] != loss_scale
                or not 0 <= saved["step"] <= total_steps):
            raise RuntimeError("Resume identity/GPU/budget differs; preserve run and use a new RUN_ID")
        model.load_state_dict(saved["model_state"], strict=True)
        opt.load_state_dict(saved["optimizer_state"])
        step, history, before = saved["step"], saved["history"], saved["buffers_signature"]
        counts.update(saved["active_feature_counts"])
    order = []
    with EncoderFeatureTap(model) as tap:
        for cursor in range(step, total_steps):
            epoch, offset = divmod(cursor, len(dataset))
            if not order or offset == 0:
                order = list(range(len(dataset)))
                random.Random(RECIPE["seed"] + epoch).shuffle(order)
            index = order[offset]
            sample = dataset.idx_list[index]
            engine.verify_files(m, sample)
            inputs, target, _ = engine.item(dataset, index, device)
            opt.zero_grad(set_to_none=True)
            outputs = engine.forward(model, inputs)
            memory = tap.take()
            gt, parts, _ = engine.gt_loss(outputs, target)
            kd, weight = memory.sum() * 0, 0.
            if role == "kd":
                row = read_feature(cache_root / f"{sample}.pt", m, sample, target, cache_report["cache_sha256"][sample])
                kd, active = feature_kd_loss(object_features(memory, target, STUDENT_SHAPES),
                                            row["features"].to(device), target["labels"], row["eligible"].to(device))
                counts.update(active)
                weight = loss_scale * RECIPE["kd_max_weight"] * min(1., (cursor + 1) / RECIPE["kd_warmup_steps"])
            total = gt + weight * kd
            if not torch.isfinite(total):
                raise RuntimeError(f"Nonfinite {role} loss at step{cursor}")
            total.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), RECIPE["clip_gradient_norm"])
            if not torch.isfinite(norm):
                raise RuntimeError("Nonfinite paired training gradient")
            opt.step()
            step = cursor + 1
            if step % 32 == 0 or step == total_steps:
                history.append(dict(step=step, epoch=epoch + 1, sample_id=sample, gt=float(gt.detach()),
                                    kd=float(kd.detach()), kd_weight=weight, gradient_norm=float(norm),
                                    components={k: float(v.detach()) for k, v in parts.items()}))
                print(f"{role} {step}/{total_steps}: GT={float(gt.detach()):.4f} featureKD={float(kd.detach()):.5f} weight={weight:.3f}", flush=True)
            if step % RECIPE["checkpoint_every_steps"] == 0 or step == total_steps:
                if engine.buffers_signature(model) != before:
                    raise RuntimeError("Frozen buffers changed")
                engine.atomic_checkpoint(path, dict(run_signature=m["signature_sha256"], role=role, step=step,
                    model_state=model.state_dict(), optimizer_state=opt.state_dict(), history=history,
                    buffers_signature=before, device_identity=environment,
                    active_feature_counts=dict(counts), feature_loss_scale=loss_scale, architecture_changed=False))
    return step, history, dict(counts), environment


def train(a, m):
    engine.tiny_gate(m)
    scale = checked_calibration(m)
    engine.setup_device(a.device)
    root, report = local_cache(m) if a.role == "kd" else (None, None)
    model = engine.build_model(m, a.device, True)
    dataset = engine.native_dataset(m, m["pilot_ids"])
    path = a.output / a.role / "last.pth"
    step, history, counts, environment = run_steps(m, model, engine.optimizer(model), dataset,
        len(dataset) * RECIPE["epochs"], path, a.role, a.device, root, report, scale)
    engine.save_report(a.output / a.role / "training_complete.json", dict(complete=True,
        run_signature=m["signature_sha256"], role=a.role, checkpoint_sha256=engine.sha256(path),
        optimizer_steps=step, fixed_endpoint_epoch=RECIPE["epochs"], train_images=len(dataset),
        same_initialization_sha256=engine.resolution.CHECKPOINT_SHA, tiny_checkpoint_used=False,
        external_teacher_used=a.role == "kd", history=history, active_feature_counts=counts,
        device_identity=environment, feature_loss_scale=scale, prediction_kd_components=[], architecture_changed=False,
        deployment_qualified=False))


def compare(a, m):
    scale = checked_calibration(m)
    results, training = {}, {}
    for role in ("control", "kd"):
        t = engine.checked_report(a.output / role / "training_complete.json", m)
        value = json.loads((a.output / "evaluation" / role / "accuracy.json").read_text())
        if (not t["complete"] or t["train_images"] != 512 or t["optimizer_steps"] != 2560
                or t["fixed_endpoint_epoch"] != 5 or t["tiny_checkpoint_used"] or t["architecture_changed"]
                or t["same_initialization_sha256"] != engine.resolution.CHECKPOINT_SHA
                or t["external_teacher_used"] != (role == "kd") or t["prediction_kd_components"]
                or t["feature_loss_scale"] != scale
                or engine.sha256(a.output / role / "last.pth") != t["checkpoint_sha256"]
                or not value["complete"] or value["evaluated_images"] != 3769
                or value["run_signature"] != m["signature_sha256"] or value["checkpoint_sha256"] != t["checkpoint_sha256"]):
            raise RuntimeError("Both fresh matched fixed5-epoch endpoints are required")
        training[role], results[role] = t, value
    if (training["control"]["device_identity"] != training["kd"]["device_identity"]
            or any(training["control"]["active_feature_counts"].values())
            or min(training["kd"]["active_feature_counts"].get(c, 0) for c in ("Pedestrian", "Vehicle")) <= 0):
        raise RuntimeError("Fresh matched-device/feature-scope checks failed")
    decision = reference.decision(results["control"], results["kd"])
    passed = decision["kd_benefit_guardrail_passed"]
    next_step = ("Review feature-KD benefit and repeatability before longer training" if passed else
                 "Do not extend this feature recipe automatically; review accuracy, recall and gradient evidence")
    decision["next_review"] = next_step
    engine.save_report(a.output / "feature_comparison.json", dict(complete=True, revision=REVISION,
        run_signature=m["signature_sha256"], recipe=RECIPE, guardrails=GUARDRAILS,
        results=results, versus_label_only=decision, benefit_guardrail_passed=passed,
        fresh_matched_control=True, single_seed=True, validation_checkpoint_selection=False,
        training_scope="512 TRAIN frames; five fixed epochs per arm; not full-data training",
        comparison_limit="One bounded feature recipe/seed; not proof that all feature KD succeeds or fails",
        full_training_authorized=False, deployment_qualified=False, phone_retest_required=True, next_review=next_step))
    print(json.dumps(dict(versus_label_only=decision, next_review=next_step), indent=2), flush=True)


def bundle(a):
    path = a.output / "a2_lite_feature_results.zip"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for source in sorted(a.output.rglob("*")):
            if source.is_file() and "assets" not in source.relative_to(a.output).parts and source.suffix in (".json", ".csv", ".txt", ".log"):
                archive.write(source, source.relative_to(a.output))
    print(path, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "smoke", "tiny", "cache", "feature-smoke", "train", "evaluate", "compare", "bundle"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--role", choices=("control", "kd"))
    for name in ("pilot", "initialization", "teacher-checkpoint", "teacher-export", "compact", "repo", "dataset", "split-dir"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--candidate", type=Path, action="append", default=[])
    a = parser.parse_args()
    a.output_verified = False
    for name in ("output", "pilot", "initialization", "teacher_checkpoint", "teacher_export", "compact", "repo", "dataset", "split_dir"):
        if getattr(a, name) is not None:
            setattr(a, name, getattr(a, name).resolve())
    a.candidate = [p.resolve() for p in a.candidate]
    if a.stage == "prepare" and any(getattr(a, k) is None for k in ("pilot", "initialization", "teacher_checkpoint", "teacher_export", "compact", "repo", "dataset", "split_dir")):
        parser.error("prepare requires original pilot, initialization, A2 checkpoint/export, compact export, new source/data and splits")
    if a.stage in ("train", "evaluate") and a.role is None:
        parser.error("train/evaluate require --role")
    try:
        with recipe_context():
            if a.stage == "prepare":
                prepare(a)
            elif a.stage == "bundle":
                saved = a.output / "feature_manifest.json"
                if saved.exists() and json.loads(saved.read_text()).get("revision") != REVISION:
                    raise RuntimeError("Bundle only this new feature output")
                a.output.mkdir(parents=True, exist_ok=True)
                bundle(a)
            else:
                saved = a.output / "feature_manifest.json"
                a.output_verified = saved.is_file() and json.loads(saved.read_text()).get("revision") == REVISION
                m = load(a.output)
                if m["execution_identity"] != device_identity(a.device):
                    raise RuntimeError("Stage device differs from the frozen matched-run identity")
                dict(smoke=engine.smoke, tiny=engine.tiny, cache=cache, **{"feature-smoke": feature_smoke},
                     train=train, evaluate=engine.evaluate, compare=compare)[a.stage](a, m)
    except Exception as error:
        if a.output_verified:
            import traceback
            engine.write_json(a.output / "failures" / f"{a.stage}_{time.time_ns()}.json",
                              dict(complete=False, error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == "__main__":
    main()
