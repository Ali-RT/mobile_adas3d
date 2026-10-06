"""M66: paired one-epoch R0-to-A2 Vehicle depth-head feature distillation."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import m64_teacher_qualification as q
import m65_a2_preservation_control as c
from third_party.monodetr.m65c_preservation import preservation, RULES as KEEP_RULES, SCALES, WEIGHTS
from third_party.monodetr.m66_vehicle_feature_kd import DepthFeatureTap, RULES, configure_trainable, feature_kd

REVISION = "M66-R0-A2-VEHICLE-FEATURE-KD-2026-10-06-r1"
R0_SHA = "fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59"
A2_INPUT_SIGNATURE = "853fc19cb6f7af250f8397bb688129efcc82a4c67eb7cee7a7daf7032ecbf724"
RECIPE = dict(epochs=1, optimizer_steps=928, seed=444, batch_size=4,
              learning_rate=1e-6, weight_decay=0.0, gradient_clip=1.0,
              amp=False, preservation_weight=0.10, feature_kd_weight=0.10,
              trainable_scope="depth_embed.* only", dropout_disabled=True,
              batchnorm_buffers_frozen=True, batchnorm_affine_frozen=True,
              teacher_class="Vehicle only", feature="last depth head pre-output activation",
              teacher_temperature=None, mixup=False)
LIMITS = dict(ap_drop=0.15, nearby_recall_drop=0.005, vehicle_3d_gain=0.15)
BASELINE = dict(vehicle_3d_moderate=15.447527335986564,
                pedestrian_3d_moderate=7.528416451699579,
                vehicle_bev_moderate=21.377606101348228,
                pedestrian_bev_moderate=8.488669175064146)
NEARBY = dict(Vehicle=0.8829344841114163, Pedestrian=0.6926807760141094)
ROLES = ("baseline", "control", "kd")


def implementation_hash():
    files = [Path(__file__), ROOT / "third_party/monodetr/m66_vehicle_feature_kd.py",
             ROOT / "third_party/monodetr/m65c_preservation.py"]
    return q.signature(dict(files={str(p.relative_to(ROOT)): q.sha(p) for p in files},
                            reused_m65_implementation=c.implementation_hash()))


def teacher_config(a2):
    cfg = copy.deepcopy(a2)
    cfg["model"].update(backbone_source="torchvision", backbone="resnet50")
    for key in ("backbone_out_indices", "backbone_pretrained"):
        cfg["model"].pop(key, None)
    return cfg


def prepare(a):
    old = q.read_json(a.a2_manifest)
    if (old.get("revision") != "M65C-A2-SCALED-PRESERVATION-2026-10-06-r1"
            or q.signature({k: v for k, v in old.items() if k != "manifest_sha256"}) != old.get("manifest_sha256")
            or old.get("manifest_sha256") != A2_INPUT_SIGNATURE
            or old.get("kd_authorized") is not False):
        raise RuntimeError("Use the unchanged original M65c manifest, not its rejected control")
    entry = copy.deepcopy(old["models"]["a2"])
    if q.sha(entry["checkpoint"]) != q.A2_SHA or entry["checkpoint_sha256"] != q.A2_SHA:
        raise RuntimeError("Original A2 checkpoint is missing or changed")
    selection = q.read_json(a.r0_selection)
    selected = selection.get("selected", {})
    if not selection.get("complete") or selected.get("epoch") != 185 or selected.get("checkpoint_sha256") != R0_SHA:
        raise RuntimeError("R0 must be the frozen epoch185 Vehicle specialist")
    r0_checkpoint = Path(selected["checkpoint"])
    if q.sha(r0_checkpoint) != R0_SHA:
        raise RuntimeError("R0 checkpoint bytes changed")
    repo, dataset, output = a.a2_repo.resolve(), a.dataset_root.resolve(), a.output.resolve()
    if not repo.name.startswith("MonoDETR_M66"):
        raise RuntimeError("Use a fresh MonoDETR_M66 checkout; preserve historical sources")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != q.A2_COMMIT:
        raise RuntimeError("Wrong MonoDETR source pin")
    if q.source_hash(repo) != entry["patched_source_sha256"]:
        raise RuntimeError("Fresh patched A2 source differs from the reviewed baseline")
    identity = c.data_identity(dataset)
    if identity != old["data_identity"]:
        raise RuntimeError("Exact reviewed Chen labels/calibration/splits differ")
    build_path = repo / "lib/models/monodetr/ops/m64_build_receipt.json"
    build = q.read_json(build_path)
    if not build.get("import_verified") or q.sha(build["binary"]) != build["binary_sha256"]:
        raise RuntimeError("Build the attention extension in this fresh checkout")
    env = q.environment()
    if env["gpu"] is None or env["cuda"] != "13.0":
        raise RuntimeError("Use the private CUDA13 runtime with an allocated GPU")
    entry.update(repo=str(repo), attention_build_receipt=str(build_path),
                 attention_build_sha256=q.sha(build_path), attention_binary=build["binary"],
                 attention_binary_sha256=build["binary_sha256"])
    entry["config"]["dataset"]["root_dir"] = str(dataset)
    cfg = entry["config"]
    if (cfg["model"]["backbone"] != "mobilenetv4_conv_medium.e500_r256_in1k"
            or cfg["model"]["hidden_dim"] != 256 or cfg["model"]["num_classes"] != 3
            or cfg["dataset"].get("meanshape") is not False
            or cfg["dataset"].get("class_mapping") != q.CLASS_MAPPING):
        raise RuntimeError("Unexpected A2 architecture/geometry/taxonomy")
    training = copy.deepcopy(cfg)
    training["dataset"].update(batch_size=4, aug_pd=True, aug_crop=True,
                                random_flip=0.5, random_crop=0.5, random_mixup3d=0.0,
                                scale=0.05, shift=0.05)
    m = dict(schema_version=1, revision=REVISION, output=str(output), dataset_root=str(dataset),
             models=dict(a2=entry), r0=dict(checkpoint=str(r0_checkpoint), checkpoint_sha256=R0_SHA,
                                          config=teacher_config(cfg), epoch=185),
             a2_manifest=str(a.a2_manifest.resolve()), a2_manifest_file_sha256=q.sha(a.a2_manifest),
             r0_selection=str(a.r0_selection.resolve()), r0_selection_file_sha256=q.sha(a.r0_selection),
             environment=env, data_identity=identity, training_config=training,
             recipe=RECIPE, limits=LIMITS, feature_rules=RULES,
             preservation_rules=KEEP_RULES, preservation_scales=SCALES, preservation_weights=WEIGHTS,
             implementation_sha256=implementation_hash(), reviewed_a2_metrics=BASELINE,
             reviewed_a2_nearby=NEARBY, product_score_threshold=0.001, topk=50,
             kd_authorized=True, authorization_scope="one paired one-epoch R0 Vehicle feature pilot",
             passing_preservation_control_required=False, historical_manifests_modified=False,
             monoprio_kd_authorized=False, full_training_authorized=False,
             deployment_authorized=False, improved_student_selected=False)
    m["manifest_sha256"] = q.signature(m)
    path = output / "m66_manifest.json"
    if path.exists() and q.read_json(path) != m:
        raise RuntimeError("Run identity differs; keep existing files and choose a fresh RUN_ID")
    q.write_json(path, m)
    print(json.dumps(m, indent=2), flush=True)


def load(path):
    m = q.read_json(path)
    if (m.get("revision") != REVISION
            or q.signature({k: v for k, v in m.items() if k != "manifest_sha256"}) != m.get("manifest_sha256")):
        raise RuntimeError("M66 manifest changed")
    expected = dict(recipe=RECIPE, limits=LIMITS, feature_rules=RULES,
                    preservation_rules=KEEP_RULES, preservation_scales=SCALES, preservation_weights=WEIGHTS,
                    reviewed_a2_metrics=BASELINE, reviewed_a2_nearby=NEARBY, kd_authorized=True,
                    passing_preservation_control_required=False, monoprio_kd_authorized=False,
                    full_training_authorized=False, deployment_authorized=False)
    if any(m.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Frozen M66 recipe or authorization scope changed")
    if m["implementation_sha256"] != implementation_hash() or m["environment"] != q.environment():
        raise RuntimeError("Implementation/GPU/runtime changed; use a new RUN_ID and baseline")
    for field in ("a2_manifest", "r0_selection"):
        if q.sha(m[field]) != m[field + "_file_sha256"]:
            raise RuntimeError(f"Historical input changed: {field}")
    entry = m["models"]["a2"]
    if q.sha(entry["checkpoint"]) != q.A2_SHA or q.sha(m["r0"]["checkpoint"]) != R0_SHA:
        raise RuntimeError("Original A2 or R0 changed")
    if q.source_hash(Path(entry["repo"])) != entry["patched_source_sha256"]:
        raise RuntimeError("MonoDETR source changed")
    for key in ("attention_build", "attention_binary"):
        field = "attention_build_receipt" if key == "attention_build" else key
        if q.sha(entry[field]) != entry[key + "_sha256"]:
            raise RuntimeError("Attention build changed")
    if c.data_identity(Path(m["dataset_root"])) != m["data_identity"]:
        raise RuntimeError("KITTI identity changed")
    return m


def checkpoint_path(m, role):
    if role not in ("control", "kd"):
        raise ValueError("Only paired arms have new checkpoints")
    return Path(m["output"]) / role / "checkpoint_epoch_1.pth"


def validate_summary(m, role, summary):
    expected = dict(complete=True, role=role, manifest_sha256=m["manifest_sha256"],
                    completed_epochs=1, optimizer_steps=928, external_kd_enabled=role == "kd",
                    original_checkpoint_sha256=q.A2_SHA, running_buffers_unchanged=True,
                    frozen_parameters_unchanged=True, anchor_unchanged=True, teacher_unchanged=True,
                    inference_graph_unchanged=True)
    if any(summary.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Incomplete or wrong-arm training summary")
    means = summary.get("loss_means", {})
    if set(means) != {"gt", "preservation", "feature_kd", "total"} or not all(math.isfinite(v) for v in means.values()):
        raise RuntimeError("Invalid loss means")
    expected_loss = means["gt"] + RECIPE["preservation_weight"] * means["preservation"]
    expected_loss += (RECIPE["feature_kd_weight"] if role == "kd" else 0) * means["feature_kd"]
    if not math.isclose(means["total"], expected_loss, rel_tol=1e-5, abs_tol=1e-5):
        raise RuntimeError("Loss arithmetic differs from the paired treatment")
    names = summary.get("trainable_parameter_names", [])
    if not names or any(not name.startswith("depth_embed.") for name in names):
        raise RuntimeError("Unexpected learned parameter scope")
    hashes = summary.get("batch_input_sha256", [])
    if len(hashes) != 928 or any(not isinstance(h, str) or len(h) != 64 for h in hashes):
        raise RuntimeError("Missing paired training-input fingerprints")
    if summary.get("eligible_kd_pairs", {}).get("Pedestrian") != 0:
        raise RuntimeError("Pedestrian external KD is forbidden")
    if summary.get("eligible_kd_pairs", {}).get("Vehicle", 0) < 8:
        raise RuntimeError("Missing Vehicle transfer coverage")


def recover_completed(m, role):
    path = checkpoint_path(m, role)
    if not path.exists():
        return False
    payload = q.safe_payload(path)
    if payload.get("m66_manifest_sha256") != m["manifest_sha256"] or payload.get("epoch") != 1:
        raise RuntimeError("Existing checkpoint lineage differs")
    summary = payload.get("training_summary", {})
    validate_summary(m, role, summary)
    receipt = dict(complete=True, manifest_sha256=m["manifest_sha256"],
                   checkpoint_sha256=q.sha(path), completed_control_epoch=1, role=role)
    sidecar = path.with_suffix(".json")
    if sidecar.exists() and q.read_json(sidecar) != receipt:
        raise RuntimeError("Completed checkpoint receipt changed")
    q.write_json(sidecar, receipt)
    q.write_json(Path(m["output"]) / role / "training_summary.json", summary)
    return True


def evaluation_view(m, role):
    if role not in ROLES:
        raise ValueError("Unknown evaluation arm")
    view = copy.deepcopy(m)
    if role != "baseline":
        if not recover_completed(m, role):
            raise RuntimeError("Arm must finish its one epoch before evaluation")
        path = checkpoint_path(m, role)
        view["models"]["a2"].update(checkpoint=str(path), checkpoint_sha256=q.sha(path))
    view["output"] = str(Path(m["output"]) / "evaluation" / role)
    view["manifest_sha256"] = q.signature(dict(m66=m["manifest_sha256"], role=role,
        checkpoint_sha256=view["models"]["a2"]["checkpoint_sha256"]))
    return view


def metrics(m, role):
    view = evaluation_view(m, role)
    predictions = q.validate_predictions(view, "a2") / "product"
    output = Path(view["output"]) / "metrics"
    shared = ["--profile", "colab_drive", "--dataset-root", m["dataset_root"],
              "--split-dir", Path(m["dataset_root"]) / "ImageSets", "--prediction-dir", predictions,
              "--split", "val", "--source-name", f"M66_{role}"]
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


def validated_metrics(m, role):
    view = evaluation_view(m, role)
    q.validate_predictions(view, "a2")
    root = Path(view["output"]) / "metrics"
    result = q.read_json(root / "summary.json")
    if (not result.get("complete") or result.get("role") != role
            or result.get("manifest_sha256") != m["manifest_sha256"]
            or result.get("evaluation_identity") != view["manifest_sha256"]):
        raise RuntimeError("Stale or incomplete full evaluation")
    product = q.read_json(root / "product/kitti_r40_summary.json")
    geometry = q.read_json(root / "nearby/nearby_geometry_summary.json")
    actual = {k: v["near_recall"] for k, v in geometry["classes"].items()}
    if (not geometry.get("complete") or geometry.get("evaluated_images") != 3769
            or geometry.get("checkpoint_sha256") != view["models"]["a2"]["checkpoint_sha256"]
            or geometry.get("split_file_sha256") != q.SPLITS["val"][1]
            or set(result["product"]) != set(BASELINE)
            or any(not math.isfinite(v) or not 0 <= v <= 1 for v in actual.values())
            or result["product"] != q.metric_rows(product) or result["nearby"] != actual or set(actual) != set(NEARBY)):
        raise RuntimeError("Metric summaries disagree")
    return result


def baseline(m):
    result = validated_metrics(m, "baseline")
    checks = {k: abs(result["product"][k] - v) <= 0.15 for k, v in BASELINE.items()}
    checks.update({k + "_nearby": abs(result["nearby"][k] - v) <= 0.01 for k, v in NEARBY.items()})
    report = dict(complete=True, manifest_sha256=m["manifest_sha256"], checks=checks,
                   passed=all(checks.values()), optimizer_steps=0)
    q.write_json(Path(m["output"]) / "m66_baseline_gate.json", report)
    if not report["passed"]:
        raise RuntimeError("Unchanged A2 must reproduce before updates")
    return result


def training_runtime(m):
    import torch
    view = copy.deepcopy(m)
    view["models"]["a2"]["config"] = m["training_config"]
    model, criterion, dataset, epoch = q.runtime(view, "a2", "train")
    if epoch != 130:
        raise RuntimeError("Each arm must start from original A2 epoch130")
    dataset.data_augmentation = True
    anchor = copy.deepcopy(model).eval().requires_grad_(False)
    from lib.helpers.model_helper import build_model
    teacher, _ = build_model(m["r0"]["config"]["model"])
    payload = q.safe_payload(m["r0"]["checkpoint"])
    if payload.get("epoch") != 185:
        raise RuntimeError("R0 epoch mismatch")
    teacher.load_state_dict(payload["model_state"], strict=True)
    del payload
    teacher = teacher.cuda().eval().requires_grad_(False)
    names = configure_trainable(model)
    c.training_mode(model)
    criterion.train()
    # Model construction consumes RNG. Reset after constructing ALL models so
    # both arms receive exactly identical sample order and online augmentation.
    c.seed_all(RECIPE["seed"])
    return model, anchor, teacher, criterion, dataset, names


def losses(model, anchor, teacher, criterion, batch, student_tap, teacher_tap, role):
    import torch
    images, calibs, raw, _ = batch
    images, calibs, sizes = images.cuda(), calibs.cuda(), raw["img_size"].cuda()
    targets = c.gt_targets(raw)
    c.training_mode(model)
    student_tap.reset()
    native = model(images, calibs, targets, sizes, dn_args=None)
    student_tap.take()  # discard grouped GT feature; KD uses one inference group
    parts = criterion(native, targets, None)
    gt = sum(v * criterion.weight_dict[k] for k, v in parts.items() if k in criterion.weight_dict)
    with torch.no_grad():
        reference = anchor(images, calibs, None, sizes, dn_args=0)
        teacher_tap.reset()
        external = teacher(images, calibs, None, sizes, dn_args=0)
        external_features = teacher_tap.take()
    model.eval()
    student_tap.reset()
    outputs = model(images, calibs, None, sizes, dn_args=0)
    features = student_tap.take()
    keep, keep_counts = preservation(outputs, reference, targets, criterion.matcher)
    kd, kd_counts = feature_kd(outputs, external, reference, features, external_features, targets, criterion.matcher)
    total = gt + RECIPE["preservation_weight"] * keep["total"]
    # The control receives no external gradient; forward-only diagnostics are
    # identical, avoiding a teacher-construction/RNG difference between arms.
    if role == "kd":
        total = total + RECIPE["feature_kd_weight"] * kd
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Non-finite pilot loss; no update taken")
    c.training_mode(model)
    return total, gt, keep, kd, keep_counts, kd_counts


def frozen_state(model):
    return q.tensor_hash({k: v for k, v in model.named_parameters() if not v.requires_grad})


def finite_gradients(grads):
    import torch
    values = [g for g in grads if g is not None]
    return bool(values and all(bool(torch.isfinite(g).all()) for g in values)
                and any(bool((g != 0).any()) for g in values))


def atomic_checkpoint(m, role, payload):
    import torch
    path = checkpoint_path(m, role)
    if path.exists():
        raise RuntimeError("Refusing to overwrite a completed M66 arm")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pth.tmp")
    with temporary.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    q.write_json(path.with_suffix(".json"), dict(complete=True, manifest_sha256=m["manifest_sha256"],
                 checkpoint_sha256=q.sha(path), completed_control_epoch=1, role=role))


def smoke(m):
    import torch
    baseline(m)
    model, anchor, teacher, criterion, dataset, names = training_runtime(m)
    state = q.tensor_hash(model.state_dict())
    anchor_state, teacher_state = q.tensor_hash(anchor.state_dict()), q.tensor_hash(teacher.state_dict())
    taps = DepthFeatureTap(model), DepthFeatureTap(teacher)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0)
    parameters = [p for p in model.parameters() if p.requires_grad]
    rows, count = [], 0
    try:
        for i, batch in enumerate(loader):
            if i >= 16:
                break
            total, gt, keep, kd, keep_counts, kd_counts = losses(model, anchor, teacher, criterion, batch, *taps, "kd")
            count += kd_counts["Vehicle"]
            gg = torch.autograd.grad(gt, parameters, retain_graph=True, allow_unused=True)
            kg = torch.autograd.grad(kd, parameters, retain_graph=True, allow_unused=True)
            if not finite_gradients(gg):
                raise RuntimeError("Native GT has no finite nonzero head gradient")
            if kd_counts["Vehicle"] and not finite_gradients(kg):
                raise RuntimeError("Eligible KD features have no finite nonzero student-head gradient")
            rows.append(dict(batch=i + 1, vehicle_pairs=kd_counts["Vehicle"], pedestrian_pairs=kd_counts["Pedestrian"],
                             gt=float(gt.detach()), initial_preservation=float(keep["total"].detach()),
                             feature_kd=float(kd.detach()), finite_gt_gradient=True,
                             finite_kd_gradient=finite_gradients(kg)))
            if float(keep["total"].detach()) > 1e-3:
                raise RuntimeError("Original A2 and its preservation anchor disagree before updates")
            del total, gt, keep, kd, gg, kg
    finally:
        for tap in taps:
            tap.close()
    checks = dict(minimum_eight_vehicle_pairs=count >= 8,
                  parameters_and_buffers_unchanged=state == q.tensor_hash(model.state_dict()),
                  anchor_unchanged=anchor_state == q.tensor_hash(anchor.state_dict()),
                  teacher_unchanged=teacher_state == q.tensor_hash(teacher.state_dict()),
                  no_pedestrian_external_targets=all(r["pedestrian_pairs"] == 0 for r in rows))
    report = dict(complete=True, manifest_sha256=m["manifest_sha256"], optimizer_steps=0,
                   checked_train_images=4 * len(rows), eligible_vehicle_pairs=count,
                   trainable_parameter_names=names, rows=rows, checks=checks, passed=all(checks.values()))
    report["report_sha256"] = q.signature(report)
    q.write_json(Path(m["output"]) / "m66_training_smoke.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("M66 zero-update CUDA smoke failed; bundle evidence, do not loosen masks")


def validated_smoke(m):
    r = q.read_json(Path(m["output"]) / "m66_training_smoke.json")
    if (q.signature({k: v for k, v in r.items() if k != "report_sha256"}) != r.get("report_sha256")
            or r.get("manifest_sha256") != m["manifest_sha256"] or not r.get("passed")
            or r.get("optimizer_steps") != 0 or r.get("eligible_vehicle_pairs", 0) < 8
            or not r.get("checks") or not all(r["checks"].values())
            or len(r.get("rows", [])) != 16 or r.get("checked_train_images") != 64
            or any(row.get("pedestrian_pairs") != 0 or not row.get("finite_gt_gradient") for row in r.get("rows", []))
            or not any(row.get("vehicle_pairs", 0) > 0 and row.get("finite_kd_gradient") for row in r.get("rows", []))):
        raise RuntimeError("Matching successful zero-update smoke required")
    return r


def train(m, role):
    import torch
    if role not in ("control", "kd"):
        raise ValueError("Train only control or kd")
    baseline(m)
    smoke_report = validated_smoke(m)
    if recover_completed(m, role):
        print(f"{role}: completed one-epoch checkpoint verified; no extra updates", flush=True)
        return
    model, anchor, teacher, criterion, dataset, names = training_runtime(m)
    buffers_before, frozen_before = q.tensor_hash(dict(model.named_buffers())), frozen_state(model)
    anchor_before, teacher_before = q.tensor_hash(anchor.state_dict()), q.tensor_hash(teacher.state_dict())
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=RECIPE["learning_rate"], weight_decay=0.0)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0, shuffle=True,
                                         generator=torch.Generator().manual_seed(RECIPE["seed"]))
    sums = dict(gt=0.0, preservation=0.0, feature_kd=0.0, total=0.0)
    counts = dict(Vehicle=0, Pedestrian=0, vehicle_near=0, vehicle_far=0)
    input_hashes = []
    keep_pairs = {key: {0: 0, 1: 0} for key in WEIGHTS}
    taps = DepthFeatureTap(model), DepthFeatureTap(teacher)
    try:
        for step, batch in enumerate(loader, start=1):
            images, calibs, raw, info = batch
            fingerprint = dict(image=images, calib=calibs, **{k: v for k, v in raw.items() if isinstance(v, torch.Tensor)})
            input_hashes.append(q.tensor_hash(fingerprint))
            optimizer.zero_grad(set_to_none=True)
            total, gt, keep, kd, kp, kc = losses(model, anchor, teacher, criterion, batch, *taps, role)
            total.backward()
            if not finite_gradients([p.grad for p in parameters]):
                raise RuntimeError(f"Non-finite/zero gradient at {role} step{step}; no update")
            norm = torch.nn.utils.clip_grad_norm_(parameters, RECIPE["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            values = dict(gt=float(gt.detach()), preservation=float(keep["total"].detach()),
                          feature_kd=float(kd.detach()), total=float(total.detach()))
            for k, value in values.items():
                sums[k] += value
            for k, value in kc.items():
                counts[k] += value
            for k in keep_pairs:
                for cls in (0, 1):
                    keep_pairs[k][cls] += kp[k][cls]
            if step == 1 or step % 20 == 0 or step == len(loader):
                print(f"M66 {role} epoch=1/1 batch={step:04d}/{len(loader):04d} "
                      f"loss={values} preclip_norm={float(norm):.5f} kd_pairs={kc}", flush=True)
            del total, gt, keep, kd
    finally:
        for tap in taps:
            tap.close()
    if len(loader) != 928 or counts["Vehicle"] < 8 or any(keep_pairs["depth"][cls] == 0 for cls in (0, 1)):
        raise RuntimeError("Wrong update count or missing KD/both-class preservation coverage")
    summary = dict(complete=True, role=role, manifest_sha256=m["manifest_sha256"],
                    completed_epochs=1, optimizer_steps=len(loader), loss_means={k: v / len(loader) for k, v in sums.items()},
                    eligible_kd_pairs=counts, preservation_pairs=keep_pairs, batch_input_sha256=input_hashes,
                    trainable_parameter_names=names, smoke_report_sha256=smoke_report["report_sha256"],
                    running_buffers_unchanged=buffers_before == q.tensor_hash(dict(model.named_buffers())),
                    frozen_parameters_unchanged=frozen_before == frozen_state(model),
                    anchor_unchanged=anchor_before == q.tensor_hash(anchor.state_dict()),
                    teacher_unchanged=teacher_before == q.tensor_hash(teacher.state_dict()),
                    external_kd_enabled=role == "kd", original_checkpoint_sha256=q.A2_SHA,
                    inference_graph_unchanged=True)
    validate_summary(m, role, summary)
    path = checkpoint_path(m, role)
    # Atomic commit helper rejects overwrites. Recover missing sidecar/summary on
    # the next invocation from the embedded signed-lineage training summary.
    atomic_checkpoint(m, role, dict(epoch=1, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                        m66_manifest_sha256=m["manifest_sha256"], training_summary=summary))
    q.write_json(Path(m["output"]) / role / "training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def acceptance(baseline_result, control, kd):
    checks = {}
    for label, source in (("original_a2", baseline_result), ("matched_control", control)):
        checks[label + "_vehicle_3d_gain"] = kd["product"]["vehicle_3d_moderate"] >= source["product"]["vehicle_3d_moderate"] + LIMITS["vehicle_3d_gain"]
        for key in BASELINE:
            if key != "vehicle_3d_moderate":
                checks[label + "_" + key] = kd["product"][key] >= source["product"][key] - LIMITS["ap_drop"]
        for key in NEARBY:
            checks[label + "_" + key + "_nearby"] = kd["nearby"][key] >= source["nearby"][key] - LIMITS["nearby_recall_drop"]
    return checks


def review(m):
    source = baseline(m)
    control, kd = (validated_metrics(m, role) for role in ("control", "kd"))
    summaries = {role: q.read_json(Path(m["output"]) / role / "training_summary.json") for role in ("control", "kd")}
    for role, summary in summaries.items():
        validate_summary(m, role, summary)
    if summaries["control"]["batch_input_sha256"] != summaries["kd"]["batch_input_sha256"]:
        raise RuntimeError("Arms used different training inputs/order; paired comparison invalid")
    checks = acceptance(source, control, kd)
    historical = dict(vehicle_3d_moderate=15.8713, pedestrian_3d_moderate=5.1493,
                      vehicle_bev_moderate=21.3134, pedestrian_bev_moderate=5.9365)
    historical_checks = {k: kd["product"][k] >= v for k, v in historical.items()}
    historical_checks["mean_3d_moderate"] = (kd["product"]["vehicle_3d_moderate"] + kd["product"]["pedestrian_3d_moderate"]) / 2 >= 10.5103
    report = dict(complete=True, revision=REVISION, manifest_sha256=m["manifest_sha256"],
                   original_a2=source, control=control, kd=kd, paired_inputs_identical=True,
                   acceptance_checks=checks, pilot_passed=all(checks.values()),
                   control_stable=all(c.control_checks(source, control).values()),
                   kd_minus_original={k: kd["product"][k] - source["product"][k] for k in BASELINE},
                   kd_minus_control={k: kd["product"][k] - control["product"][k] for k in BASELINE},
                   historical_accuracy_gates=historical_checks,
                   nearby_product_gates={"Vehicle": kd["nearby"]["Vehicle"] >= 0.85,
                                         "Pedestrian": kd["nearby"]["Pedestrian"] >= 0.80},
                   improved_student_selected=False, original_a2_remains_selected=True,
                   second_seed_required_if_promising=True, full_training_authorized=False,
                   deployment_authorized=False, optimizer_steps_per_arm=928)
    q.write_json(Path(m["output"]) / "m66_paired_gate.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("STOP FOR REVIEW regardless of pass. No extra epochs, new teacher or automatic promotion.", flush=True)


def bundle(output):
    path = output / "m66_results.zip"
    files = [p for p in output.rglob("*") if p.is_file() and p.suffix in {".json", ".csv", ".log", ".yaml"}
             and "predictions" not in p.relative_to(output).parts]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(files):
            archive.write(p, p.relative_to(output))
    print(f"Return {path}; raw checkpoints/predictions remain on Drive.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "infer", "metrics", "baseline", "smoke", "train", "review", "bundle"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--a2-manifest", type=Path)
    parser.add_argument("--r0-selection", type=Path)
    parser.add_argument("--a2-repo", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--model", choices=ROLES)
    a = parser.parse_args()
    if a.stage == "bundle":
        if a.output is None:
            parser.error("bundle requires --output")
        bundle(a.output)
        return
    if a.stage == "prepare":
        if any(getattr(a, key) is None for key in ("output", "a2_manifest", "r0_selection", "a2_repo", "dataset_root")):
            parser.error("prepare requires all source/data/output arguments")
        prepare(a)
        return
    if a.manifest is None:
        parser.error("--manifest is required")
    if a.stage in ("infer", "metrics", "train") and a.model is None:
        parser.error("--model is required")
    m = load(a.manifest)
    if a.stage == "infer":
        q.infer(evaluation_view(m, a.model), "a2")
    elif a.stage == "metrics":
        metrics(m, a.model)
    elif a.stage == "baseline":
        baseline(m)
    elif a.stage == "smoke":
        smoke(m)
    elif a.stage == "train":
        train(m, a.model)
    else:
        review(m)


if __name__ == "__main__":
    main()
