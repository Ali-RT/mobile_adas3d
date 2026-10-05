"""M65: one teacher-independent GT + original-A2 preservation epoch."""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import zipfile

import m64_teacher_qualification as q

ROOT = Path(__file__).resolve().parents[1]
REVISION = "M65-A2-PRESERVATION-2026-10-05-r1"
REVIEWED_M64_MANIFEST = "10cfd5663b8cb999794d2b0600f71ecc6df976d55f7ce806f666e71bb13d8ce8"
REVIEWED_M64_REPORT = "4b256b1b27050198e787618f732c3398929084357838261aaa8a95a19dfc8554"
RECIPE = dict(epochs=1, seed=444, batch_size=4, learning_rate=1e-6,
              weight_decay=0.0, gradient_clip=1.0, preservation_weight=1.0,
              amp=False, batchnorm_buffers_frozen=True, batchnorm_affine_frozen=True,
              dropout_disabled=True, external_teacher=False, mixup=False)
CONTROL_LIMITS = dict(ap_drop=0.15, nearby_recall_drop=0.005)


def implementation_hash():
    paths = [Path(__file__), ROOT / "third_party/monodetr/m65_preservation.py",
             ROOT / "scripts/audit_m65_prior_provenance.py"]
    return q.signature(dict(files={str(p.relative_to(ROOT)): q.sha(p) for p in paths},
                            reused_m64_implementation=q.implementation_hash()))


def reviewed_m64(root):
    manifest, report = root / "m64_manifest.json", root / "m64_teacher_qualification.json"
    if q.sha(manifest) != REVIEWED_M64_MANIFEST or q.sha(report) != REVIEWED_M64_REPORT:
        raise RuntimeError("Use the exact reviewed M64 r2 result, not a rewritten or different run")
    m, r = q.read_json(manifest), q.read_json(report)
    plain = {k: v for k, v in m.items() if k != "manifest_sha256"}
    if q.signature(plain) != m["manifest_sha256"] or r["manifest_sha256"] != m["manifest_sha256"]:
        raise RuntimeError("M64 result identity mismatch")
    if not (r.get("complete") and r.get("a2_baseline_reproduced") and r.get("teacher_native_reproduced")):
        raise RuntimeError("M64 qualification prerequisites did not pass")
    if r.get("optimizer_steps") != 0 or r.get("kd_authorized") is not False:
        raise RuntimeError("Unexpected M64 scope")
    return m, r


def data_identity(dataset):
    ids = {s: q.split_ids(dataset / "ImageSets" / f"{s}.txt", s) for s in q.SPLITS}
    if set(ids["train"]) & set(ids["val"]):
        raise RuntimeError("Train/val overlap")
    identity = {}
    for split, samples in ids.items():
        files = {}
        for sample in samples:
            for folder, suffix in (("image_2", ".png"), ("label_2", ".txt"), ("calib", ".txt")):
                p = dataset / "training" / folder / (sample + suffix)
                if not p.is_file():
                    raise FileNotFoundError(p)
                if folder != "image_2":
                    files[f"{folder}/{sample}{suffix}"] = q.sha(p)
        identity[split] = q.signature(files)
    return identity


def prepare(a):
    from third_party.monodetr.m65_preservation import RULES, WEIGHTS
    old, review = reviewed_m64(a.m64_output)
    output, repo, dataset = a.output.resolve(), a.a2_repo.resolve(), a.dataset_root.resolve()
    if not repo.name.startswith("MonoDETR_M65"):
        raise RuntimeError("Use a fresh MonoDETR_M65 checkout; do not modify historical sources")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != q.A2_COMMIT:
        raise RuntimeError("Wrong A2 upstream pin")
    entry = copy.deepcopy(old["models"]["a2"])
    if entry["config"]["dataset"].get("meanshape") is not False:
        raise RuntimeError("M65 expects physical dimension targets, not mean-shape residuals")
    if q.sha(entry["checkpoint"]) != q.A2_SHA or q.source_hash(repo) != entry["patched_source_sha256"]:
        raise RuntimeError("Original A2 weights or repaired inference source changed")
    build_path = repo / "lib/models/monodetr/ops/m64_build_receipt.json"
    build = q.read_json(build_path)
    if not build.get("import_verified") or q.sha(build["binary"]) != build["binary_sha256"]:
        raise RuntimeError("Missing or changed locally built attention binary")
    identity = data_identity(dataset)
    if identity != old["data_identity"]:
        raise RuntimeError("KITTI labels/calibration/splits differ from the reviewed M64 run")
    entry.update(repo=str(repo), attention_build_receipt=str(build_path),
                 attention_build_sha256=q.sha(build_path), attention_binary=build["binary"],
                 attention_binary_sha256=build["binary_sha256"])
    entry["config"]["dataset"]["root_dir"] = str(dataset)
    train_cfg = copy.deepcopy(entry["config"])
    train_cfg["dataset"].update(batch_size=4, aug_pd=True, aug_crop=True,
                                random_flip=0.5, random_crop=0.5, random_mixup3d=0.0,
                                scale=0.05, shift=0.05)
    env = q.environment()
    if env["gpu"] is None or env["cuda"] != "13.0":
        raise RuntimeError("Use the working isolated CUDA13 runtime and a GPU")
    m = dict(schema_version=1, revision=REVISION, output=str(output), dataset_root=str(dataset),
             models=dict(a2=entry), training_config=train_cfg, data_identity=identity,
             m64_output=str(a.m64_output.resolve()), reviewed_m64_manifest_sha256=REVIEWED_M64_MANIFEST,
             reviewed_m64_report_sha256=REVIEWED_M64_REPORT, environment=env,
             implementation_sha256=implementation_hash(), recipe=RECIPE,
             preservation_rules=RULES, preservation_weights=WEIGHTS, limits=CONTROL_LIMITS,
             reviewed_a2_metrics=review["rows"]["a2"]["product"],
             reviewed_a2_nearby={k: v["near_recall"] for k, v in review["rows"]["a2"]["nearby"].items()},
             product_score_threshold=0.001, topk=50, kd_authorized=False,
             external_teacher_selected=False, full_training_authorized=False,
             deployment_authorized=False, baseline_checkpoint_overwritten=False)
    m["manifest_sha256"] = q.signature(m)
    p = output / "m65_manifest.json"
    if p.exists() and q.read_json(p) != m:
        raise RuntimeError("M65 identity changed; preserve this run and choose a new RUN_ID")
    q.write_json(p, m)
    print(json.dumps(m, indent=2), flush=True)


def load(path):
    from third_party.monodetr.m65_preservation import RULES, WEIGHTS
    m = q.read_json(path)
    digest = m["manifest_sha256"]
    if m.get("revision") != REVISION or q.signature({k: v for k, v in m.items() if k != "manifest_sha256"}) != digest:
        raise RuntimeError("M65 manifest changed")
    if (m["recipe"] != RECIPE or m["limits"] != CONTROL_LIMITS or m["kd_authorized"] is not False
            or m["preservation_rules"] != RULES or m["preservation_weights"] != WEIGHTS):
        raise RuntimeError("Fixed control recipe changed or external KD enabled")
    if m["implementation_sha256"] != implementation_hash() or m["environment"] != q.environment():
        raise RuntimeError("Source/runtime identity differs; use a new RUN_ID, do not rewrite the manifest")
    entry = m["models"]["a2"]
    if q.sha(entry["checkpoint"]) != q.A2_SHA or q.source_hash(Path(entry["repo"])) != entry["patched_source_sha256"]:
        raise RuntimeError("Frozen original A2 identity changed")
    if q.sha(entry["attention_build_receipt"]) != entry["attention_build_sha256"] or q.sha(entry["attention_binary"]) != entry["attention_binary_sha256"]:
        raise RuntimeError("Attention build changed")
    if data_identity(Path(m["dataset_root"])) != m["data_identity"]:
        raise RuntimeError("KITTI data identity changed")
    reviewed_m64(Path(m["m64_output"]))
    return m


def checkpoint_path(m):
    return Path(m["output"]) / "control/checkpoint_epoch_1.pth"


def evaluation_view(m, role):
    view = copy.deepcopy(m)
    if role == "control":
        path = checkpoint_path(m)
        receipt = q.read_json(path.with_suffix(".json"))
        if receipt["manifest_sha256"] != m["manifest_sha256"] or receipt["checkpoint_sha256"] != q.sha(path):
            raise RuntimeError("Missing or changed control checkpoint transaction")
        view["models"]["a2"].update(checkpoint=str(path), checkpoint_sha256=q.sha(path))
    view["output"] = str(Path(m["output"]) / "evaluation" / role)
    view["manifest_sha256"] = q.signature(dict(m65=m["manifest_sha256"], role=role,
        checkpoint_sha256=view["models"]["a2"]["checkpoint_sha256"]))
    return view


def evaluate(m, role):
    view = evaluation_view(m, role)
    predictions = q.validate_predictions(view, "a2") / "product"
    output = Path(view["output"]) / "metrics"
    shared = ["--profile", "colab_drive", "--dataset-root", m["dataset_root"],
              "--split-dir", Path(m["dataset_root"]) / "ImageSets", "--prediction-dir", predictions,
              "--split", "val", "--source-name", f"M65_{role}"]
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
    directory = Path(view["output"]) / "metrics"
    summary = q.read_json(directory / "summary.json")
    if not summary.get("complete") or summary["manifest_sha256"] != m["manifest_sha256"] or summary["evaluation_identity"] != view["manifest_sha256"]:
        raise RuntimeError("Missing or stale complete evaluation")
    if summary["product"] != q.metric_rows(q.read_json(directory / "product/kitti_r40_summary.json")):
        raise RuntimeError("Evaluation summary changed")
    actual_nearby = {k: v["near_recall"] for k, v in q.read_json(directory / "nearby/nearby_geometry_summary.json")["classes"].items()}
    if summary["nearby"] != actual_nearby or set(actual_nearby) != {"Vehicle", "Pedestrian"}:
        raise RuntimeError("Nearby summary changed")
    return summary


def baseline(m):
    result = validated_metrics(m, "baseline")
    checks = {key: abs(result["product"][key] - value) <= 0.15 for key, value in m["reviewed_a2_metrics"].items()}
    checks.update({key + "_nearby": abs(result["nearby"][key] - value) <= 0.01 for key, value in m["reviewed_a2_nearby"].items()})
    report = dict(complete=True, manifest_sha256=m["manifest_sha256"], checks=checks,
                   passed=all(checks.values()), optimizer_steps=0)
    q.write_json(Path(m["output"]) / "m65_baseline_gate.json", report)
    if not report["passed"]:
        raise RuntimeError("Unchanged A2 does not reproduce M64; stop before any optimizer update")
    return result


def training_mode(model):
    import torch
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()
            for parameter in module.parameters(recurse=False):
                parameter.requires_grad_(False)
        if isinstance(module, torch.nn.modules.dropout._DropoutNd):
            module.eval()
        if isinstance(module, torch.nn.MultiheadAttention):
            # Attention uses functional dropout, not a child nn.Dropout.
            module.dropout = 0.0


def seed_all(value):
    import numpy as np
    import torch
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.cuda.manual_seed_all(value)


def gt_targets(raw):
    return [{k: raw[k][i][raw["mask_2d"][i]].cuda() for k in q.TARGET_KEYS}
            for i in range(len(raw["labels"]))]


def training_runtime(m):
    import torch
    view = copy.deepcopy(m)
    view["models"]["a2"]["config"] = m["training_config"]
    model, criterion, dataset, epoch = q.runtime(view, "a2", "train")
    if epoch != 130:
        raise RuntimeError("Control must initialize from A2 epoch130")
    dataset.data_augmentation = True
    anchor = copy.deepcopy(model).eval().requires_grad_(False)
    training_mode(model)
    criterion.train()
    return model, anchor, criterion, dataset


def losses(model, anchor, criterion, batch):
    from third_party.monodetr.m65_preservation import preservation
    import torch
    images, calibs, raw, _ = batch
    images, calibs, sizes = images.cuda(), calibs.cuda(), raw["img_size"].cuda()
    targets = gt_targets(raw)
    training_mode(model)
    train_outputs = model(images, calibs, targets, sizes, dn_args=None)
    parts = criterion(train_outputs, targets, None)
    gt = sum(value * criterion.weight_dict[key] for key, value in parts.items() if key in criterion.weight_dict)
    # The anchor and differentiable student inference group see the exact same
    # augmented tensors. No stale/untransformed prediction cache is used.
    with torch.no_grad():
        reference = anchor(images, calibs, None, sizes, dn_args=0)
    model.eval()
    inference_outputs = model(images, calibs, None, sizes, dn_args=0)
    keep, counts = preservation(inference_outputs, reference, targets, criterion.matcher)
    training_mode(model)
    total = gt + m_preservation_weight() * keep["total"]
    if not bool(torch.isfinite(total)):
        raise RuntimeError("Non-finite GT/preservation loss; no optimizer update taken")
    return total, gt, keep, counts, parts, inference_outputs, reference, targets


def m_preservation_weight():
    return RECIPE["preservation_weight"]


def smoke(m):
    import torch
    from third_party.monodetr.m65_preservation import preservation
    baseline(m)
    seed_all(RECIPE["seed"])
    model, anchor, criterion, dataset = training_runtime(m)
    before = q.tensor_hash(dict(model.named_parameters()))
    buffers = q.tensor_hash(dict(model.named_buffers()))
    anchor_before = q.tensor_hash(anchor.state_dict())
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)
    eligible = False
    for index, batch in enumerate(loader):
        total, gt, keep, counts, _, outputs, reference, targets = losses(model, anchor, criterion, batch)
        if sum(counts["classification"].values()):
            eligible = True
            break
        if index == 31:
            break
    if not eligible:
        raise RuntimeError("No reliable preservation pairs in first32 train images; stop and review masks")
    initial_preservation = float(keep["total"].detach())
    if initial_preservation > 1e-5:
        raise RuntimeError("Identical original-A2 copies disagree before training")
    # At exact equality preservation gradient is correctly zero. A small output
    # perturbation exercises a nonzero gradient through the real student graph.
    perturbed = dict(outputs, pred_logits=outputs["pred_logits"] + 0.1,
                     pred_boxes=outputs["pred_boxes"] + 0.005)
    probe, _ = preservation(perturbed, reference, targets, criterion.matcher)
    grads = torch.autograd.grad(probe["total"], [p for p in model.parameters() if p.requires_grad],
                                retain_graph=True, allow_unused=True)
    gradient_ok = any(g is not None and bool((g != 0).any()) for g in grads)
    gradient_ok &= all(g is None or bool(torch.isfinite(g).all()) for g in grads)
    total.backward()
    gt_gradient_ok = any(p.grad is not None for p in model.parameters()) and all(
        p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters())
    unchanged = before == q.tensor_hash(dict(model.named_parameters()))
    buffers_unchanged = buffers == q.tensor_hash(dict(model.named_buffers()))
    anchor_unchanged = anchor_before == q.tensor_hash(anchor.state_dict())
    report = dict(complete=True, manifest_sha256=m["manifest_sha256"], optimizer_steps=0,
                   sample_id=f"{int(batch[3]['img_id'][0]):06d}", gt_loss=float(gt.detach()),
                   initial_preservation_loss=initial_preservation, paired_counts=counts,
                   finite_nonzero_preservation_probe_gradients=bool(gradient_ok),
                   finite_gt_gradients=bool(gt_gradient_ok), parameters_unchanged=unchanged,
                   running_buffers_unchanged=buffers_unchanged, anchor_unchanged=anchor_unchanged,
                   same_transformed_inputs=True, external_kd_enabled=False,
                   passed=bool(gradient_ok and gt_gradient_ok and unchanged and buffers_unchanged and anchor_unchanged))
    q.write_json(Path(m["output"]) / "m65_training_smoke.json", report)
    print(json.dumps(report, indent=2), flush=True)
    if not report["passed"]:
        raise RuntimeError("M65 CUDA smoke failed; stop before training")


def atomic_checkpoint(path, payload, m):
    import torch
    if path.exists():
        raise RuntimeError("Refusing to overwrite a completed M65 checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".pth.tmp")
    with temp.open("wb") as stream:
        torch.save(payload, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)
    q.write_json(path.with_suffix(".json"), dict(complete=True, manifest_sha256=m["manifest_sha256"],
                 checkpoint_sha256=q.sha(path), completed_control_epoch=1))


def validate_training_summary(m, summary):
    expected = dict(complete=True, manifest_sha256=m["manifest_sha256"], completed_epochs=1,
                    optimizer_steps=928, running_buffers_unchanged=True, anchor_unchanged=True,
                    external_kd_enabled=False, original_checkpoint_sha256=q.A2_SHA)
    if any(summary.get(key) != value for key, value in expected.items()):
        raise RuntimeError("Incomplete or changed one-epoch training summary")
    means = summary.get("loss_means", {})
    if set(means) != {"gt", "preservation", "total"} or not all(math.isfinite(value) for value in means.values()):
        raise RuntimeError("Invalid completed-control loss summary")


def train(m):
    import torch
    baseline(m)
    smoke_report = q.read_json(Path(m["output"]) / "m65_training_smoke.json")
    if not smoke_report.get("passed") or smoke_report["manifest_sha256"] != m["manifest_sha256"]:
        raise RuntimeError("Matching successful CUDA smoke required before training")
    path = checkpoint_path(m)
    if path.exists():
        payload = q.safe_payload(path)
        if payload.get("m65_manifest_sha256") != m["manifest_sha256"] or payload.get("epoch") != 1:
            raise RuntimeError("Existing checkpoint is not this completed control")
        validate_training_summary(m, payload.get("training_summary", {}))
        sidecar = path.with_suffix(".json")
        if sidecar.exists():
            receipt = q.read_json(sidecar)
            if (receipt.get("checkpoint_sha256") != q.sha(path)
                    or receipt.get("manifest_sha256") != m["manifest_sha256"]
                    or receipt.get("completed_control_epoch") != 1 or receipt.get("complete") is not True):
                raise RuntimeError("Existing checkpoint changed; preserve it for review")
        else:
            # Recover an interrupted sidecar after a complete atomic checkpoint.
            q.write_json(sidecar, dict(complete=True, manifest_sha256=m["manifest_sha256"],
                         checkpoint_sha256=q.sha(path), completed_control_epoch=1))
        q.write_json(Path(m["output"]) / "m65_training_summary.json", payload["training_summary"])
        print("One-epoch control already complete; no additional optimizer updates.", flush=True)
        return
    seed_all(RECIPE["seed"])
    model, anchor, criterion, dataset = training_runtime(m)
    buffers_before = q.tensor_hash(dict(model.named_buffers()))
    anchor_before = q.tensor_hash(anchor.state_dict())
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=RECIPE["learning_rate"], weight_decay=RECIPE["weight_decay"])
    generator = torch.Generator().manual_seed(RECIPE["seed"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, generator=generator,
                                         num_workers=0, drop_last=False)
    sums = dict(gt=0.0, preservation=0.0, total=0.0)
    counts_total = {key: {0: 0, 1: 0} for key in m["preservation_weights"]}
    for step, batch in enumerate(loader, start=1):
        optimizer.zero_grad(set_to_none=True)
        total, gt, keep, counts, parts, outputs, reference, targets = losses(model, anchor, criterion, batch)
        total.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise RuntimeError(f"Non-finite gradient at step{step}; no optimizer update")
        norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], RECIPE["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        values = dict(gt=float(gt.detach()), preservation=float(keep["total"].detach()), total=float(total.detach()))
        for key in sums:
            sums[key] += values[key]
        for key in counts_total:
            for cls in (0, 1):
                counts_total[key][cls] += counts[key][cls]
        if step == 1 or step % 20 == 0 or step == len(loader):
            primary = {k: round(float(v.detach()), 5) for k, v in parts.items()
                       if k in criterion.weight_dict and not k[-1:].isdigit()}
            print(f"M65 epoch=001/001 batch={step:04d}/{len(loader):04d} lr={RECIPE['learning_rate']:.8f} "
                  f"gt={values['gt']:.6f} preservation={values['preservation']:.6f} total={values['total']:.6f} "
                  f"grad_norm={float(norm):.4f} pairs={counts} native_parts={primary}", flush=True)
        del total, gt, keep, parts, outputs, reference, targets
    if len(loader) != math.ceil(3712 / RECIPE["batch_size"]) or not all(counts_total["classification"].values()):
        raise RuntimeError("Wrong update count or a product class has no preservation pairs")
    buffers_unchanged = buffers_before == q.tensor_hash(dict(model.named_buffers()))
    anchor_unchanged = anchor_before == q.tensor_hash(anchor.state_dict())
    if not buffers_unchanged or not anchor_unchanged:
        raise RuntimeError("Frozen running buffers or original-A2 anchor changed; stop")
    summary = dict(complete=True, manifest_sha256=m["manifest_sha256"], completed_epochs=1,
                    optimizer_steps=len(loader), loss_means={k: v / len(loader) for k, v in sums.items()},
                    preservation_pairs=counts_total, trainable_parameter_names=trainable_names,
                    running_buffers_unchanged=buffers_unchanged, anchor_unchanged=anchor_unchanged,
                    external_kd_enabled=False, original_checkpoint_sha256=q.A2_SHA)
    atomic_checkpoint(path, dict(epoch=1, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(),
                       m65_manifest_sha256=m["manifest_sha256"], training_summary=summary), m)
    q.write_json(Path(m["output"]) / "m65_training_summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


def control_checks(source, candidate):
    checks = {k: candidate["product"][k] >= value - CONTROL_LIMITS["ap_drop"] for k, value in source["product"].items()}
    checks.update({k + "_nearby": candidate["nearby"][k] >= value - CONTROL_LIMITS["nearby_recall_drop"]
                   for k, value in source["nearby"].items()})
    return checks


def review(m):
    source, candidate = baseline(m), validated_metrics(m, "control")
    summary = q.read_json(Path(m["output"]) / "m65_training_summary.json")
    validate_training_summary(m, summary)
    checks = control_checks(source, candidate)
    historical = dict(vehicle_3d_moderate=15.8713, pedestrian_3d_moderate=5.1493,
                      vehicle_bev_moderate=21.3134, pedestrian_bev_moderate=5.9365)
    gates = {k: candidate["product"][k] >= v for k, v in historical.items()}
    gates["mean_3d_moderate"] = (candidate["product"]["vehicle_3d_moderate"] + candidate["product"]["pedestrian_3d_moderate"]) / 2 >= 10.5103
    report = dict(complete=True, revision=REVISION, manifest_sha256=m["manifest_sha256"],
                   baseline=source, control=candidate, preservation_checks=checks,
                   control_stable=all(checks.values()), historical_accuracy_gates=gates,
                   nearby_product_gates={"Vehicle": candidate["nearby"]["Vehicle"] >= 0.85,
                                         "Pedestrian": candidate["nearby"]["Pedestrian"] >= 0.80},
                   candidate_minus_baseline={k: candidate["product"][k] - v for k, v in source["product"].items()},
                   optimizer_steps=928, improved_student_selected=False, kd_authorized=False,
                   full_training_authorized=False, deployment_authorized=False,
                   next_step="Review control stability and resolve teacher prior provenance before freezing one paired feature-KD treatment")
    q.write_json(Path(m["output"]) / "m65_control_gate.json", report)
    print(json.dumps(report, indent=2), flush=True)
    print("STOP FOR REVIEW. A stable control is not teacher qualification or permission for a full run.", flush=True)


def bundle(output):
    path = output / "m65_results.zip"
    files = [p for p in output.rglob("*") if p.is_file() and p.suffix in {".json", ".log", ".csv", ".yaml"}
             and "predictions" not in p.relative_to(output).parts]
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            z.write(p, p.relative_to(output))
    print(f"Return {path}", flush=True)


def main():
    sys.path.insert(0, str(ROOT))
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "infer", "metrics", "baseline", "smoke", "train", "review", "bundle"))
    p.add_argument("--manifest", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--m64-output", type=Path)
    p.add_argument("--a2-repo", type=Path)
    p.add_argument("--dataset-root", type=Path)
    p.add_argument("--model", choices=("baseline", "control"))
    a = p.parse_args()
    if a.stage == "bundle":
        bundle(a.output)
        return
    if a.stage == "prepare":
        prepare(a)
        return
    m = load(a.manifest)
    if a.stage in ("infer", "metrics") and not a.model:
        p.error("--model baseline|control is required")
    if a.stage == "infer":
        q.infer(evaluation_view(m, a.model), "a2")
    elif a.stage == "metrics":
        evaluate(m, a.model)
    elif a.stage == "baseline":
        baseline(m)
    elif a.stage == "smoke":
        smoke(m)
    elif a.stage == "train":
        train(m)
    elif a.stage == "review":
        review(m)


if __name__ == "__main__":
    main()
