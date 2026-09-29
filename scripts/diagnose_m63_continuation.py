"""M63b post-hoc continuation diagnosis. No training or promotion authorization."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from m63_common import (AP_GATES, ROOT, check_split, load_manifest,
                        read_json, sha256, write_json)


def decide(source, control, student):
    """Predeclared epoch-10 rule. A pass permits review, never automatic training."""
    checks = {}
    checks["vehicle_3d_gain_over_control_ge_0_10"] = student["vehicle_3d_moderate"] >= max(source["vehicle_3d_moderate"], control["vehicle_3d_moderate"]) + 0.10
    for k in ("pedestrian_3d_moderate", "pedestrian_bev_moderate", "pedestrian_near_recall", "mean_3d_moderate"):
        checks[k + "_preserved_vs_source_and_control"] = student[k] >= max(source[k], control[k])
    checks["vehicle_bev_drop_vs_control_le_0_15"] = student["vehicle_bev_moderate"] >= max(source["vehicle_bev_moderate"], control["vehicle_bev_moderate"]) - 0.15
    checks["vehicle_recall_drop_le_0_01"] = student["vehicle_near_recall"] >= max(source["vehicle_near_recall"], control["vehicle_near_recall"]) - 0.01
    return checks


def checkpoint_for(m, variant, epoch):
    if variant == "baseline":
        if epoch != 0:
            raise ValueError("Baseline evaluation uses epoch 0 (frozen A2 epoch130 weights)")
        return Path(m["student"]["checkpoint"])
    if variant not in m["variants"] or epoch not in (1, 3, 5, 10):
        raise ValueError("Unregistered variant/evaluation epoch")
    return Path(m["variants"][variant]["run_dir"]) / f"checkpoint_epoch_{epoch}.pth"


def inference(m, variant, epoch, directory):
    import torch
    from m63_common import build_runtime, environment, seed_all
    seed_all(m["seed"])
    model, _, _ = build_runtime(m, "student")
    checkpoint = checkpoint_for(m, variant, epoch)
    if variant != "baseline":
        if checkpoint.with_suffix(".sha256").read_text().strip() != sha256(checkpoint):
            raise RuntimeError("Evaluation checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        binding = payload["m63"]
        if (binding["manifest_sha256"] != m["manifest_sha256"] or binding["variant"] != variant
                or binding["environment"] != environment() or payload["epoch"] != epoch
                or binding["audit_sha256"] != sha256(Path(m["output_dir"]) / "m63_teacher_audit.json")):
            raise RuntimeError("Evaluation checkpoint lineage mismatch")
        from m63_common import require_baseline
        if binding["baseline_sha256"] != require_baseline(m):
            raise RuntimeError("Training/evaluation baseline changed")
        model.load_state_dict(payload["model_state"], strict=True)
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    from lib.helpers.tester_helper import Tester
    cfg = copy.deepcopy(m["student"]["config"])
    dataset = KITTI_Dataset("val", cfg["dataset"])
    loader = torch.utils.data.DataLoader(dataset, batch_size=m["batch_size"], shuffle=False, num_workers=0)
    repo = Path(m["student"]["repo"])
    # Native Tester prepends './' to save_path; supply a relative path from its repo.
    cfg["trainer"]["save_path"] = os.path.relpath(directory, repo)
    os.chdir(repo)
    tester = Tester(cfg["tester"], model, loader, logging.getLogger("M63"),
                    train_cfg=cfg["trainer"], model_name="native")
    tester.inference()


def run(command, cwd=ROOT):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def evaluate_one(m, manifest_path, variant, epoch):
    output = Path(m["output_dir"])
    directory = output / "diagnostics_m63b" / "evaluation" / f"{variant}_epoch{epoch:03d}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_for(m, variant, epoch)
    prediction_dir = directory / "native/outputs/data"
    ids = check_split(Path(m["dataset_root"]) / "ImageSets/val.txt", "val")
    identity = dict(manifest_sha256=m["manifest_sha256"], variant=variant, epoch=epoch,
                    checkpoint_sha256=sha256(checkpoint))
    done = directory / "inference_manifest.json"
    if done.exists():
        old = read_json(done)
        if any(old.get(k) != v for k, v in identity.items()):
            raise RuntimeError("Cached evaluation belongs to different weights/provenance")
        if (set(old["prediction_files"]) != set(ids)
                or {p.stem for p in prediction_dir.glob("*.txt")} != set(ids)):
            raise RuntimeError("Cached validation predictions are incomplete")
        for image_id, digest in old["prediction_files"].items():
            if sha256(prediction_dir / f"{image_id}.txt") != digest:
                raise RuntimeError("Changed cached prediction file")
    else:
        run([sys.executable, "-u", Path(__file__), "--manifest", manifest_path,
             "--infer", variant, "--epoch", epoch])
        if {p.stem for p in prediction_dir.glob("*.txt")} != set(ids):
            raise RuntimeError("Expected exactly 3,769 validation prediction IDs")
        identity["prediction_files"] = {i: sha256(prediction_dir / f"{i}.txt") for i in ids}
        write_json(done, identity)
    split_dir = Path(m["dataset_root"]) / "ImageSets"
    run([sys.executable, "-u", ROOT / "scripts/evaluate_kitti_prediction_dir.py",
         "--config", ROOT / "configs/kitti_mobileadas3d_s1.yaml", "--profile", "colab_drive",
         "--dataset-root", m["dataset_root"], "--split-dir", split_dir,
         "--prediction-dir", prediction_dir, "--split", "val", "--classes", "Vehicle", "Pedestrian",
         "--source-name", f"M63_{variant}_{epoch}", "--output-dir", directory / "product"])
    run([sys.executable, "-u", ROOT / "scripts/audit_product_prediction_geometry.py",
         "--dataset-root", m["dataset_root"], "--split-file", split_dir / "val.txt",
         "--prediction-dir", prediction_dir, "--output-dir", directory / "nearby",
         "--checkpoint", checkpoint, "--expected-images", "3769",
         "--score-threshold", "0.001", "--match-iou-threshold", "0.5"])
    summary = read_json(directory / "product/kitti_r40_summary.json")
    if not summary.get("complete_split") or summary["evaluated_images"] != 3769:
        raise RuntimeError("Incomplete product evaluation")
    from sweep_monodetr_r0_product_checkpoints import metric_value
    metrics = {f"{cls.lower()}_{metric}_moderate": metric_value(summary, metric, cls)
               for cls in ("Vehicle", "Pedestrian") for metric in ("3d", "bev")}
    metrics["mean_3d_moderate"] = (metrics["vehicle_3d_moderate"] + metrics["pedestrian_3d_moderate"]) / 2
    nearby = read_json(directory / "nearby/nearby_geometry_summary.json")
    for cls in ("Vehicle", "Pedestrian"):
        metrics[f"{cls.lower()}_near_recall"] = nearby["classes"][cls]["near_recall"]
    if any(not __import__("math").isfinite(v) for v in metrics.values()):
        raise RuntimeError("Non-finite evaluation metric")
    return dict(variant=variant, epoch=epoch, checkpoint_sha256=sha256(checkpoint), manifest_sha256=m["manifest_sha256"], **metrics)


REVIEWED_COMPARISON = "7cd41fff09c27564252cfc61b3dce15272aad925bd5a2cc74c0ac744a6f3e3e5"
EPOCHS = (1, 3, 5)
REVISION = "M63b-2026-09-29-r1"

def buffer_drift(source, candidate):
    """Descriptive only: no statistics are restored/recalibrated."""
    import torch
    rows = []
    for name, original in source.items():
        if not name.endswith(("running_mean", "running_var", "num_batches_tracked")):
            continue
        if name not in candidate or candidate[name].shape != original.shape:
            raise RuntimeError(f"Normalization state mismatch: {name}")
        delta = candidate[name].detach().cpu().double() - original.detach().cpu().double()
        if not torch.isfinite(delta).all():
            raise RuntimeError(f"Non-finite normalization state: {name}")
        rows.append(dict(name=name, mean_absolute_change=float(delta.abs().mean()),
                         max_absolute_change=float(delta.abs().max())))
    return rows

def load_bound_checkpoint(m, variant, epoch):
    import torch
    from m63_common import environment, require_baseline
    path = checkpoint_for(m, variant, epoch)
    if path.with_suffix(".sha256").read_text().strip() != sha256(path):
        raise RuntimeError(f"Checkpoint checksum mismatch: {path}")
    p = torch.load(path, map_location="cpu", weights_only=True)
    b = p["m63"]
    if (p["epoch"] != epoch or b["variant"] != variant
            or b["manifest_sha256"] != m["manifest_sha256"]
            or b["environment"] != environment()
            or b["baseline_sha256"] != require_baseline(m)
            or b["audit_sha256"] != sha256(Path(m["output_dir"]) / "m63_teacher_audit.json")):
        raise RuntimeError("Diagnostic checkpoint lineage mismatch")
    return p

def recipe_audit(m, directory):
    import torch
    import yaml
    path = Path(m["student"]["checkpoint"])
    # Exact trusted A2 digest has already been checked by load_manifest().
    base = torch.load(path, map_location="cpu", weights_only=False)
    original_manifest = path.parent / "experiment_manifest.json"
    original = read_json(original_manifest) if original_manifest.is_file() else None
    upstream_path = Path(m["student"]["repo"]) / "configs/monodetr.yaml"
    upstream = yaml.safe_load(upstream_path.read_text())
    keys = ("batch_size", "aug_pd", "aug_crop", "random_flip", "random_crop")
    original_dataset = {k: upstream["dataset"].get(k) for k in keys}
    if original:
        original_dataset["batch_size"] = original.get("batch_size")
    source_optimizer = base.get("optimizer_state") or {}
    report = dict(complete=True, revision=REVISION, manifest_sha256=m["manifest_sha256"],
        source_checkpoint_sha256=sha256(path),
        original_experiment_manifest=original,
        original_experiment_manifest_sha256=sha256(original_manifest) if original else None,
        original_recipe_evidence="Dataset augmentation reconstructed from pinned upstream defaults and A2 preparer; not a saved runtime-config verification",
        upstream_config_sha256=sha256(upstream_path),
        original_dataset_reconstructed=original_dataset,
        continuation_dataset={k: m["student"]["config"]["dataset"].get(k) for k in keys},
        source_saved_optimizer_learning_rates=[g.get("lr") for g in source_optimizer.get("param_groups", [])],
        continuation_optimizer=m["student"]["config"]["optimizer"],
        continuation_optimizer_state="fresh; source optimizer state not restored",
        continuation_normalization="model.train(); no special BatchNorm freeze in M63 trainer",
        causality_established=False, training_authorized=False,
        note="Differences and buffer drift cannot isolate which change caused validation regression.")
    drift = []
    for variant in ("control", "vehicle_kd"):
        for epoch in (*EPOCHS, 10):
            payload = load_bound_checkpoint(m, variant, epoch)
            drift.append(dict(variant=variant, epoch=epoch,
                checkpoint_sha256=sha256(checkpoint_for(m,variant,epoch)),
                buffers=buffer_drift(base["model_state"],payload["model_state"])))
            del payload
            print(f"Audited normalization buffers: {variant} epoch {epoch}", flush=True)
    report["normalization_drift"] = drift
    write_json(directory / "m63b_recipe_and_normalization.json", report)
    return report

def summarize(rows):
    baseline = next(r for r in rows if r["variant"] == "baseline")
    metrics = ("vehicle_3d_moderate", "pedestrian_3d_moderate", "mean_3d_moderate",
               "vehicle_bev_moderate", "pedestrian_bev_moderate",
               "vehicle_near_recall", "pedestrian_near_recall")
    return [dict(variant=r["variant"], epoch=r["epoch"],
                 delta_vs_baseline={k:r[k]-baseline[k] for k in metrics})
            for r in rows if r["variant"] != "baseline"]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--infer", choices=("control","vehicle_kd"))
    p.add_argument("--epoch", type=int, choices=EPOCHS)
    args = p.parse_args()
    m = load_manifest(args.manifest)
    output = Path(m["output_dir"])
    comparison = output / "m63_pilot_comparison.json"
    if sha256(comparison) != REVIEWED_COMPARISON:
        raise RuntimeError("Not the reviewed M63 comparison; preserve it for inspection")
    old = read_json(comparison)
    if old["pilot_passed"] or old["manifest_sha256"] != m["manifest_sha256"]:
        raise RuntimeError("Unexpected source result")
    directory = output / "diagnostics_m63b"
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(revision=REVISION, manifest_sha256=m["manifest_sha256"],
                    comparison_sha256=REVIEWED_COMPARISON,
                    diagnostic_script_sha256=sha256(Path(__file__)))
    identity_path = directory / "identity.json"
    if identity_path.exists() and read_json(identity_path) != identity:
        raise RuntimeError("Diagnostic identity changed; preserve old results")
    write_json(identity_path, identity)
    if args.infer:
        if args.epoch is None:
            p.error("--infer requires --epoch")
        inference(m, args.infer, args.epoch,
                  directory / "evaluation" / f"{args.infer}_epoch{args.epoch:03d}")
        return
    recipe_audit(m, directory)
    rows = list(old["rows"])
    for epoch in EPOCHS:
        for variant in ("control", "vehicle_kd"):
            rows.append(evaluate_one(m, args.manifest, variant, epoch))
    report = dict(complete=True, **identity, rows=rows, deltas=summarize(rows),
        post_hoc_diagnostic=True, original_epoch10_pilot_passed=False,
        selected_checkpoint=None, training_authorized=False, deployment_authorized=False,
        causality_established=False,
        next_action="Review onset of regression before designing a separate controlled experiment")
    write_json(directory / "m63b_continuation_diagnostic.json", report)
    print(json.dumps(report,indent=2))
    print("STOP. No checkpoint promoted, no training authorized, M63 remains failed.")

if __name__ == "__main__":
    main()
