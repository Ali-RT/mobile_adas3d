"""Full Chen-val evaluation at fixed epochs; no validation-based teacher filtering."""
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

from m61_common import (AP_GATES, ROOT, check_split, load_manifest,
                        read_json, sha256, write_json)


def decide(source, control, student):
    """Predeclared epoch-10 rule. A pass permits review, never automatic training."""
    checks = {k: student[k] >= v for k, v in AP_GATES.items()}
    checks["vehicle_3d_gain_over_control_ge_0_10"] = student["vehicle_3d_moderate"] >= control["vehicle_3d_moderate"] + 0.10
    for k in ("pedestrian_3d_moderate", "pedestrian_bev_moderate", "pedestrian_near_recall", "mean_3d_moderate"):
        checks[k + "_preserved_vs_source_and_control"] = student[k] >= max(source[k], control[k])
    checks["vehicle_bev_drop_vs_control_le_0_15"] = student["vehicle_bev_moderate"] >= control["vehicle_bev_moderate"] - 0.15
    checks["vehicle_recall_drop_le_0_01"] = student["vehicle_near_recall"] >= max(source["vehicle_near_recall"], control["vehicle_near_recall"]) - 0.01
    return checks


def checkpoint_for(m, variant, epoch):
    if variant == "baseline":
        if epoch != 0:
            raise ValueError("Baseline evaluation uses epoch 0 (frozen A2 epoch130 weights)")
        return Path(m["student"]["checkpoint"])
    if variant not in m["variants"] or epoch not in m["evaluation_epochs"]:
        raise ValueError("Unregistered variant/evaluation epoch")
    return Path(m["variants"][variant]["run_dir"]) / f"checkpoint_epoch_{epoch}.pth"


def inference(m, variant, epoch, directory):
    import torch
    from m61_common import build_runtime, environment, seed_all
    seed_all(m["seed"])
    model, _, _ = build_runtime(m, "student")
    checkpoint = checkpoint_for(m, variant, epoch)
    if variant != "baseline":
        if checkpoint.with_suffix(".sha256").read_text().strip() != sha256(checkpoint):
            raise RuntimeError("Evaluation checkpoint hash mismatch")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        binding = payload["m61"]
        if (binding["manifest_sha256"] != m["manifest_sha256"] or binding["variant"] != variant
                or binding["environment"] != environment() or payload["epoch"] != epoch
                or binding["audit_sha256"] != sha256(Path(m["output_dir"]) / "m61_teacher_audit.json")):
            raise RuntimeError("Evaluation checkpoint lineage mismatch")
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
    tester = Tester(cfg["tester"], model, loader, logging.getLogger("M61"),
                    train_cfg=cfg["trainer"], model_name="native")
    tester.inference()


def run(command, cwd=ROOT):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def evaluate_one(m, manifest_path, variant, epoch):
    output = Path(m["output_dir"])
    directory = output / "evaluation" / f"{variant}_epoch{epoch:03d}"
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
         "--source-name", f"M61_{variant}_{epoch}", "--output-dir", directory / "product"])
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
    return dict(variant=variant, epoch=epoch, checkpoint_sha256=sha256(checkpoint), **metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--infer", choices=("baseline", "control", "vehicle_kd"))
    p.add_argument("--epoch", type=int)
    p.add_argument("--baseline-only", action="store_true")
    args = p.parse_args()
    m = load_manifest(args.manifest)
    output = Path(m["output_dir"])
    if args.infer:
        inference(m, args.infer, args.epoch, output / "evaluation" / f"{args.infer}_epoch{args.epoch:03d}")
        return
    baseline = evaluate_one(m, args.manifest, "baseline", 0)
    write_json(output / "m61_baseline_metrics.json", baseline)
    if args.baseline_only:
        print(json.dumps(baseline, indent=2))
        return
    rows = [baseline]
    for epoch in m["evaluation_epochs"]:
        for variant in ("control", "vehicle_kd"):
            summary = read_json(Path(m["variants"][variant]["run_dir"]) / "training_summary.json")
            if not summary["complete"] or summary["epoch"] != 10 or summary["manifest_sha256"] != m["manifest_sha256"]:
                raise RuntimeError("Both ten-epoch runs must complete before final comparison")
            rows.append(evaluate_one(m, args.manifest, variant, epoch))
    control = next(r for r in rows if r["variant"] == "control" and r["epoch"] == 10)
    student = next(r for r in rows if r["variant"] == "vehicle_kd" and r["epoch"] == 10)
    gates = decide(baseline, control, student)
    report = dict(schema_version=1, complete=True, manifest_sha256=m["manifest_sha256"],
                  decision_epoch=10, epoch5_is_diagnostic_only=True, rows=rows,
                  gate_results=gates, pilot_passed=all(gates.values()),
                  confirmation_review_recommended=all(gates.values()),
                  full_run_authorized=False, phone_deployment_authorized=False,
                  product_pedestrian_recall_target_met=student["pedestrian_near_recall"] >= 0.80)
    write_json(output / "m61_pilot_comparison.json", report)
    with (output / "m61_pilot_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))
    print("STOP for review. Do not start a longer run, threshold sweep, or phone deployment.")


if __name__ == "__main__":
    main()
