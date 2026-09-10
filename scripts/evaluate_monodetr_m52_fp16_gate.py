from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


R0 = {
    "vehicle_3d_moderate": 17.634769196266316,
    "pedestrian_3d_moderate": 5.721371354710236,
    "mean_3d_moderate": 11.678070275488276,
    "vehicle_bev_moderate": 23.68156332656625,
    "pedestrian_bev_moderate": 6.596148868813419,
    "vehicle_near_recall": 0.8824637112593173,
    "pedestrian_near_recall": 0.6834215167548501,
    "pedestrian_localization_failure_rate": 0.24603174603174602,
}
MINIMUMS = {key: value * 0.95 for key, value in R0.items() if "recall" not in key and "failure" not in key}
MINIMUMS.update({"vehicle_near_recall": R0["vehicle_near_recall"] - 0.01, "pedestrian_near_recall": R0["pedestrian_near_recall"] - 0.01})
MAX_LOCALIZATION = R0["pedestrian_localization_failure_rate"] + 0.01


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate M52 FP16 against frozen R0.")
    parser.add_argument("--mobile-repo", type=Path, required=True)
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    args = parser.parse_args()
    mobile, monodetr = args.mobile_repo.resolve(), args.monodetr_repo.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.manifest.resolve().parent
    evaluation = output / "evaluation"
    run([sys.executable, "-u", "scripts/sweep_monodetr_r0_product_checkpoints.py", "--monodetr-repo", str(monodetr), "--mobile-repo", str(mobile), "--training-config", manifest["config"], "--run-dir", manifest["run_dir"], "--dataset-root", str(args.dataset_root), "--split-dir", str(args.split_dir), "--output-dir", str(evaluation), "--product-config", "configs/kitti_mobileadas3d_s1.yaml", "--profile", "colab_drive", "--score-threshold", "0.001", "--topk", "50", "--source-name-prefix", "MonoDETR_M52_FP16", "--epochs", "185"], mobile)
    prediction = Path(manifest["run_dir"]) / "outputs/data"
    nearby_dir = evaluation / "nearby_geometry"
    run([sys.executable, "-u", "scripts/audit_product_prediction_geometry.py", "--dataset-root", str(args.dataset_root), "--split-file", str(args.split_dir / "val.txt"), "--prediction-dir", str(prediction), "--output-dir", str(nearby_dir), "--checkpoint", manifest["evaluation_checkpoint"], "--expected-images", "3769", "--score-threshold", "0.001", "--match-iou-threshold", "0.5"], mobile)
    miss_dir = evaluation / "pedestrian_false_negatives"
    run([sys.executable, "-u", "scripts/diagnose_a2_pedestrian_false_negatives.py", "--dataset-root", str(args.dataset_root), "--split-file", str(args.split_dir / "val.txt"), "--prediction-dir", str(prediction), "--output-dir", str(miss_dir), "--checkpoint", manifest["evaluation_checkpoint"], "--expected-checkpoint-sha256", manifest["r0_checkpoint_sha256"], "--expected-images", "3769", "--score-threshold", "0.001", "--iou-threshold", "0.5", "--weak-iou-threshold", "0.1"], mobile)
    metrics = json.loads((evaluation / "r0_product_selection.json").read_text())["selected"]
    nearby = json.loads((nearby_dir / "nearby_geometry_summary.json").read_text())
    miss = json.loads((miss_dir / "a2_pedestrian_false_negative_summary.json").read_text())
    row = {key: float(metrics[key]) for key in MINIMUMS if key in metrics}
    row.update({"vehicle_near_recall": nearby["classes"]["Vehicle"]["near_recall"], "pedestrian_near_recall": nearby["classes"]["Pedestrian"]["near_recall"], "pedestrian_localization_failure_rate": miss["near_failure_rates"].get("localization_failure", 0.0)})
    gates = {key: row[key] >= floor for key, floor in MINIMUMS.items()}
    gates["pedestrian_localization_failure_rate"] = row["pedestrian_localization_failure_rate"] <= MAX_LOCALIZATION
    gates["complete_prediction_set"] = len(list(prediction.glob("*.txt"))) == 3769
    report = {"schema_version": 1, "complete": True, "precision": "fp16_autocast", "source_r0": R0, "requirements": {**MINIMUMS, "pedestrian_localization_failure_rate_max": MAX_LOCALIZATION, "prediction_files": 3769}, "candidate": row, "gate_results": gates, "all_gates_passed": all(gates.values()), "compression_rung_authorized": all(gates.values()), "product_safety_qualified": False}
    (output / "m52_fp16_gate_comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with (output / "m52_fp16_gate_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row) + list(gates)); writer.writeheader(); writer.writerow({**row, **gates})
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
