from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path


SOURCE_A2 = {
    "vehicle_3d_moderate": 15.457290707384603,
    "pedestrian_3d_moderate": 7.532768186994984,
    "mean_3d_moderate": 11.495029447189793,
    "vehicle_bev_moderate": 21.37498561663574,
    "pedestrian_bev_moderate": 8.489171811764757,
    "vehicle_near_recall": 0.8829344841114162,
    "pedestrian_near_recall": 0.6922398589065256,
    "pedestrian_localization_failure_rate": 0.23015873015873015,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def is_eligible(row: dict, control: dict) -> bool:
    return bool(
        row["variant"] != "control_w1_0"
        and row["pedestrian_near_gain_vs_control"] >= 0.02
        and row["pedestrian_localization_reduction_vs_control"] >= 0.02
        and row["vehicle_near_recall"] >= control["vehicle_near_recall"] - 0.01
        and row["vehicle_3d_delta_vs_control"] >= -0.15
        and row["vehicle_bev_delta_vs_control"] >= -0.15
        and row["pedestrian_3d_moderate"] >= control["pedestrian_3d_moderate"]
        and row["pedestrian_bev_moderate"] >= control["pedestrian_bev_moderate"]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the paired A2d gate.")
    parser.add_argument("--mobile-repo", type=Path, required=True)
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--product-config", default="configs/kitti_mobileadas3d_s1.yaml")
    parser.add_argument("--profile", default="colab_drive")
    args = parser.parse_args()

    mobile_repo = args.mobile_repo.resolve()
    monodetr_repo = args.monodetr_repo.resolve()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    gate_epochs = int(manifest["gate_epochs"])
    output_root = manifest_path.parent
    rows = []
    for name, variant in manifest["variants"].items():
        run_dir = Path(variant["run_dir"])
        checkpoint = run_dir / f"checkpoint_epoch_{gate_epochs}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        checkpoint_sha256 = sha256_file(checkpoint)
        eval_dir = output_root / "evaluation" / name
        run(
            [
                sys.executable,
                "-u",
                "scripts/sweep_monodetr_r0_product_checkpoints.py",
                "--monodetr-repo",
                str(monodetr_repo),
                "--mobile-repo",
                str(mobile_repo),
                "--training-config",
                variant["config"],
                "--run-dir",
                str(run_dir),
                "--dataset-root",
                str(args.dataset_root),
                "--split-dir",
                str(args.split_dir),
                "--output-dir",
                str(eval_dir),
                "--product-config",
                args.product_config,
                "--profile",
                args.profile,
                "--score-threshold",
                "0.001",
                "--topk",
                "50",
                "--source-name-prefix",
                f"MobileMonoDETR_A2d_{name}",
                "--epochs",
                str(gate_epochs),
            ],
            mobile_repo,
        )
        prediction_dir = run_dir / "outputs/data"
        nearby_dir = eval_dir / "nearby_geometry"
        run(
            [
                sys.executable,
                "-u",
                "scripts/audit_product_prediction_geometry.py",
                "--dataset-root",
                str(args.dataset_root),
                "--split-file",
                str(args.split_dir / "val.txt"),
                "--prediction-dir",
                str(prediction_dir),
                "--output-dir",
                str(nearby_dir),
                "--checkpoint",
                str(checkpoint),
                "--expected-images",
                "3769",
                "--score-threshold",
                "0.001",
                "--match-iou-threshold",
                "0.5",
            ],
            mobile_repo,
        )
        miss_dir = eval_dir / "pedestrian_false_negatives"
        run(
            [
                sys.executable,
                "-u",
                "scripts/diagnose_a2_pedestrian_false_negatives.py",
                "--dataset-root",
                str(args.dataset_root),
                "--split-file",
                str(args.split_dir / "val.txt"),
                "--prediction-dir",
                str(prediction_dir),
                "--output-dir",
                str(miss_dir),
                "--checkpoint",
                str(checkpoint),
                "--expected-checkpoint-sha256",
                checkpoint_sha256,
                "--expected-images",
                "3769",
                "--score-threshold",
                "0.001",
                "--iou-threshold",
                "0.5",
                "--weak-iou-threshold",
                "0.1",
            ],
            mobile_repo,
        )
        selection = json.loads((eval_dir / "r0_product_selection.json").read_text())
        metrics = selection["selected"]
        nearby = json.loads((nearby_dir / "nearby_geometry_summary.json").read_text())
        miss = json.loads(
            (miss_dir / "a2_pedestrian_false_negative_summary.json").read_text()
        )
        rates = miss["near_failure_rates"]
        rows.append(
            {
                "variant": name,
                "pedestrian_box_loss_weight": variant["pedestrian_box_loss_weight"],
                **{key: float(metrics[key]) for key in SOURCE_A2 if "recall" not in key and "failure" not in key},
                "vehicle_near_recall": nearby["classes"]["Vehicle"]["near_recall"],
                "pedestrian_near_recall": nearby["classes"]["Pedestrian"]["near_recall"],
                "pedestrian_localization_failure_rate": rates.get("localization_failure", 0.0),
                "pedestrian_missing_query_rate": rates.get("missing_query", 0.0),
                "pedestrian_subthreshold_rate": rates.get("subthreshold_well_localized_query", 0.0),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": checkpoint_sha256,
            }
        )

    control = next(row for row in rows if row["variant"] == "control_w1_0")
    for row in rows:
        row["pedestrian_near_gain_vs_control"] = row["pedestrian_near_recall"] - control["pedestrian_near_recall"]
        row["pedestrian_localization_reduction_vs_control"] = control["pedestrian_localization_failure_rate"] - row["pedestrian_localization_failure_rate"]
        row["vehicle_3d_delta_vs_control"] = row["vehicle_3d_moderate"] - control["vehicle_3d_moderate"]
        row["vehicle_bev_delta_vs_control"] = row["vehicle_bev_moderate"] - control["vehicle_bev_moderate"]
        row["eligible"] = is_eligible(row, control)
    eligible = [row for row in rows if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["pedestrian_near_recall"],
            -row["pedestrian_localization_failure_rate"],
            row["mean_3d_moderate"],
        ),
        default=None,
    )
    report = {
        "schema_version": 1,
        "complete": True,
        "source_a2": SOURCE_A2,
        "eligibility_rule": {
            "pedestrian_near_gain_vs_control_min": 0.02,
            "pedestrian_localization_reduction_vs_control_min": 0.02,
            "vehicle_near_recall_drop_max": 0.01,
            "vehicle_3d_drop_max": 0.15,
            "vehicle_bev_drop_max": 0.15,
            "pedestrian_3d_no_regression": True,
            "pedestrian_bev_no_regression": True,
        },
        "variants": rows,
        "selected": selected,
        "full_run_authorized": selected is not None,
    }
    (output_root / "a2d_gate_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "a2d_gate_comparison.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
