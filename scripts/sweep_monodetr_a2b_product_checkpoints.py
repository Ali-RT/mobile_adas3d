from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


GATES = {
    "vehicle_3d_moderate": 15.8713,
    "pedestrian_3d_moderate": 5.1493,
    "mean_3d_moderate": 10.5103,
    "vehicle_bev_moderate": 21.3134,
    "pedestrian_bev_moderate": 5.9365,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the restartable product sweep and apply A2b accuracy gates."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--mobile-repo", type=Path, required=True)
    parser.add_argument("--training-config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--product-config", default="configs/kitti_mobileadas3d_s1.yaml")
    parser.add_argument("--profile", default="colab_drive")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--epochs", type=int, nargs="*")
    args = parser.parse_args()

    command = [
        sys.executable,
        "-u",
        "scripts/sweep_monodetr_r0_product_checkpoints.py",
        "--monodetr-repo",
        str(args.monodetr_repo),
        "--mobile-repo",
        str(args.mobile_repo),
        "--training-config",
        str(args.training_config),
        "--run-dir",
        str(args.run_dir),
        "--dataset-root",
        str(args.dataset_root),
        "--split-dir",
        str(args.split_dir),
        "--output-dir",
        str(args.output_dir),
        "--product-config",
        args.product_config,
        "--profile",
        args.profile,
        "--score-threshold",
        str(args.score_threshold),
        "--topk",
        str(args.topk),
        "--source-name-prefix",
        "MobileMonoDETR_A2b",
    ]
    if args.epochs:
        command.extend(["--epochs", *map(str, args.epochs)])
    subprocess.run(command, cwd=args.mobile_repo, check=True)

    selection_path = args.output_dir / "r0_product_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection["selected"]
    gate_results = {
        name: float(selected[name]) >= threshold
        for name, threshold in GATES.items()
    }
    report = {
        "schema_version": 1,
        "complete": bool(selection.get("complete")),
        "selected_epoch": selected["epoch"],
        "selected_checkpoint": selected["checkpoint"],
        "metrics": {name: float(selected[name]) for name in GATES},
        "gates": GATES,
        "gate_results": gate_results,
        "all_accuracy_gates_passed": all(gate_results.values()),
        "nearby_recall_review_required": True,
    }
    report_path = args.output_dir / "a2b_product_selection.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
