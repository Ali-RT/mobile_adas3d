from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.kitti_parser import parse_kitti_label_file
from data.class_taxonomy import map_objects
from data.kitti_prediction_parser import load_kitti_prediction_directory
from data.kitti_r40 import (
    PRODUCT_IOU_THRESHOLDS,
    PRODUCT_NEIGHBOR_CLASSES,
    evaluate_kitti_r40,
)
from data.split_resolver import get_split_file
from tools.config import apply_runtime_overrides, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate an external KITTI-format prediction directory with the "
            "MobileADAS3D AP_R40 implementation."
        )
    )
    parser.add_argument("--config", default="configs/kitti_mnv4_quality_scoring_v3.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument(
        "--classes",
        nargs="+",
        default=None,
        help="Classes to evaluate. Defaults to dataset.classes from the config.",
    )
    parser.add_argument("--source-name", default="external_kitti_predictions")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Treat missing prediction files as empty. Results remain marked incomplete.",
    )
    return parser.parse_args()


def write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = apply_runtime_overrides(
        load_config(args.config),
        profile=args.profile,
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
    )
    dataset_cfg = config["dataset"]
    profile = dataset_cfg["active_profile"]
    dataset_root = Path(dataset_cfg["profiles"][profile]["root_dir"])
    split_file = Path(get_split_file(config, args.split))
    sample_ids = [
        Path(line.strip()).stem
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not sample_ids:
        raise ValueError(f"Split file is empty: {split_file}")

    classes = list(args.classes or dataset_cfg["classes"])
    prediction_dir = Path(args.prediction_dir)
    missing_ids = [
        sample_id
        for sample_id in sample_ids
        if not (prediction_dir / f"{sample_id}.txt").is_file()
    ]
    class_mapping = dataset_cfg.get("class_mapping")
    prediction_input_classes = (
        list(class_mapping) if class_mapping else classes
    )
    predictions = load_kitti_prediction_directory(
        prediction_dir=prediction_dir,
        sample_ids=sample_ids,
        allowed_classes=prediction_input_classes,
        require_all_files=not args.allow_missing,
    )
    if class_mapping:
        predictions = {
            sample_id: map_objects(items, class_mapping)
            for sample_id, items in predictions.items()
        }

    label_dir = dataset_root / dataset_cfg["label_dir"]
    ground_truth = {}
    for sample_id in sample_ids:
        raw_objects = [
            asdict(obj)
            for obj in parse_kitti_label_file(label_dir / f"{sample_id}.txt")
        ]
        ground_truth[sample_id] = (
            map_objects(raw_objects, class_mapping)
            if class_mapping
            else raw_objects
        )
    product_taxonomy = bool(class_mapping)
    results = evaluate_kitti_r40(
        ground_truth=ground_truth,
        predictions=predictions,
        classes=classes,
        iou_thresholds=PRODUCT_IOU_THRESHOLDS if product_taxonomy else None,
        neighbor_classes=PRODUCT_NEIGHBOR_CLASSES if product_taxonomy else None,
    )
    rows = [result.to_dict() for result in results]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "kitti_r40_metrics.csv")

    complete_split = len(missing_ids) == 0
    summary = {
        "protocol": (
            "KITTI-difficulty product-taxonomy AP_R40 external predictions"
            if product_taxonomy
            else "KITTI AP_R40 external prediction directory"
        ),
        "official_kitti_leaderboard_metric": not product_taxonomy,
        "split_protocol": dataset_cfg["splits"].get("protocol", "unknown"),
        "split": args.split,
        "source_name": args.source_name,
        "prediction_dir": str(prediction_dir),
        "classes": classes,
        "evaluated_images": len(sample_ids),
        "split_images": len(sample_ids),
        "prediction_files": len(sample_ids) - len(missing_ids),
        "missing_prediction_files": len(missing_ids),
        "missing_prediction_ids_preview": missing_ids[:20],
        "complete_split": complete_split,
        "metrics": rows,
    }
    (output_dir / "kitti_r40_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Source: {args.source_name}")
    print(f"Prediction directory: {prediction_dir}")
    print(f"Split file: {split_file}")
    print(
        f"Prediction files: {len(sample_ids) - len(missing_ids)}/{len(sample_ids)}"
    )
    print("\nKITTI AP_R40 (%):")
    for metric in ("bev", "3d"):
        print(f"  {metric.upper()}")
        for class_name in classes:
            class_results = [
                result
                for result in results
                if result.metric == metric and result.class_name == class_name
            ]
            values = " / ".join(f"{result.ap_r40:.2f}" for result in class_results)
            print(f"    {class_name}: {values}  (easy/moderate/hard)")
    if not complete_split:
        print("WARNING: missing prediction files; results are not reportable.")
    print(f"Saved evaluation artifacts to {output_dir}")


if __name__ == "__main__":
    main()
