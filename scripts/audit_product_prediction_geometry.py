from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.class_taxonomy import KITTI_PRODUCTION_CLASS_MAPPING, map_objects
from data.kitti_parser import parse_kitti_label_file
from data.kitti_prediction_parser import load_kitti_prediction_directory
from data.kitti_r40 import bev_iou, iou_3d
from data.matching import bbox_iou
from data.splits import read_split_file

CLASSES = ("Vehicle", "Pedestrian")
NEAR_LIMITS_M = {"Vehicle": 40.0, "Pedestrian": 30.0}
NEAR_RECALL_GATES = {"Vehicle": 0.85, "Pedestrian": 0.80}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_label_dir(dataset_root: Path) -> Path:
    for relative in ("training/label_2", "training/label_02"):
        candidate = dataset_root / relative
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"KITTI labels not found under {dataset_root}")


def angle_error_degrees(prediction: float, target: float) -> float:
    difference = (prediction - target + math.pi) % (2.0 * math.pi) - math.pi
    return abs(math.degrees(difference))


def distance_bucket(depth_m: float) -> str:
    if depth_m < 20.0:
        return "00_20m"
    if depth_m < 40.0:
        return "20_40m"
    if depth_m < 60.0:
        return "40_60m"
    return "60m_plus"


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(np.mean(values)) if values else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    values = list(values)
    return float(np.percentile(values, q)) if values else 0.0


def match_class(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    score_threshold: float,
    match_iou_threshold: float,
) -> tuple[list[tuple[Mapping[str, Any], Mapping[str, Any], float]], int, list[Mapping[str, Any]]]:
    candidates = sorted(
        (item for item in predictions if float(item["score"]) >= score_threshold),
        key=lambda item: float(item["score"]),
        reverse=True,
    )
    unmatched_gt = set(range(len(ground_truth)))
    matches = []
    false_positives = 0
    for prediction in candidates:
        overlaps = [
            (bbox_iou(prediction["bbox_2d"], ground_truth[index]["bbox_2d"]), index)
            for index in unmatched_gt
        ]
        if not overlaps:
            false_positives += 1
            continue
        overlap, target_index = max(overlaps, key=lambda item: (item[0], -item[1]))
        if overlap < match_iou_threshold:
            false_positives += 1
            continue
        unmatched_gt.remove(target_index)
        matches.append((prediction, ground_truth[target_index], overlap))
    return matches, false_positives, [ground_truth[index] for index in sorted(unmatched_gt)]


def geometry_row(
    sample_id: str,
    class_name: str,
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    overlap: float,
) -> dict[str, Any]:
    pred_location = np.asarray(prediction["location_3d"], dtype=float)
    gt_location = np.asarray(target["location_3d"], dtype=float)
    pred_dimensions = np.asarray(prediction["dimensions_3d_hwl"], dtype=float)
    gt_dimensions = np.asarray(target["dimensions_3d"], dtype=float)
    depth = float(gt_location[2])
    return {
        "sample_id": sample_id,
        "class_name": class_name,
        "score": float(prediction["score"]),
        "distance_bucket": distance_bucket(depth),
        "gt_depth_m": depth,
        "near_field": depth < NEAR_LIMITS_M[class_name],
        "iou_2d": float(overlap),
        "iou_bev": float(bev_iou(prediction, target)),
        "iou_3d": float(iou_3d(prediction, target)),
        "depth_abs_error_m": abs(float(pred_location[2] - gt_location[2])),
        "depth_relative_error": abs(float(pred_location[2] - gt_location[2])) / max(abs(depth), 1e-6),
        "dimension_mae_m": float(np.mean(np.abs(pred_dimensions - gt_dimensions))),
        "yaw_abs_error_deg": angle_error_degrees(float(prediction["rotation_y"]), float(target["rotation_y"])),
        "loc_x_abs_error_m": abs(float(pred_location[0] - gt_location[0])),
        "loc_y_abs_error_m": abs(float(pred_location[1] - gt_location[1])),
        "loc_z_abs_error_m": abs(float(pred_location[2] - gt_location[2])),
        "center3d_error_m": float(np.linalg.norm(pred_location - gt_location)),
    }


def summarize(rows: Sequence[Mapping[str, Any]], class_name: str, bucket: str = "ALL") -> dict[str, Any]:
    selected = [
        row for row in rows
        if row["class_name"] == class_name
        and (bucket == "ALL" or row["distance_bucket"] == bucket)
    ]
    return {
        "class_name": class_name,
        "distance_bucket": bucket,
        "matched": len(selected),
        "iou_2d_mean": mean(row["iou_2d"] for row in selected),
        "iou_bev_mean": mean(row["iou_bev"] for row in selected),
        "iou_3d_mean": mean(row["iou_3d"] for row in selected),
        "depth_mae_m": mean(row["depth_abs_error_m"] for row in selected),
        "depth_p90_m": percentile((row["depth_abs_error_m"] for row in selected), 90),
        "depth_relative_error_mean": mean(row["depth_relative_error"] for row in selected),
        "dimension_mae_m": mean(row["dimension_mae_m"] for row in selected),
        "yaw_mae_deg": mean(row["yaw_abs_error_deg"] for row in selected),
        "yaw_p90_deg": percentile((row["yaw_abs_error_deg"] for row in selected), 90),
        "loc_x_mae_m": mean(row["loc_x_abs_error_m"] for row in selected),
        "loc_y_mae_m": mean(row["loc_y_abs_error_m"] for row in selected),
        "loc_z_mae_m": mean(row["loc_z_abs_error_m"] for row in selected),
        "center3d_mae_m": mean(row["center3d_error_m"] for row in selected),
    }


def audit(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    ground_truth: Mapping[str, Sequence[Mapping[str, Any]]],
    score_threshold: float,
    match_iou_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    matched_rows = []
    false_negative_rows = []
    counts = {
        name: {"gt": 0, "matched": 0, "fp": 0, "near_gt": 0, "near_matched": 0}
        for name in CLASSES
    }
    for sample_id, targets in ground_truth.items():
        sample_predictions = predictions.get(sample_id, ())
        for class_name in CLASSES:
            class_targets = [item for item in targets if item["class_name"] == class_name]
            class_predictions = [item for item in sample_predictions if item["class_name"] == class_name]
            matches, false_positives, unmatched = match_class(
                class_predictions, class_targets, score_threshold, match_iou_threshold
            )
            counts[class_name]["gt"] += len(class_targets)
            counts[class_name]["matched"] += len(matches)
            counts[class_name]["fp"] += false_positives
            counts[class_name]["near_gt"] += sum(
                float(item["location_3d"][2]) < NEAR_LIMITS_M[class_name]
                for item in class_targets
            )
            counts[class_name]["near_matched"] += sum(
                float(target["location_3d"][2]) < NEAR_LIMITS_M[class_name]
                for _, target, _ in matches
            )
            matched_rows.extend(
                geometry_row(sample_id, class_name, prediction, target, overlap)
                for prediction, target, overlap in matches
            )
            false_negative_rows.extend(
                {
                    "sample_id": sample_id,
                    "class_name": class_name,
                    "gt_depth_m": float(target["location_3d"][2]),
                    "distance_bucket": distance_bucket(float(target["location_3d"][2])),
                    "near_field": float(target["location_3d"][2]) < NEAR_LIMITS_M[class_name],
                }
                for target in unmatched
            )

    class_reports = {}
    for class_name, values in counts.items():
        recall = values["matched"] / max(values["gt"], 1)
        near_recall = values["near_matched"] / max(values["near_gt"], 1)
        precision = values["matched"] / max(values["matched"] + values["fp"], 1)
        class_reports[class_name] = {
            **values,
            "precision": precision,
            "recall": recall,
            "near_limit_m": NEAR_LIMITS_M[class_name],
            "near_recall": near_recall,
            "near_recall_gate": NEAR_RECALL_GATES[class_name],
            "near_recall_gate_passed": near_recall >= NEAR_RECALL_GATES[class_name],
        }

    geometry = []
    for class_name in CLASSES:
        geometry.append(summarize(matched_rows, class_name))
        for bucket in ("00_20m", "20_40m", "40_60m", "60m_plus"):
            geometry.append(summarize(matched_rows, class_name, bucket))
    report = {
        "schema_version": 1,
        "complete": True,
        "score_threshold": score_threshold,
        "match_2d_iou_threshold": match_iou_threshold,
        "classes": class_reports,
        "all_near_recall_gates_passed": all(
            item["near_recall_gate_passed"] for item in class_reports.values()
        ),
        "geometry_summary": geometry,
    }
    return report, matched_rows, false_negative_rows


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit product nearby recall and geometry from KITTI predictions.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-images", type=int, default=3769)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--match-iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    sample_ids = [Path(item).stem for item in read_split_file(args.split_file)]
    if len(sample_ids) != args.expected_images or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(f"Expected {args.expected_images} unique images, found {len(sample_ids)}")
    missing = [item for item in sample_ids if not (args.prediction_dir / f"{item}.txt").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} prediction files; first={missing[:10]}")
    checkpoint_hash = None
    if args.checkpoint:
        checkpoint_hash = file_sha256(args.checkpoint)
        if args.expected_checkpoint_sha256 and checkpoint_hash != args.expected_checkpoint_sha256:
            raise RuntimeError(f"Checkpoint SHA-256 mismatch: {checkpoint_hash}")

    label_dir = resolve_label_dir(args.dataset_root)
    ground_truth = {
        sample_id: map_objects(
            [asdict(item) for item in parse_kitti_label_file(label_dir / f"{sample_id}.txt")],
            KITTI_PRODUCTION_CLASS_MAPPING,
        )
        for sample_id in sample_ids
    }
    raw_predictions = load_kitti_prediction_directory(
        args.prediction_dir,
        sample_ids,
        allowed_classes=list(KITTI_PRODUCTION_CLASS_MAPPING),
        require_all_files=True,
    )
    predictions = {
        sample_id: map_objects(items, KITTI_PRODUCTION_CLASS_MAPPING)
        for sample_id, items in raw_predictions.items()
    }
    report, matched_rows, false_negative_rows = audit(
        predictions, ground_truth, args.score_threshold, args.match_iou_threshold
    )
    report.update({
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": checkpoint_hash,
        "prediction_dir": str(args.prediction_dir),
        "evaluated_images": len(sample_ids),
        "split_file": str(args.split_file),
        "split_file_sha256": file_sha256(args.split_file),
    })
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "nearby_geometry_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(matched_rows, output_dir / "matched_geometry.csv")
    write_csv(false_negative_rows, output_dir / "false_negatives.csv")
    write_csv(report["geometry_summary"], output_dir / "geometry_summary.csv")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
