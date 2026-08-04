from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.kitti_parser import parse_kitti_label_file
from data.kitti_prediction_parser import parse_kitti_prediction_file
from data.kitti_r40 import bev_iou, iou_3d
from scripts.create_teacher_prediction_cache import (
    load_split_ids,
    prediction_tree_sha256,
)


DEFAULT_THRESHOLDS = (0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, float(box_a[2]) - float(box_a[0])) * max(
        0.0, float(box_a[3]) - float(box_a[1])
    )
    area_b = max(0.0, float(box_b[2]) - float(box_b[0])) * max(
        0.0, float(box_b[3]) - float(box_b[1])
    )
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


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


def percentile(values: Iterable[float], quantile: float) -> float:
    values = list(values)
    return float(np.percentile(values, quantile)) if values else 0.0


def resolve_label_dir(dataset_root: Path) -> Path:
    candidates = (
        dataset_root / "training/label_2",
        dataset_root / "training/label_02",
    )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"KITTI label directory not found under {dataset_root}")


def load_ground_truth(
    label_dir: Path,
    sample_ids: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        sample_id: [
            asdict(obj)
            for obj in parse_kitti_label_file(
                label_dir / f"{sample_id}.txt",
                allowed_classes=["Car"],
            )
        ]
        for sample_id in sample_ids
    }


def load_predictions(
    prediction_dir: Path,
    sample_ids: Sequence[str],
) -> Dict[str, List[Dict[str, Any]]]:
    return {
        sample_id: parse_kitti_prediction_file(
            prediction_dir / f"{sample_id}.txt",
            allowed_classes=["Car"],
        )
        for sample_id in sample_ids
    }


def match_sample(
    predictions: Sequence[Mapping[str, Any]],
    ground_truth: Sequence[Mapping[str, Any]],
    score_threshold: float,
    match_iou_threshold: float,
) -> Tuple[List[Tuple[Mapping[str, Any], Mapping[str, Any], float]], int, int]:
    candidates = sorted(
        (
            prediction
            for prediction in predictions
            if float(prediction["score"]) >= score_threshold
        ),
        key=lambda prediction: float(prediction["score"]),
        reverse=True,
    )
    unmatched_gt = set(range(len(ground_truth)))
    matches: List[Tuple[Mapping[str, Any], Mapping[str, Any], float]] = []
    unmatched_predictions = 0

    for prediction in candidates:
        overlaps = [
            (
                bbox_iou(prediction["bbox_2d"], ground_truth[index]["bbox_2d"]),
                index,
            )
            for index in unmatched_gt
        ]
        if not overlaps:
            unmatched_predictions += 1
            continue
        overlap, gt_index = max(overlaps, key=lambda item: (item[0], -item[1]))
        if overlap < match_iou_threshold:
            unmatched_predictions += 1
            continue
        unmatched_gt.remove(gt_index)
        matches.append((prediction, ground_truth[gt_index], overlap))

    return matches, unmatched_predictions, len(unmatched_gt)


def matched_geometry_row(
    sample_id: str,
    prediction: Mapping[str, Any],
    target: Mapping[str, Any],
    iou_2d: float,
) -> Dict[str, Any]:
    pred_depth = float(prediction["location_3d"][2])
    gt_depth = float(target["location_3d"][2])
    pred_dimensions = np.asarray(prediction["dimensions_3d_hwl"], dtype=float)
    gt_dimensions = np.asarray(target["dimensions_3d"], dtype=float)
    return {
        "sample_id": sample_id,
        "score": float(prediction["score"]),
        "distance_bucket": distance_bucket(gt_depth),
        "gt_depth_m": gt_depth,
        "iou_2d": iou_2d,
        "iou_bev": bev_iou(prediction, target),
        "iou_3d": iou_3d(prediction, target),
        "depth_abs_error_m": abs(pred_depth - gt_depth),
        "depth_relative_error": abs(pred_depth - gt_depth) / max(abs(gt_depth), 1e-6),
        "yaw_abs_error_deg": angle_error_degrees(
            float(prediction["yaw"]), float(target["rotation_y"])
        ),
        "dimension_mae_m": float(np.mean(np.abs(pred_dimensions - gt_dimensions))),
    }


def audit_threshold(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    ground_truth: Mapping[str, Sequence[Mapping[str, Any]]],
    score_threshold: float,
    match_iou_threshold: float,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    matched_rows: List[Dict[str, Any]] = []
    false_positives = 0
    false_negatives = 0
    total_predictions = 0

    bucket_gt = {bucket: 0 for bucket in ("00_20m", "20_40m", "40_60m", "60m_plus")}
    for sample_targets in ground_truth.values():
        for target in sample_targets:
            bucket_gt[distance_bucket(float(target["location_3d"][2]))] += 1

    for sample_id, sample_targets in ground_truth.items():
        sample_predictions = predictions.get(sample_id, ())
        total_predictions += sum(
            float(prediction["score"]) >= score_threshold
            for prediction in sample_predictions
        )
        matches, sample_fp, sample_fn = match_sample(
            sample_predictions,
            sample_targets,
            score_threshold,
            match_iou_threshold,
        )
        false_positives += sample_fp
        false_negatives += sample_fn
        matched_rows.extend(
            matched_geometry_row(sample_id, prediction, target, overlap)
            for prediction, target, overlap in matches
        )

    true_positives = len(matched_rows)
    total_gt = true_positives + false_negatives
    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(total_gt, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    summary = {
        "score_threshold": score_threshold,
        "match_2d_iou_threshold": match_iou_threshold,
        "ground_truth_cars": total_gt,
        "teacher_predictions": total_predictions,
        "matched": true_positives,
        "unmatched_teacher_predictions": false_positives,
        "unmatched_ground_truth": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou_2d_mean": mean(row["iou_2d"] for row in matched_rows),
        "iou_bev_mean": mean(row["iou_bev"] for row in matched_rows),
        "iou_3d_mean": mean(row["iou_3d"] for row in matched_rows),
        "depth_mae_m": mean(row["depth_abs_error_m"] for row in matched_rows),
        "depth_relative_error_mean": mean(
            row["depth_relative_error"] for row in matched_rows
        ),
        "yaw_mae_deg": mean(row["yaw_abs_error_deg"] for row in matched_rows),
        "dimension_mae_m": mean(row["dimension_mae_m"] for row in matched_rows),
    }

    distance_rows = []
    for bucket, gt_count in bucket_gt.items():
        bucket_matches = [
            row for row in matched_rows if row["distance_bucket"] == bucket
        ]
        distance_rows.append(
            {
                "score_threshold": score_threshold,
                "distance_bucket": bucket,
                "ground_truth_cars": gt_count,
                "matched": len(bucket_matches),
                "recall": len(bucket_matches) / max(gt_count, 1),
                "iou_3d_mean": mean(row["iou_3d"] for row in bucket_matches),
                "depth_mae_m": mean(
                    row["depth_abs_error_m"] for row in bucket_matches
                ),
                "yaw_mae_deg": mean(
                    row["yaw_abs_error_deg"] for row in bucket_matches
                ),
            }
        )
    return summary, distance_rows, matched_rows


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_recommendations(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    best_f1 = max(rows, key=lambda row: (row["f1"], row["score_threshold"]))
    high_recall_candidates = [row for row in rows if row["recall"] >= 0.95]
    high_recall = (
        max(
            high_recall_candidates,
            key=lambda row: (row["score_threshold"], row["precision"]),
        )
        if high_recall_candidates
        else max(rows, key=lambda row: (row["recall"], row["precision"]))
    )
    return {
        "max_f1": dict(best_f1),
        "high_recall_95": dict(high_recall),
    }


def run_audit(
    *,
    cache_dir: Path,
    dataset_root: Path,
    split_file: Path,
    output_dir: Path,
    thresholds: Sequence[float],
    match_iou_threshold: float,
) -> Dict[str, Any]:
    manifest_path = cache_dir / "teacher_cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise RuntimeError(f"Teacher cache is incomplete: {manifest_path}")
    if manifest.get("inference_data_augmentation") is not False:
        raise RuntimeError("Teacher cache was not produced with clean inference")

    sample_ids = load_split_ids(split_file)
    if len(sample_ids) != int(manifest.get("prediction_files", -1)):
        raise RuntimeError("Teacher cache count does not match the requested split")
    prediction_dir = cache_dir / "predictions"
    actual_tree_digest = prediction_tree_sha256(prediction_dir, sample_ids)
    if actual_tree_digest != manifest.get("prediction_tree_sha256"):
        raise RuntimeError("Teacher prediction-tree SHA-256 mismatch")

    ground_truth = load_ground_truth(resolve_label_dir(dataset_root), sample_ids)
    predictions = load_predictions(prediction_dir, sample_ids)
    threshold_rows = []
    all_distance_rows = []
    matched_by_threshold = {}
    for threshold in sorted(set(float(value) for value in thresholds)):
        summary, distance_rows, matched_rows = audit_threshold(
            predictions,
            ground_truth,
            threshold,
            match_iou_threshold,
        )
        threshold_rows.append(summary)
        all_distance_rows.extend(distance_rows)
        matched_by_threshold[threshold] = matched_rows

    recommendations = select_recommendations(threshold_rows)
    selected_threshold = float(recommendations["high_recall_95"]["score_threshold"])
    selected_matches = matched_by_threshold[selected_threshold]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "teacher_threshold_sweep.csv", threshold_rows)
    write_csv(output_dir / "teacher_distance_coverage.csv", all_distance_rows)
    write_csv(output_dir / "teacher_selected_matches.csv", selected_matches)

    report = {
        "schema_version": 1,
        "complete": True,
        "cache_manifest": str(manifest_path),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "prediction_tree_sha256": actual_tree_digest,
        "split_file": str(split_file),
        "split_images": len(sample_ids),
        "class_name": "Car",
        "matching": {
            "algorithm": "greedy descending teacher score",
            "association_metric": "2d_iou",
            "association_threshold": match_iou_threshold,
        },
        "thresholds": list(sorted(set(float(value) for value in thresholds))),
        "recommendations": recommendations,
        "selected_match_rows": len(selected_matches),
    }
    (output_dir / "teacher_matching_audit.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a clean KITTI teacher cache against ground truth."
    )
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
    )
    parser.add_argument("--match-2d-iou-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_audit(
        cache_dir=args.cache_dir,
        dataset_root=args.dataset_root,
        split_file=args.split_file,
        output_dir=args.output_dir,
        thresholds=args.thresholds,
        match_iou_threshold=args.match_2d_iou_threshold,
    )
    print(json.dumps(report, indent=2))
    print(f"Teacher matching audit: {args.output_dir}")


if __name__ == "__main__":
    main()
