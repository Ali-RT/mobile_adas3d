from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.class_taxonomy import KITTI_PRODUCTION_CLASS_MAPPING, map_objects
from data.kitti_parser import parse_kitti_label_file
from data.kitti_prediction_parser import load_kitti_prediction_directory
from data.matching import bbox_iou
from data.splits import read_split_file
from scripts.audit_product_prediction_geometry import file_sha256, resolve_label_dir


PEDESTRIAN = "Pedestrian"
NEAR_LIMIT_M = 30.0


def kitti_difficulty(target: Mapping[str, Any]) -> str:
    height = float(target["bbox_2d"][3]) - float(target["bbox_2d"][1])
    occlusion = int(target["occluded"])
    truncation = float(target["truncated"])
    if height >= 40 and occlusion == 0 and truncation <= 0.15:
        return "easy"
    if height >= 25 and occlusion <= 1 and truncation <= 0.30:
        return "moderate"
    if height >= 25 and occlusion <= 2 and truncation <= 0.50:
        return "hard"
    return "outside_hard"


def size_bucket(height: float) -> str:
    if height < 25:
        return "small_lt25px"
    if height < 40:
        return "medium_25_40px"
    return "large_ge40px"


def depth_bucket(depth: float) -> str:
    if depth < 10:
        return "00_10m"
    if depth < 20:
        return "10_20m"
    if depth < 30:
        return "20_30m"
    if depth < 40:
        return "30_40m"
    return "40m_plus"


def truncation_bucket(truncation: float) -> str:
    if truncation <= 0.15:
        return "low_le_0_15"
    if truncation <= 0.30:
        return "medium_0_15_0_30"
    if truncation <= 0.50:
        return "high_0_30_0_50"
    return "extreme_gt_0_50"


def best_prediction(
    target: Mapping[str, Any], predictions: Sequence[Mapping[str, Any]]
) -> tuple[Mapping[str, Any] | None, float]:
    if not predictions:
        return None, 0.0
    ranked = [
        (bbox_iou(target["bbox_2d"], prediction["bbox_2d"]), prediction)
        for prediction in predictions
    ]
    overlap, prediction = max(
        ranked, key=lambda item: (item[0], float(item[1]["score"]))
    )
    return prediction, float(overlap)


def greedy_pedestrian_matches(
    targets: Sequence[Mapping[str, Any]],
    predictions: Sequence[Mapping[str, Any]],
    iou_threshold: float,
    score_threshold: float,
) -> set[int]:
    unmatched = set(range(len(targets)))
    matched = set()
    pedestrian_predictions = sorted(
        (
            p
            for p in predictions
            if p["class_name"] == PEDESTRIAN
            and float(p["score"]) >= score_threshold
        ),
        key=lambda p: float(p["score"]),
        reverse=True,
    )
    for prediction in pedestrian_predictions:
        candidates = [
            (bbox_iou(prediction["bbox_2d"], targets[index]["bbox_2d"]), index)
            for index in unmatched
        ]
        if not candidates:
            continue
        overlap, index = max(candidates, key=lambda item: (item[0], -item[1]))
        if overlap >= iou_threshold:
            unmatched.remove(index)
            matched.add(index)
    return matched


def classify_target(
    target: Mapping[str, Any],
    predictions: Sequence[Mapping[str, Any]],
    class_matched: bool,
    iou_threshold: float,
    weak_iou_threshold: float,
    score_threshold: float,
) -> dict[str, Any]:
    prediction, overlap = best_prediction(target, predictions)
    best_class = prediction["class_name"] if prediction else None
    best_score = float(prediction["score"]) if prediction else 0.0
    if class_matched:
        failure_mode = "detected_pedestrian"
    elif overlap >= iou_threshold and best_score < score_threshold:
        failure_mode = "subthreshold_well_localized_query"
    elif overlap >= iou_threshold and best_class != PEDESTRIAN:
        failure_mode = "wrong_class_classification"
    elif overlap >= iou_threshold:
        failure_mode = "pedestrian_assignment_conflict"
    elif overlap >= weak_iou_threshold:
        failure_mode = "localization_failure"
    else:
        failure_mode = "missing_query"
    height = float(target["bbox_2d"][3]) - float(target["bbox_2d"][1])
    depth = float(target["location_3d"][2])
    return {
        "source_class_name": target["source_class_name"],
        "gt_height_px": height,
        "size_bucket": size_bucket(height),
        "difficulty": kitti_difficulty(target),
        "occluded": int(target["occluded"]),
        "truncated": float(target["truncated"]),
        "truncation_bucket": truncation_bucket(float(target["truncated"])),
        "gt_depth_m": depth,
        "depth_bucket": depth_bucket(depth),
        "near_field": depth < NEAR_LIMIT_M,
        "class_matched": class_matched,
        "best_any_iou_2d": overlap,
        "best_any_class": best_class,
        "best_any_score": best_score,
        "failure_mode": failure_mode,
    }


def summarize(rows: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    output = []
    for value, selected in sorted(groups.items()):
        modes = Counter(row["failure_mode"] for row in selected)
        count = len(selected)
        output.append(
            {
                "group_name": field,
                "group_value": value,
                "count": count,
                "detected_rate": modes["detected_pedestrian"] / count,
                "wrong_class_rate": modes["wrong_class_classification"] / count,
                "assignment_conflict_rate": modes["pedestrian_assignment_conflict"] / count,
                "localization_failure_rate": modes["localization_failure"] / count,
                "missing_query_rate": modes["missing_query"] / count,
                "subthreshold_well_localized_rate": modes[
                    "subthreshold_well_localized_query"
                ]
                / count,
            }
        )
    return output


def diagnose(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    ground_truth: Mapping[str, Sequence[Mapping[str, Any]]],
    iou_threshold: float = 0.5,
    weak_iou_threshold: float = 0.1,
    score_threshold: float = 0.001,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    for sample_id, objects in ground_truth.items():
        targets = [obj for obj in objects if obj["class_name"] == PEDESTRIAN]
        sample_predictions = list(predictions.get(sample_id, ()))
        matched = greedy_pedestrian_matches(
            targets, sample_predictions, iou_threshold, score_threshold
        )
        for index, target in enumerate(targets):
            row = classify_target(
                target,
                sample_predictions,
                index in matched,
                iou_threshold,
                weak_iou_threshold,
                score_threshold,
            )
            row["sample_id"] = sample_id
            rows.append(row)
    near_rows = [row for row in rows if row["near_field"]]
    near_modes = Counter(row["failure_mode"] for row in near_rows)
    summaries = []
    for field in (
        "failure_mode",
        "size_bucket",
        "difficulty",
        "occluded",
        "truncation_bucket",
        "depth_bucket",
        "source_class_name",
    ):
        summaries.extend(summarize(near_rows, field))
    report = {
        "schema_version": 1,
        "complete": True,
        "scope": "Pedestrian ground truth; primary summary restricted to depth <30m",
        "prediction_scope": "saved decoded top-50 predictions; best IoU ignores predicted class",
        "iou_threshold": iou_threshold,
        "weak_iou_threshold": weak_iou_threshold,
        "score_threshold": score_threshold,
        "pedestrian_ground_truth": len(rows),
        "near_pedestrian_ground_truth": len(near_rows),
        "near_failure_modes": dict(sorted(near_modes.items())),
        "near_failure_rates": {
            mode: count / max(len(near_rows), 1) for mode, count in sorted(near_modes.items())
        },
    }
    return report, rows, summaries


def write_csv(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose frozen-A2 Pedestrian false negatives.")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--expected-images", type=int, default=3769)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--weak-iou-threshold", type=float, default=0.1)
    parser.add_argument("--score-threshold", type=float, default=0.001)
    args = parser.parse_args()

    checkpoint_hash = file_sha256(args.checkpoint)
    if checkpoint_hash != args.expected_checkpoint_sha256:
        raise RuntimeError(f"Checkpoint SHA-256 mismatch: {checkpoint_hash}")
    sample_ids = [Path(item).stem for item in read_split_file(args.split_file)]
    if len(sample_ids) != args.expected_images or len(set(sample_ids)) != len(sample_ids):
        raise RuntimeError(f"Expected {args.expected_images} unique images, found {len(sample_ids)}")
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
    report, rows, summaries = diagnose(
        predictions,
        ground_truth,
        args.iou_threshold,
        args.weak_iou_threshold,
        args.score_threshold,
    )
    report.update(
        {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "prediction_dir": str(args.prediction_dir),
            "evaluated_images": len(sample_ids),
            "split_file": str(args.split_file),
            "split_file_sha256": file_sha256(args.split_file),
        }
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "a2_pedestrian_false_negative_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(rows, args.output_dir / "a2_pedestrian_ground_truth_diagnostics.csv")
    write_csv(summaries, args.output_dir / "a2_pedestrian_false_negative_groups.csv")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
