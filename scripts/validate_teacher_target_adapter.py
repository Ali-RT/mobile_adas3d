from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.kitti_parser import parse_kitti_label_file
from data.teacher_target_adapter import TeacherTargetAdapter


def resolve_label_dir(dataset_root: Path) -> Path:
    for name in ("training/label_2", "training/label_02"):
        candidate = dataset_root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"KITTI label directory not found under {dataset_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate object-aligned teacher targets.")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--expected-prediction-tree-sha256")
    parser.add_argument("--expected-matches", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter = TeacherTargetAdapter(
        args.cache_dir,
        args.split_file,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        expected_prediction_tree_sha256=args.expected_prediction_tree_sha256,
    )
    label_dir = resolve_label_dir(args.dataset_root)
    object_count = 0
    class_counts = {"Car": 0, "Pedestrian": 0, "Cyclist": 0}
    teacher_association_count = 0
    teacher_match_count = 0
    for index, sample_id in enumerate(adapter.sample_ids, start=1):
        objects = [
            asdict(obj)
            for obj in parse_kitti_label_file(
                label_dir / f"{sample_id}.txt",
                allowed_classes=list(class_counts),
            )
        ]
        targets = adapter.build_for_sample(sample_id, objects)
        object_count += len(objects)
        teacher_association_count += int(
            targets["teacher_association_mask"].sum().item()
        )
        teacher_match_count += int(targets["teacher_valid_mask"].sum().item())
        for obj in objects:
            class_counts[obj["class_name"]] += 1
        if index % 500 == 0 or index == len(adapter.sample_ids):
            print(
                f"Validated {index}/{len(adapter.sample_ids)} samples; "
                f"approved teacher matches={teacher_match_count}",
                flush=True,
            )

    report = {
        "schema_version": 1,
        "complete": True,
        "samples": len(adapter.sample_ids),
        "ground_truth_objects": object_count,
        "ground_truth_by_class": class_counts,
        "teacher_associations_before_distance_mask": teacher_association_count,
        "approved_teacher_matches": teacher_match_count,
        "teacher_associations_masked_by_distance": (
            teacher_association_count - teacher_match_count
        ),
        "policy": {
            "class_name": adapter.class_name,
            "score_threshold": adapter.score_threshold,
            "match_2d_iou_threshold": adapter.match_iou_threshold,
            "max_gt_depth_m_exclusive": adapter.max_gt_depth_m,
            "matching": "greedy descending teacher score, one-to-one",
        },
        "checkpoint_sha256": adapter.manifest["checkpoint_sha256"],
        "prediction_tree_sha256": adapter.manifest["prediction_tree_sha256"],
    }
    if (
        args.expected_matches is not None
        and teacher_match_count != args.expected_matches
    ):
        raise RuntimeError(
            f"Expected {args.expected_matches} matches, found {teacher_match_count}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Adapter validation report: {args.output}")


if __name__ == "__main__":
    main()
