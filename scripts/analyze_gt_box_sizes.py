from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from data.kitti_dataset import KITTIDataset
from data.split_resolver import get_split_file
from data.target_builder import scale_bbox_2d
from tools.config import load_config, apply_runtime_overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze KITTI GT box sizes, stride coverage, and feature-cell collisions."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/kitti_mobileadas3d.yaml",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Runtime profile, e.g. local_mac or colab_drive.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="Use -1 for all images.",
    )

    parser.add_argument(
        "--strides",
        type=str,
        default="16,32",
        help="Comma-separated strides to analyze.",
    )

    parser.add_argument(
        "--size-thresholds",
        type=str,
        default="16,32,64",
        help="Comma-separated pixel thresholds.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    return parser.parse_args()


def parse_int_list(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        raise ValueError(f"No rows to save for {output_path}")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")


def percentile(values: List[float], q: float) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), q))


def summarize_numeric(values: List[float], prefix: str) -> Dict[str, float]:
    if len(values) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_p10": 0.0,
            f"{prefix}_p25": 0.0,
            f"{prefix}_p50": 0.0,
            f"{prefix}_p75": 0.0,
            f"{prefix}_p90": 0.0,
        }

    arr = np.asarray(values, dtype=np.float32)

    return {
        f"{prefix}_mean": float(arr.mean()),
        f"{prefix}_p10": percentile(values, 10),
        f"{prefix}_p25": percentile(values, 25),
        f"{prefix}_p50": percentile(values, 50),
        f"{prefix}_p75": percentile(values, 75),
        f"{prefix}_p90": percentile(values, 90),
    }


def fraction_less_than(values: List[float], threshold: float) -> float:
    if len(values) == 0:
        return 0.0

    arr = np.asarray(values, dtype=np.float32)
    return float((arr < threshold).mean())


def assign_center_cell(
    x_center: float,
    y_center: float,
    input_width: int,
    input_height: int,
    stride: int,
) -> Tuple[int, int, int, int]:
    grid_w = input_width // stride
    grid_h = input_height // stride

    cell_x = int(x_center // stride)
    cell_y = int(y_center // stride)

    cell_x = max(0, min(cell_x, grid_w - 1))
    cell_y = max(0, min(cell_y, grid_h - 1))

    return cell_x, cell_y, grid_w, grid_h


def build_object_rows(
    dataset: KITTIDataset,
    input_width: int,
    input_height: int,
    strides: List[int],
    max_images: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    num_images = len(dataset) if max_images < 0 else min(max_images, len(dataset))

    for idx in range(num_images):
        sample = dataset[idx]

        original_width = int(sample["original_size"]["width"])
        original_height = int(sample["original_size"]["height"])

        for obj_idx, obj in enumerate(sample["objects"]):
            bbox = scale_bbox_2d(
                bbox=obj["bbox_2d"],
                original_width=original_width,
                original_height=original_height,
                input_width=input_width,
                input_height=input_height,
            )

            x1, y1, x2, y2 = [float(v) for v in bbox]

            width_px = max(0.0, x2 - x1)
            height_px = max(0.0, y2 - y1)
            area_px2 = width_px * height_px
            short_side_px = min(width_px, height_px)
            long_side_px = max(width_px, height_px)

            x_center = 0.5 * (x1 + x2)
            y_center = 0.5 * (y1 + y2)

            row: Dict[str, Any] = {
                "sample_id": sample["sample_id"],
                "object_index": obj_idx,
                "class_name": obj["class_name"],
                "class_id": obj["class_id"],
                "depth_m": float(obj["location_3d"][2]),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "x_center": x_center,
                "y_center": y_center,
                "width_px": width_px,
                "height_px": height_px,
                "area_px2": area_px2,
                "short_side_px": short_side_px,
                "long_side_px": long_side_px,
            }

            for stride in strides:
                cell_x, cell_y, grid_w, grid_h = assign_center_cell(
                    x_center=x_center,
                    y_center=y_center,
                    input_width=input_width,
                    input_height=input_height,
                    stride=stride,
                )

                row[f"stride{stride}_grid_w"] = grid_w
                row[f"stride{stride}_grid_h"] = grid_h
                row[f"stride{stride}_cell_x"] = cell_x
                row[f"stride{stride}_cell_y"] = cell_y
                row[f"stride{stride}_cell_key"] = f"{cell_x}_{cell_y}"
                row[f"stride{stride}_width_cells"] = width_px / stride
                row[f"stride{stride}_height_cells"] = height_px / stride
                row[f"stride{stride}_area_cells"] = area_px2 / float(stride * stride)

            rows.append(row)

        if (idx + 1) % 100 == 0 or idx + 1 == num_images:
            print(f"Processed {idx + 1}/{num_images} images.")

    return rows


def summarize_by_class(
    object_rows: List[Dict[str, Any]],
    classes: List[str],
    strides: List[int],
    size_thresholds: List[int],
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []

    class_names = ["ALL"] + classes

    for class_name in class_names:
        if class_name == "ALL":
            rows = object_rows
        else:
            rows = [r for r in object_rows if r["class_name"] == class_name]

        count = len(rows)

        width_values = [float(r["width_px"]) for r in rows]
        height_values = [float(r["height_px"]) for r in rows]
        short_values = [float(r["short_side_px"]) for r in rows]
        area_values = [float(r["area_px2"]) for r in rows]
        depth_values = [float(r["depth_m"]) for r in rows]

        summary: Dict[str, Any] = {
            "class_name": class_name,
            "count": count,
            "count_fraction": count / max(len(object_rows), 1),
        }

        summary.update(summarize_numeric(width_values, "width_px"))
        summary.update(summarize_numeric(height_values, "height_px"))
        summary.update(summarize_numeric(short_values, "short_side_px"))
        summary.update(summarize_numeric(area_values, "area_px2"))
        summary.update(summarize_numeric(depth_values, "depth_m"))

        for threshold in size_thresholds:
            summary[f"frac_width_lt_{threshold}px"] = fraction_less_than(
                width_values, threshold
            )
            summary[f"frac_height_lt_{threshold}px"] = fraction_less_than(
                height_values, threshold
            )
            summary[f"frac_short_side_lt_{threshold}px"] = fraction_less_than(
                short_values, threshold
            )

        for stride in strides:
            width_cells = [float(r[f"stride{stride}_width_cells"]) for r in rows]
            height_cells = [float(r[f"stride{stride}_height_cells"]) for r in rows]
            area_cells = [float(r[f"stride{stride}_area_cells"]) for r in rows]

            summary.update(
                summarize_numeric(width_cells, f"stride{stride}_width_cells")
            )
            summary.update(
                summarize_numeric(height_cells, f"stride{stride}_height_cells")
            )
            summary.update(
                summarize_numeric(area_cells, f"stride{stride}_area_cells")
            )

            summary[f"stride{stride}_frac_height_lt_1_cell"] = fraction_less_than(
                height_cells, 1.0
            )
            summary[f"stride{stride}_frac_height_lt_2_cells"] = fraction_less_than(
                height_cells, 2.0
            )
            summary[f"stride{stride}_frac_width_lt_1_cell"] = fraction_less_than(
                width_cells, 1.0
            )
            summary[f"stride{stride}_frac_width_lt_2_cells"] = fraction_less_than(
                width_cells, 2.0
            )
            summary[f"stride{stride}_frac_area_lt_1_cell"] = fraction_less_than(
                area_cells, 1.0
            )
            summary[f"stride{stride}_frac_area_lt_4_cells"] = fraction_less_than(
                area_cells, 4.0
            )

        summary_rows.append(summary)

    return summary_rows


def compute_cell_collision_summary(
    object_rows: List[Dict[str, Any]],
    classes: List[str],
    strides: List[int],
) -> List[Dict[str, Any]]:
    summary_rows: List[Dict[str, Any]] = []

    for stride in strides:
        # sample_id -> cell_key -> list[object row]
        cells_by_sample: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )

        for row in object_rows:
            sample_id = row["sample_id"]
            cell_key = row[f"stride{stride}_cell_key"]
            cells_by_sample[sample_id][cell_key].append(row)

        total_positive_cells = 0
        collided_cells = 0
        total_objects = len(object_rows)
        objects_in_collided_cells = 0
        same_class_collision_cells = 0
        mixed_class_collision_cells = 0
        cells_with_car_and_small_object = 0

        per_class_objects_in_collided_cells = defaultdict(int)
        per_class_total_objects = defaultdict(int)

        for row in object_rows:
            per_class_total_objects[row["class_name"]] += 1

        for sample_id, cell_map in cells_by_sample.items():
            for cell_key, cell_objects in cell_map.items():
                total_positive_cells += 1

                if len(cell_objects) > 1:
                    collided_cells += 1
                    objects_in_collided_cells += len(cell_objects)

                    cell_classes = [obj["class_name"] for obj in cell_objects]
                    unique_classes = set(cell_classes)

                    if len(unique_classes) == 1:
                        same_class_collision_cells += 1
                    else:
                        mixed_class_collision_cells += 1

                    if (
                        "Car" in unique_classes
                        and (
                            "Pedestrian" in unique_classes
                            or "Cyclist" in unique_classes
                        )
                    ):
                        cells_with_car_and_small_object += 1

                    for obj in cell_objects:
                        per_class_objects_in_collided_cells[obj["class_name"]] += 1

        row: Dict[str, Any] = {
            "stride": stride,
            "total_objects": total_objects,
            "total_positive_cells": total_positive_cells,
            "collided_cells": collided_cells,
            "collision_cell_fraction": collided_cells / max(total_positive_cells, 1),
            "objects_in_collided_cells": objects_in_collided_cells,
            "object_collision_fraction": objects_in_collided_cells
            / max(total_objects, 1),
            "same_class_collision_cells": same_class_collision_cells,
            "mixed_class_collision_cells": mixed_class_collision_cells,
            "cells_with_car_and_ped_or_cyclist": cells_with_car_and_small_object,
        }

        for class_name in classes:
            row[f"{class_name}_objects"] = per_class_total_objects[class_name]
            row[f"{class_name}_objects_in_collided_cells"] = (
                per_class_objects_in_collided_cells[class_name]
            )
            row[f"{class_name}_object_collision_fraction"] = (
                per_class_objects_in_collided_cells[class_name]
                / max(per_class_total_objects[class_name], 1)
            )

        summary_rows.append(row)

    return summary_rows


def print_key_summary(
    summary_rows: List[Dict[str, Any]],
    collision_rows: List[Dict[str, Any]],
    strides: List[int],
) -> None:
    print("\nGT Box Size Summary:")
    for row in summary_rows:
        class_name = row["class_name"]

        print(f"\nClass: {class_name}")
        print(f"  count: {row['count']}")
        print(f"  height px p50/p75/p90: "
              f"{row['height_px_p50']:.1f} / "
              f"{row['height_px_p75']:.1f} / "
              f"{row['height_px_p90']:.1f}")
        print(f"  width px p50/p75/p90: "
              f"{row['width_px_p50']:.1f} / "
              f"{row['width_px_p75']:.1f} / "
              f"{row['width_px_p90']:.1f}")
        print(f"  frac height < 32px: {row.get('frac_height_lt_32px', 0.0):.3f}")
        print(f"  frac short side < 32px: {row.get('frac_short_side_lt_32px', 0.0):.3f}")

        for stride in strides:
            print(
                f"  stride {stride}: "
                f"height_cells_p50={row[f'stride{stride}_height_cells_p50']:.2f}, "
                f"frac_height<1cell={row[f'stride{stride}_frac_height_lt_1_cell']:.3f}, "
                f"frac_height<2cells={row[f'stride{stride}_frac_height_lt_2_cells']:.3f}"
            )

    print("\nCell Collision Summary:")
    for row in collision_rows:
        stride = row["stride"]
        print(f"\nStride: {stride}")
        print(f"  total objects: {row['total_objects']}")
        print(f"  total positive cells: {row['total_positive_cells']}")
        print(f"  collided cells: {row['collided_cells']}")
        print(f"  collision cell fraction: {row['collision_cell_fraction']:.4f}")
        print(f"  object collision fraction: {row['object_collision_fraction']:.4f}")
        print(f"  mixed-class collision cells: {row['mixed_class_collision_cells']}")
        print(
            f"  cells with Car + Pedestrian/Cyclist: "
            f"{row['cells_with_car_and_ped_or_cyclist']}"
        )


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    config = apply_runtime_overrides(
        config=config,
        profile=args.profile,
        run_name=None,
    )

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]

    input_height = int(model_cfg["input_height"])
    input_width = int(model_cfg["input_width"])

    strides = parse_int_list(args.strides)
    size_thresholds = parse_int_list(args.size_thresholds)

    split_file = get_split_file(config, args.split)

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=split_file,
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["outputs"]["visualization_dir"]) / "diagnostics"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("Starting GT box-size diagnostic.")
    print(f"Config: {args.config}")
    print(f"Profile: {active_profile}")
    print(f"Dataset root: {root_dir}")
    print(f"Split: {args.split}")
    print(f"Split file: {split_file}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Input size: {input_width} x {input_height}")
    print(f"Strides: {strides}")
    print(f"Size thresholds: {size_thresholds}")
    print(f"Output dir: {output_dir}")

    object_rows = build_object_rows(
        dataset=dataset,
        input_width=input_width,
        input_height=input_height,
        strides=strides,
        max_images=args.max_images,
    )

    summary_rows = summarize_by_class(
        object_rows=object_rows,
        classes=dataset_cfg["classes"],
        strides=strides,
        size_thresholds=size_thresholds,
    )

    collision_rows = compute_cell_collision_summary(
        object_rows=object_rows,
        classes=dataset_cfg["classes"],
        strides=strides,
    )

    object_csv = output_dir / f"gt_objects_{args.split}.csv"
    summary_csv = output_dir / f"gt_box_summary_{args.split}.csv"
    collision_csv = output_dir / f"cell_collision_summary_{args.split}.csv"

    save_csv(object_rows, object_csv)
    save_csv(summary_rows, summary_csv)
    save_csv(collision_rows, collision_csv)

    print_key_summary(
        summary_rows=summary_rows,
        collision_rows=collision_rows,
        strides=strides,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()