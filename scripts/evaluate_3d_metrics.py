from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from data.kitti_dataset import KITTIDataset
from data.split_resolver import get_split_file
from data.target_builder import scale_bbox_2d
from models.build import build_model
from models.decode import decode_mobile_adas3d_outputs, box_iou
from tools.config import load_config, apply_runtime_overrides
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MobileADAS3D depth/yaw/dimension metrics on matched detections."
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
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=-1,
        help="Use -1 for all images.",
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.55,
    )

    parser.add_argument(
        "--match-iou-threshold",
        type=float,
        default=0.50,
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    return parser.parse_args()


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        print(f"No rows to save for {output_path}")
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


def mean(values: List[float]) -> float:
    if len(values) == 0:
        return 0.0
    return float(sum(values) / len(values))


def percentile(values: List[float], q: float) -> float:
    if len(values) == 0:
        return 0.0

    values_sorted = sorted(values)

    if len(values_sorted) == 1:
        return float(values_sorted[0])

    position = (len(values_sorted) - 1) * q / 100.0
    lower = int(math.floor(position))
    upper = int(math.ceil(position))

    if lower == upper:
        return float(values_sorted[lower])

    weight = position - lower

    return float(
        values_sorted[lower] * (1.0 - weight)
        + values_sorted[upper] * weight
    )


def distance_bucket(depth_m: float) -> str:
    if depth_m < 20.0:
        return "00_20m"
    if depth_m < 40.0:
        return "20_40m"
    if depth_m < 60.0:
        return "40_60m"
    return "60m_plus"


def size_bucket(height_px: float) -> str:
    if height_px < 32.0:
        return "small_h_lt_32px"
    if height_px < 96.0:
        return "medium_h_32_96px"
    return "large_h_ge_96px"


def angle_diff_rad(pred: float, gt: float) -> float:
    """
    Smallest absolute angular difference in radians.
    """
    diff = pred - gt
    diff = (diff + math.pi) % (2.0 * math.pi) - math.pi
    return abs(diff)


def scale_gt_objects_to_input(
    sample: Dict[str, Any],
    input_width: int,
    input_height: int,
) -> List[Dict[str, Any]]:
    gt_objects = []

    original_width = int(sample["original_size"]["width"])
    original_height = int(sample["original_size"]["height"])

    for gt_index, obj in enumerate(sample["objects"]):
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

        depth_m = float(obj["location_3d"][2])

        gt_objects.append(
            {
                "sample_id": sample["sample_id"],
                "gt_index": gt_index,
                "class_name": obj["class_name"],
                "class_id": obj["class_id"],
                "bbox_2d": [x1, y1, x2, y2],
                "depth_m": depth_m,
                "yaw_rad": float(obj["rotation_y"]),
                "dimensions_3d_hwl": [
                    float(obj["dimensions_3d"][0]),
                    float(obj["dimensions_3d"][1]),
                    float(obj["dimensions_3d"][2]),
                ],
                "width_px": width_px,
                "height_px": height_px,
                "area_px2": area_px2,
                "distance_bucket": distance_bucket(depth_m),
                "size_bucket": size_bucket(height_px),
            }
        )

    return gt_objects


def greedy_match_predictions_to_gt(
    predictions: List[Dict[str, Any]],
    gt_objects: List[Dict[str, Any]],
    match_iou_threshold: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Class-aware greedy matching.

    Returns:
      matches:
        [
          {
            "prediction": pred,
            "gt": gt,
            "iou": iou,
          }
        ]

      unmatched_predictions:
        false positives

      unmatched_gt:
        false negatives
    """
    predictions = sorted(predictions, key=lambda p: p["score"], reverse=True)

    gt_boxes = torch.tensor(
        [g["bbox_2d"] for g in gt_objects],
        dtype=torch.float32,
    )

    matched_gt_indices = set()
    matches = []
    unmatched_predictions = []

    for pred in predictions:
        if len(gt_objects) == 0:
            unmatched_predictions.append(
                {
                    **pred,
                    "best_iou": 0.0,
                    "best_iou_class_agnostic": 0.0,
                }
            )
            continue

        pred_box = torch.tensor(pred["bbox_2d"], dtype=torch.float32)
        ious = box_iou(pred_box, gt_boxes)

        best_same_class_iou = 0.0
        best_same_class_gt_idx = -1

        best_class_agnostic_iou = float(ious.max().item()) if ious.numel() else 0.0

        for gt_idx, iou_value in enumerate(ious.tolist()):
            if gt_idx in matched_gt_indices:
                continue

            if pred["class_name"] != gt_objects[gt_idx]["class_name"]:
                continue

            if iou_value > best_same_class_iou:
                best_same_class_iou = float(iou_value)
                best_same_class_gt_idx = gt_idx

        if (
            best_same_class_gt_idx >= 0
            and best_same_class_iou >= match_iou_threshold
        ):
            matched_gt_indices.add(best_same_class_gt_idx)

            matches.append(
                {
                    "prediction": pred,
                    "gt": gt_objects[best_same_class_gt_idx],
                    "iou": best_same_class_iou,
                }
            )
        else:
            unmatched_predictions.append(
                {
                    **pred,
                    "best_iou": best_same_class_iou,
                    "best_iou_class_agnostic": best_class_agnostic_iou,
                }
            )

    unmatched_gt = [
        gt
        for gt_idx, gt in enumerate(gt_objects)
        if gt_idx not in matched_gt_indices
    ]

    return matches, unmatched_predictions, unmatched_gt


def compute_match_metric_row(
    split_name: str,
    match: Dict[str, Any],
) -> Dict[str, Any]:
    pred = match["prediction"]
    gt = match["gt"]
    iou = float(match["iou"])

    gt_depth = float(gt["depth_m"])
    pred_depth = float(pred["depth"])

    depth_abs_error = abs(pred_depth - gt_depth)
    depth_rel_error = depth_abs_error / max(gt_depth, 1e-6)

    log_depth_abs_error = abs(
        math.log(max(pred_depth, 1e-6)) - math.log(max(gt_depth, 1e-6))
    )

    gt_dims = gt["dimensions_3d_hwl"]
    pred_dims = pred["dimensions_3d_hwl"]

    dim_h_abs_error = abs(float(pred_dims[0]) - float(gt_dims[0]))
    dim_w_abs_error = abs(float(pred_dims[1]) - float(gt_dims[1]))
    dim_l_abs_error = abs(float(pred_dims[2]) - float(gt_dims[2]))

    dim_mae_m = mean(
        [
            dim_h_abs_error,
            dim_w_abs_error,
            dim_l_abs_error,
        ]
    )

    dim_h_rel_error = dim_h_abs_error / max(float(gt_dims[0]), 1e-6)
    dim_w_rel_error = dim_w_abs_error / max(float(gt_dims[1]), 1e-6)
    dim_l_rel_error = dim_l_abs_error / max(float(gt_dims[2]), 1e-6)

    dim_mean_rel_error = mean(
        [
            dim_h_rel_error,
            dim_w_rel_error,
            dim_l_rel_error,
        ]
    )

    gt_yaw = float(gt["yaw_rad"])
    pred_yaw = float(pred["yaw"])

    yaw_abs_error_rad = angle_diff_rad(pred_yaw, gt_yaw)
    yaw_abs_error_deg = yaw_abs_error_rad * 180.0 / math.pi

    return {
        "split": split_name,
        "sample_id": gt["sample_id"],
        "class_name": gt["class_name"],
        "score": float(pred["score"]),
        "iou_2d": iou,
        "distance_bucket": gt["distance_bucket"],
        "size_bucket": gt["size_bucket"],
        "gt_depth_m": gt_depth,
        "pred_depth_m": pred_depth,
        "depth_abs_error_m": depth_abs_error,
        "depth_rel_error": depth_rel_error,
        "log_depth_abs_error": log_depth_abs_error,
        "gt_yaw_rad": gt_yaw,
        "pred_yaw_rad": pred_yaw,
        "yaw_abs_error_rad": yaw_abs_error_rad,
        "yaw_abs_error_deg": yaw_abs_error_deg,
        "gt_h_m": float(gt_dims[0]),
        "gt_w_m": float(gt_dims[1]),
        "gt_l_m": float(gt_dims[2]),
        "pred_h_m": float(pred_dims[0]),
        "pred_w_m": float(pred_dims[1]),
        "pred_l_m": float(pred_dims[2]),
        "dim_h_abs_error_m": dim_h_abs_error,
        "dim_w_abs_error_m": dim_w_abs_error,
        "dim_l_abs_error_m": dim_l_abs_error,
        "dim_mae_m": dim_mae_m,
        "dim_h_rel_error": dim_h_rel_error,
        "dim_w_rel_error": dim_w_rel_error,
        "dim_l_rel_error": dim_l_rel_error,
        "dim_mean_rel_error": dim_mean_rel_error,
        "gt_box_width_px": float(gt["width_px"]),
        "gt_box_height_px": float(gt["height_px"]),
        "gt_box_area_px2": float(gt["area_px2"]),
        "pred_cell_x": int(pred.get("cell_x", -1)),
        "pred_cell_y": int(pred.get("cell_y", -1)),
    }


def summarize_metric_rows(
    rows: List[Dict[str, Any]],
    group_name: str,
    group_key: str | None = None,
) -> List[Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    if group_key is None:
        groups["ALL"] = rows
    else:
        for row in rows:
            groups[str(row[group_key])].append(row)

    summary_rows: List[Dict[str, Any]] = []

    for group_value, group_rows in sorted(groups.items(), key=lambda x: x[0]):
        if len(group_rows) == 0:
            continue

        depth_errors = [float(r["depth_abs_error_m"]) for r in group_rows]
        depth_rel_errors = [float(r["depth_rel_error"]) for r in group_rows]
        log_depth_errors = [float(r["log_depth_abs_error"]) for r in group_rows]
        yaw_errors = [float(r["yaw_abs_error_deg"]) for r in group_rows]
        dim_errors = [float(r["dim_mae_m"]) for r in group_rows]
        dim_rel_errors = [float(r["dim_mean_rel_error"]) for r in group_rows]
        ious = [float(r["iou_2d"]) for r in group_rows]
        scores = [float(r["score"]) for r in group_rows]

        summary_rows.append(
            {
                "group_name": group_name,
                "group_value": group_value,
                "count": len(group_rows),
                "score_mean": mean(scores),
                "iou_2d_mean": mean(ious),
                "iou_2d_p50": percentile(ious, 50),
                "depth_mae_m": mean(depth_errors),
                "depth_abs_error_p50_m": percentile(depth_errors, 50),
                "depth_abs_error_p90_m": percentile(depth_errors, 90),
                "depth_rel_error_mean": mean(depth_rel_errors),
                "depth_rel_error_p50": percentile(depth_rel_errors, 50),
                "log_depth_abs_error_mean": mean(log_depth_errors),
                "yaw_abs_error_mean_deg": mean(yaw_errors),
                "yaw_abs_error_p50_deg": percentile(yaw_errors, 50),
                "yaw_abs_error_p90_deg": percentile(yaw_errors, 90),
                "dim_mae_m": mean(dim_errors),
                "dim_mae_p50_m": percentile(dim_errors, 50),
                "dim_mae_p90_m": percentile(dim_errors, 90),
                "dim_mean_rel_error": mean(dim_rel_errors),
            }
        )

    return summary_rows


def summarize_class_distance(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        key = (row["class_name"], row["distance_bucket"])
        groups[key].append(row)

    summary_rows: List[Dict[str, Any]] = []

    for (class_name, bucket), group_rows in sorted(groups.items(), key=lambda x: x[0]):
        depth_errors = [float(r["depth_abs_error_m"]) for r in group_rows]
        yaw_errors = [float(r["yaw_abs_error_deg"]) for r in group_rows]
        dim_errors = [float(r["dim_mae_m"]) for r in group_rows]
        ious = [float(r["iou_2d"]) for r in group_rows]

        summary_rows.append(
            {
                "group_name": "class_distance_bucket",
                "class_name": class_name,
                "distance_bucket": bucket,
                "count": len(group_rows),
                "iou_2d_mean": mean(ious),
                "depth_mae_m": mean(depth_errors),
                "depth_abs_error_p50_m": percentile(depth_errors, 50),
                "depth_abs_error_p90_m": percentile(depth_errors, 90),
                "yaw_abs_error_mean_deg": mean(yaw_errors),
                "yaw_abs_error_p50_deg": percentile(yaw_errors, 50),
                "yaw_abs_error_p90_deg": percentile(yaw_errors, 90),
                "dim_mae_m": mean(dim_errors),
                "dim_mae_p50_m": percentile(dim_errors, 50),
                "dim_mae_p90_m": percentile(dim_errors, 90),
            }
        )

    return summary_rows


def load_model(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loaded checkpoint.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Global step: {checkpoint.get('global_step', 'unknown')}")
    print(f"Metric value: {checkpoint.get('metric_value', checkpoint.get('loss', 'unknown'))}")

    return model


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
    target_cfg = config["targets"]
    training_cfg = config["training"]

    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]

    split_file = get_split_file(config, args.split)

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=split_file,
    )

    if args.max_images is None or args.max_images < 0:
        num_images = len(dataset)
    else:
        num_images = min(args.max_images, len(dataset))

    device = get_device(training_cfg.get("device", "auto"))

    model = load_model(
        config=config,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    input_height = int(model_cfg["input_height"])
    input_width = int(model_cfg["input_width"])

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["outputs"]["visualization_dir"]) / "evaluation_3d"

    output_dir.mkdir(parents=True, exist_ok=True)

    matched_rows: List[Dict[str, Any]] = []
    false_positive_rows: List[Dict[str, Any]] = []
    false_negative_rows: List[Dict[str, Any]] = []

    print("\nStarting 3D metrics evaluation.")
    print(f"Config: {args.config}")
    print(f"Profile: {active_profile}")
    print(f"Dataset root: {root_dir}")
    print(f"Split: {args.split}")
    print(f"Split file: {split_file}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Evaluating images: {num_images}")
    print(f"Input size: {input_width} x {input_height}")
    print(f"Device: {device}")
    print(f"Score threshold: {args.score_threshold}")
    print(f"Match IoU threshold: {args.match_iou_threshold}")
    print(f"TopK: {args.topk}")
    print(f"NMS IoU threshold: {args.nms_iou_threshold}")
    print(f"Output dir: {output_dir}")

    with torch.no_grad():
        for idx in range(num_images):
            sample = dataset[idx]

            image = sample["image"].unsqueeze(0)

            image = F.interpolate(
                image,
                size=(input_height, input_width),
                mode="bilinear",
                align_corners=False,
            ).to(device)

            outputs = model(image)

            predictions = decode_mobile_adas3d_outputs(
                outputs=outputs,
                classes=dataset_cfg["classes"],
                class_mean_dims=target_cfg["class_mean_dims"],
                input_height=input_height,
                input_width=input_width,
                score_threshold=args.score_threshold,
                topk=args.topk,
                nms_iou_threshold=args.nms_iou_threshold,
            )[0]

            gt_objects = scale_gt_objects_to_input(
                sample=sample,
                input_width=input_width,
                input_height=input_height,
            )

            matches, unmatched_predictions, unmatched_gt = greedy_match_predictions_to_gt(
                predictions=predictions,
                gt_objects=gt_objects,
                match_iou_threshold=args.match_iou_threshold,
            )

            for match in matches:
                matched_rows.append(
                    compute_match_metric_row(
                        split_name=args.split,
                        match=match,
                    )
                )

            for pred in unmatched_predictions:
                false_positive_rows.append(
                    {
                        "split": args.split,
                        "sample_id": sample["sample_id"],
                        "class_name": pred["class_name"],
                        "score": float(pred["score"]),
                        "bbox_x1": float(pred["bbox_2d"][0]),
                        "bbox_y1": float(pred["bbox_2d"][1]),
                        "bbox_x2": float(pred["bbox_2d"][2]),
                        "bbox_y2": float(pred["bbox_2d"][3]),
                        "pred_depth_m": float(pred["depth"]),
                        "pred_yaw_rad": float(pred["yaw"]),
                        "best_iou": float(pred.get("best_iou", 0.0)),
                        "best_iou_class_agnostic": float(
                            pred.get("best_iou_class_agnostic", 0.0)
                        ),
                    }
                )

            for gt in unmatched_gt:
                false_negative_rows.append(
                    {
                        "split": args.split,
                        "sample_id": sample["sample_id"],
                        "class_name": gt["class_name"],
                        "gt_depth_m": float(gt["depth_m"]),
                        "distance_bucket": gt["distance_bucket"],
                        "size_bucket": gt["size_bucket"],
                        "gt_box_width_px": float(gt["width_px"]),
                        "gt_box_height_px": float(gt["height_px"]),
                        "gt_box_area_px2": float(gt["area_px2"]),
                    }
                )

            if (idx + 1) % 50 == 0 or idx + 1 == num_images:
                print(
                    f"Processed {idx + 1}/{num_images} images. "
                    f"matches={len(matched_rows)} "
                    f"fp={len(false_positive_rows)} "
                    f"fn={len(false_negative_rows)}"
                )

    matched_csv = output_dir / f"matched_3d_metrics_{args.split}.csv"
    fp_csv = output_dir / f"false_positives_{args.split}.csv"
    fn_csv = output_dir / f"false_negatives_{args.split}.csv"

    save_csv(matched_rows, matched_csv)
    save_csv(false_positive_rows, fp_csv)
    save_csv(false_negative_rows, fn_csv)

    summary_rows: List[Dict[str, Any]] = []
    summary_rows.extend(
        summarize_metric_rows(
            matched_rows,
            group_name="ALL",
            group_key=None,
        )
    )
    summary_rows.extend(
        summarize_metric_rows(
            matched_rows,
            group_name="class_name",
            group_key="class_name",
        )
    )
    summary_rows.extend(
        summarize_metric_rows(
            matched_rows,
            group_name="distance_bucket",
            group_key="distance_bucket",
        )
    )
    summary_rows.extend(
        summarize_metric_rows(
            matched_rows,
            group_name="size_bucket",
            group_key="size_bucket",
        )
    )

    summary_csv = output_dir / f"summary_3d_metrics_{args.split}.csv"
    save_csv(summary_rows, summary_csv)

    class_distance_rows = summarize_class_distance(matched_rows)
    class_distance_csv = output_dir / f"summary_3d_class_distance_{args.split}.csv"
    save_csv(class_distance_rows, class_distance_csv)

    print("\n3D Metrics Summary:")
    for row in summary_rows:
        if row["group_name"] == "ALL":
            print(
                f"ALL matched={row['count']} "
                f"depth_mae={row['depth_mae_m']:.3f}m "
                f"depth_rel={row['depth_rel_error_mean']:.3f} "
                f"yaw_mae={row['yaw_abs_error_mean_deg']:.2f}deg "
                f"dim_mae={row['dim_mae_m']:.3f}m "
                f"iou_mean={row['iou_2d_mean']:.3f}"
            )

    print("\nPer-class summary:")
    for row in summary_rows:
        if row["group_name"] == "class_name":
            print(
                f"{row['group_value']}: "
                f"count={row['count']} "
                f"depth_mae={row['depth_mae_m']:.3f}m "
                f"depth_rel={row['depth_rel_error_mean']:.3f} "
                f"yaw_mae={row['yaw_abs_error_mean_deg']:.2f}deg "
                f"dim_mae={row['dim_mae_m']:.3f}m "
                f"iou_mean={row['iou_2d_mean']:.3f}"
            )

    print("\nPer-distance summary:")
    for row in summary_rows:
        if row["group_name"] == "distance_bucket":
            print(
                f"{row['group_value']}: "
                f"count={row['count']} "
                f"depth_mae={row['depth_mae_m']:.3f}m "
                f"depth_rel={row['depth_rel_error_mean']:.3f} "
                f"yaw_mae={row['yaw_abs_error_mean_deg']:.2f}deg "
                f"dim_mae={row['dim_mae_m']:.3f}m"
            )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()