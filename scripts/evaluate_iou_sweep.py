from __future__ import annotations

import argparse
import csv
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
        description="Evaluate MobileADAS3D with confidence-threshold sweep and per-class IoU metrics."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/kitti_mobileadas3d.yaml",
        help="Path to config file.",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Runtime profile, e.g. local_mac or colab_drive.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint, usually best.pt.",
    )

    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Split to evaluate.",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=500,
        help="Maximum number of images to evaluate.",
    )

    parser.add_argument(
        "--score-thresholds",
        type=str,
        default="0.03,0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
        help="Comma-separated confidence thresholds.",
    )

    parser.add_argument(
        "--iou-thresholds",
        type=str,
        default="0.25,0.50",
        help="Comma-separated IoU thresholds.",
    )

    parser.add_argument(
        "--topk",
        type=int,
        default=200,
        help="Top-k predictions to decode per image before threshold filtering.",
    )

    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.5,
        help="NMS IoU threshold.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory for CSV files. If omitted, uses config visualization_dir/evaluation.",
    )

    return parser.parse_args()


def parse_float_list(value: str) -> List[float]:
    return [float(x.strip()) for x in value.split(",") if x.strip()]


def scale_gt_objects_to_input(
    objects: List[Dict[str, Any]],
    original_width: int,
    original_height: int,
    input_width: int,
    input_height: int,
) -> List[Dict[str, Any]]:
    scaled = []

    for obj in objects:
        bbox = scale_bbox_2d(
            bbox=obj["bbox_2d"],
            original_width=original_width,
            original_height=original_height,
            input_width=input_width,
            input_height=input_height,
        )

        scaled.append(
            {
                "class_name": obj["class_name"],
                "class_id": obj["class_id"],
                "bbox_2d": bbox,
                "depth": float(obj["location_3d"][2]),
            }
        )

    return scaled


def initialize_stats(classes: List[str]) -> Dict[str, Dict[str, Any]]:
    stats: Dict[str, Dict[str, Any]] = {
        "ALL": {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_ious": [],
        }
    }

    for class_name in classes:
        stats[class_name] = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "matched_ious": [],
        }

    return stats


def update_stats(
    stats: Dict[str, Dict[str, Any]],
    class_name: str,
    tp: int,
    fp: int,
    fn: int,
    matched_ious: List[float],
) -> None:
    stats[class_name]["tp"] += tp
    stats[class_name]["fp"] += fp
    stats[class_name]["fn"] += fn
    stats[class_name]["matched_ious"].extend(matched_ious)

    stats["ALL"]["tp"] += tp
    stats["ALL"]["fp"] += fp
    stats["ALL"]["fn"] += fn
    stats["ALL"]["matched_ious"].extend(matched_ious)


def greedy_match_one_class(
    predictions: List[Dict[str, Any]],
    gt_objects: List[Dict[str, Any]],
    class_name: str,
    iou_threshold: float,
) -> Tuple[int, int, int, List[float]]:
    """
    Greedy same-class matching for one class.
    """
    pred_cls = [
        p for p in predictions
        if p["class_name"] == class_name
    ]

    gt_cls = [
        g for g in gt_objects
        if g["class_name"] == class_name
    ]

    if len(pred_cls) == 0:
        return 0, 0, len(gt_cls), []

    if len(gt_cls) == 0:
        return 0, len(pred_cls), 0, []

    pred_cls = sorted(pred_cls, key=lambda x: x["score"], reverse=True)

    gt_boxes = torch.tensor(
        [g["bbox_2d"] for g in gt_cls],
        dtype=torch.float32,
    )

    matched_gt = set()
    tp = 0
    fp = 0
    matched_ious: List[float] = []

    for pred in pred_cls:
        pred_box = torch.tensor(pred["bbox_2d"], dtype=torch.float32)

        ious = box_iou(pred_box, gt_boxes)

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, iou in enumerate(ious.tolist()):
            if gt_idx in matched_gt:
                continue

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
            matched_ious.append(float(best_iou))
        else:
            fp += 1

    fn = len(gt_cls) - len(matched_gt)

    return tp, fp, fn, matched_ious


def compute_summary_row(
    split_name: str,
    score_threshold: float,
    iou_threshold: float,
    class_name: str,
    stat: Dict[str, Any],
) -> Dict[str, Any]:
    tp = int(stat["tp"])
    fp = int(stat["fp"])
    fn = int(stat["fn"])
    ious = stat["matched_ious"]

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    mean_iou = sum(ious) / max(len(ious), 1)

    return {
        "split": split_name,
        "score_threshold": score_threshold,
        "iou_threshold": iou_threshold,
        "class_name": class_name,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": mean_iou,
    }


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to save.")

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


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

    score_thresholds = parse_float_list(args.score_thresholds)
    iou_thresholds = parse_float_list(args.iou_thresholds)

    if not score_thresholds:
        raise ValueError("No score thresholds provided.")

    if not iou_thresholds:
        raise ValueError("No IoU thresholds provided.")

    min_score_threshold = min(score_thresholds)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    target_cfg = config["targets"]
    training_cfg = config["training"]

    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]
    classes = dataset_cfg["classes"]

    split_file = get_split_file(config, args.split)

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=classes,
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=split_file,
    )

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
        output_dir = Path(config["outputs"]["visualization_dir"]) / "evaluation"

    output_dir.mkdir(parents=True, exist_ok=True)

    if args.max_images is None or args.max_images < 0:
        num_images = len(dataset)
    else:
        num_images = min(args.max_images, len(dataset))

    # Nested results:
    # results[(score_thr, iou_thr)] = stats
    results: Dict[Tuple[float, float], Dict[str, Dict[str, Any]]] = {}

    for score_thr in score_thresholds:
        for iou_thr in iou_thresholds:
            results[(score_thr, iou_thr)] = initialize_stats(classes)

    print("\nStarting IoU sweep evaluation.")
    print(f"Config: {args.config}")
    print(f"Profile: {active_profile}")
    print(f"Dataset root: {root_dir}")
    print(f"Split: {args.split}")
    print(f"Split file: {split_file}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Evaluating images: {num_images}")
    print(f"Input size: {input_width} x {input_height}")
    print(f"Device: {device}")
    print(f"Score thresholds: {score_thresholds}")
    print(f"IoU thresholds: {iou_thresholds}")
    print(f"Decode min score threshold: {min_score_threshold}")
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

            # Decode once at minimum threshold, then filter for each threshold.
            decoded_predictions = decode_mobile_adas3d_outputs(
                outputs=outputs,
                classes=classes,
                class_mean_dims=target_cfg["class_mean_dims"],
                input_height=input_height,
                input_width=input_width,
                score_threshold=min_score_threshold,
                topk=args.topk,
                nms_iou_threshold=args.nms_iou_threshold,
            )[0]

            gt_scaled = scale_gt_objects_to_input(
                objects=sample["objects"],
                original_width=int(sample["original_size"]["width"]),
                original_height=int(sample["original_size"]["height"]),
                input_width=input_width,
                input_height=input_height,
            )

            for score_thr in score_thresholds:
                predictions = [
                    p for p in decoded_predictions
                    if p["score"] >= score_thr
                ]

                for iou_thr in iou_thresholds:
                    stats = results[(score_thr, iou_thr)]

                    for class_name in classes:
                        tp, fp, fn, matched_ious = greedy_match_one_class(
                            predictions=predictions,
                            gt_objects=gt_scaled,
                            class_name=class_name,
                            iou_threshold=iou_thr,
                        )

                        update_stats(
                            stats=stats,
                            class_name=class_name,
                            tp=tp,
                            fp=fp,
                            fn=fn,
                            matched_ious=matched_ious,
                        )

            if (idx + 1) % 50 == 0 or idx + 1 == num_images:
                print(f"Processed {idx + 1}/{num_images} images.")

    rows: List[Dict[str, Any]] = []

    for score_thr in score_thresholds:
        for iou_thr in iou_thresholds:
            stats = results[(score_thr, iou_thr)]

            for class_name in ["ALL"] + classes:
                rows.append(
                    compute_summary_row(
                        split_name=args.split,
                        score_threshold=score_thr,
                        iou_threshold=iou_thr,
                        class_name=class_name,
                        stat=stats[class_name],
                    )
                )

    output_csv = output_dir / f"iou_sweep_{args.split}.csv"
    save_csv(rows, output_csv)

    # Print concise summary for ALL classes.
    print("\nOverall IoU Sweep Summary:")
    print(
        "score_thr | iou_thr | precision | recall | f1 | mean_iou | TP | FP | FN"
    )

    for row in rows:
        if row["class_name"] != "ALL":
            continue

        print(
            f"{row['score_threshold']:.2f}      "
            f"| {row['iou_threshold']:.2f}   "
            f"| {row['precision']:.4f}    "
            f"| {row['recall']:.4f} "
            f"| {row['f1']:.4f} "
            f"| {row['mean_matched_iou']:.4f} "
            f"| {row['tp']} "
            f"| {row['fp']} "
            f"| {row['fn']}"
        )

    # Print per-class summary at two commonly useful thresholds.
    preferred_score_threshold = 0.25 if 0.25 in score_thresholds else score_thresholds[-1]

    print(f"\nPer-class summary at score_threshold={preferred_score_threshold}:")
    for iou_thr in iou_thresholds:
        print(f"\nIoU threshold: {iou_thr}")
        for row in rows:
            if (
                row["score_threshold"] == preferred_score_threshold
                and row["iou_threshold"] == iou_thr
                and row["class_name"] != "ALL"
            ):
                print(
                    f"{row['class_name']}: "
                    f"P={row['precision']:.4f}, "
                    f"R={row['recall']:.4f}, "
                    f"F1={row['f1']:.4f}, "
                    f"mIoU={row['mean_matched_iou']:.4f}, "
                    f"TP={row['tp']}, FP={row['fp']}, FN={row['fn']}"
                )

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()