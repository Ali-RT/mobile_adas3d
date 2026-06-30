from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from data.kitti_dataset import KITTIDataset
from data.split_resolver import get_split_file
from data.target_builder import scale_bbox_2d
from models.build import build_model
from models.decode import decode_mobile_adas3d_outputs, box_iou
from tools.cli import parse_config_profile_args
from tools.config import load_runtime_config_from_args
from tools.device import get_device


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
                "depth": obj["location_3d"][2],
            }
        )

    return scaled


def greedy_match_predictions_to_gt(
    predictions: List[Dict[str, Any]],
    gt_objects: List[Dict[str, Any]],
    iou_threshold: float,
) -> Tuple[int, int, int, List[float]]:
    """
    Match predictions to GT using same-class greedy matching.

    Returns:
      tp, fp, fn, matched_ious
    """
    if len(predictions) == 0:
        return 0, 0, len(gt_objects), []

    if len(gt_objects) == 0:
        return 0, len(predictions), 0, []

    predictions = sorted(predictions, key=lambda x: x["score"], reverse=True)

    gt_boxes = torch.tensor(
        [g["bbox_2d"] for g in gt_objects],
        dtype=torch.float32,
    )

    gt_classes = [g["class_name"] for g in gt_objects]
    matched_gt = set()

    tp = 0
    fp = 0
    matched_ious = []

    for pred in predictions:
        pred_box = torch.tensor(pred["bbox_2d"], dtype=torch.float32)

        ious = box_iou(pred_box, gt_boxes)

        best_iou = 0.0
        best_gt_idx = -1

        for gt_idx, iou in enumerate(ious.tolist()):
            if gt_idx in matched_gt:
                continue

            if pred["class_name"] != gt_classes[gt_idx]:
                continue

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx >= 0 and best_iou >= iou_threshold:
            tp += 1
            matched_gt.add(best_gt_idx)
            matched_ious.append(best_iou)
        else:
            fp += 1

    fn = len(gt_objects) - len(matched_gt)

    return tp, fp, fn, matched_ious


def main() -> None:
    args = parse_config_profile_args("Evaluate MobileADAS3D 2D IoU")
    config = load_runtime_config_from_args(args)

    checkpoint_path = args.checkpoint
    split_name = args.split
    max_images = args.max_images
    image_id = args.image_id
    score_threshold = args.score_threshold

    if checkpoint_path is None:
        raise ValueError("Please pass --checkpoint path/to/best.pt")

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    target_cfg = config["targets"]
    training_cfg = config["training"]

    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]

    split_file = get_split_file(config, split_name)

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=split_file,
    )

    device = get_device(training_cfg.get("device", "auto"))

    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    input_height = model_cfg["input_height"]
    input_width = model_cfg["input_width"]

    iou_thresholds = [0.25, 0.50]

    if image_id is not None:
        image_id = dataset.resolve_sample_id(image_id)
        selected_indices = [dataset.sample_ids.index(image_id)]
    else:
        num_images = (
            len(dataset)
            if max_images is None or max_images < 0
            else min(max_images, len(dataset))
        )
        selected_indices = list(range(num_images))

    num_images = len(selected_indices)

    totals = {
        thr: {"tp": 0, "fp": 0, "fn": 0, "ious": []}
        for thr in iou_thresholds
    }

    print("Evaluating IoU.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split_name}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Max images: {max_images}")
    print(f"Image ID: {image_id}")
    print(f"Score threshold: {score_threshold}")
    print(f"Input size: {input_width} x {input_height}")
    print(f"Device: {device}")

    with torch.no_grad():
        for output_idx, dataset_idx in enumerate(selected_indices):
            sample = dataset[dataset_idx]

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
                score_threshold=score_threshold,
                topk=100,
                nms_iou_threshold=0.5,
            )[0]

            gt_scaled = scale_gt_objects_to_input(
                objects=sample["objects"],
                original_width=sample["original_size"]["width"],
                original_height=sample["original_size"]["height"],
                input_width=input_width,
                input_height=input_height,
            )

            print(
                f"[{output_idx + 1}/{num_images}] "
                f"sample={sample['sample_id']} "
                f"image={sample['image_path']} "
                f"gt={len(gt_scaled)} pred={len(predictions)}"
            )

            for thr in iou_thresholds:
                tp, fp, fn, matched_ious = greedy_match_predictions_to_gt(
                    predictions=predictions,
                    gt_objects=gt_scaled,
                    iou_threshold=thr,
                )

                totals[thr]["tp"] += tp
                totals[thr]["fp"] += fp
                totals[thr]["fn"] += fn
                totals[thr]["ious"].extend(matched_ious)

    print("\nIoU Summary:")
    for thr in iou_thresholds:
        tp = totals[thr]["tp"]
        fp = totals[thr]["fp"]
        fn = totals[thr]["fn"]
        ious = totals[thr]["ious"]

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        mean_iou = sum(ious) / max(len(ious), 1)

        print(f"\nIoU threshold: {thr}")
        print(f"  TP: {tp}")
        print(f"  FP: {fp}")
        print(f"  FN: {fn}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  Mean matched IoU: {mean_iou:.4f}")


if __name__ == "__main__":
    main()
