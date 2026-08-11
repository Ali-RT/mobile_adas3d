from pathlib import Path
import sys
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F

from data.kitti_dataset import KITTIDataset
from data.split_resolver import get_split_file
from data.target_builder import scale_bbox_2d
from data.geometry import scale_p2_for_resize
from data.visualization import draw_gt_and_predictions_2d, draw_projected_3d_boxes
from models.build import build_model
from models.decode import decode_mobile_adas3d_outputs
from tools.cli import parse_config_profile_args
from tools.config import load_runtime_config_from_args
from tools.device import get_device


def resize_image_tensor_to_rgb_uint8(
    image_tensor: torch.Tensor,
    input_height: int,
    input_width: int,
) -> np.ndarray:
    """
    Resize image tensor to model input size and convert to RGB uint8.

    Args:
      image_tensor: [3, H, W], float in [0, 1]

    Returns:
      image_rgb: [input_height, input_width, 3], uint8
    """
    resized = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(input_height, input_width),
        mode="bilinear",
        align_corners=False,
    )

    image_rgb = (
        resized.squeeze(0)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    image_rgb = (image_rgb * 255.0).clip(0, 255).astype("uint8")

    return image_rgb


def scale_gt_objects_to_model_input(
    sample: Dict[str, Any],
    input_height: int,
    input_width: int,
) -> List[Dict[str, Any]]:
    """
    Scale original KITTI GT boxes to the resized model input coordinate frame.

    This is important because predictions are decoded in model input space:
      input_width x input_height
    """
    gt_scaled = []

    original_width = int(sample["original_size"]["width"])
    original_height = int(sample["original_size"]["height"])

    for obj in sample["objects"]:
        bbox = scale_bbox_2d(
            bbox=obj["bbox_2d"],
            original_width=original_width,
            original_height=original_height,
            input_width=input_width,
            input_height=input_height,
        )

        gt_scaled.append(
            {
                "class_name": obj["class_name"],
                "class_id": obj.get("class_id", None),
                "bbox_2d": bbox,
                "depth": float(obj["location_3d"][2]),
                "dimensions_3d": obj["dimensions_3d"],
                "rotation_y": obj["rotation_y"],
            }
        )

    return gt_scaled


def load_model_from_checkpoint(
    config: Dict[str, Any],
    checkpoint_path: str,
    device: torch.device,
) -> torch.nn.Module:
    model = build_model(config)

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain 'model_state_dict': {checkpoint_path}"
        )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loaded checkpoint successfully.")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Checkpoint global_step: {checkpoint.get('global_step', 'unknown')}")
    print(f"Checkpoint loss: {checkpoint.get('loss', 'unknown')}")

    return model


def main() -> None:
    args = parse_config_profile_args("Visualize MobileADAS3D predictions")
    config = load_runtime_config_from_args(args)

    checkpoint_path = args.checkpoint
    split_name = args.split
    max_images = int(args.max_images)
    image_id = args.image_id
    score_threshold = float(args.score_threshold)

    if checkpoint_path is None:
        raise ValueError("Please pass --checkpoint path/to/best.pt")

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    target_cfg = config["targets"]
    training_cfg = config["training"]

    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]
    split_file = get_split_file(config, split_name)

    input_height = int(model_cfg["input_height"])
    input_width = int(model_cfg["input_width"])

    device = get_device(training_cfg.get("device", "auto"))

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=split_file,
        class_mapping=dataset_cfg.get("class_mapping"),
    )

    model = load_model_from_checkpoint(
        config=config,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["outputs"]["visualization_dir"]) / "predictions_overlay"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning prediction visualization.")
    print(f"Active profile: {active_profile}")
    print(f"Dataset root: {root_dir}")
    print(f"Split: {split_name}")
    print(f"Split file: {split_file}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Max images: {max_images}")
    print(f"Image ID: {image_id}")
    print(f"Score threshold: {score_threshold}")
    print(f"Input size: {input_width} x {input_height}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")

    if image_id is not None:
        image_id = dataset.resolve_sample_id(image_id)
        selected_indices = [dataset.sample_ids.index(image_id)]
    else:
        num_images = len(dataset) if max_images < 0 else min(max_images, len(dataset))
        selected_indices = list(range(num_images))

    num_images = len(selected_indices)

    with torch.no_grad():
        for output_idx, dataset_idx in enumerate(selected_indices):
            sample = dataset[dataset_idx]

            # Resize input for model.
            image = sample["image"].unsqueeze(0)

            image_resized = F.interpolate(
                image,
                size=(input_height, input_width),
                mode="bilinear",
                align_corners=False,
            ).to(device)

            outputs = model(image_resized)

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

            # Scale GT boxes to the same coordinate frame as predictions.
            gt_scaled = scale_gt_objects_to_model_input(
                sample=sample,
                input_height=input_height,
                input_width=input_width,
            )

            # Draw both GT and predictions on the resized model input image.
            resized_rgb = resize_image_tensor_to_rgb_uint8(
                image_tensor=sample["image"],
                input_height=input_height,
                input_width=input_width,
            )

            overlay_path = output_dir / f"{sample['sample_id']}_overlay.png"

            draw_gt_and_predictions_2d(
                image_rgb=resized_rgb,
                gt_objects=gt_scaled,
                predictions=predictions,
                output_path=overlay_path,
            )

            # Predicted cuboid overlay reconstructed from physical 3D pose
            # (decoded location_3d + dimensions + yaw), projected with P2.
            # Do not backproject the 2D bbox center anymore.
            P2_model = scale_p2_for_resize(
                P2=np.asarray(sample["P2"], dtype=np.float32),
                orig_w=int(sample["original_size"]["width"]),
                orig_h=int(sample["original_size"]["height"]),
                input_w=input_width,
                input_h=input_height,
            )

            pred_cuboid_objects = [
                {
                    "class_name": pred["class_name"],
                    "bbox_2d": pred["bbox_2d"],
                    "dimensions_3d": pred["dimensions_3d_hwl"],
                    "location_3d": pred["location_3d"],
                    "rotation_y": pred["yaw"],
                }
                for pred in predictions
            ]

            cuboid_path = output_dir / f"{sample['sample_id']}_cuboid_pred.png"

            draw_projected_3d_boxes(
                image_rgb=resized_rgb,
                objects=pred_cuboid_objects,
                P2=P2_model,
                output_path=cuboid_path,
            )

            print(
                f"[{output_idx + 1}/{num_images}] "
                f"sample={sample['sample_id']} "
                f"image={sample['image_path']} "
                f"gt={len(gt_scaled)} "
                f"pred={len(predictions)} "
                f"saved={overlay_path}"
            )


if __name__ == "__main__":
    main()
