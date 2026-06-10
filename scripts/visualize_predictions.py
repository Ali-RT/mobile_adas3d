from pathlib import Path

import torch
import torch.nn.functional as F

from data.kitti_dataset import KITTIDataset
from data.split_resolver import get_split_file
from data.visualization import draw_2d_boxes, draw_predictions_2d
from models.build import build_model
from models.decode import decode_mobile_adas3d_outputs
from tools.cli import parse_config_profile_args
from tools.config import load_runtime_config_from_args
from tools.device import get_device


def main() -> None:
    args = parse_config_profile_args("Visualize MobileADAS3D predictions")
    config = load_runtime_config_from_args(args)

    # Add manual fields if argparse does not include them yet.
    checkpoint_path = getattr(args, "checkpoint", None)
    split_name = getattr(args, "split", "val")
    max_images = int(getattr(args, "max_images", 20))
    score_threshold = float(getattr(args, "score_threshold", 0.25))

    if checkpoint_path is None:
        raise ValueError(
            "Please pass --checkpoint path/to/best.pt"
        )

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

    output_dir = Path(config["outputs"]["visualization_dir"]) / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Running prediction visualization.")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Split: {split_name}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Device: {device}")
    print(f"Output dir: {output_dir}")

    input_height = model_cfg["input_height"]
    input_width = model_cfg["input_width"]

    with torch.no_grad():
        for idx in range(min(max_images, len(dataset))):
            sample = dataset[idx]

            image = sample["image"].unsqueeze(0)
            image = F.interpolate(
                image,
                size=(input_height, input_width),
                mode="bilinear",
                align_corners=False,
            )

            image = image.to(device)

            outputs = model(image)

            batch_predictions = decode_mobile_adas3d_outputs(
                outputs=outputs,
                classes=dataset_cfg["classes"],
                class_mean_dims=target_cfg["class_mean_dims"],
                input_height=input_height,
                input_width=input_width,
                score_threshold=score_threshold,
                topk=100,
                nms_iou_threshold=0.5,
            )

            predictions = batch_predictions[0]

            # Ground-truth visualization on original image.
            gt_path = output_dir / f"{sample['sample_id']}_gt.png"
            draw_2d_boxes(
                image_rgb=sample["image_rgb"],
                objects=sample["objects"],
                output_path=gt_path,
            )

            # Prediction visualization uses resized coordinate space.
            # So we visualize on resized image.
            resized_rgb = (
                F.interpolate(
                    sample["image"].unsqueeze(0),
                    size=(input_height, input_width),
                    mode="bilinear",
                    align_corners=False,
                )
                .squeeze(0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )

            resized_rgb = (resized_rgb * 255).clip(0, 255).astype("uint8")

            pred_path = output_dir / f"{sample['sample_id']}_pred.png"
            draw_predictions_2d(
                image_rgb=resized_rgb,
                predictions=predictions,
                output_path=pred_path,
            )

            print(
                f"[{idx + 1}/{max_images}] "
                f"sample={sample['sample_id']} "
                f"predictions={len(predictions)} "
                f"saved={pred_path}"
            )


if __name__ == "__main__":
    main()