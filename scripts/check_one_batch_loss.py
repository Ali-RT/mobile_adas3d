from __future__ import annotations

from functools import partial
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from data.collate import mobile_adas3d_collate_fn
from data.kitti_dataset import KITTIDataset
from data.split_resolver import get_split_file
from losses.mobile_adas3d_loss import MobileADAS3DLoss
from models.build import build_model
from tools.cli import parse_config_profile_args
from tools.config import load_runtime_config_from_args
from tools.device import get_device


def move_targets_to_device(
    targets: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in targets.items()
    }


def build_criterion(config: Dict[str, Any]) -> MobileADAS3DLoss:
    loss_cfg = config.get("loss", {})

    return MobileADAS3DLoss(
        input_height=int(config["model"]["input_height"]),
        input_width=int(config["model"]["input_width"]),
        classes=config["dataset"]["classes"],
        class_weights=loss_cfg.get("class_weights", {}),
        cls_weight=float(loss_cfg.get("cls_weight", 1.0)),
        box2d_weight=float(loss_cfg.get("box2d_weight", 2.0)),
        depth_weight=float(loss_cfg.get("depth_weight", 1.0)),
        depth_uncertainty_weight=float(loss_cfg.get("depth_uncertainty_weight", 0.0)),
        dim_weight=float(loss_cfg.get("dim_weight", 1.0)),
        yaw_weight=float(loss_cfg.get("yaw_weight", 1.0)),
        offset_weight=float(loss_cfg.get("offset_weight", 0.5)),
    )


def main() -> None:
    args = parse_config_profile_args("Check one-batch MobileADAS3D loss")
    config = load_runtime_config_from_args(args)

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    target_cfg = config["targets"]
    loss_cfg = config.get("loss", {})
    training_cfg = config["training"]

    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]

    split_name = getattr(args, "split", "val")
    split_file = get_split_file(config, split_name)

    device = get_device(training_cfg.get("device", "auto"))

    center_sampling_cfg = target_cfg.get("center_sampling", {})
    center_sampling_radius = 0

    if center_sampling_cfg.get("enabled", False):
        center_sampling_radius = int(center_sampling_cfg.get("radius", 1))

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=split_file,
    )

    batch_size = min(2, len(dataset))

    collate_fn = partial(
        mobile_adas3d_collate_fn,
        classes=dataset_cfg["classes"],
        input_height=int(model_cfg["input_height"]),
        input_width=int(model_cfg["input_width"]),
        output_stride=int(model_cfg["output_stride"]),
        class_mean_dims=target_cfg["class_mean_dims"],
        center_sampling_radius=center_sampling_radius,
        class_weights=loss_cfg.get("class_weights", {}),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        drop_last=False,
    )

    batch = next(iter(dataloader))

    images = batch["images"].to(device)
    targets = move_targets_to_device(batch["targets"], device)

    model = build_model(config)
    model.to(device)
    model.train()

    criterion = build_criterion(config)

    outputs = model(images)
    losses = criterion(outputs, targets)

    print("One-batch loss check")
    print(f"Active profile: {active_profile}")
    print(f"Dataset root: {root_dir}")
    print(f"Split: {split_name}")
    print(f"Split file: {split_file}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print(f"Input images shape: {tuple(images.shape)}")
    print(f"Output stride: {model_cfg['output_stride']}")

    print("\nOutput shapes:")
    for key, value in outputs.items():
        print(f"  {key}: {tuple(value.shape)}")

    print("\nTarget shapes:")
    for key, value in targets.items():
        print(f"  {key}: {tuple(value.shape)}")

    print("\nPositive cells:")
    print(f"  valid_mask sum: {float(targets['valid_mask'].sum().item()):.1f}")
    print(f"  cls_target sum: {float(targets['cls_target'].sum().item()):.1f}")

    if "loss_weight_target" in targets:
        positive_weights = targets["loss_weight_target"][targets["valid_mask"] > 0]
        if positive_weights.numel() > 0:
            print(
                "  positive loss weights: "
                f"min={positive_weights.min().item():.2f}, "
                f"max={positive_weights.max().item():.2f}, "
                f"mean={positive_weights.mean().item():.2f}"
            )

    print("\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {float(value.item()):.6f}")

    total_loss = losses["total_loss"]

    if not torch.isfinite(total_loss):
        raise RuntimeError(f"Non-finite total loss: {total_loss.item()}")

    print("\nStatus: one-batch loss check passed.")


if __name__ == "__main__":
    main()