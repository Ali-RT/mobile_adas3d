from functools import partial
from pathlib import Path
from typing import Dict
import argparse

import torch
from torch.utils.data import DataLoader

from data.collate import mobile_adas3d_collate_fn
from data.kitti_dataset import KITTIDataset
from losses.mobile_adas3d_loss import MobileADAS3DLoss
from models.build import build_model
from tools.config import load_config
from tools.device import get_device

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MobileADAS3D")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/kitti_mobileadas3d.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()

def move_targets_to_device(
    targets: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device)
        for key, value in targets.items()
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    loss_value: float,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_value,
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    print(f"Using config: {args.config}")

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    target_cfg = config["targets"]
    training_cfg = config["training"]
    outputs_cfg = config["outputs"]

    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]

    device = get_device(training_cfg.get("device", "auto"))

    print("Starting MobileADAS3D training.")
    print(f"Active dataset profile: {active_profile}")
    print(f"Dataset root: {root_dir}")
    print(f"Device: {device}")

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
    )

    collate_fn = partial(
        mobile_adas3d_collate_fn,
        classes=dataset_cfg["classes"],
        input_height=model_cfg["input_height"],
        input_width=model_cfg["input_width"],
        output_stride=model_cfg["output_stride"],
        class_mean_dims=target_cfg["class_mean_dims"],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=True,
        num_workers=training_cfg["num_workers"],
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model(config)
    model.to(device)
    model.train()

    criterion = MobileADAS3DLoss(
        input_height=model_cfg["input_height"],
        input_width=model_cfg["input_width"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    checkpoint_dir = Path(outputs_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    epochs = training_cfg["epochs"]
    log_interval = training_cfg.get("log_interval", 10)
    save_interval = training_cfg.get("save_interval", 1)

    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader, start=1):
            images = batch["images"].to(device)
            targets = move_targets_to_device(batch["targets"], device)

            outputs = model(images)
            losses = criterion(outputs, targets)

            total_loss = losses["total_loss"]

            if not torch.isfinite(total_loss):
                raise RuntimeError(f"Non-finite loss detected: {total_loss.item()}")

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            optimizer.step()

            loss_value = float(total_loss.item())
            epoch_loss += loss_value
            num_batches += 1
            global_step += 1

            if batch_idx % log_interval == 0:
                print(
                    f"epoch={epoch:03d}/{epochs:03d} "
                    f"batch={batch_idx:04d}/{len(dataloader):04d} "
                    f"step={global_step:06d} "
                    f"total={losses['total_loss'].item():.4f} "
                    f"cls={losses['cls_loss'].item():.4f} "
                    f"box2d={losses['box2d_loss'].item():.4f} "
                    f"depth={losses['depth_loss'].item():.4f} "
                    f"dim={losses['dim_loss'].item():.4f} "
                    f"yaw={losses['yaw_loss'].item():.4f} "
                    f"offset={losses['offset_loss'].item():.4f}"
                )

        avg_epoch_loss = epoch_loss / max(num_batches, 1)

        print(
            f"Epoch {epoch:03d} complete. "
            f"Average loss: {avg_epoch_loss:.4f}"
        )

        if epoch % save_interval == 0:
            checkpoint_path = checkpoint_dir / f"mobile_adas3d_epoch_{epoch:03d}.pt"

            save_checkpoint(
                checkpoint_path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                loss_value=avg_epoch_loss,
            )

            print(f"Saved checkpoint: {checkpoint_path}")

    final_checkpoint_path = checkpoint_dir / "mobile_adas3d_final.pt"

    save_checkpoint(
        checkpoint_path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        global_step=global_step,
        loss_value=avg_epoch_loss,
    )

    print(f"Training complete. Final checkpoint: {final_checkpoint_path}")


if __name__ == "__main__":
    main()