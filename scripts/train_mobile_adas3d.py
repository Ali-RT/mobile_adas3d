from functools import partial
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.collate import mobile_adas3d_collate_fn
from data.kitti_dataset import KITTIDataset
from data.split_resolver import get_split_file
from losses.mobile_adas3d_loss import MobileADAS3DLoss
from models.build import build_model
from tools.cli import parse_config_profile_args
from tools.device import get_device
from tools.metrics_logger import MetricsLogger
from tools.run_manager import create_run_dir
from tools.seed import seed_everything
from tools.config import load_runtime_config_from_args, apply_runtime_overrides


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
    config: Dict,
    is_best: bool = False,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss_value,
            "config": config,
            "is_best": is_best,
        },
        checkpoint_path,
    )


def build_dataloader(
    config: Dict,
    split_name: str,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    target_cfg = config["targets"]

    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]

    split_file = None

    if "splits" in dataset_cfg:
        split_file = get_split_file(config, split_name)
        print(f"Using {split_name} split file: {split_file}")

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=split_file,
    )

    collate_fn = partial(
        mobile_adas3d_collate_fn,
        classes=dataset_cfg["classes"],
        input_height=model_cfg["input_height"],
        input_width=model_cfg["input_width"],
        output_stride=model_cfg["output_stride"],
        class_mean_dims=target_cfg["class_mean_dims"],
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    print(f"{split_name} samples: {len(dataset)}")
    print(f"{split_name} batches: {len(loader)}")

    return loader


def average_losses(loss_sums: Dict[str, float], num_batches: int) -> Dict[str, float]:
    return {
        key: value / max(num_batches, 1)
        for key, value in loss_sums.items()
    }


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: MobileADAS3DLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    global_step: int,
    log_interval: int,
    writer: Optional[SummaryWriter],
    use_amp: bool,
    gradient_clip_norm: Optional[float],
) -> tuple[Dict[str, float], int]:
    model.train()

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    loss_sums: Dict[str, float] = {}

    for batch_idx, batch in enumerate(dataloader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        targets = move_targets_to_device(batch["targets"], device)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)
            losses = criterion(outputs, targets)
            total_loss = losses["total_loss"]

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite train loss detected: {total_loss.item()}")

        scaler.scale(total_loss).backward()

        if gradient_clip_norm is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        scaler.step(optimizer)
        scaler.update()

        global_step += 1

        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.item())

        if batch_idx % log_interval == 0:
            msg = (
                f"epoch={epoch:03d}/{epochs:03d} "
                f"batch={batch_idx:04d}/{len(dataloader):04d} "
                f"step={global_step:06d} "
                f"train_total={losses['total_loss'].item():.4f} "
                f"cls={losses['cls_loss'].item():.4f} "
                f"box2d={losses['box2d_loss'].item():.4f} "
                f"depth={losses['depth_loss'].item():.4f} "
                f"dim={losses['dim_loss'].item():.4f} "
                f"yaw={losses['yaw_loss'].item():.4f} "
                f"offset={losses['offset_loss'].item():.4f}"
            )
            print(msg)

            if writer is not None:
                for name, value in losses.items():
                    writer.add_scalar(f"train_step/{name}", value.item(), global_step)

    return average_losses(loss_sums, len(dataloader)), global_step


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: MobileADAS3DLoss,
    device: torch.device,
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()

    loss_sums: Dict[str, float] = {}

    for batch in dataloader:
        images = batch["images"].to(device, non_blocking=True)
        targets = move_targets_to_device(batch["targets"], device)

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)
            losses = criterion(outputs, targets)

        total_loss = losses["total_loss"]

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite val loss detected: {total_loss.item()}")

        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.item())

    return average_losses(loss_sums, len(dataloader))


def is_better_metric(
    current: float,
    best: Optional[float],
    mode: str,
) -> bool:
    if best is None:
        return True

    if mode == "min":
        return current < best

    if mode == "max":
        return current > best

    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_config_profile_args("Train MobileADAS3D")
    config = load_runtime_config_from_args(args)

    training_cfg = config["training"]
    validation_cfg = config["validation"]
    logging_cfg = config["logging"]

    seed_everything(int(training_cfg.get("seed", 42)))

    device = get_device(training_cfg.get("device", "auto"))

    run_dirs = create_run_dir(config=config, config_path=args.config)
    metrics_logger = MetricsLogger(run_dirs["log_dir"])

    writer = None
    if logging_cfg.get("use_tensorboard", True):
        writer = SummaryWriter(log_dir=str(run_dirs["tensorboard_dir"]))

    active_profile = config["dataset"]["active_profile"]
    root_dir = config["dataset"]["profiles"][active_profile]["root_dir"]

    print("Starting MobileADAS3D training.")
    print(f"Using config: {args.config}")
    print(f"Active profile: {active_profile}")
    print(f"Dataset root: {root_dir}")
    print(f"Device: {device}")
    print(f"Run dir: {run_dirs['run_dir']}")

    train_loader = build_dataloader(
        config=config,
        split_name="train",
        batch_size=training_cfg["batch_size"],
        num_workers=training_cfg["num_workers"],
        shuffle=True,
        device=device,
    )

    val_loader = None
    if validation_cfg.get("enabled", True):
        val_loader = build_dataloader(
            config=config,
            split_name="val",
            batch_size=validation_cfg["batch_size"],
            num_workers=validation_cfg["num_workers"],
            shuffle=False,
            device=device,
        )

    model = build_model(config)
    model.to(device)

    criterion = MobileADAS3DLoss(
        input_height=config["model"]["input_height"],
        input_width=config["model"]["input_width"],
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_cfg["learning_rate"],
        weight_decay=training_cfg["weight_decay"],
    )

    epochs = int(training_cfg["epochs"])
    log_interval = int(training_cfg.get("log_interval", 20))
    save_interval = int(training_cfg.get("save_interval", 1))
    use_amp = bool(training_cfg.get("use_amp", True))
    gradient_clip_norm = training_cfg.get("gradient_clip_norm", None)

    monitor_metric = validation_cfg.get("monitor_metric", "val_total_loss")
    monitor_mode = validation_cfg.get("mode", "min")

    best_metric = None
    global_step = 0

    for epoch in range(1, epochs + 1):
        train_losses, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=epochs,
            global_step=global_step,
            log_interval=log_interval,
            writer=writer,
            use_amp=use_amp,
            gradient_clip_norm=gradient_clip_norm,
        )

        epoch_metrics = {
            "epoch": epoch,
            "global_step": global_step,
        }

        for name, value in train_losses.items():
            epoch_metrics[f"train_{name}"] = value

        print(
            f"Epoch {epoch:03d} train complete. "
            f"train_total_loss={train_losses['total_loss']:.4f}"
        )

        if val_loader is not None and epoch % int(validation_cfg["interval_epochs"]) == 0:
            val_losses = validate_one_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                use_amp=use_amp,
            )

            for name, value in val_losses.items():
                epoch_metrics[f"val_{name}"] = value

            print(
                f"Epoch {epoch:03d} validation complete. "
                f"val_total_loss={val_losses['total_loss']:.4f}"
            )

        if writer is not None:
            for name, value in epoch_metrics.items():
                if name not in {"epoch", "global_step"}:
                    writer.add_scalar(f"epoch/{name}", value, epoch)

        metrics_logger.log(epoch_metrics)

        latest_path = run_dirs["checkpoint_dir"] / "latest.pt"

        current_metric = epoch_metrics.get(monitor_metric)

        is_best = False
        if current_metric is not None:
            is_best = is_better_metric(
                current=float(current_metric),
                best=best_metric,
                mode=monitor_mode,
            )

            if is_best:
                best_metric = float(current_metric)

        save_checkpoint(
            checkpoint_path=latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            loss_value=float(epoch_metrics.get(monitor_metric, train_losses["total_loss"])),
            config=config,
            is_best=False,
        )

        if epoch % save_interval == 0:
            epoch_path = run_dirs["checkpoint_dir"] / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                checkpoint_path=epoch_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                loss_value=float(epoch_metrics.get(monitor_metric, train_losses["total_loss"])),
                config=config,
                is_best=False,
            )

        if is_best:
            best_path = run_dirs["checkpoint_dir"] / "best.pt"
            save_checkpoint(
                checkpoint_path=best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                loss_value=float(current_metric),
                config=config,
                is_best=True,
            )
            print(f"New best checkpoint saved: {best_path}")
            print(f"Best {monitor_metric}: {best_metric:.6f}")

    if writer is not None:
        writer.close()

    print("\nTraining complete.")
    print(f"Run directory: {run_dirs['run_dir']}")
    print(f"Best {monitor_metric}: {best_metric}")


if __name__ == "__main__":
    main()