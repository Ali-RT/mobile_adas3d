import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from functools import partial
from pathlib import Path
from typing import Dict, Optional, Any

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data.collate import mobile_adas3d_collate_fn
from data.kitti_dataset import KITTIDataset
from data.teacher_target_adapter import TeacherTargetAdapter
from data.class_taxonomy import normalize_class_mapping, validate_taxonomy_manifest
from data.split_resolver import get_split_file
from losses.mobile_adas3d_loss import MobileADAS3DLoss
from models.build import build_model
from tools.cli import parse_config_profile_args
from tools.config import load_runtime_config_from_args
from tools.device import get_device
from tools.metrics_logger import MetricsLogger
from tools.run_manager import create_run_dir, resume_run_dir
from tools.seed import seed_everything
from tools.training_control import EarlyStopping, get_current_lr


def move_targets_to_device(
    targets: Dict[str, torch.Tensor],
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    return {
        key: value.to(device, non_blocking=True)
        for key, value in targets.items()
    }


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[ReduceLROnPlateau],
    epoch: int,
    global_step: int,
    metric_value: float,
    best_metric: Optional[float],
    config: Dict[str, Any],
    scaler: torch.cuda.amp.GradScaler,
    early_stopper: Optional[EarlyStopping],
    is_best: bool = False,
) -> None:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metric_value": metric_value,
        "best_metric": best_metric,
        "config": config,
        "is_best": is_best,
        "scaler_state_dict": scaler.state_dict(),
    }

    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()

    if early_stopper is not None:
        payload["early_stopping_state"] = {
            "best": early_stopper.best,
            "num_bad_epochs": early_stopper.num_bad_epochs,
            "should_stop": early_stopper.should_stop,
            "stop_reason": early_stopper.stop_reason,
        }

    temporary_path = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(checkpoint_path)


def build_dataloader(
    config: Dict[str, Any],
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

    class_mapping = normalize_class_mapping(
        dataset_cfg.get("class_mapping"), dataset_cfg["classes"]
    )
    if class_mapping and dataset_cfg.get("require_taxonomy_manifest", False):
        manifest_paths = dataset_cfg.get("taxonomy_manifest_paths", {})
        manifest_path = manifest_paths.get(
            active_profile, dataset_cfg.get("taxonomy_manifest")
        )
        if not manifest_path:
            raise ValueError(
                f"No taxonomy manifest configured for profile {active_profile!r}"
            )
        validate_taxonomy_manifest(
            manifest_path=manifest_path,
            classes=dataset_cfg["classes"],
            mapping=class_mapping,
            split_files={
                "train": get_split_file(config, "train"),
                "val": get_split_file(config, "val"),
            },
            label_dir=Path(root_dir) / dataset_cfg["label_dir"],
        )

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=split_file,
        class_mapping=class_mapping,
    )

    loss_cfg = config.get("loss", {})
    center_sampling_cfg = target_cfg.get("center_sampling", {})

    center_sampling_radius = 0
    if center_sampling_cfg.get("enabled", False):
        center_sampling_radius = int(center_sampling_cfg.get("radius", 1))

    teacher_adapter = None
    distillation_cfg = config.get("distillation", {})
    if distillation_cfg.get("enabled", False) and split_name == "train":
        cache_dirs = distillation_cfg.get("profile_cache_dirs", {})
        cache_dir = cache_dirs.get(active_profile, distillation_cfg.get("cache_dir"))
        if not cache_dir:
            raise ValueError(
                f"No teacher cache configured for active profile {active_profile!r}"
            )
        teacher_adapter = TeacherTargetAdapter(
            cache_dir=cache_dir,
            split_file=split_file,
            score_threshold=float(distillation_cfg.get("score_threshold", 0.30)),
            match_iou_threshold=float(
                distillation_cfg.get("match_2d_iou_threshold", 0.50)
            ),
            max_gt_depth_m=float(distillation_cfg.get("max_gt_depth_m", 60.0)),
            expected_checkpoint_sha256=distillation_cfg.get(
                "checkpoint_sha256"
            ),
            expected_prediction_tree_sha256=distillation_cfg.get(
                "prediction_tree_sha256"
            ),
        )

    collate_fn = partial(
        mobile_adas3d_collate_fn,
        classes=dataset_cfg["classes"],
        input_height=model_cfg["input_height"],
        input_width=model_cfg["input_width"],
        output_stride=model_cfg["output_stride"],
        class_mean_dims=target_cfg["class_mean_dims"],
        center_sampling_radius=center_sampling_radius,
        class_weights=loss_cfg.get("class_weights", {}),
        quality_center_sigma=float(target_cfg.get("quality_center_sigma", 1.0)),
        teacher_adapter=teacher_adapter,
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


def build_criterion(config: Dict[str, Any]) -> MobileADAS3DLoss:
    loss_cfg = config.get("loss", {})
    distillation_cfg = config.get("distillation", {})
    teacher_loss_cfg = distillation_cfg.get("loss_weights", {})

    return MobileADAS3DLoss(
        input_height=config["model"]["input_height"],
        input_width=config["model"]["input_width"],
        classes=config["dataset"]["classes"],
        class_weights=loss_cfg.get("class_weights", {}),
        cls_weight=loss_cfg.get("cls_weight", 1.0),
        box2d_weight=loss_cfg.get("box2d_weight", 2.0),
        depth_weight=loss_cfg.get("depth_weight", 1.0),
        depth_uncertainty_weight=loss_cfg.get("depth_uncertainty_weight", 0.0),
        dim_weight=loss_cfg.get("dim_weight", 1.0),
        yaw_weight=loss_cfg.get("yaw_weight", 1.0),
        yaw_cosine_weight=loss_cfg.get("yaw_cosine_weight", 0.0),
        yaw_direction_weight=loss_cfg.get("yaw_direction_weight", 0.0),
        offset_weight=loss_cfg.get("offset_weight", 0.5),
        loc_xy_weight=loss_cfg.get("loc_xy_weight", 1.0),
        projected_center_weight=loss_cfg.get("projected_center_weight", 0.0),
        quality_weight=loss_cfg.get("quality_weight", 0.0),
        corner3d_weight=loss_cfg.get("corner3d_weight", 0.0),
        detach_yaw_in_corner3d=loss_cfg.get("detach_yaw_in_corner3d", False),
        yaw_pred_is_direct_sincos=loss_cfg.get(
            "yaw_pred_is_direct_sincos", False
        ),
        class_mean_dims=config["targets"]["class_mean_dims"],
        distillation_enabled=distillation_cfg.get("enabled", False),
        teacher_depth_weight=teacher_loss_cfg.get("depth", 0.0),
        teacher_dim_weight=teacher_loss_cfg.get("dimensions", 0.0),
        teacher_loc_xy_weight=teacher_loss_cfg.get("location_xy", 0.0),
        teacher_yaw_weight=teacher_loss_cfg.get("yaw", 0.0),
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
) -> Optional[ReduceLROnPlateau]:
    scheduler_cfg = config.get("scheduler", {})

    if not scheduler_cfg.get("enabled", False):
        return None

    if scheduler_cfg.get("name") != "ReduceLROnPlateau":
        raise ValueError(f"Unsupported scheduler: {scheduler_cfg.get('name')}")

    return ReduceLROnPlateau(
        optimizer,
        mode=scheduler_cfg.get("mode", "min"),
        factor=float(scheduler_cfg.get("factor", 0.5)),
        patience=int(scheduler_cfg.get("patience", 4)),
        threshold=float(scheduler_cfg.get("threshold", 0.0005)),
        threshold_mode=scheduler_cfg.get("threshold_mode", "rel"),
        cooldown=int(scheduler_cfg.get("cooldown", 1)),
        min_lr=float(scheduler_cfg.get("min_lr", 1e-6)),
    )


def build_early_stopper(config: Dict[str, Any]) -> Optional[EarlyStopping]:
    early_cfg = config.get("early_stopping", {})

    if not early_cfg.get("enabled", False):
        return None

    return EarlyStopping(
        mode=early_cfg.get("mode", "min"),
        patience=int(early_cfg.get("patience", 12)),
        min_delta=float(early_cfg.get("min_delta", 0.0005)),
        start_epoch=int(early_cfg.get("start_epoch", 10)),
    )


def average_losses(
    loss_sums: Dict[str, float],
    num_batches: int,
) -> Dict[str, float]:
    return {
        key: value / max(num_batches, 1)
        for key, value in loss_sums.items()
    }


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    criterion: MobileADAS3DLoss,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    epoch: int,
    epochs: int,
    global_step: int,
    log_interval: int,
    writer: Optional[SummaryWriter],
    use_amp: bool,
    gradient_clip_norm: Optional[float],
    gradient_accumulation_steps: int,
) -> tuple[Dict[str, float], int]:
    model.train()

    loss_sums: Dict[str, float] = {}

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(dataloader, start=1):
        images = batch["images"].to(device, non_blocking=True)
        targets = move_targets_to_device(batch["targets"], device)
        if criterion.distillation_enabled and "teacher_valid_mask" not in targets:
            raise RuntimeError(
                "Distillation is enabled but the training batch has no teacher targets"
            )

        with torch.cuda.amp.autocast(enabled=(use_amp and device.type == "cuda")):
            outputs = model(images)
            losses = criterion(outputs, targets)
            total_loss = losses["total_loss"]

        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite train loss detected: {total_loss.item()}")

        scaler.scale(total_loss / gradient_accumulation_steps).backward()

        should_step = (
            batch_idx % gradient_accumulation_steps == 0
            or batch_idx == len(dataloader)
        )

        if should_step:
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

        for name, value in losses.items():
            loss_sums[name] = loss_sums.get(name, 0.0) + float(value.item())

        if batch_idx % log_interval == 0:
            lr = get_current_lr(optimizer)

            print(
                f"epoch={epoch:03d}/{epochs:03d} "
                f"batch={batch_idx:04d}/{len(dataloader):04d} "
                f"step={global_step:06d} "
                f"lr={lr:.8f} "
                f"train_total={losses['total_loss'].item():.6f} "
                f"cls={losses['cls_loss'].item():.6f} "
                f"box2d={losses['box2d_loss'].item():.6f} "
                f"depth={losses['depth_loss'].item():.6f} "
                f"unc={losses.get('depth_uncertainty_loss', torch.tensor(0.0)).item():.6f} "
                f"dim={losses['dim_loss'].item():.6f} "
                f"yaw={losses['yaw_loss'].item():.6f} "
                f"yaw_cos={losses['yaw_cosine_loss'].item():.6f} "
                f"yaw_dir={losses['yaw_direction_loss'].item():.6f} "
                f"offset={losses['offset_loss'].item():.6f} "
                f"loc_xy={losses['loc_xy_loss'].item():.6f} "
                f"proj_center={losses.get('projected_center_loss', torch.tensor(0.0)).item():.6f} "
                f"quality={losses.get('quality_loss', torch.tensor(0.0)).item():.6f} "
                f"corner3d={losses.get('corner3d_loss', torch.tensor(0.0)).item():.6f}"
                f" teacher_depth={losses.get('teacher_depth_loss', torch.tensor(0.0)).item():.6f}"
                f" teacher_dim={losses.get('teacher_dim_loss', torch.tensor(0.0)).item():.6f}"
                f" teacher_loc={losses.get('teacher_loc_xy_loss', torch.tensor(0.0)).item():.6f}"
                f" teacher_yaw={losses.get('teacher_yaw_loss', torch.tensor(0.0)).item():.6f}"
            )

            if writer is not None:
                writer.add_scalar("train_step/learning_rate", lr, global_step)

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
    min_delta: float = 0.0,
) -> bool:
    if best is None:
        return True

    if mode == "min":
        return current < best - min_delta

    if mode == "max":
        return current > best + min_delta

    raise ValueError(f"Unsupported mode: {mode}")


def main() -> None:
    args = parse_config_profile_args("Train MobileADAS3D")
    config = load_runtime_config_from_args(args)

    training_cfg = config["training"]
    validation_cfg = config["validation"]
    logging_cfg = config["logging"]
    scheduler_cfg = config.get("scheduler", {})
    early_cfg = config.get("early_stopping", {})

    seed_everything(int(training_cfg.get("seed", 42)))

    device = get_device(training_cfg.get("device", "auto"))

    if args.resume is not None:
        run_dirs = resume_run_dir(args.resume)
    else:
        run_dirs = create_run_dir(config=config, config_path=args.config)
    metrics_logger = MetricsLogger(run_dirs["log_dir"])

    writer = None
    if logging_cfg.get("use_tensorboard", True):
        writer = SummaryWriter(log_dir=str(run_dirs["tensorboard_dir"]))

    active_profile = config["dataset"]["active_profile"]
    root_dir = config["dataset"]["profiles"][active_profile]["root_dir"]

    print("Starting MobileADAS3D training.")
    print(f"Using config: {args.config}")
    print(f"Requested profile: {args.profile}")
    print(f"Resolved active profile: {active_profile}")
    print(f"Dataset root: {root_dir}")
    print(f"Device: {device}")
    print(f"Run dir: {run_dirs['run_dir']}")

    train_loader = build_dataloader(
        config=config,
        split_name="train",
        batch_size=int(training_cfg["batch_size"]),
        num_workers=int(training_cfg["num_workers"]),
        shuffle=True,
        device=device,
    )

    val_loader = None
    if validation_cfg.get("enabled", True):
        val_loader = build_dataloader(
            config=config,
            split_name="val",
            batch_size=int(validation_cfg["batch_size"]),
            num_workers=int(validation_cfg["num_workers"]),
            shuffle=False,
            device=device,
        )

    model = build_model(config)
    model.to(device)

    criterion = build_criterion(config)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )

    scheduler = build_scheduler(optimizer, config)
    early_stopper = build_early_stopper(config)

    epochs = int(training_cfg["epochs"])
    log_interval = int(training_cfg.get("log_interval", 20))
    save_interval = int(training_cfg.get("save_interval", 1))
    use_amp = bool(training_cfg.get("use_amp", True))
    gradient_clip_norm = training_cfg.get("gradient_clip_norm", None)
    gradient_accumulation_steps = int(
        training_cfg.get("gradient_accumulation_steps", 1)
    )
    if gradient_accumulation_steps < 1:
        raise ValueError("training.gradient_accumulation_steps must be >= 1")

    if gradient_clip_norm is not None:
        gradient_clip_norm = float(gradient_clip_norm)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device.type == "cuda"))

    monitor_metric = validation_cfg.get("monitor_metric", "val_total_loss")
    monitor_mode = validation_cfg.get("mode", "min")
    monitor_min_delta = float(early_cfg.get("min_delta", 0.0))

    best_metric: Optional[float] = None
    global_step = 0
    start_epoch = 1

    if args.resume is not None:
        resume_path = Path(args.resume)
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        global_step = int(checkpoint.get("global_step", 0))
        start_epoch = int(checkpoint["epoch"]) + 1
        best_metric = checkpoint.get("best_metric", checkpoint.get("metric_value"))
        early_state = checkpoint.get("early_stopping_state", {})
        if early_stopper is not None and early_state:
            early_stopper.best = early_state.get("best")
            early_stopper.num_bad_epochs = int(early_state.get("num_bad_epochs", 0))
            early_stopper.should_stop = False
            early_stopper.stop_reason = ""
        print(
            f"Resumed {resume_path} at epoch {start_epoch}, "
            f"global_step={global_step}, best_metric={best_metric}"
        )

    print("\nTraining controls:")
    print(f"  Max epochs: {epochs}")
    print(f"  AMP enabled: {use_amp}")
    print(f"  Gradient clip norm: {gradient_clip_norm}")
    print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
    print(f"  Scheduler enabled: {scheduler is not None}")
    print(f"  Early stopping enabled: {early_stopper is not None}")
    print(f"  Monitor metric: {monitor_metric}")
    print(f"  Monitor mode: {monitor_mode}")

    if start_epoch > epochs:
        print(f"Checkpoint already reached epoch {start_epoch - 1}; nothing to do.")
        if writer is not None:
            writer.close()
        return

    for epoch in range(start_epoch, epochs + 1):
        lr_before_epoch = get_current_lr(optimizer)

        train_losses, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            epochs=epochs,
            global_step=global_step,
            log_interval=log_interval,
            writer=writer,
            use_amp=use_amp,
            gradient_clip_norm=gradient_clip_norm,
            gradient_accumulation_steps=gradient_accumulation_steps,
        )

        epoch_metrics: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": lr_before_epoch,
        }

        for name, value in train_losses.items():
            epoch_metrics[f"train_{name}"] = value

        print(
            f"\nEpoch {epoch:03d} train complete. "
            f"train_total_loss={train_losses['total_loss']:.6f}"
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
                f"val_total_loss={val_losses['total_loss']:.6f}"
            )

        current_metric = epoch_metrics.get(monitor_metric)

        if current_metric is None:
            raise KeyError(
                f"Monitor metric '{monitor_metric}' was not found in epoch metrics. "
                f"Available keys: {list(epoch_metrics.keys())}"
            )

        current_metric = float(current_metric)

        is_best = is_better_metric(
            current=current_metric,
            best=best_metric,
            mode=monitor_mode,
            min_delta=monitor_min_delta,
        )

        if is_best:
            best_metric = current_metric

        # Step LR scheduler after validation.
        if scheduler is not None:
            scheduler_metric_name = scheduler_cfg.get("monitor_metric", monitor_metric)
            scheduler_metric = epoch_metrics.get(scheduler_metric_name)

            if scheduler_metric is None:
                raise KeyError(
                    f"Scheduler monitor metric '{scheduler_metric_name}' not found. "
                    f"Available keys: {list(epoch_metrics.keys())}"
                )

            old_lr = get_current_lr(optimizer)
            scheduler.step(float(scheduler_metric))
            new_lr = get_current_lr(optimizer)

            epoch_metrics["learning_rate_after_scheduler"] = new_lr

            if new_lr != old_lr:
                print(
                    f"LR reduced by scheduler: {old_lr:.8f} -> {new_lr:.8f} "
                    f"based on {scheduler_metric_name}={float(scheduler_metric):.6f}"
                )
        else:
            epoch_metrics["learning_rate_after_scheduler"] = get_current_lr(optimizer)

        # Early stopping state update.
        should_stop = False
        if early_stopper is not None:
            early_metric_name = early_cfg.get("monitor_metric", monitor_metric)
            early_metric = epoch_metrics.get(early_metric_name)

            if early_metric is None:
                raise KeyError(
                    f"Early stopping monitor metric '{early_metric_name}' not found. "
                    f"Available keys: {list(epoch_metrics.keys())}"
                )

            should_stop = early_stopper.step(
                current=float(early_metric),
                epoch=epoch,
            )

            epoch_metrics["early_stopping_bad_epochs"] = early_stopper.num_bad_epochs
            epoch_metrics["early_stopping_best"] = early_stopper.best
        else:
            epoch_metrics["early_stopping_bad_epochs"] = 0
            epoch_metrics["early_stopping_best"] = best_metric

        # TensorBoard epoch logging.
        if writer is not None:
            for name, value in epoch_metrics.items():
                if name in {"epoch", "global_step"}:
                    continue

                if isinstance(value, (int, float)):
                    writer.add_scalar(f"epoch/{name}", value, epoch)

        # CSV/JSONL logging.
        metrics_logger.log(epoch_metrics)

        # Save latest checkpoint every epoch.
        latest_path = run_dirs["checkpoint_dir"] / "latest.pt"
        save_checkpoint(
            checkpoint_path=latest_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            metric_value=current_metric,
            best_metric=best_metric,
            config=config,
            scaler=scaler,
            early_stopper=early_stopper,
            is_best=False,
        )

        # Save epoch checkpoint at interval.
        if epoch % save_interval == 0:
            epoch_path = run_dirs["checkpoint_dir"] / f"epoch_{epoch:03d}.pt"
            save_checkpoint(
                checkpoint_path=epoch_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                metric_value=current_metric,
                best_metric=best_metric,
                config=config,
                scaler=scaler,
                early_stopper=early_stopper,
                is_best=False,
            )

        # Save best checkpoint.
        if is_best:
            best_path = run_dirs["checkpoint_dir"] / "best.pt"
            save_checkpoint(
                checkpoint_path=best_path,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                metric_value=current_metric,
                best_metric=best_metric,
                config=config,
                scaler=scaler,
                early_stopper=early_stopper,
                is_best=True,
            )

            print(f"New best checkpoint saved: {best_path}")
            print(f"Best {monitor_metric}: {best_metric:.6f}")

        print(
            f"Epoch {epoch:03d} summary: "
            f"{monitor_metric}={current_metric:.6f}, "
            f"best={best_metric:.6f}, "
            f"lr={get_current_lr(optimizer):.8f}, "
            f"bad_epochs={epoch_metrics['early_stopping_bad_epochs']}"
        )

        if should_stop:
            print(early_stopper.stop_reason)
            break

    if writer is not None:
        writer.close()

    print("\nTraining complete.")
    print(f"Run directory: {run_dirs['run_dir']}")
    print(f"Best {monitor_metric}: {best_metric}")


if __name__ == "__main__":
    main()
