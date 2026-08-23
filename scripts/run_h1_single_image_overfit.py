from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.build import build_model
from scripts.train_mobile_adas3d import (
    build_criterion,
    build_dataloader,
    move_targets_to_device,
)
from tools.config import apply_runtime_overrides, load_config
from tools.device import get_device
from tools.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exact-step H1 single-image overfit gate.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=100)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--report", required=True)
    return parser.parse_args()


def _save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    step: int,
    config: Dict[str, Any],
    latest_losses: Dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": step,
            "global_step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "config": config,
            "single_image_overfit": True,
            "latest_losses": latest_losses,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    if args.steps < 1 or args.save_interval < 1 or args.log_interval < 1:
        raise ValueError("steps, save-interval, and log-interval must be positive")
    config = apply_runtime_overrides(
        load_config(args.config),
        profile=args.profile,
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
    )
    if config["model"]["name"] != "MobileADAS3D-H1":
        raise RuntimeError("Single-image overfit requires MobileADAS3D-H1")
    if config.get("distillation", {}).get("enabled", False):
        raise RuntimeError("Single-image overfit requires distillation=false")
    if config.get("loss", {}).get("classification_mode") != "implicit_background_softmax":
        raise RuntimeError("Single-image overfit requires the H1-v2 objective")

    seed_everything(int(config["training"].get("seed", 42)))
    device = get_device(config["training"].get("device", "auto"))
    loader = build_dataloader(
        config=config,
        split_name="train",
        batch_size=1,
        num_workers=0,
        shuffle=False,
        device=device,
    )
    if len(loader.dataset) != 1 or len(loader) != 1:
        raise RuntimeError(f"Expected exactly one training sample, got {len(loader.dataset)}")
    batch = next(iter(loader))
    object_classes = {obj["class_name"] for obj in batch["metadata"][0]["objects"]}
    required_classes = set(config["dataset"]["classes"])
    if not required_classes.issubset(object_classes):
        raise RuntimeError(
            f"Single sample must contain {sorted(required_classes)}; got {sorted(object_classes)}"
        )

    images = batch["images"].to(device)
    targets = move_targets_to_device(batch["targets"], device)
    model = build_model(config).to(device).train()
    criterion = build_criterion(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    amp_enabled = bool(config["training"].get("use_amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    checkpoint_path = Path(args.checkpoint)
    start_step = 0
    latest_losses: Dict[str, float] = {}
    if checkpoint_path.is_file():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not checkpoint.get("single_image_overfit", False):
            raise RuntimeError(f"Refusing non-single-image checkpoint: {checkpoint_path}")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(checkpoint.get("scaler_state_dict", {}))
        start_step = int(checkpoint.get("global_step", checkpoint.get("epoch", 0)))
        latest_losses = {
            name: float(value)
            for name, value in checkpoint.get("latest_losses", {}).items()
        }
        if start_step > args.steps:
            raise RuntimeError(
                f"Checkpoint step {start_step} exceeds requested steps {args.steps}"
            )
        print(f"Resuming {checkpoint_path} at step {start_step}")

    gradient_clip = float(config["training"].get("gradient_clip_norm", 1.0))
    for step in range(start_step + 1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            outputs = model(images)
            losses = criterion(outputs, targets)
            total_loss = losses["total_loss"]
        if not torch.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {float(total_loss)}")
        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        scaler.step(optimizer)
        scaler.update()
        latest_losses = {name: float(value.detach()) for name, value in losses.items()}
        if step % args.log_interval == 0 or step == 1 or step == args.steps:
            print(
                f"step={step:04d}/{args.steps:04d} "
                f"total={latest_losses['total_loss']:.6f} "
                f"cls={latest_losses['cls_loss']:.6f} "
                f"box2d={latest_losses['box2d_loss']:.6f} "
                f"depth={latest_losses['depth_loss']:.6f} "
                f"quality={latest_losses['quality_loss']:.6f}",
                flush=True,
            )
        if step % args.save_interval == 0 or step == args.steps:
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                scaler,
                step,
                config,
                latest_losses,
            )

    report = {
        "schema_version": 1,
        "complete": True,
        "sample_id": batch["metadata"][0]["sample_id"],
        "object_classes": sorted(object_classes),
        "objects": int(targets["object_mask"].sum()),
        "device": str(device),
        "amp_enabled": amp_enabled,
        "steps": args.steps,
        "checkpoint": str(checkpoint_path),
        "latest_losses": latest_losses,
        "distillation_enabled": False,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
