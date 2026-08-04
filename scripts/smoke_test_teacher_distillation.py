from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.build import build_model
from scripts.train_mobile_adas3d import (
    build_criterion,
    build_dataloader,
    move_targets_to_device,
)
from tools.config import apply_runtime_overrides, load_config
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one real-data teacher-distillation forward/backward pass."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default="colab_drive")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-batches", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_runtime_overrides(
        load_config(args.config),
        profile=args.profile,
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
    )
    distillation = config.setdefault("distillation", {})
    distillation["enabled"] = True
    distillation["cache_dir"] = args.cache_dir
    distillation.setdefault("profile_cache_dirs", {})[args.profile] = args.cache_dir
    config["model"]["pretrained"] = False

    device = get_device(config["training"].get("device", "auto"))
    loader = build_dataloader(
        config,
        split_name="train",
        batch_size=1,
        num_workers=0,
        shuffle=False,
        device=device,
    )
    batch = None
    teacher_cells = 0
    for batch_index, candidate in enumerate(loader, start=1):
        teacher_cells = int(candidate["targets"]["teacher_valid_mask"].sum().item())
        if teacher_cells > 0:
            batch = candidate
            break
        if batch_index >= args.max_batches:
            break
    if batch is None or teacher_cells == 0:
        raise RuntimeError(
            f"No approved teacher cells found in the first {args.max_batches} batches"
        )

    model = build_model(config).to(device).train()
    criterion = build_criterion(config).to(device)
    images = batch["images"].to(device)
    targets = move_targets_to_device(batch["targets"], device)
    outputs = model(images)
    losses = criterion(outputs, targets)
    total_loss = losses["total_loss"]
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"Non-finite smoke loss: {float(total_loss.detach())}")
    total_loss.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    if not finite_gradients:
        raise RuntimeError("Non-finite gradients in distillation smoke test")

    report = {
        "schema_version": 1,
        "complete": True,
        "sample_id": batch["metadata"][0]["sample_id"],
        "device": str(device),
        "approved_teacher_cells": teacher_cells,
        "total_loss": float(total_loss.detach()),
        "teacher_depth_loss": float(losses["teacher_depth_loss"].detach()),
        "teacher_dim_loss": float(losses["teacher_dim_loss"].detach()),
        "teacher_loc_xy_loss": float(losses["teacher_loc_xy_loss"].detach()),
        "teacher_yaw_loss": float(losses["teacher_yaw_loss"].detach()),
        "finite_gradients": finite_gradients,
        "optimizer_steps": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Distillation smoke report: {args.output}")


if __name__ == "__main__":
    main()
