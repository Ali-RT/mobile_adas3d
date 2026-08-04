from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

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
from tools.seed import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a bounded paired control/distillation fine-tuning gate."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default="colab_drive")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split-dir", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-steps", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=2)
    return parser.parse_args()


def mean_losses(sums: Dict[str, float], batches: int) -> Dict[str, float]:
    return {key: value / max(batches, 1) for key, value in sums.items()}


def evaluate_branch(
    model: torch.nn.Module,
    config: Dict[str, Any],
    device: torch.device,
    eval_batches: int,
) -> Dict[str, float]:
    eval_config = copy.deepcopy(config)
    eval_config["distillation"]["enabled"] = False
    loader = build_dataloader(
        eval_config,
        split_name="val",
        batch_size=int(eval_config["validation"]["batch_size"]),
        num_workers=0,
        shuffle=False,
        device=device,
    )
    criterion = build_criterion(eval_config).to(device)
    sums: Dict[str, float] = {}
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader, start=1):
            images = batch["images"].to(device)
            targets = move_targets_to_device(batch["targets"], device)
            losses = criterion(model(images), targets)
            for key, value in losses.items():
                sums[key] = sums.get(key, 0.0) + float(value.detach())
            if batch_index >= eval_batches:
                return mean_losses(sums, batch_index)
    return mean_losses(sums, len(loader))


def run_branch(
    *,
    name: str,
    config: Dict[str, Any],
    state_dict: Dict[str, torch.Tensor],
    source_checkpoint: Path,
    output_dir: Path,
    device: torch.device,
    train_steps: int,
    eval_batches: int,
    batch_size: int,
) -> Dict[str, Any]:
    seed = int(config["training"].get("seed", 42))
    seed_everything(seed)
    branch_config = copy.deepcopy(config)
    branch_config["distillation"]["enabled"] = name == "distillation"
    branch_config["model"]["pretrained"] = False
    loader = build_dataloader(
        branch_config,
        split_name="train",
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
        device=device,
    )
    model = build_model(branch_config).to(device)
    model.load_state_dict(state_dict)
    criterion = build_criterion(branch_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(branch_config["training"]["learning_rate"]),
        weight_decay=float(branch_config["training"]["weight_decay"]),
    )
    model.train()
    sums: Dict[str, float] = {}
    completed_steps = 0
    teacher_cells = 0
    for batch in loader:
        images = batch["images"].to(device)
        targets = move_targets_to_device(batch["targets"], device)
        if name == "distillation":
            teacher_cells += int(targets["teacher_valid_mask"].sum().item())
        optimizer.zero_grad(set_to_none=True)
        losses = criterion(model(images), targets)
        if not torch.isfinite(losses["total_loss"]):
            raise RuntimeError(f"Non-finite {name} loss at step {completed_steps + 1}")
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        completed_steps += 1
        for key, value in losses.items():
            sums[key] = sums.get(key, 0.0) + float(value.detach())
        if completed_steps % 20 == 0 or completed_steps == train_steps:
            print(
                f"branch={name} step={completed_steps}/{train_steps} "
                f"loss={float(losses['total_loss'].detach()):.6f} "
                f"teacher_cells={teacher_cells}",
                flush=True,
            )
        if completed_steps >= train_steps:
            break
    if completed_steps != train_steps:
        raise RuntimeError(f"Only completed {completed_steps}/{train_steps} {name} steps")

    eval_losses = evaluate_branch(model, branch_config, device, eval_batches)
    checkpoint_path = output_dir / f"{name}_after_{train_steps}_steps.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": branch_config,
            "source_checkpoint": str(source_checkpoint),
            "branch": name,
            "gate_train_steps": train_steps,
        },
        checkpoint_path,
    )
    return {
        "name": name,
        "train_steps": completed_steps,
        "teacher_positive_cells": teacher_cells,
        "mean_train_losses": mean_losses(sums, completed_steps),
        "supervised_eval_losses": eval_losses,
        "checkpoint": str(checkpoint_path),
    }


def main() -> None:
    args = parse_args()
    if args.train_steps < 1 or args.eval_batches < 1:
        raise ValueError("train-steps and eval-batches must be positive")
    config = apply_runtime_overrides(
        load_config(args.config),
        profile=args.profile,
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        output_dir=str(args.output_dir),
    )
    if config.get("distillation", {}).get("enabled") is not True:
        raise RuntimeError("Task 6 requires an explicitly enabled distillation config")
    device = get_device(config["training"].get("device", "auto"))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name in ("control", "distillation"):
        results[name] = run_branch(
            name=name,
            config=config,
            state_dict=state_dict,
            source_checkpoint=args.checkpoint,
            output_dir=args.output_dir,
            device=device,
            train_steps=args.train_steps,
            eval_batches=args.eval_batches,
            batch_size=args.batch_size,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()

    control_eval = results["control"]["supervised_eval_losses"]["total_loss"]
    distill_eval = results["distillation"]["supervised_eval_losses"]["total_loss"]
    report = {
        "schema_version": 1,
        "complete": True,
        "device": str(device),
        "seed": int(config["training"].get("seed", 42)),
        "source_checkpoint": str(args.checkpoint),
        "train_steps_per_branch": args.train_steps,
        "eval_batches_per_branch": args.eval_batches,
        "results": results,
        "comparison": {
            "control_supervised_eval_total_loss": control_eval,
            "distillation_supervised_eval_total_loss": distill_eval,
            "distillation_minus_control": distill_eval - control_eval,
            "relative_change": (distill_eval - control_eval) / max(control_eval, 1e-12),
        },
        "optimizer_steps_total": args.train_steps * 2,
        "full_training_started": False,
    }
    report_path = args.output_dir / "distillation_experiment_gate.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Distillation experiment gate: {report_path}")


if __name__ == "__main__":
    main()
