from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

from models.build import build_model
from models.torchscript_wrapper import TORCHSCRIPT_OUTPUT_NAMES, MobileADAS3DTupleWrapper
from tools.config import load_config, apply_runtime_overrides
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate PyTorch eager vs TorchScript output parity."
    )

    parser.add_argument("--config", type=str, default="configs/kitti_mobileadas3d.yaml")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--torchscript-path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default=None)

    return parser.parse_args()


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])


def compare_tensor(
    name: str,
    eager_tensor: torch.Tensor,
    scripted_tensor: torch.Tensor,
) -> Dict[str, Any]:
    eager = eager_tensor.detach().float().cpu()
    scripted = scripted_tensor.detach().float().cpu()

    diff = (eager - scripted).abs()

    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    rms = float(torch.sqrt(torch.mean(diff ** 2)).item())

    denom = eager.abs().clamp(min=1e-6)
    rel = diff / denom

    max_rel = float(rel.max().item())
    mean_rel = float(rel.mean().item())

    allclose_1e_4 = bool(torch.allclose(eager, scripted, atol=1e-4, rtol=1e-4))
    allclose_1e_5 = bool(torch.allclose(eager, scripted, atol=1e-5, rtol=1e-5))

    return {
        "output_name": name,
        "shape": list(eager.shape),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "rms_diff": rms,
        "max_rel_diff": max_rel,
        "mean_rel_diff": mean_rel,
        "allclose_atol_rtol_1e_4": allclose_1e_4,
        "allclose_atol_rtol_1e_5": allclose_1e_5,
    }


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {path}")


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved JSON: {path}")


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    config = apply_runtime_overrides(
        config=config,
        profile=args.profile,
        run_name=None,
    )

    model_cfg = config["model"]

    input_height = int(model_cfg["input_height"])
    input_width = int(model_cfg["input_width"])

    device = get_device(args.device)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["outputs"]["visualization_dir"]) / "benchmarks"

    output_dir.mkdir(parents=True, exist_ok=True)

    eager_model = build_model(config)
    load_checkpoint(
        model=eager_model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    eager_model.to(device)
    eager_model.eval()

    eager_wrapper = MobileADAS3DTupleWrapper(eager_model)
    eager_wrapper.to(device)
    eager_wrapper.eval()

    scripted_model = torch.jit.load(args.torchscript_path, map_location=device)
    scripted_model.to(device)
    scripted_model.eval()

    rows: List[Dict[str, Any]] = []

    print("\nRunning parity validation...")
    print(f"Device: {device}")
    print(f"Num runs: {args.num_runs}")
    print(f"TorchScript path: {args.torchscript_path}")

    with torch.no_grad():
        for run_idx in range(args.num_runs):
            x = torch.randn(
                1,
                3,
                input_height,
                input_width,
                device=device,
                dtype=torch.float32,
            )

            eager_outputs = eager_wrapper(x)
            scripted_outputs = scripted_model(x)

            for name, eager_tensor, scripted_tensor in zip(
                TORCHSCRIPT_OUTPUT_NAMES,
                eager_outputs,
                scripted_outputs,
            ):
                row = compare_tensor(
                    name=name,
                    eager_tensor=eager_tensor,
                    scripted_tensor=scripted_tensor,
                )
                row["run_idx"] = run_idx
                rows.append(row)

    summary_rows: List[Dict[str, Any]] = []

    for name in TORCHSCRIPT_OUTPUT_NAMES:
        name_rows = [r for r in rows if r["output_name"] == name]

        summary_rows.append(
            {
                "output_name": name,
                "max_abs_diff_over_runs": max(r["max_abs_diff"] for r in name_rows),
                "mean_abs_diff_over_runs": sum(r["mean_abs_diff"] for r in name_rows) / len(name_rows),
                "max_rel_diff_over_runs": max(r["max_rel_diff"] for r in name_rows),
                "mean_rel_diff_over_runs": sum(r["mean_rel_diff"] for r in name_rows) / len(name_rows),
                "all_runs_allclose_1e_4": all(r["allclose_atol_rtol_1e_4"] for r in name_rows),
                "all_runs_allclose_1e_5": all(r["allclose_atol_rtol_1e_5"] for r in name_rows),
            }
        )

    report = {
        "config": args.config,
        "profile": args.profile,
        "checkpoint": args.checkpoint,
        "torchscript_path": args.torchscript_path,
        "device": str(device),
        "num_runs": args.num_runs,
        "input_height": input_height,
        "input_width": input_width,
        "summary": summary_rows,
    }

    detail_csv = output_dir / "torchscript_parity_detail.csv"
    summary_csv = output_dir / "torchscript_parity_summary.csv"
    summary_json = output_dir / "torchscript_parity_summary.json"

    save_csv(rows, detail_csv)
    save_csv(summary_rows, summary_csv)
    save_json(report, summary_json)

    print("\nTorchScript Parity Summary")
    for row in summary_rows:
        print(
            f"{row['output_name']}: "
            f"max_abs={row['max_abs_diff_over_runs']:.8f}, "
            f"mean_abs={row['mean_abs_diff_over_runs']:.8f}, "
            f"allclose_1e-4={row['all_runs_allclose_1e_4']}, "
            f"allclose_1e-5={row['all_runs_allclose_1e_5']}"
        )

    print("\nFiles:")
    print(detail_csv)
    print(summary_csv)
    print(summary_json)


if __name__ == "__main__":
    main()