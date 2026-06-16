from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from models.build import build_model
from tools.config import load_config, apply_runtime_overrides
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark static MobileADAS3D model complexity."
    )

    parser.add_argument(
        "--config",
        type=str,
        default="configs/kitti_mobileadas3d.yaml",
    )

    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Runtime profile, e.g. local_mac or colab_drive.",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint path, usually best.pt.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device: cpu, cuda, mps, or auto. If omitted, uses config training.device.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
    )

    return parser.parse_args()


def count_parameters(model: nn.Module) -> Dict[str, int]:
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    for param in model.parameters():
        n = param.numel()
        total_params += n

        if param.requires_grad:
            trainable_params += n
        else:
            non_trainable_params += n

    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
    }


def estimate_state_dict_size_mb(model: nn.Module) -> float:
    total_bytes = 0

    for tensor in model.state_dict().values():
        total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes / (1024.0 * 1024.0)


def get_checkpoint_size_mb(checkpoint_path: Optional[str]) -> Optional[float]:
    if checkpoint_path is None:
        return None

    path = Path(checkpoint_path)

    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    return path.stat().st_size / (1024.0 * 1024.0)


def module_parameter_breakdown(model: nn.Module) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
        size_mb = sum(
            p.numel() * p.element_size()
            for p in module.parameters()
        ) / (1024.0 * 1024.0)

        rows.append(
            {
                "module": name,
                "params": params,
                "params_m": params / 1e6,
                "trainable_params": trainable,
                "size_mb_fp32": size_mb,
            }
        )

    rows = sorted(rows, key=lambda r: r["params"], reverse=True)

    return rows


def tensor_size_mb(tensor: torch.Tensor) -> float:
    return tensor.numel() * tensor.element_size() / (1024.0 * 1024.0)


def run_dummy_forward(
    model: nn.Module,
    batch_size: int,
    input_height: int,
    input_width: int,
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()

    dummy = torch.randn(
        batch_size,
        3,
        input_height,
        input_width,
        device=device,
    )

    with torch.no_grad():
        outputs = model(dummy)

    output_rows = []
    total_output_mb = 0.0

    for name, tensor in outputs.items():
        size_mb = tensor_size_mb(tensor)
        total_output_mb += size_mb

        output_rows.append(
            {
                "output_name": name,
                "shape": list(tensor.shape),
                "numel": int(tensor.numel()),
                "size_mb": size_mb,
                "dtype": str(tensor.dtype).replace("torch.", ""),
            }
        )

    return {
        "dummy_input_shape": list(dummy.shape),
        "dummy_input_size_mb": tensor_size_mb(dummy),
        "outputs": output_rows,
        "total_output_size_mb": total_output_mb,
    }


class TupleOutputWrapper(nn.Module):
    """
    Some FLOP counters prefer tensor/tuple outputs instead of dict outputs.
    This wrapper keeps the same computation but converts dict output to tuple.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        outputs = self.model(x)
        return tuple(outputs.values())


def try_compute_macs_with_ptflops(
    config: Dict[str, Any],
    checkpoint_path: Optional[str],
    input_height: int,
    input_width: int,
) -> Dict[str, Any]:
    """
    Computes MACs using ptflops if available.

    Uses CPU because FLOP counting is a static-ish analysis and is more portable on CPU.
    """
    try:
        from ptflops import get_model_complexity_info
    except ImportError:
        return {
            "ptflops_available": False,
            "macs": None,
            "flops": None,
            "macs_g": None,
            "flops_g": None,
            "ptflops_error": "ptflops is not installed. Run: pip install ptflops",
        }

    try:
        model = build_model(config)
        model.eval()

        if checkpoint_path is not None:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model.load_state_dict(checkpoint["model_state_dict"])

        wrapper = TupleOutputWrapper(model)
        wrapper.eval()

        macs, params = get_model_complexity_info(
            wrapper,
            (3, input_height, input_width),
            as_strings=False,
            print_per_layer_stat=False,
            verbose=False,
        )

        # ptflops returns MACs. FLOPs are often reported as 2 * MACs.
        flops = 2 * macs

        return {
            "ptflops_available": True,
            "macs": int(macs),
            "flops": int(flops),
            "macs_g": float(macs) / 1e9,
            "flops_g": float(flops) / 1e9,
            "ptflops_error": None,
        }

    except Exception as exc:
        return {
            "ptflops_available": True,
            "macs": None,
            "flops": None,
            "macs_g": None,
            "flops_g": None,
            "ptflops_error": repr(exc),
        }


def save_json(data: Dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved JSON: {output_path}")


def save_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(rows) == 0:
        print(f"No rows to save for {output_path}")
        return

    fieldnames = list(rows[0].keys())

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved CSV: {output_path}")


def load_model_from_checkpoint_if_needed(
    model: nn.Module,
    checkpoint_path: Optional[str],
    device: torch.device,
) -> None:
    if checkpoint_path is None:
        return

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded checkpoint.")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Global step: {checkpoint.get('global_step', 'unknown')}")
    print(f"Metric value: {checkpoint.get('metric_value', checkpoint.get('loss', 'unknown'))}")


def main() -> None:
    args = parse_args()

    config = load_config(args.config)
    config = apply_runtime_overrides(
        config=config,
        profile=args.profile,
        run_name=None,
    )

    model_cfg = config["model"]
    training_cfg = config["training"]

    input_height = int(model_cfg["input_height"])
    input_width = int(model_cfg["input_width"])
    output_stride = int(model_cfg["output_stride"])

    if args.device is not None:
        device_str = args.device
    else:
        device_str = training_cfg.get("device", "auto")

    device = get_device(device_str)

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(config["outputs"]["visualization_dir"]) / "benchmarks"

    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_model(config)
    load_model_from_checkpoint_if_needed(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model.to(device)
    model.eval()

    param_counts = count_parameters(model)
    state_dict_size_mb = estimate_state_dict_size_mb(model)
    checkpoint_size_mb = get_checkpoint_size_mb(args.checkpoint)

    output_info = run_dummy_forward(
        model=model,
        batch_size=args.batch_size,
        input_height=input_height,
        input_width=input_width,
        device=device,
    )

    module_rows = module_parameter_breakdown(model)

    macs_info = try_compute_macs_with_ptflops(
        config=config,
        checkpoint_path=args.checkpoint,
        input_height=input_height,
        input_width=input_width,
    )

    report = {
        "config": args.config,
        "profile": args.profile,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "batch_size": args.batch_size,
        "input_height": input_height,
        "input_width": input_width,
        "output_stride": output_stride,
        "feature_height": input_height // output_stride,
        "feature_width": input_width // output_stride,
        "model_name": model_cfg.get("name", "unknown"),
        "backbone": model_cfg.get("backbone", "unknown"),
        "neck": model_cfg.get("neck", "unknown"),
        "box_encoding": model_cfg.get("box_encoding", "unknown"),
        "params": param_counts,
        "total_params_m": param_counts["total_params"] / 1e6,
        "trainable_params_m": param_counts["trainable_params"] / 1e6,
        "state_dict_size_mb_fp32": state_dict_size_mb,
        "checkpoint_size_mb": checkpoint_size_mb,
        "dummy_forward": output_info,
        "macs": macs_info,
    }

    json_path = output_dir / "benchmark_model_complexity.json"
    module_csv_path = output_dir / "benchmark_module_params.csv"
    output_csv_path = output_dir / "benchmark_output_tensors.csv"

    save_json(report, json_path)
    save_csv(module_rows, module_csv_path)
    save_csv(output_info["outputs"], output_csv_path)

    print("\nModel Complexity Summary")
    print(f"Model: {report['model_name']}")
    print(f"Backbone: {report['backbone']}")
    print(f"Neck: {report['neck']}")
    print(f"Box encoding: {report['box_encoding']}")
    print(f"Input: {input_width} x {input_height}")
    print(f"Output stride: {output_stride}")
    print(f"Feature map: {report['feature_width']} x {report['feature_height']}")
    print(f"Device used for dummy forward: {device}")
    print(f"Total params: {param_counts['total_params']:,} ({report['total_params_m']:.3f}M)")
    print(
        f"Trainable params: "
        f"{param_counts['trainable_params']:,} ({report['trainable_params_m']:.3f}M)"
    )
    print(f"State dict size fp32: {state_dict_size_mb:.2f} MB")

    if checkpoint_size_mb is not None:
        print(f"Checkpoint file size: {checkpoint_size_mb:.2f} MB")

    print(f"Dummy input shape: {output_info['dummy_input_shape']}")
    print(f"Dummy input size: {output_info['dummy_input_size_mb']:.2f} MB")
    print(f"Total output tensor size: {output_info['total_output_size_mb']:.2f} MB")

    print("\nOutput shapes:")
    for row in output_info["outputs"]:
        print(
            f"  {row['output_name']}: "
            f"shape={row['shape']} "
            f"size={row['size_mb']:.3f} MB"
        )

    print("\nTop-level module parameter breakdown:")
    for row in module_rows:
        print(
            f"  {row['module']}: "
            f"{row['params']:,} params "
            f"({row['params_m']:.3f}M), "
            f"{row['size_mb_fp32']:.2f} MB"
        )

    print("\nMACs/FLOPs:")
    if macs_info["macs_g"] is not None:
        print(f"  MACs:  {macs_info['macs_g']:.3f} GMACs")
        print(f"  FLOPs: {macs_info['flops_g']:.3f} GFLOPs")
    else:
        print(f"  Not available: {macs_info['ptflops_error']}")

    print("\nBenchmark files:")
    print(f"  {json_path}")
    print(f"  {module_csv_path}")
    print(f"  {output_csv_path}")


if __name__ == "__main__":
    main()