from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from models.build import build_model
from models.torchscript_wrapper import TORCHSCRIPT_OUTPUT_NAMES, MobileADAS3DTupleWrapper
from tools.config import load_config, apply_runtime_overrides
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MobileADAS3D to TorchScript."
    )

    parser.add_argument("--config", type=str, default="configs/kitti_mobileadas3d.yaml")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--output-path", type=str, default=None)
    parser.add_argument("--filename", type=str, default="mobileadas3d_torchscript.pt")
    parser.add_argument("--optimize-for-inference", action="store_true")

    return parser.parse_args()


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint missing model_state_dict: {checkpoint_path}")

    model.load_state_dict(checkpoint["model_state_dict"])

    return checkpoint


def tensor_summary(tensor: torch.Tensor) -> Dict[str, Any]:
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "device": str(tensor.device),
        "numel": int(tensor.numel()),
    }


def resolve_export_paths(
    config: Dict[str, Any],
    output_dir_arg: Optional[str],
    output_path_arg: Optional[str],
    filename: str,
) -> tuple[Path, Path, Path]:
    if output_path_arg is not None:
        export_path = Path(output_path_arg)
        output_dir = export_path.parent
    elif output_dir_arg is not None:
        requested_path = Path(output_dir_arg)

        if requested_path.suffix in {".pt", ".pth", ".torchscript"}:
            print(
                "Warning: --output-dir received a file-like path. "
                "Prefer --output-path for exact export files."
            )
            output_dir = requested_path.parent
            export_path = requested_path
        else:
            output_dir = requested_path
            export_path = output_dir / filename
    else:
        output_dir = Path(config["outputs"]["output_dir"]) / "exports" / "torchscript"
        export_path = output_dir / filename

    if export_path.exists() and export_path.is_dir():
        raise IsADirectoryError(
            "Requested TorchScript export path is a directory, but it must be "
            f"a final model file: {export_path}. Pass --output-dir with a "
            "directory plus --filename, or pass --output-path with a file path. "
            "If this directory was created by an earlier export, remove or "
            "rename it before exporting to that .pt file path."
        )

    metadata_path = output_dir / "mobileadas3d_torchscript_metadata.json"

    return output_dir, export_path, metadata_path


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

    output_dir, export_path, metadata_path = resolve_export_paths(
        config=config,
        output_dir_arg=args.output_dir,
        output_path_arg=args.output_path,
        filename=args.filename,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    model = build_model(config)
    checkpoint = load_checkpoint(
        model=model,
        checkpoint_path=args.checkpoint,
        device=device,
    )

    model.to(device)
    model.eval()

    wrapper = MobileADAS3DTupleWrapper(model)
    wrapper.to(device)
    wrapper.eval()

    example_input = torch.randn(
        1,
        3,
        input_height,
        input_width,
        device=device,
        dtype=torch.float32,
    )

    print("Tracing TorchScript model...")
    print(f"Device: {device}")
    print(f"Input shape: {list(example_input.shape)}")

    with torch.no_grad():
        traced = torch.jit.trace(
            wrapper,
            example_input,
            strict=False,
        )

        traced.eval()

        if args.optimize_for_inference:
            print("Applying torch.jit.optimize_for_inference...")
            traced = torch.jit.optimize_for_inference(traced)

        traced_outputs = traced(example_input)

    torch.jit.save(traced, str(export_path))

    output_summaries = {
        name: tensor_summary(tensor)
        for name, tensor in zip(TORCHSCRIPT_OUTPUT_NAMES, traced_outputs)
    }

    metadata = {
        "config": args.config,
        "profile": args.profile,
        "checkpoint": args.checkpoint,
        "export_path": str(export_path),
        "device_used_for_export": str(device),
        "input_height": input_height,
        "input_width": input_width,
        "input_shape": [1, 3, input_height, input_width],
        "output_names": TORCHSCRIPT_OUTPUT_NAMES,
        "output_summaries": output_summaries,
        "optimize_for_inference": bool(args.optimize_for_inference),
        "checkpoint_epoch": checkpoint.get("epoch", None),
        "checkpoint_global_step": checkpoint.get("global_step", None),
        "checkpoint_metric_value": checkpoint.get(
            "metric_value",
            checkpoint.get("loss", None),
        ),
    }

    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    size_mb = export_path.stat().st_size / (1024.0 * 1024.0)

    print("\nTorchScript export complete")
    print(f"Export path: {export_path}")
    print(f"File size: {size_mb:.2f} MB")
    print(f"Metadata: {metadata_path}")

    print("\nOutputs:")
    for name, summary in output_summaries.items():
        print(f"  {name}: {summary['shape']}")


if __name__ == "__main__":
    main()
