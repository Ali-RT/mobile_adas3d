from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export MobileADAS3D TorchScript model to Core ML."
    )

    parser.add_argument("--torchscript-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--filename", type=str, default="MobileADAS3D.mlpackage")

    parser.add_argument("--input-height", type=int, default=384)
    parser.add_argument("--input-width", type=int, default=1280)

    parser.add_argument(
        "--precision",
        type=str,
        default="fp16",
        choices=["fp32", "fp16"],
    )

    parser.add_argument(
        "--target",
        type=str,
        default="ios15",
        choices=["ios15", "ios16", "ios17", "ios18"],
    )

    parser.add_argument(
        "--compute-units",
        type=str,
        default="all",
        choices=["all", "cpu_only", "cpu_and_gpu", "cpu_and_ne"],
    )

    return parser.parse_args()


def get_coreml_target(ct: Any, target: str) -> Any:
    if target == "ios15":
        return ct.target.iOS15
    if target == "ios16":
        return ct.target.iOS16
    if target == "ios17":
        return ct.target.iOS17

    # Some coremltools versions may not expose iOS18.
    if target == "ios18":
        if hasattr(ct.target, "iOS18"):
            return ct.target.iOS18
        print("ct.target.iOS18 not available; falling back to iOS17.")
        return ct.target.iOS17

    raise ValueError(f"Unsupported target: {target}")


def get_compute_units(ct: Any, compute_units: str) -> Any:
    if compute_units == "all":
        return ct.ComputeUnit.ALL
    if compute_units == "cpu_only":
        return ct.ComputeUnit.CPU_ONLY
    if compute_units == "cpu_and_gpu":
        return ct.ComputeUnit.CPU_AND_GPU
    if compute_units == "cpu_and_ne":
        return ct.ComputeUnit.CPU_AND_NE

    raise ValueError(f"Unsupported compute units: {compute_units}")


def main() -> None:
    args = parse_args()

    import coremltools as ct

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / args.filename
    metadata_path = output_dir / "MobileADAS3D_coreml_metadata.json"

    print("Loading TorchScript model...")
    ts_model = torch.jit.load(args.torchscript_path, map_location="cpu")
    ts_model.eval()

    example_input = torch.randn(
        1,
        3,
        args.input_height,
        args.input_width,
        dtype=torch.float32,
    )

    print("Running one TorchScript check...")
    with torch.no_grad():
        outputs = ts_model(example_input)

    print("TorchScript output shapes:")
    output_names = [
        "cls_logits",
        "box2d",
        "log_depth",
        "dim",
        "yaw",
        "center_offset",
        "depth_uncertainty",
    ]

    for name, tensor in zip(output_names, outputs):
        print(f"  {name}: {list(tensor.shape)}")

    minimum_deployment_target = get_coreml_target(ct, args.target)
    compute_units = get_compute_units(ct, args.compute_units)

    if args.precision == "fp16":
        compute_precision = ct.precision.FLOAT16
    else:
        compute_precision = ct.precision.FLOAT32

    print("\nConverting to Core ML...")
    print(f"TorchScript path: {args.torchscript_path}")
    print(f"Output path: {output_path}")
    print(f"Input shape: [1, 3, {args.input_height}, {args.input_width}]")
    print(f"Precision: {args.precision}")
    print(f"Target: {args.target}")
    print(f"Compute units: {args.compute_units}")

    mlmodel = ct.convert(
        ts_model,
        source="pytorch",
        inputs=[
            ct.TensorType(
                name="image",
                shape=example_input.shape,
            )
        ],
        outputs=[
            ct.TensorType(name="cls_logits"),
            ct.TensorType(name="box2d"),
            ct.TensorType(name="log_depth"),
            ct.TensorType(name="dim"),
            ct.TensorType(name="yaw"),
            ct.TensorType(name="center_offset"),
            ct.TensorType(name="depth_uncertainty"),
        ],
        convert_to="mlprogram",
        minimum_deployment_target=minimum_deployment_target,
        compute_precision=compute_precision,
        compute_units=compute_units,
    )

    mlmodel.short_description = "MobileADAS3D monocular 3D object detector"
    mlmodel.input_description["image"] = "RGB image tensor in NCHW format, shape [1, 3, 384, 1280]"

    for name in output_names:
        mlmodel.output_description[name] = f"MobileADAS3D output tensor: {name}"

    mlmodel.save(str(output_path))

    metadata: Dict[str, Any] = {
        "torchscript_path": args.torchscript_path,
        "output_path": str(output_path),
        "input_shape": [1, 3, args.input_height, args.input_width],
        "output_names": output_names,
        "precision": args.precision,
        "target": args.target,
        "compute_units": args.compute_units,
        "coremltools_version": ct.__version__,
        "package_size_mb": get_directory_size_mb(output_path),
    }

    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)

    print("\nCore ML export complete")
    print(f"Saved: {output_path}")
    print(f"Metadata: {metadata_path}")
    print(f"Package size: {metadata['package_size_mb']:.2f} MB")


def get_directory_size_mb(path: Path) -> float:
    if path.is_file():
        return path.stat().st_size / (1024.0 * 1024.0)

    total_bytes = 0

    for item in path.rglob("*"):
        if item.is_file():
            total_bytes += item.stat().st_size

    return total_bytes / (1024.0 * 1024.0)


if __name__ == "__main__":
    main()