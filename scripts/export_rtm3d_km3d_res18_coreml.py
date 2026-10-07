"""Export the pinned RTM3D/KM3D ResNet-18 neural heads to a Core ML package.

This gate exports raw neural-network head tensors only. The upstream
calibration-dependent 3D decoder is intentionally not embedded in the package.
Run this exporter in Colab (conversion only), then run
``audit_rtm3d_km3d_res18_coreml.py`` on macOS for native prediction parity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_rtm3d_km3d_res18_smoke import (
    CLASS_NAMES,
    HEADS,
    RTM3D_COMMIT,
    RTM3D_URL,
    build_strict_resnet18,
    find_sample_paths,
    preprocess,
    read_split,
    safe_load_state,
    verify_source,
)


OUTPUT_NAMES = tuple(HEADS)
INPUT_NAME = "image"
TRACE_TOLERANCE = 1e-6
EXPECTED_CHECKPOINT_SHA256 = "5fa355845f79c1afeffab427de32933758e5b4c1e7c9ec19a94a13737691d05b"
EXPECTED_VAL_SPLIT_SHA256 = "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8"


class RTM3DHeadsWrapper(nn.Module):
    """Expose the final raw head tensors as a stable, ordered tuple."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image):
        outputs = self.model(image)[-1]
        return tuple(outputs[name] for name in OUTPUT_NAMES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def package_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def operation_counts(program) -> dict[str, int]:
    counts: dict[str, int] = {}

    def visit(block):
        for operation in block.operations:
            counts[operation.op_type] = counts.get(operation.op_type, 0) + 1
            for child in operation.blocks:
                visit(child)

    visit(program.functions["main"])
    return dict(sorted(counts.items()))


def max_abs_delta(actual, expected) -> float:
    return max(
        float((got.detach().cpu().float() - want.detach().cpu().float()).abs().max())
        for got, want in zip(actual, expected)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import coremltools as ct
    import cv2

    repo = args.repo.resolve()
    checkpoint = args.checkpoint.resolve()
    dataset = args.dataset_root.resolve()
    split_path = args.split_dir.resolve() / "val.txt"
    output_dir = args.output_dir.resolve()
    if not repo.is_dir() or not checkpoint.is_file() or not split_path.is_file():
        raise FileNotFoundError("Pinned RTM3D source, checkpoint, or Chen val.txt is missing")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise RuntimeError(f"Output directory is not empty; preserve it and choose a new run directory: {output_dir}")

    source_commit = verify_source(repo)
    checkpoint_sha256 = sha256_file(checkpoint)
    split_sha256 = sha256_file(split_path)
    if checkpoint_sha256 != EXPECTED_CHECKPOINT_SHA256:
        raise RuntimeError(
            "Checkpoint differs from the passed r7 smoke artifact: "
            f"expected {EXPECTED_CHECKPOINT_SHA256}, got {checkpoint_sha256}"
        )
    if split_sha256 != EXPECTED_VAL_SPLIT_SHA256:
        raise RuntimeError(
            "Chen validation split differs from the passed r7 smoke artifact: "
            f"expected {EXPECTED_VAL_SPLIT_SHA256}, got {split_sha256}"
        )
    sample_id = read_split(split_path)[0]
    image_path, calibration_path = find_sample_paths(dataset, sample_id)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"OpenCV could not read representative image {image_path}")

    sys.path.insert(0, str(repo / "src" / "lib"))
    from utils.image import get_affine_transform

    input_tensor, _, _, _ = preprocess(
        image_bgr, cv2, np, torch, get_affine_transform, torch.device("cpu")
    )
    if list(input_tensor.shape) != [1, 3, 384, 1280]:
        raise RuntimeError(f"Unexpected preprocessed input shape: {list(input_tensor.shape)}")
    if not bool(torch.isfinite(input_tensor).all()):
        raise RuntimeError("Preprocessed representative input contains non-finite values")

    state, checkpoint_epoch = safe_load_state(checkpoint, torch)
    model, head_conv = build_strict_resnet18(repo, state, torch)
    model = model.cpu().eval()
    wrapper = RTM3DHeadsWrapper(model).eval()
    with torch.inference_mode():
        reference = wrapper(input_tensor)
        traced = torch.jit.trace(wrapper, input_tensor, strict=True, check_trace=False)
        traced_outputs = traced(input_tensor)
    trace_delta = max_abs_delta(traced_outputs, reference)
    if trace_delta > TRACE_TOLERANCE:
        raise RuntimeError(f"TorchScript parity failed: max abs delta {trace_delta} > {TRACE_TOLERANCE}")

    expected_shapes = {
        name: [1, channels, 96, 320] for name, channels in HEADS.items()
    }
    output_shapes = {
        name: list(value.shape) for name, value in zip(OUTPUT_NAMES, reference)
    }
    if output_shapes != expected_shapes:
        raise RuntimeError(f"Unexpected RTM3D head output shapes: {output_shapes}")
    if not all(bool(torch.isfinite(value).all()) for value in reference):
        raise RuntimeError("One or more representative raw head outputs are non-finite")

    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / "RTM3D_KM3D_ResNet18.mlpackage"
    trace_path = output_dir / "RTM3D_KM3D_ResNet18.pt"
    fixture_path = output_dir / "parity_reference.npz"
    traced.save(str(trace_path))
    np.savez_compressed(
        fixture_path,
        image=input_tensor.detach().cpu().numpy().astype(np.float32),
        **{
            name: value.detach().cpu().numpy().astype(np.float32)
            for name, value in zip(OUTPUT_NAMES, reference)
        },
    )

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
        inputs=[ct.TensorType(name=INPUT_NAME, shape=input_tensor.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name=name) for name in OUTPUT_NAMES],
    )
    mlmodel.save(str(package_path))
    counts = operation_counts(mlmodel._mil_program)
    if "custom" in counts:
        raise RuntimeError(f"Converted Core ML graph contains a custom op: {counts}")

    report = {
        "schema_version": 1,
        "experiment": "RTM3D/KM3D ResNet-18 trained-checkpoint Core ML head export",
        "export_revision": "2026-10-07-r1",
        "complete": True,
        "scope": "fixed-shape FP32 neural heads; calibration-based decode is excluded",
        "source_url": RTM3D_URL,
        "source_commit": source_commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_epoch": checkpoint_epoch,
        "safe_weights_only_load": True,
        "strict_state_dict_load": True,
        "head_conv": head_conv,
        "classes": list(CLASS_NAMES),
        "input": {
            "name": INPUT_NAME,
            "shape": list(input_tensor.shape),
            "dtype": "float32",
            "preprocessing": "pinned smoke preprocess: OpenCV BGR channel order, affine warp to 1280x384, then (pixel/255 - ImageNet mean)/ImageNet std",
            "fixture_sample_id": sample_id,
            "fixture_image_sha256": sha256_file(image_path),
            "fixture_calibration_sha256": sha256_file(calibration_path),
        },
        "outputs": {name: {"shape": shape, "dtype": "float32", "meaning": "raw pre-decode network head"}
                    for name, shape in output_shapes.items()},
        "trace_max_abs_delta": trace_delta,
        "trace_parity_tolerance": TRACE_TOLERANCE,
        "trace_parity_passed": trace_delta <= TRACE_TOLERANCE,
        "coremltools_version": ct.__version__,
        "torch_version": torch.__version__,
        "coreml_format": "ML Program",
        "minimum_deployment_target": "iOS 17",
        "compute_precision": "FP32",
        "coreml_model_load_or_prediction_performed": False,
        "mil_operation_counts": counts,
        "mil_has_custom_op": "custom" in counts,
        "mlpackage": package_path.name,
        "mlpackage_sha256": sha256_tree(package_path),
        "mlpackage_size_bytes": package_size(package_path),
        "torchscript": trace_path.name,
        "torchscript_sha256": sha256_file(trace_path),
        "parity_fixture": fixture_path.name,
        "parity_fixture_sha256": sha256_file(fixture_path),
        "parity_fixture_contains_one_real_chen_val_image": True,
        "training_performed": False,
        "full_val_ap_evaluated": False,
        "geometry_decode_included": False,
        "iphone_performance_measured": False,
        "deployment_qualified": False,
        "next_step": "run Mac Core ML prediction against parity_reference.npz; only after raw-head parity passes, port and validate calibration-dependent decode before iPhone integration",
    }
    report_path = output_dir / "rtm3d_coreml_export.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    bundle_path = output_dir / "rtm3d_km3d_res18_coreml_export.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, report_path.name)
        archive.write(fixture_path, fixture_path.name)
        archive.write(trace_path, trace_path.name)
        for item in sorted(package_path.rglob("*")):
            if item.is_file():
                archive.write(item, (Path(package_path.name) / item.relative_to(package_path)).as_posix())

    print(json.dumps(report, indent=2), flush=True)
    print(f"\nMac parity bundle: {bundle_path}", flush=True)


if __name__ == "__main__":
    main()
