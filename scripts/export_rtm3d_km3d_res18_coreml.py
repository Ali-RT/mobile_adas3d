"""Export the pinned RTM3D/KM3D ResNet-18 neural heads to a Core ML package.

This gate exports raw neural-network head tensors only. The upstream
calibration-dependent 3D decoder is intentionally not embedded in the package.
Run this exporter in Colab (conversion only), then run
``audit_rtm3d_km3d_res18_coreml.py`` on macOS for native prediction parity.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
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
    find_sample_paths,
    safe_load_state,
    verify_source,
)


OUTPUT_NAMES = tuple(HEADS)
INPUT_NAME = "image"
TRACE_TOLERANCE = 1e-6
EXPECTED_CHECKPOINT_SHA256 = "5fa355845f79c1afeffab427de32933758e5b4c1e7c9ec19a94a13737691d05b"
EXPECTED_VAL_SPLIT_SHA256 = "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8"
INPUT_HEIGHT = 384
INPUT_WIDTH = 1280
FIXTURE_SAMPLE_COUNT = 16


def load_pinned_module(module_name: str, source_path: Path):
    """Load an upstream module by file path, bypassing ambiguous ``models`` imports.

    RTM3D's ``src/lib/models`` is a namespace package, while MobileADAS3D has a
    regular top-level ``models`` package. Python selects the regular package
    even when RTM3D's source directory is earlier on ``sys.path``. Loading the
    exact pinned files under private names avoids this collision.
    """
    if not source_path.is_file():
        raise FileNotFoundError(f"Pinned RTM3D source file is missing: {source_path}")
    spec = importlib.util.spec_from_file_location(module_name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load pinned RTM3D source file: {source_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_pinned_resnet18(repo: Path, state: dict, torch):
    """Build the checkpoint-matched ResNet-18 from the pinned RTM3D file."""
    upstream = load_pinned_module(
        "rtm3d_pinned_msra_resnet_888c379",
        repo / "src/lib/models/networks/msra_resnet.py",
    )
    block, layers = upstream.resnet_spec[18]
    failures = []
    for head_conv in (64, 256, 128, 0):
        model = upstream.PoseResNet(block, layers, HEADS, head_conv=head_conv)
        try:
            model.load_state_dict(state, strict=True)
        except RuntimeError as exc:
            failures.append(f"head_conv={head_conv}: {exc}")
            del model
            continue
        return model, head_conv
    raise RuntimeError(
        "No strict pinned RTM3D ResNet-18/head configuration matched the checkpoint:\n"
        + "\n".join(failures)
    )


def preprocess_pinned_image(image, cv2, np, torch, repo: Path):
    """Match the RTM3D smoke preprocessing using its pinned affine routine."""
    image_module = load_pinned_module(
        "rtm3d_pinned_image_888c379", repo / "src/lib/utils/image.py"
    )
    height, width = image.shape[:2]
    center = np.array([width / 2.0, height / 2.0], dtype=np.float32)
    scale = float(max(height, width))
    transform = image_module.get_affine_transform(
        center, scale, 0, [INPUT_WIDTH, INPUT_HEIGHT]
    )
    resized = cv2.resize(image, (width, height))
    warped = cv2.warpAffine(
        resized, transform, (INPUT_WIDTH, INPUT_HEIGHT), flags=cv2.INTER_LINEAR
    )
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    normalized = ((warped / 255.0 - mean) / std).astype(np.float32)
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)


def select_fixture_ids(split_path: Path, sample_count: int = FIXTURE_SAMPLE_COUNT) -> list[str]:
    """Choose deterministic, evenly spaced IDs from the complete Chen val split."""
    ids = [line.strip() for line in split_path.read_text().splitlines() if line.strip()]
    if len(ids) != 3769 or len(set(ids)) != 3769:
        raise RuntimeError(f"Expected 3769 unique Chen val IDs, got {len(ids)}")
    if sample_count < 1 or sample_count > len(ids):
        raise ValueError(f"Invalid parity fixture sample count: {sample_count}")
    positions = np.linspace(0, len(ids) - 1, num=sample_count, dtype=np.int64)
    selected = [ids[int(index)] for index in positions]
    if len(set(selected)) != sample_count:
        raise RuntimeError("Parity fixture selection did not produce unique Chen val IDs")
    return selected


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
    fixture_sample_ids = select_fixture_ids(split_path)
    state, checkpoint_epoch = safe_load_state(checkpoint, torch)
    model, head_conv = build_pinned_resnet18(repo, state, torch)
    model = model.cpu().eval()
    wrapper = RTM3DHeadsWrapper(model).eval()

    # Do not import ``models.*`` or ``utils.*`` here: MobileADAS3D's regular
    # packages shadow RTM3D's namespace directories in this Colab process.
    first_image_path, _ = find_sample_paths(dataset, fixture_sample_ids[0])
    first_image_bgr = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
    if first_image_bgr is None:
        raise RuntimeError(f"OpenCV could not read representative image {first_image_path}")
    first_input = preprocess_pinned_image(first_image_bgr, cv2, np, torch, repo)
    if list(first_input.shape) != [1, 3, INPUT_HEIGHT, INPUT_WIDTH]:
        raise RuntimeError(f"Unexpected preprocessed input shape: {list(first_input.shape)}")
    if not bool(torch.isfinite(first_input).all()):
        raise RuntimeError("Preprocessed representative input contains non-finite values")

    with torch.inference_mode():
        reference = wrapper(first_input)
        traced = torch.jit.trace(wrapper, first_input, strict=True, check_trace=False)
        traced_outputs = traced(first_input)
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

    image_values = np.empty(
        (len(fixture_sample_ids), 3, INPUT_HEIGHT, INPUT_WIDTH), dtype=np.float32
    )
    head_values = {
        name: np.empty((len(fixture_sample_ids), channels, 96, 320), dtype=np.float32)
        for name, channels in HEADS.items()
    }
    fixture_records = []
    with torch.inference_mode():
        for image_index, sample_id in enumerate(fixture_sample_ids):
            image_path, calibration_path = find_sample_paths(dataset, sample_id)
            image_bgr = first_image_bgr if image_index == 0 else cv2.imread(
                str(image_path), cv2.IMREAD_COLOR
            )
            if image_bgr is None:
                raise RuntimeError(f"OpenCV could not read parity image {image_path}")
            input_tensor = first_input if image_index == 0 else preprocess_pinned_image(
                image_bgr, cv2, np, torch, repo
            )
            if list(input_tensor.shape) != [1, 3, INPUT_HEIGHT, INPUT_WIDTH]:
                raise RuntimeError(
                    f"Unexpected preprocessed input shape for {sample_id}: {list(input_tensor.shape)}"
                )
            outputs = reference if image_index == 0 else wrapper(input_tensor)
            if not all(bool(torch.isfinite(value).all()) for value in outputs):
                raise RuntimeError(f"Non-finite raw head output for Chen val image {sample_id}")
            for name, value in zip(OUTPUT_NAMES, outputs):
                if list(value.shape) != expected_shapes[name]:
                    raise RuntimeError(
                        f"Unexpected {name} shape for {sample_id}: {list(value.shape)}"
                    )
            image_values[image_index] = input_tensor[0].detach().cpu().numpy()
            for name, value in zip(OUTPUT_NAMES, outputs):
                head_values[name][image_index] = value[0].detach().cpu().numpy()
            fixture_records.append({
                "sample_id": sample_id,
                "image_sha256": sha256_file(image_path),
                "calibration_sha256": sha256_file(calibration_path),
            })
            if image_index == 0:
                reference = None
            del outputs, input_tensor, image_bgr
            print(
                f"Prepared parity fixture {image_index + 1}/{len(fixture_sample_ids)}: {sample_id}",
                flush=True,
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    package_path = output_dir / "RTM3D_KM3D_ResNet18.mlpackage"
    trace_path = output_dir / "RTM3D_KM3D_ResNet18.pt"
    fixture_path = output_dir / "parity_reference.npz"
    traced.save(str(trace_path))
    np.savez_compressed(
        fixture_path,
        image=image_values,
        **head_values,
    )

    # Import only after RTM3D's top-level ``models`` package is loaded. Some
    # Core ML/TensorFlow dependencies expose their own ``models`` namespace,
    # which otherwise shadows RTM3D's ``src/lib/models`` during construction.
    import coremltools as ct

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=True,
        inputs=[ct.TensorType(name=INPUT_NAME, shape=first_input.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name=name) for name in OUTPUT_NAMES],
    )
    mlmodel.save(str(package_path))
    counts = operation_counts(mlmodel._mil_program)
    if "custom" in counts:
        raise RuntimeError(f"Converted Core ML graph contains a custom op: {counts}")

    report = {
        "schema_version": 1,
        "experiment": "RTM3D/KM3D ResNet-18 trained-checkpoint Core ML head export",
        "export_revision": "2026-10-07-r3",
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
            "shape": list(first_input.shape),
            "dtype": "float32",
            "preprocessing": "pinned smoke preprocess: OpenCV BGR channel order, affine warp to 1280x384, then (pixel/255 - ImageNet mean)/ImageNet std",
            "fixture_sample_id": fixture_sample_ids[0],
            "fixture_sample_ids": fixture_sample_ids,
            "fixture_samples": fixture_records,
            "fixture_image_sha256": fixture_records[0]["image_sha256"],
            "fixture_calibration_sha256": fixture_records[0]["calibration_sha256"],
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
        "parity_fixture_real_chen_val_image_count": len(fixture_sample_ids),
        "parity_fixture_selection": "16 evenly spaced unique IDs from the frozen Chen val split",
        "training_performed": False,
        "full_val_ap_evaluated": False,
        "geometry_decode_included": False,
        "iphone_performance_measured": False,
        "deployment_qualified": False,
        "next_step": "run multi-image Mac Core ML prediction against parity_reference.npz; only after raw-head parity passes, port and validate calibration-dependent decode before iPhone integration",
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
