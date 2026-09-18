"""Run the M59c backbone-focused audit on the frozen M59b Core ML package.

M59b uses one absolute tolerance for every tap.  The deepest backbone feature
map can have values several orders of magnitude larger than later tensors, so
an absolute-only failure can be scale-proportional numerical drift rather than
the first material graph mismatch.  M59c keeps the strict result, adds a
scale-aware audit for the backbone and input projections, and reports missing
diagnostic aliases separately.  It does not authorize deployment or precision
compression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np


INPUT_NAMES = ("image", "calibration", "image_size")
BACKBONE_NAMES = (
    "backbone_feat_0",
    "backbone_feat_1",
    "backbone_feat_2",
    "backbone_pos_0",
    "backbone_pos_1",
    "backbone_pos_2",
)
PROJECTION_NAMES = ("input_proj_0", "input_proj_1", "input_proj_2", "input_proj_3")
STRICT_LIMIT = 1.0e-3
RELATIVE_LIMIT = 1.0e-5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.read_bytes())
    return digest.hexdigest()


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def compare_backbone_tensors(
    reference: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    names: tuple[str, ...] | list[str],
    strict_limit: float = STRICT_LIMIT,
    relative_limit: float = RELATIVE_LIMIT,
) -> dict[str, dict[str, Any]]:
    """Compare tensors and retain both absolute and scale-aware diagnostics."""

    report: dict[str, dict[str, Any]] = {}
    for name in names:
        expected = reference.get(name)
        actual = prediction.get(name)
        if expected is None or actual is None:
            report[name] = {
                "strict_passed": False,
                "scale_aware_passed": False,
                "error": "missing tensor",
                "strict_limit": strict_limit,
                "relative_limit": relative_limit,
            }
            continue
        expected = np.asarray(expected, dtype=np.float32)
        actual = np.asarray(actual, dtype=np.float32)
        if expected.shape != actual.shape:
            report[name] = {
                "strict_passed": False,
                "scale_aware_passed": False,
                "error": "shape mismatch",
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
                "strict_limit": strict_limit,
                "relative_limit": relative_limit,
            }
            continue
        delta = np.abs(actual - expected)
        max_abs = float(delta.max()) if delta.size else 0.0
        mean_abs = float(delta.mean()) if delta.size else 0.0
        reference_scale = max(float(np.max(np.abs(expected))), 1.0) if expected.size else 1.0
        normalized_max_abs = max_abs / reference_scale
        finite = bool(np.isfinite(actual).all())
        report[name] = {
            "shape": list(actual.shape),
            "reference_max_abs": reference_scale,
            "max_abs_delta": max_abs,
            "mean_abs_delta": mean_abs,
            "normalized_max_abs_delta": normalized_max_abs,
            "strict_passed": bool(finite and max_abs <= strict_limit),
            "scale_aware_passed": bool(
                finite and max_abs <= max(strict_limit, relative_limit * reference_scale)
            ),
            "strict_limit": strict_limit,
            "relative_limit": relative_limit,
        }
    return report


def first_failed(comparison: dict[str, dict[str, Any]], field: str) -> str | None:
    return next((name for name, row in comparison.items() if row.get(field) is not True), None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the M59c backbone-focused Core ML audit.")
    parser.add_argument("--mlpackage", type=Path, required=True)
    parser.add_argument("--reference-io", type=Path, required=True)
    parser.add_argument("--export-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compute-units", choices=("CPU_ONLY", "CPU_AND_GPU", "ALL"), default="ALL"
    )
    args = parser.parse_args()

    if platform.system() != "Darwin":
        raise RuntimeError("M59c Core ML audit requires macOS")
    if not args.mlpackage.is_dir():
        raise FileNotFoundError(args.mlpackage)
    if not args.reference_io.is_file():
        raise FileNotFoundError(args.reference_io)
    if not args.export_gate.is_file():
        raise FileNotFoundError(args.export_gate)

    import coremltools as ct

    gate = json.loads(args.export_gate.read_text(encoding="utf-8"))
    if gate.get("complete") is not True or gate.get("all_export_gates_passed") is not True:
        raise RuntimeError("M59b export gate is incomplete or failed")
    expected_io = gate.get("artifacts", {}).get("reference_io_sha256")
    if expected_io and sha256_file(args.reference_io) != expected_io:
        raise RuntimeError("Reference I/O hash differs from M59b export gate")
    expected_package = gate.get("artifacts", {}).get("mlpackage_tree_sha256")
    if expected_package and tree_sha256(args.mlpackage) != expected_package:
        raise RuntimeError("ML package tree hash differs from M59b export gate")

    output_names = list(gate.get("output_names", []))
    with np.load(args.reference_io, allow_pickle=False) as archive:
        reference_inputs = {name: np.asarray(archive[name]) for name in INPUT_NAMES}
        reference = {
            name: np.asarray(archive[name]) for name in BACKBONE_NAMES + PROJECTION_NAMES
        }

    units = getattr(ct.ComputeUnit, args.compute_units)
    started = time.perf_counter()
    model = ct.models.MLModel(str(args.mlpackage), compute_units=units)
    prediction = model.predict(reference_inputs)
    prediction_seconds = time.perf_counter() - started
    prediction_arrays = {
        name: np.asarray(prediction[name])
        for name in BACKBONE_NAMES + PROJECTION_NAMES
        if name in prediction
    }
    backbone = compare_backbone_tensors(reference, prediction_arrays, BACKBONE_NAMES)
    projections = compare_backbone_tensors(reference, prediction_arrays, PROJECTION_NAMES)
    required = BACKBONE_NAMES + PROJECTION_NAMES
    model_output_names = [item.name for item in model.get_spec().description.output]
    missing_reference_outputs = [name for name in output_names if name not in model_output_names]
    report = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M59c backbone-focused Core ML audit",
        "scope": "frozen M59b FP32 package; no training or weight changes",
        "platform": platform.platform(),
        "coremltools_version": ct.__version__,
        "compute_units": args.compute_units,
        "mlpackage": str(args.mlpackage.resolve()),
        "mlpackage_tree_sha256": tree_sha256(args.mlpackage),
        "mlpackage_size_bytes": tree_size(args.mlpackage),
        "reference_io": str(args.reference_io.resolve()),
        "reference_io_sha256": sha256_file(args.reference_io),
        "prediction_seconds": prediction_seconds,
        "backbone": backbone,
        "input_projections": projections,
        "first_strict_diverging_tensor": first_failed(
            {**backbone, **projections}, "strict_passed"
        ),
        "first_material_diverging_tensor": first_failed(
            {**backbone, **projections}, "scale_aware_passed"
        ),
        "model_output_names": model_output_names,
        "missing_reference_outputs": missing_reference_outputs,
        "backbone_strict_passed": all(row.get("strict_passed") is True for row in backbone.values()),
        "backbone_scale_aware_passed": all(
            row.get("scale_aware_passed") is True for row in backbone.values()
        ),
        "input_projections_strict_passed": all(
            row.get("strict_passed") is True for row in projections.values()
        ),
        "required_backbone_and_projection_outputs_present": all(
            name in prediction_arrays for name in required
        ),
        "backbone_audit_passed": all(
            row.get("scale_aware_passed") is True for row in backbone.values()
        )
        and all(row.get("strict_passed") is True for row in projections.values())
        and all(name in prediction_arrays for name in required),
        "physical_device_testing_authorized": False,
        "fp16_or_quantization_authorized": False,
        "product_safety_qualified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
