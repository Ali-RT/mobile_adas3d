"""Run the M59b intermediate-tensor Core ML diagnostic on macOS.

This is an investigation gate only.  It reports the first causal tensor whose
Core ML value differs from the frozen PyTorch reference; it never authorizes an
iPhone, FP16, quantization, or product-safety experiment.
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
PROBE_LIMIT = 1.0e-3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.read_bytes())
    return digest.hexdigest()


def compare_diagnostic_tensors(
    reference: dict[str, np.ndarray],
    prediction: dict[str, np.ndarray],
    output_names: list[str],
    probe_limit: float = PROBE_LIMIT,
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name in output_names:
        expected = reference.get(name)
        actual = prediction.get(name)
        if expected is None or actual is None:
            report[name] = {"passed": False, "error": "missing tensor", "limit": probe_limit}
            continue
        expected = np.asarray(expected, dtype=np.float32)
        actual = np.asarray(actual, dtype=np.float32)
        if expected.shape != actual.shape:
            report[name] = {
                "passed": False,
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
                "limit": probe_limit,
            }
            continue
        delta = np.abs(actual - expected)
        max_abs = float(delta.max()) if delta.size else 0.0
        mean_abs = float(delta.mean()) if delta.size else 0.0
        report[name] = {
            "passed": bool(np.isfinite(actual).all() and max_abs <= probe_limit),
            "shape": list(actual.shape),
            "max_abs_delta": max_abs,
            "mean_abs_delta": mean_abs,
            "limit": probe_limit,
        }
    return report


def first_diverging_tensor(comparison: dict[str, dict[str, Any]]) -> str | None:
    for name, row in comparison.items():
        if row.get("passed") is not True:
            return name
    return None


def _require_shapes(
    values: dict[str, np.ndarray], expected: dict[str, list[int]], label: str
) -> None:
    for name, shape in expected.items():
        if name not in values:
            raise RuntimeError(f"{label} is missing {name}")
        if list(values[name].shape) != shape:
            raise RuntimeError(
                f"{label} {name} shape changed: expected {shape}, found {list(values[name].shape)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the M59b intermediate-tensor Core ML diagnostic on macOS."
    )
    parser.add_argument("--mlpackage", type=Path, required=True)
    parser.add_argument("--reference-io", type=Path, required=True)
    parser.add_argument("--export-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--compute-units", choices=("CPU_ONLY", "CPU_AND_GPU", "ALL"), default="ALL"
    )
    args = parser.parse_args()

    if platform.system() != "Darwin":
        raise RuntimeError("M59b Core ML diagnostics require macOS; Colab cannot execute Core ML")
    if not args.mlpackage.is_dir():
        raise FileNotFoundError(args.mlpackage)
    if not args.reference_io.is_file():
        raise FileNotFoundError(args.reference_io)
    if not args.export_gate.is_file():
        raise FileNotFoundError(args.export_gate)

    import coremltools as ct

    export_gate = json.loads(args.export_gate.read_text(encoding="utf-8"))
    if export_gate.get("complete") is not True or export_gate.get("all_export_gates_passed") is not True:
        raise RuntimeError("M59b diagnostic export gate is incomplete or failed")
    output_names = list(export_gate.get("output_names", []))
    output_shapes = dict(export_gate.get("output_shapes", {}))
    if not output_names or set(output_names) != set(output_shapes):
        raise RuntimeError("M59b export gate has no frozen diagnostic interface")

    with np.load(args.reference_io, allow_pickle=False) as archive:
        reference_inputs = {name: np.asarray(archive[name]) for name in INPUT_NAMES}
        reference_outputs = {
            name: np.asarray(archive[name]) for name in output_names if name in archive.files
        }
    _require_shapes(reference_inputs, {
        "image": [1, 3, 384, 1280],
        "calibration": [1, 3, 4],
        "image_size": [1, 2],
    }, "Reference input")
    _require_shapes(reference_outputs, output_shapes, "Reference output")
    if not all(value.dtype == np.float32 for value in reference_inputs.values()):
        raise RuntimeError("M59b reference inputs must be float32")
    if not all(np.isfinite(value).all() for value in reference_outputs.values()):
        raise RuntimeError("M59b reference outputs contain non-finite values")
    expected_ref_sha = export_gate.get("artifacts", {}).get("reference_io_sha256")
    if expected_ref_sha and sha256_file(args.reference_io) != expected_ref_sha:
        raise RuntimeError("M59b reference I/O hash differs from export gate")
    expected_pkg_sha = export_gate.get("artifacts", {}).get("mlpackage_tree_sha256")
    if expected_pkg_sha and tree_sha256(args.mlpackage) != expected_pkg_sha:
        raise RuntimeError("M59b ML package tree hash differs from export gate")

    units = getattr(ct.ComputeUnit, args.compute_units)
    prediction_start = time.perf_counter()
    model = ct.models.MLModel(str(args.mlpackage), compute_units=units)
    prediction = model.predict(reference_inputs)
    prediction_seconds = time.perf_counter() - prediction_start
    prediction_arrays = {
        name: np.asarray(prediction[name]) for name in output_names if name in prediction
    }
    comparison = compare_diagnostic_tensors(reference_outputs, prediction_arrays, output_names)
    first = first_diverging_tensor(comparison)
    report = {
        "schema_version": 1,
        "complete": first is None,
        "experiment": "M59b macOS Core ML intermediate-tensor diagnostic",
        "scope": "fixed-shape FP32 diagnostic package; one frozen M58 validation sample",
        "platform": platform.platform(),
        "coremltools_version": ct.__version__,
        "compute_units": args.compute_units,
        "mlpackage": str(args.mlpackage.resolve()),
        "mlpackage_tree_sha256": tree_sha256(args.mlpackage),
        "mlpackage_size_bytes": tree_size(args.mlpackage),
        "reference_io": str(args.reference_io.resolve()),
        "reference_io_sha256": sha256_file(args.reference_io),
        "export_gate": str(args.export_gate.resolve()),
        "export_gate_sha256": sha256_file(args.export_gate),
        "prediction_seconds": prediction_seconds,
        "output_names": output_names,
        "raw_intermediate_parity": comparison,
        "first_diverging_tensor": first,
        "all_intermediate_gates_passed": first is None,
        "physical_device_testing_authorized": False,
        "fp16_or_quantization_authorized": False,
        "product_safety_qualified": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if first is not None:
        raise RuntimeError(f"M59b diagnostic found first divergence at {first}; see {args.output}")


if __name__ == "__main__":
    main()
