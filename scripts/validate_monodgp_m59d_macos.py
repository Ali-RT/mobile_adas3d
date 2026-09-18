"""Validate M59d 2D-transformer intermediate tensors on macOS."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np


INPUT_NAMES = ("image", "calibration", "image_size")
PROBE_LIMIT = 1.0e-3


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


def compare_tensors(reference, prediction, names):
    result = {}
    for name in names:
        expected, actual = reference.get(name), prediction.get(name)
        if expected is None or actual is None:
            result[name] = {"passed": False, "error": "missing tensor", "limit": PROBE_LIMIT}
            continue
        expected, actual = np.asarray(expected, dtype=np.float32), np.asarray(actual, dtype=np.float32)
        if expected.shape != actual.shape:
            result[name] = {"passed": False, "error": "shape mismatch", "expected_shape": list(expected.shape), "actual_shape": list(actual.shape), "limit": PROBE_LIMIT}
            continue
        delta = np.abs(actual - expected)
        result[name] = {
            "passed": bool(np.isfinite(actual).all() and (not delta.size or delta.max() <= PROBE_LIMIT)),
            "shape": list(actual.shape),
            "max_abs_delta": float(delta.max()) if delta.size else 0.0,
            "mean_abs_delta": float(delta.mean()) if delta.size else 0.0,
            "limit": PROBE_LIMIT,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the M59d 2D-transformer Core ML diagnostic.")
    parser.add_argument("--mlpackage", type=Path, required=True)
    parser.add_argument("--reference-io", type=Path, required=True)
    parser.add_argument("--export-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compute-units", choices=("CPU_ONLY", "CPU_AND_GPU", "ALL"), default="ALL")
    args = parser.parse_args()
    if platform.system() != "Darwin":
        raise RuntimeError("M59d Core ML validation requires macOS")
    if not args.mlpackage.is_dir() or not args.reference_io.is_file() or not args.export_gate.is_file():
        raise FileNotFoundError("M59d package, reference I/O, or export gate is missing")
    import coremltools as ct

    gate = json.loads(args.export_gate.read_text(encoding="utf-8"))
    if gate.get("complete") is not True or gate.get("all_export_gates_passed") is not True:
        raise RuntimeError("M59d export gate is incomplete or failed")
    output_names = list(gate.get("output_names", []))
    output_shapes = gate.get("output_shapes", {})
    if not output_names or set(output_names) != set(output_shapes):
        raise RuntimeError("M59d gate has no frozen output interface")
    expected_io = gate.get("artifacts", {}).get("reference_io_sha256")
    expected_pkg = gate.get("artifacts", {}).get("mlpackage_tree_sha256")
    if expected_io and sha256_file(args.reference_io) != expected_io:
        raise RuntimeError("M59d reference I/O hash differs from export gate")
    if expected_pkg and tree_sha256(args.mlpackage) != expected_pkg:
        raise RuntimeError("M59d ML package hash differs from export gate")
    with np.load(args.reference_io, allow_pickle=False) as archive:
        inputs = {name: np.asarray(archive[name]) for name in INPUT_NAMES}
        reference = {name: np.asarray(archive[name]) for name in output_names}
    units = getattr(ct.ComputeUnit, args.compute_units)
    started = time.perf_counter()
    model = ct.models.MLModel(str(args.mlpackage), compute_units=units)
    prediction = model.predict(inputs)
    prediction_seconds = time.perf_counter() - started
    candidate = {name: np.asarray(prediction[name]) for name in output_names if name in prediction}
    comparison = compare_tensors(reference, candidate, output_names)
    first = next((name for name, row in comparison.items() if row.get("passed") is not True), None)
    report = {
        "schema_version": 1,
        "complete": first is None,
        "experiment": "M59d macOS 2D-transformer layer diagnostic",
        "scope": "fixed-shape FP32 diagnostic package; one frozen M58 validation sample",
        "platform": platform.platform(),
        "coremltools_version": ct.__version__,
        "compute_units": args.compute_units,
        "mlpackage": str(args.mlpackage.resolve()),
        "mlpackage_tree_sha256": tree_sha256(args.mlpackage),
        "reference_io": str(args.reference_io.resolve()),
        "reference_io_sha256": sha256_file(args.reference_io),
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
        raise RuntimeError(f"M59d diagnostic found first divergence at {first}; see {args.output}")


if __name__ == "__main__":
    main()
