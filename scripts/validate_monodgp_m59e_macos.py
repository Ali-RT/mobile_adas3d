"""Compare M59e encoder-internal Core ML tensors with the frozen trace."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_monodgp_m59d_2d_transformer import INPUT_SHAPES, PROBE_LIMIT, sha256_file, tree_sha256
from scripts.export_monodgp_m59e_encoder_internals import TAP_NAMES
from scripts.validate_monodgp_m59d_macos import compare_tensors


def summarize(reference, candidate):
    comparison = compare_tensors(reference, candidate, TAP_NAMES)
    for name, row in comparison.items():
        if "max_abs_delta" in row:
            scale = float(np.max(np.abs(reference[name])))
            row["reference_max_abs"] = scale
            row["max_delta_over_reference_scale"] = row["max_abs_delta"] / max(scale, 1.0)
    first = next((name for name in TAP_NAMES if not comparison[name]["passed"]), None)
    return comparison, first


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--compute-units", choices=("ALL", "CPU_AND_GPU", "CPU_ONLY"), default="ALL")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--save-predictions", type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        raise RuntimeError("Core ML execution requires macOS")
    root = args.artifact_dir.resolve()
    package = root / "MonoDGP_M59e_encoder_internal_fp32.mlpackage"
    reference_path = root / "m59e_encoder_internal_reference_io.npz"
    gate_path = root / "m59e_encoder_internal_export_gate.json"
    report = {"schema_version": 1, "complete": False,
              "experiment": "M59e macOS first encoder internal diagnostic",
              "scope": "one frozen M59d input; diagnostic outputs only",
              "compute_units": args.compute_units, "platform": platform.platform(),
              "physical_device_testing_authorized": False,
              "fp16_or_quantization_authorized": False, "product_safety_qualified": False}
    try:
        gate = json.loads(gate_path.read_text())
        if gate.get("complete") is not True or gate.get("all_export_gates_passed") is not True:
            raise RuntimeError("M59e export gate is incomplete or failed")
        if gate.get("output_names") != list(TAP_NAMES):
            raise RuntimeError("M59e output interface differs from the diagnostic contract")
        if tree_sha256(package) != gate["artifacts"]["mlpackage_tree_sha256"]:
            raise RuntimeError("M59e package hash changed")
        if sha256_file(reference_path) != gate["artifacts"]["reference_io_sha256"]:
            raise RuntimeError("M59e reference I/O hash changed")
        with np.load(reference_path, allow_pickle=False) as archive:
            inputs = {n: archive[n].copy() for n in INPUT_SHAPES}
            reference = {n: archive[n].copy() for n in TAP_NAMES}
        if any(list(inputs[n].shape) != s for n, s in INPUT_SHAPES.items()):
            raise RuntimeError("M59e input signature differs")
        if any(list(reference[n].shape) != gate["output_shapes"][n] for n in TAP_NAMES):
            raise RuntimeError("M59e reference output shapes differ")
        import coremltools as ct
        started = time.perf_counter()
        model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, args.compute_units))
        load_seconds = time.perf_counter() - started
        started = time.perf_counter()
        predictions = model.predict(inputs)
        inference_seconds = time.perf_counter() - started
        comparison, first = summarize(reference, predictions)
        if args.save_predictions:
            args.save_predictions.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save_predictions, **predictions)
            report["predictions_sha256"] = sha256_file(args.save_predictions)
        report.update({"complete": True, "all_internal_gates_passed": first is None,
                       "first_diverging_tensor": first, "raw_internal_parity": comparison,
                       "probe_limit_max_abs": PROBE_LIMIT, "coremltools_version": ct.__version__,
                       "export_gate_sha256": sha256_file(gate_path),
                       "mlpackage_tree_sha256": gate["artifacts"]["mlpackage_tree_sha256"],
                       "reference_io_sha256": gate["artifacts"]["reference_io_sha256"],
                       "load_compile_seconds": load_seconds,
                       "first_prediction_seconds": inference_seconds,
                       "timing_is_benchmark": False})
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    if first is not None:
        raise SystemExit(f"M59e parity failed first at {first}; report saved to {args.output}")


if __name__ == "__main__":
    main()
