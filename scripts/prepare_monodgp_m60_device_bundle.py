"""Prepare exact M59i model and fixed16 fixtures for an isolated iPhone app.

No conversion, training, camera preprocessing, or tolerance change is performed.
Large generated resources stay under outputs/, not in git.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_monodgp_m59i_validation import write_json
from scripts.prepare_monodgp_m59i_tensor_inputs import load_reviewed_inputs
from scripts.evaluate_monodgp_m59i_coreml import PACKAGE_SHA256, load_policy, numerical_gate
from scripts.validate_monodgp_m58_macos_parity import (
    INPUT_SHAPES, OUTPUT_NAMES, OUTPUT_SHAPES, sha256_file, tree_sha256, compare_tensor_outputs,
)
from scripts.collect_monodgp_m59g_inputs import TRACE_SHA256
from scripts.diagnose_monodgp_m59h_geometry import upstream_decoder, compare_geometry, geometry_rows

M59I_REPORT_SHA256 = "2aff5f2603d44d30dea1edce5cf6104375b1550eb560f0497296b2f6b21c4ad2"


def write_tensor(root, relative, value):
    value = np.asarray(value)
    if value.dtype != np.float32 or not np.isfinite(value).all():
        raise ValueError("Device fixtures must be finite FP32")
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(np.ascontiguousarray(value, dtype="<f4").tobytes())
    return {"file": relative, "shape": list(value.shape), "sha256": sha256_file(path)}


def read_tensor(root, row):
    path = (root / row["file"]).resolve()
    if not path.is_relative_to(root.resolve()) or sha256_file(path) != row["sha256"]:
        raise RuntimeError("Tensor path or checksum changed")
    value = np.frombuffer(path.read_bytes(), dtype="<f4")
    if value.size != np.prod(row["shape"]) or not np.isfinite(value).all():
        raise RuntimeError("Tensor shape or finiteness changed")
    return value.reshape(row["shape"]).copy()


def parity_row(reference, actual, inputs, extract, native_decode):
    expected, expected_ids = extract(reference)
    observed, observed_ids = extract(actual)
    row = {"raw_output_parity": compare_tensor_outputs(reference, actual),
           "same_identity_order": bool(np.array_equal(expected_ids, observed_ids)),
           "decoder_port_max_abs_delta": 0.0}
    for candidates in (expected, observed):
        decoded, _ = geometry_rows(candidates, inputs)
        if not np.array_equal(decoded, native_decode(candidates, inputs)):
            raise RuntimeError("Native decoder parity failed")
    if row["same_identity_order"]:
        row.update(compare_geometry(expected, observed, inputs, expected_ids, observed_ids)[0])
    row["gates"] = numerical_gate(row, load_policy())
    row["passed"] = all(row["gates"].values())
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--reviewed-input-archive", type=Path, required=True)
    parser.add_argument("--upstream-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report_path = ROOT / "artifacts/m59i_full_validation_gate_20260921.json"
    if sha256_file(report_path) != M59I_REPORT_SHA256:
        raise RuntimeError("Reviewed M59i report changed")
    parent = json.loads(report_path.read_text())
    if not parent["complete"] or not parent["all_validation_gates_passed"]:
        raise RuntimeError("M59i must pass before M60")
    if tree_sha256(args.package) != PACKAGE_SHA256 or sha256_file(args.trace) != TRACE_SHA256:
        raise RuntimeError("Frozen model changed")
    inputs = load_reviewed_inputs(args.reviewed_input_archive)
    policy = load_policy()
    decode, extract, decoder_hashes = upstream_decoder(args.upstream_repo)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import torch
    import coremltools as ct
    torch.set_num_threads(4)
    trace = torch.jit.load(str(args.trace), map_location="cpu").eval()
    model = ct.models.MLModel(str(args.package), compute_units=ct.ComputeUnit.ALL)
    manifest = {"schema_version": 1, "revision": "M60-2026-09-21-r1", "complete": False,
                "experiment": "M60a fixed-input physical-device feasibility",
                "m59i_report_sha256": M59I_REPORT_SHA256, "mlpackage_tree_sha256": PACKAGE_SHA256,
                "torchscript_sha256": TRACE_SHA256, "policy": policy,
                "upstream_decoder_hashes": decoder_hashes,
                "model_directory": "MonoDGP_M59f_fp32.mlpackage", "compute_units": "ALL",
                "warmups": 5, "timed_predictions": 100, "sustain_seconds": 60,
                "inference_p95_target_ms": 50, "sample_ids": list(inputs), "samples": [],
                "camera_integration": False, "deployment_authorized": False}
    for sample_id, values in inputs.items():
        print(f"M60 fixture {sample_id}", flush=True)
        with torch.inference_mode():
            raw = trace(*(torch.from_numpy(values[name]) for name in INPUT_SHAPES))
        reference = {name: value.numpy().copy() for name, value in zip(OUTPUT_NAMES, raw)}
        actual = model.predict(values)
        parity = parity_row(reference, actual, values, extract, decode)
        if not parity["passed"]:
            raise RuntimeError(f"Fresh fixed-input Mac parity failed: {sample_id}: {parity}")
        row = {"sample_id": sample_id, "inputs": {}, "mac_outputs": {}, "pytorch_outputs": {},
               "mac_parity": parity}
        for group, tensors in (("inputs", values), ("mac_outputs", actual), ("pytorch_outputs", reference)):
            for name in (INPUT_SHAPES if group == "inputs" else OUTPUT_NAMES):
                shape = INPUT_SHAPES[name] if group == "inputs" else OUTPUT_SHAPES[name]
                if list(tensors[name].shape) != shape:
                    raise RuntimeError(f"Unexpected {name} shape")
                row[group][name] = write_tensor(args.output_dir, f"{sample_id}/{group}/{name}.bin", tensors[name])
        manifest["samples"].append(row)
    shutil.copytree(args.package, args.output_dir / manifest["model_directory"])
    if tree_sha256(args.output_dir / manifest["model_directory"]) != PACKAGE_SHA256:
        raise RuntimeError("Model copy checksum failed")
    manifest["model_files"] = {p.relative_to(args.output_dir).as_posix(): sha256_file(p)
                               for p in sorted((args.output_dir / manifest["model_directory"]).rglob("*")) if p.is_file()}
    manifest["complete"] = True
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Prepared {len(inputs)} exact fixtures: {args.output_dir}")


if __name__ == "__main__":
    main()
