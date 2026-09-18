"""Execute repaired M59f Core ML exports with unchanged diagnostic/full limits."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_monodgp_m59d_2d_transformer import INPUT_SHAPES, compare_tensors, sha256_file, tree_sha256
from scripts.diagnose_monodgp_m59e_position_layout import SPATIAL_SHAPES, error_summary
from scripts.validate_monodgp_m58_macos_parity import (
    OUTPUT_NAMES, compare_tensor_outputs, compare_candidates, decode_candidates, _sigmoid,
)

DECODED_FIELDS = {"class": (0, 1), "score": (1, 2), "center_2d_normalized": (2, 4),
                  "size_2d_normalized": (4, 6), "depth_m": (6, 7), "angle_code": (7, 31),
                  "dimensions": (31, 34), "center_3d_normalized": (34, 36),
                  "depth_confidence": (36, 37)}


def compare_stage(stage, reference, prediction, names):
    if stage not in ("internals", "layers", "full") or not names:
        raise ValueError("Unknown stage or empty diagnostic interface")
    if stage == "full":
        if list(names) != list(OUTPUT_NAMES):
            raise ValueError("Unexpected full-model interface")
        comparison = compare_tensor_outputs(reference, prediction)
        if all(name in prediction and np.asarray(prediction[name]).shape == reference[name].shape
               and np.isfinite(prediction[name]).all() for name in names):
            expected, actual = decode_candidates(reference), decode_candidates(prediction)
            decoded = compare_candidates(expected, actual)
            decoded["class_rank_changes"] = int((expected[..., 0] != actual[..., 0]).sum())
            decoded["fields"] = {field: compare_candidates(expected[..., start:end], actual[..., start:end])
                                  for field, (start, end) in DECODED_FIELDS.items()}
            orders = [np.argsort(-_sigmoid(values["pred_logits"]).reshape(values["pred_logits"].shape[0], -1), axis=1, kind="stable")[:, :50]
                      for values in (reference, prediction)]
            decoded["topk_flat_index_changes"] = int((orders[0] != orders[1]).sum())
        else:
            decoded = {"passed": False, "error": "invalid raw outputs"}
        return {"raw_output_parity": comparison, "decoded_candidate_parity": decoded,
                "all_parity_gates_passed": all(row["passed"] for row in comparison.values()) and decoded["passed"]}
    comparison = compare_tensors(reference, prediction, names)
    result = {"raw_intermediate_parity": comparison,
              "first_diverging_tensor": next((name for name in names if not comparison[name]["passed"]), None),
              "all_parity_gates_passed": all(row["passed"] for row in comparison.values())}
    if stage == "internals" and comparison.get("enc0_pos", {}).get("shape") == [1, 10200, 256]:
        start, scales = 0, []
        for height, width in SPATIAL_SHAPES:
            end = start + height * width
            scales.append({"spatial_shape": [height, width],
                           **error_summary(reference["enc0_pos"][:, start:end], prediction["enc0_pos"][:, start:end])})
            start = end
        result["positional_scales"] = scales
    return result


def cpu_control(source_dir, gate, inputs, reference):
    """Separate cross-backend rounding from any change caused by the rewrite."""
    from scripts.export_monodgp_m59f_position_interleave import verify_full_source, rewrite_positions
    import torch
    source = verify_full_source(source_dir)
    if source["artifacts"]["torchscript_sha256"] != gate["source_torchscript_sha256"]:
        raise RuntimeError("CPU control source differs from the exported source")
    torch.set_num_threads(4)
    args = tuple(torch.from_numpy(inputs[name]).float() for name in INPUT_SHAPES)
    path = source_dir / "MonoDGP_M58_fixed_fp32.pt"
    original = torch.jit.load(str(path), map_location="cpu").eval()
    patched = torch.jit.load(str(path), map_location="cpu").eval()
    rewrite_positions(patched, torch)
    with torch.inference_mode():
        before = {name: value.numpy() for name, value in zip(OUTPUT_NAMES, original(*args))}
        after = {name: value.numpy() for name, value in zip(OUTPUT_NAMES, patched(*args))}
    equivalent = compare_tensors(before, after, OUTPUT_NAMES)
    return {"torch_version": torch.__version__,
            "unmodified_cpu_vs_frozen_reference": compare_stage("full", reference, before, OUTPUT_NAMES),
            "rewrite_cpu_vs_unmodified_cpu": equivalent,
            "same_backend_rewrite_bit_exact": all(row["max_abs_delta"] == 0.0 for row in equivalent.values()),
            "note": "Diagnostic control only; frozen reference and acceptance thresholds are NOT replaced"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--compute-units", choices=("ALL", "CPU_AND_GPU", "CPU_ONLY"), default="ALL")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-dir", type=Path, help="Optional original M58 directory for full-stage CPU control")
    args = parser.parse_args()
    root = args.artifact_dir.resolve()
    gate_path, io_path = root / "m59f_export_gate.json", root / "m59f_reference_io.npz"
    package = root / "MonoDGP_M59f_fp32.mlpackage"
    report = {"schema_version": 1, "complete": False,
              "experiment": "M59f macOS positional-interleaving repair parity",
              "scope": "one frozen sample; not full validation or device qualification",
              "compute_units": args.compute_units, "deployment_authorized": False,
              "physical_device_testing_authorized": False, "product_safety_qualified": False}
    try:
        if platform.system() != "Darwin":
            raise RuntimeError("Core ML execution requires macOS")
        gate = json.loads(gate_path.read_text())
        if gate.get("complete") is not True or gate.get("all_export_gates_passed") is not True:
            raise RuntimeError("Passed M59f export required")
        if sha256_file(io_path) != gate["artifacts"]["reference_io_sha256"]:
            raise RuntimeError("Reference I/O changed")
        if tree_sha256(package) != gate["artifacts"]["mlpackage_tree_sha256"]:
            raise RuntimeError("Core ML package changed")
        with np.load(io_path, allow_pickle=False) as archive:
            inputs = {name: archive[name].copy() for name in INPUT_SHAPES}
            reference = {name: archive[name].copy() for name in gate["output_names"]}
        if any(list(inputs[name].shape) != shape or inputs[name].dtype != np.float32
               or not np.isfinite(inputs[name]).all() for name, shape in INPUT_SHAPES.items()):
            raise RuntimeError("Frozen input interface changed")
        if any(list(value.shape) != gate["output_shapes"][name] or not np.isfinite(value).all()
               for name, value in reference.items()):
            raise RuntimeError("Frozen output interface changed")
        import coremltools as ct
        started = time.perf_counter()
        model = ct.models.MLModel(str(package), compute_units=getattr(ct.ComputeUnit, args.compute_units))
        loaded = time.perf_counter()
        prediction = model.predict(inputs)
        finished = time.perf_counter()
        report.update(compare_stage(gate["stage"], reference, prediction, gate["output_names"]))
        if args.source_dir is not None:
            if gate["stage"] != "full":
                raise ValueError("CPU control --source-dir is only supported for the full stage")
            report["cpu_control"] = cpu_control(args.source_dir, gate, inputs, reference)
        report.update({"complete": True, "stage": gate["stage"],
                       "export_gate_sha256": sha256_file(gate_path),
                       "source_torchscript_sha256": gate["source_torchscript_sha256"],
                       "mlpackage_tree_sha256": gate["artifacts"]["mlpackage_tree_sha256"],
                       "reference_io_sha256": sha256_file(io_path),
                       "platform": platform.platform(), "coremltools_version": ct.__version__,
                       "load_compile_seconds": loaded - started,
                       "first_prediction_seconds": finished - loaded, "timing_is_benchmark": False})
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    if not report["all_parity_gates_passed"]:
        raise SystemExit("M59f parity failed; preserve report and do not proceed to deployment")


if __name__ == "__main__":
    main()
