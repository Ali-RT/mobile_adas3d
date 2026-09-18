"""M59g bounded numerical audit of the unchanged repaired M59f package."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_monodgp_m59g_inputs import (
    ANCHOR_SHA256, TRACE_SHA256, SAMPLE_COUNT, PREPROCESSING, VAL_SHA256,
    choose_samples, require_anchor_match, validate_inputs,
)
from scripts.export_monodgp_m59f_position_interleave import verify_full_source, rewrite_positions
from scripts.validate_monodgp_m58_macos_parity import (
    INPUT_SHAPES, OUTPUT_NAMES, OUTPUT_SHAPES, sha256_file, tree_sha256, _sigmoid,
)
from scripts.validate_monodgp_m59f_macos import compare_stage


def load_inputs(path):
    with np.load(path, allow_pickle=False) as archive:
        values = {name: archive[name].copy() for name in INPUT_SHAPES}
    validate_inputs(values)
    return values


def validate_outputs(values):
    for name, shape in OUTPUT_SHAPES.items():
        value = np.asarray(values[name])
        if list(value.shape) != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid output: {name}")


def verify_bundle(root, anchor):
    manifest = json.loads((root / "m59g_input_manifest.json").read_text())
    indices, ids = choose_samples(root / "val.txt")
    required = {"schema_version": 1, "complete": True, "sample_count": SAMPLE_COUNT,
                "sample_ids": ids, "sample_indices": indices, "val_split_sha256": VAL_SHA256,
                "source_torchscript_sha256": TRACE_SHA256, "frozen_anchor_sha256": ANCHOR_SHA256,
                "preprocessing": PREPROCESSING, "anchor_preprocessing_bit_exact": True}
    if any(manifest.get(key) != value for key, value in required.items()):
        raise RuntimeError("Input bundle provenance differs from the frozen audit protocol")
    rows = manifest["samples"]
    if len(rows) != SAMPLE_COUNT or [row["sample_id"] for row in rows] != ids:
        raise RuntimeError("Input bundle sample list changed")
    for row in rows:
        if row["file"] != f'{row["sample_id"]}.npz' or sha256_file(root / row["file"]) != row["sha256"]:
            raise RuntimeError("Input bundle file path or checksum changed")
        inputs = load_inputs(root / row["file"])
        if inputs["image_size"][0].tolist() != row["original_size_wh"]:
            raise RuntimeError("Image size metadata changed")
        if row["sample_id"] == "000001":
            require_anchor_match(inputs, anchor)
    return rows


def repeat_summary(first, second):
    validate_outputs(first)
    validate_outputs(second)
    deltas = {name: float(np.abs(first[name] - second[name]).max()) for name in OUTPUT_NAMES}
    return {"bit_exact": all(value == 0 for value in deltas.values()), "max_abs_deltas": deltas}


def selection_summary(reference, actual):
    orders = [np.argsort(-_sigmoid(x["pred_logits"]).reshape(-1), kind="stable")[:50]
              for x in (reference, actual)]
    ref_set, actual_set = (set(x.tolist()) for x in orders)
    return {"rank_positions_changed": int((orders[0] != orders[1]).sum()),
            "selected_identities_removed": sorted(ref_set - actual_set),
            "selected_identities_added": sorted(actual_set - ref_set),
            "same_selected_identity_set": ref_set == actual_set}


def cpu_only_control(artifact_dir, output_dir):
    """Bound compilation/inference wall time; timeout is not a parity result."""
    path = output_dir / "m59g_cpu_only_anchor.json"
    command = [sys.executable, "-u", str(Path(__file__).with_name("validate_monodgp_m59f_macos.py")),
               "--artifact-dir", str(artifact_dir), "--compute-units", "CPU_ONLY", "--output", str(path)]
    log_path = output_dir / "m59g_cpu_only_anchor.log"
    with log_path.open("w") as log:
        try:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=45)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "timeout_seconds": 45, "parity_result_available": False,
                    "log": log_path.name, "note": "Bounded control stopped; not a measured latency benchmark"}
    if not path.is_file():
        return {"status": "process_error", "returncode": result.returncode, "log": log_path.name,
                "parity_result_available": False}
    report = json.loads(path.read_text())
    return {"status": "completed" if report.get("complete") else "execution_error",
            "returncode": result.returncode, "report": path.name, "log": log_path.name,
            "parity_result_available": report.get("complete", False),
            "all_parity_gates_passed": report.get("all_parity_gates_passed", False),
            "decoded_candidate_parity": report.get("decoded_candidate_parity")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--m58-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--bundle-dir", type=Path)
    mode.add_argument("--anchor-only", action="store_true", help="Partial one-input control, never multi-input acceptance")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if platform.system() != "Darwin":
        raise RuntimeError("Run this audit on macOS; the collection notebook runs on Colab CPU")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report_path = args.output_dir / "m59g_precision_audit.json"
    report = {"schema_version": 1, "complete": False, "experiment": "M59g bounded precision audit",
              "multi_input_complete": False, "expected_samples": SAMPLE_COUNT, "samples": [],
              "thresholds_changed": False, "training_performed": False, "weights_changed": False,
              "ap_evaluation_performed": False, "timing_is_benchmark": False,
              "deployment_authorized": False, "physical_device_testing_authorized": False,
              "product_safety_qualified": False, "all_parity_gates_passed": False}
    try:
        source = verify_full_source(args.m58_dir)
        if (source["artifacts"]["torchscript_sha256"] != TRACE_SHA256
                or source["artifacts"]["reference_io_sha256"] != ANCHOR_SHA256):
            raise RuntimeError("Original M58 source changed")
        gate_path = args.artifact_dir / "m59f_export_gate.json"
        gate = json.loads(gate_path.read_text())
        package = args.artifact_dir / "MonoDGP_M59f_fp32.mlpackage"
        if (gate.get("complete") is not True or gate.get("all_export_gates_passed") is not True
                or gate.get("stage") != "full" or gate.get("source_torchscript_sha256") != TRACE_SHA256
                or gate.get("output_names") != list(OUTPUT_NAMES)
                or tree_sha256(package) != gate["artifacts"]["mlpackage_tree_sha256"]
                or sha256_file(args.artifact_dir / "m59f_reference_io.npz") != ANCHOR_SHA256):
            raise RuntimeError("M59f full package or frozen reference provenance changed")
        with np.load(args.m58_dir / "m58_reference_io.npz", allow_pickle=False) as archive:
            anchor = {name: archive[name].copy() for name in (*INPUT_SHAPES, *OUTPUT_NAMES)}
        rows = verify_bundle(args.bundle_dir, anchor) if args.bundle_dir else [{"sample_id": "000001"}]
        if args.bundle_dir:
            report["input_manifest_sha256"] = sha256_file(args.bundle_dir / "m59g_input_manifest.json")
        import torch
        import coremltools as ct
        torch.set_num_threads(4)
        original = torch.jit.load(str(args.m58_dir / "MonoDGP_M58_fixed_fp32.pt"), map_location="cpu").eval()
        patched = torch.jit.load(str(args.m58_dir / "MonoDGP_M58_fixed_fp32.pt"), map_location="cpu").eval()
        report["rewritten_pairs"] = rewrite_positions(patched, torch)
        report.update({"source_torchscript_sha256": TRACE_SHA256, "frozen_anchor_sha256": ANCHOR_SHA256,
                       "export_gate_sha256": sha256_file(gate_path),
                       "mlpackage_tree_sha256": gate["artifacts"]["mlpackage_tree_sha256"],
                       "compute_units": "ALL", "repeats_per_input": {"pytorch_cpu": 2, "coreml": 3},
                       "reference_policy": "unchanged frozen anchor plus unmodified CPU TorchScript for new inputs",
                       "software": {"python": platform.python_version(), "platform": platform.platform(),
                                    "torch": torch.__version__, "numpy": np.__version__, "coremltools": ct.__version__}})
        print("Loading verified M59f Core ML package (ALL)", flush=True)
        model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)
        for number, row in enumerate(rows, 1):
            inputs = load_inputs(args.bundle_dir / row["file"]) if args.bundle_dir else {n: anchor[n] for n in INPUT_SHAPES}
            tensors = tuple(torch.from_numpy(inputs[name]).float() for name in INPUT_SHAPES)
            with torch.inference_mode():
                before = {n: v.numpy().copy() for n, v in zip(OUTPUT_NAMES, original(*tensors))}
                repeated = {n: v.numpy().copy() for n, v in zip(OUTPUT_NAMES, original(*tensors))}
                after = {n: v.numpy().copy() for n, v in zip(OUTPUT_NAMES, patched(*tensors))}
            prediction = model.predict(inputs)
            validate_outputs(prediction)
            result = {"sample_id": row["sample_id"], "pytorch_repeatability": repeat_summary(before, repeated),
                      "rewrite_equivalence": repeat_summary(before, after),
                      "coreml_vs_pytorch_cpu": compare_stage("full", before, prediction, OUTPUT_NAMES),
                      "selection_vs_pytorch_cpu": selection_summary(before, prediction),
                      "coreml_repeatability": [repeat_summary(prediction, model.predict(inputs)) for _ in range(2)]}
            if row["sample_id"] == "000001":
                result["coreml_vs_unchanged_frozen_anchor"] = compare_stage("full", anchor, prediction, OUTPUT_NAMES)
                result["pytorch_cpu_vs_unchanged_frozen_anchor"] = compare_stage("full", anchor, before, OUTPUT_NAMES)
            report["samples"].append(result)
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f'Audited {number}/{len(rows)}: {row["sample_id"]}; decoded CPU delta='
                  f'{result["coreml_vs_pytorch_cpu"]["decoded_candidate_parity"]["max_abs_delta"]}', flush=True)
        print("Running one CPU_ONLY anchor control (45-second budget)", flush=True)
        report["cpu_only_anchor_control"] = cpu_only_control(args.artifact_dir.resolve(), args.output_dir.resolve())
        report["multi_input_complete"] = args.bundle_dir is not None and len(rows) == SAMPLE_COUNT
        report["complete"] = True
        # A partial audit cannot pass. Neither CPU references nor repeatability replace the old anchor gate.
        report["all_parity_gates_passed"] = bool(report["multi_input_complete"] and all(
            row["coreml_vs_pytorch_cpu"]["all_parity_gates_passed"]
            and row["rewrite_equivalence"]["bit_exact"] and row["pytorch_repeatability"]["bit_exact"]
            and all(x["bit_exact"] for x in row["coreml_repeatability"])
            and row["selection_vs_pytorch_cpu"]["rank_positions_changed"] == 0
            and row.get("coreml_vs_unchanged_frozen_anchor", {"all_parity_gates_passed": True})["all_parity_gates_passed"]
            for row in report["samples"]))
    except Exception as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n")
        print(f"Audit report: {report_path}", flush=True)
    if not report["all_parity_gates_passed"]:
        raise SystemExit("M59g not passed: retain evidence; no threshold relaxation or deployment")


if __name__ == "__main__":
    main()
