"""Restartable paired full Chen-val validation of the frozen M59f Core ML model.

M59i applies the explicitly approved unit-aware policy without rewriting M59g.
No training, threshold tuning, output-format change, or deployment approval.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_monodgp_m59i_validation import verify_bundle, reviewed_samples, write_json
from scripts.collect_monodgp_m59g_inputs import preprocess, require_anchor_match, TRACE_SHA256, ANCHOR_SHA256
from scripts.audit_monodgp_m59g_precision import validate_outputs
from scripts.diagnose_monodgp_m59h_geometry import (
    CONFIG_SHA256, upstream_decoder, geometry_rows, compare_geometry, filter_masks, serialize_rows,
)
from scripts.validate_monodgp_m58_macos_parity import (
    INPUT_SHAPES, OUTPUT_NAMES, PARITY_LIMITS, sha256_file, tree_sha256, compare_tensor_outputs,
)
from scripts.evaluate_monodgp_m56_fp16_storage import (
    metric_value, preservation_gate_results, prediction_tree_sha256, run_logged,
)
from scripts.prepare_monodgp_m55_feasibility import PRESERVATION_GATES
from scripts.prepare_monodgp_m59i_tensor_inputs import verify_tensor_bundle, load_tensor

POLICY_PATH = ROOT / "configs/monodgp_m59i_validation_policy.json"
POLICY_SHA256 = "8306ec063a55b81d4db9b852242144fd42d078e7ca4644d3a6bcf18e0d7c340f"
M59H_SHA256 = "4de8f0e2939763dc4562185c120f1c473c2979ddb1f40c8a1dddecff91f32c07"
PACKAGE_SHA256 = "90a5146acc25eb96d8bdafd27c3434898f737082ef5c1c8140d3ba97aaba1459"


def load_policy():
    if sha256_file(POLICY_PATH) != POLICY_SHA256:
        raise RuntimeError("Approved M59i policy changed; do not tune after validation")
    policy = json.loads(POLICY_PATH.read_text())
    if policy["raw_limits"] != PARITY_LIMITS or policy["preservation_gates"] != PRESERVATION_GATES:
        raise RuntimeError("Historical raw/accuracy preservation gates changed")
    return policy


def numerical_gate(sample, policy):
    """Fixed diagnostic-set gates. Missing or nonfinite measurements fail closed."""
    geometry = sample.get("continuous_geometry", {})
    results = {}
    for name, limit in policy["geometry_limits"].items():
        value = geometry.get(name, {}).get("max")
        results[name] = bool(isinstance(value, (float, int)) and np.isfinite(value) and 0 <= value <= limit)
    raw = sample.get("raw_output_parity", sample.get("unchanged_m59g_checks", {}).get("raw_output_parity", {}))
    results["raw_outputs"] = set(raw) == set(OUTPUT_NAMES) and all(x.get("passed") is True for x in raw.values())
    results["same_identity_order"] = sample.get("same_identity_order", sample.get("selection", {}).get("rank_positions_changed") == 0) is True
    results["heading_bins"] = sample.get("heading_bin_changes") == 0
    filters = sample.get("filter_decisions", {})
    results["filters"] = set(filters) == {"native_score", "product_export", "nearby_score_before_text", "nearby_score_after_native_text"} and all(x.get("changed_count") == 0 for x in filters.values())
    results["valid_geometry"] = sample.get("invalid_geometry") == {"reference": 0, "actual": 0}
    results["native_decoder_exact"] = sample.get("decoder_port_max_abs_delta") == 0.0
    return results


def approved_diagnostic_report(policy):
    path = ROOT / "artifacts/m59h_geometry_diagnostic_20260919.json"
    if sha256_file(path) != M59H_SHA256:
        raise RuntimeError("Historical M59h evidence changed")
    old = json.loads(path.read_text())
    ids = list(reviewed_samples())
    if old.get("complete") is not True or [x["sample_id"] for x in old["samples"]] != ids:
        raise RuntimeError("Incomplete fixed-set evidence")
    gates = {row["sample_id"]: numerical_gate(row, policy) for row in old["samples"]}
    return {"schema_version": 1, "complete": True, "policy_id": policy["policy_id"],
            "policy_sha256": POLICY_SHA256, "m59h_report_sha256": M59H_SHA256,
            "sample_gate_results": gates, "approved_diagnostic_gate_passed": all(all(x.values()) for x in gates.values()),
            "historical_m59g_gate_passed": False, "full_validation_performed": False,
            "deployment_authorized": False}


def code_fingerprint():
    # Includes evaluators, preprocessing, decoder, config inheritance and runner.
    digest = hashlib.sha256()
    for directory, suffix in (("scripts", ".py"), ("data", ".py"), ("tools", ".py"), ("configs", ".yaml")):
        for path in sorted((ROOT / directory).rglob("*" + suffix)):
            digest.update(path.relative_to(ROOT).as_posix().encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def verify_cached_sample(path, binding, root, sample_id):
    row = json.loads(path.read_text())
    if row.get("complete") is not True or row.get("binding") != binding or row.get("sample_id") != sample_id:
        raise RuntimeError(f"Saved sample provenance changed: {path}")
    required = {f"pytorch/data/{sample_id}.txt", f"coreml/data/{sample_id}.txt", f"candidates/{sample_id}.npz"}
    if set(row.get("files", {})) != required:
        raise RuntimeError(f"Saved sample artifact list changed: {path}")
    for name, expected in row["files"].items():
        if sha256_file(root / name) != expected:
            raise RuntimeError(f"Saved sample file changed: {name}")
    return row


def exact_prediction_set(directory, ids):
    if {path.stem for path in directory.glob("*.txt")} != set(ids):
        raise RuntimeError(f"Prediction IDs are missing or extra: {directory}")


def evaluate_predictions(side, dataset, output, trace):
    prediction_dir = output / side / "data"
    base = output / side
    common = ["--dataset-root", dataset, "--prediction-dir", prediction_dir]
    run_logged([sys.executable, "-u", ROOT / "scripts/evaluate_kitti_prediction_dir.py",
                "--config", ROOT / "configs/kitti_mobileadas3d_s1.yaml", "--profile", "colab_drive",
                *common, "--split-dir", dataset / "ImageSets", "--split", "val",
                "--classes", "Vehicle", "Pedestrian", "--source-name", "M59i_" + side,
                "--output-dir", base / "product_ap"], ROOT, base / "logs/ap.log")
    # These legacy diagnostics call this argument 'checkpoint'. Here it is the
    # hash-pinned TorchScript inference artifact, not an invented .pth checkpoint.
    diagnostic = [*common, "--split-file", dataset / "ImageSets/val.txt", "--checkpoint", trace,
                  "--expected-checkpoint-sha256", TRACE_SHA256, "--expected-images", "3769", "--score-threshold", "0.001"]
    run_logged([sys.executable, "-u", ROOT / "scripts/audit_product_prediction_geometry.py", *diagnostic,
                "--match-iou-threshold", "0.5", "--output-dir", base / "nearby_geometry"], ROOT, base / "logs/nearby.log")
    run_logged([sys.executable, "-u", ROOT / "scripts/diagnose_a2_pedestrian_false_negatives.py", *diagnostic,
                "--iou-threshold", "0.5", "--weak-iou-threshold", "0.1", "--output-dir", base / "pedestrian_misses"], ROOT, base / "logs/misses.log")
    ap = json.loads((base / "product_ap/kitti_r40_summary.json").read_text())
    nearby = json.loads((base / "nearby_geometry/nearby_geometry_summary.json").read_text())
    misses = json.loads((base / "pedestrian_misses/a2_pedestrian_false_negative_summary.json").read_text())
    if not ap["complete_split"] or ap["evaluated_images"] != 3769:
        raise RuntimeError("AP evaluation did not cover the complete split")
    vehicle, pedestrian = metric_value(ap, "3d", "Vehicle"), metric_value(ap, "3d", "Pedestrian")
    metrics = {"vehicle_3d_moderate": vehicle, "pedestrian_3d_moderate": pedestrian,
               "mean_3d_moderate": (vehicle + pedestrian) / 2,
               "vehicle_bev_moderate": metric_value(ap, "bev", "Vehicle"),
               "pedestrian_bev_moderate": metric_value(ap, "bev", "Pedestrian"),
               "vehicle_near_recall": float(nearby["classes"]["Vehicle"]["near_recall"]),
               "pedestrian_near_recall": float(nearby["classes"]["Pedestrian"]["near_recall"]),
               "pedestrian_localization_failure_rate": float(misses["near_failure_rates"]["localization_failure"])}
    if not all(np.isfinite(x) for x in metrics.values()):
        raise RuntimeError("Nonfinite accuracy metric")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-bundle", "artifact-dir", "m58-dir", "upstream-repo"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tensor-input-dir", type=Path,
                        help="Source-bound, bit-exact Linux input bundle; never approximate inputs")
    parser.add_argument("--reviewed-input-archive", type=Path,
                        help="Original M59g ZIP for independent fixed16 verification")
    parser.add_argument("--check-policy-only", action="store_true",
                        help="Apply the approved policy to hash-pinned existing M59h evidence; no model execution")
    args = parser.parse_args()
    if (args.tensor_input_dir is None) != (args.reviewed_input_archive is None):
        parser.error("--tensor-input-dir and --reviewed-input-archive must be supplied together")
    policy = load_policy()
    reviewed = approved_diagnostic_report(policy)
    if args.check_policy_only:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        path = args.output_dir / "m59i_approved_diagnostic_gate.json"
        write_json(path, reviewed)
        print(json.dumps(reviewed, indent=2))
        if not reviewed["approved_diagnostic_gate_passed"]:
            raise SystemExit("Approved diagnostic gate failed")
        return
    if any(getattr(args, name) is None for name in ("dataset_bundle", "artifact_dir", "m58_dir", "upstream_repo")):
        parser.error("Full validation requires --dataset-bundle, --artifact-dir, --m58-dir and --upstream-repo")
    if platform.system() != "Darwin":
        raise RuntimeError("Core ML validation must run on macOS, not Colab")
    if not reviewed["approved_diagnostic_gate_passed"]:
        raise RuntimeError("Approved diagnostic policy did not pass; stop for review")
    dataset = args.dataset_bundle.resolve()
    manifest = verify_bundle(dataset)
    ids = manifest["sample_ids"]
    trace = args.m58_dir / "MonoDGP_M58_fixed_fp32.pt"
    anchor_file = args.m58_dir / "m58_reference_io.npz"
    package = args.artifact_dir / "MonoDGP_M59f_fp32.mlpackage"
    if sha256_file(trace) != TRACE_SHA256 or sha256_file(anchor_file) != ANCHOR_SHA256 or tree_sha256(package) != PACKAGE_SHA256:
        raise RuntimeError("Reviewed trace, anchor or Core ML package changed")
    if sha256_file(ROOT / "configs/monodgp_m56d_runtime_frozen.yaml") != CONFIG_SHA256:
        raise RuntimeError("Frozen inference config changed")
    native_decode, extract, decoder_hashes = upstream_decoder(args.upstream_repo)
    tensor_manifest = None
    tensor_rows = {}
    if args.tensor_input_dir is not None:
        print("Verifying complete Linux tensor bundle before inference...", flush=True)
        tensor_manifest = verify_tensor_bundle(args.tensor_input_dir, dataset, manifest,
                                              args.reviewed_input_archive, anchor_file)
        tensor_rows = {row["sample_id"]: row for row in tensor_manifest["samples"]}
    def inputs_for(sample_id):
        if tensor_manifest is not None:
            return load_tensor(args.tensor_input_dir, tensor_rows[sample_id])
        return preprocess(dataset / "training/image_2" / (sample_id + ".png"),
                          dataset / "training/calib" / (sample_id + ".txt"))
    with np.load(anchor_file, allow_pickle=False) as anchor:
        require_anchor_match(inputs_for("000001"), anchor)
    import torch
    import coremltools as ct
    import PIL
    import cv2
    software = {"python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
                "numpy": np.__version__, "coremltools": ct.__version__, "pillow": PIL.__version__, "opencv": cv2.__version__}
    binding = {"policy_sha256": POLICY_SHA256, "source_torchscript_sha256": TRACE_SHA256,
               "mlpackage_tree_sha256": PACKAGE_SHA256, "dataset_manifest_sha256": sha256_file(dataset / "m59i_dataset_manifest.json"),
               "runtime_config_sha256": CONFIG_SHA256, "upstream_decoder_hashes": decoder_hashes,
               "code_fingerprint": code_fingerprint(), "compute_units": "ALL", "software": software}
    if tensor_manifest is not None:
        binding["input_source"] = "verified_linux_tensor_bundle"
        binding["tensor_manifest_sha256"] = sha256_file(args.tensor_input_dir / "m59i_tensor_manifest.json")
        binding["preprocessing_binding"] = tensor_manifest["binding"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    lock = output / "m59i_run_binding.json"
    if lock.exists() and json.loads(lock.read_text()) != binding:
        raise RuntimeError("Run binding changed. Do not mix runs or overwrite previous results")
    if not lock.exists() and any(output.iterdir()):
        raise RuntimeError("Nonempty output without run binding; use a dedicated output directory")
    write_json(lock, binding)
    write_json(output / "m59i_approved_diagnostic_gate.json", reviewed)
    report_path = output / "m59i_full_validation_gate.json"
    report = {"schema_version": 1, "complete": False, "binding": binding, "training_performed": False,
              "weights_changed": False, "prediction_format_changed": False, "historical_m59g_gate_passed": False,
              "product_safety_qualified": False, "deployment_authorized": False, "samples_completed": 0}
    write_json(report_path, report)  # A prior finished run cannot remain green during a failed rerun.
    samples = []
    try:
        for folder in ("pytorch/data", "coreml/data", "samples", "candidates"):
            (output / folder).mkdir(parents=True, exist_ok=True)
        torch.set_num_threads(4)
        original = None
        model = None
        fixed_ids = set(reviewed_samples())
        for number, sample_id in enumerate(ids, 1):
            sample_path = output / "samples" / (sample_id + ".json")
            if sample_path.exists():
                row = verify_cached_sample(sample_path, binding, output, sample_id)
            else:
                if original is None:
                    original = torch.jit.load(str(trace), map_location="cpu").eval()
                    model = ct.models.MLModel(str(package), compute_units=ct.ComputeUnit.ALL)
                inputs = inputs_for(sample_id)
                with torch.inference_mode():
                    values = original(*(torch.from_numpy(inputs[name]) for name in INPUT_SHAPES))
                reference = {name: value.numpy().copy() for name, value in zip(OUTPUT_NAMES, values)}
                actual = model.predict(inputs)
                validate_outputs(reference); validate_outputs(actual)
                expected, expected_ids = extract(reference)
                observed, observed_ids = extract(actual)
                ref_rows, _ = geometry_rows(expected, inputs)
                actual_rows, _ = geometry_rows(observed, inputs)
                for candidates, rows in ((expected, ref_rows), (observed, actual_rows)):
                    if not np.array_equal(rows, native_decode(candidates, inputs)):
                        raise RuntimeError(f"Decoder parity failed: {sample_id}")
                    if not np.array_equal(rows[filter_masks(candidates, rows)["native_score"]], native_decode(candidates, inputs, .001)):
                        raise RuntimeError(f"Native threshold parity failed: {sample_id}")
                row = {"complete": True, "sample_id": sample_id, "binding": binding, "decoder_port_max_abs_delta": 0.0,
                       "raw_output_parity": compare_tensor_outputs(reference, actual),
                       "same_identity_order": bool(np.array_equal(expected_ids, observed_ids)), "files": {}}
                if row["same_identity_order"]:
                    measured, _, _, _ = compare_geometry(expected, observed, inputs, expected_ids, observed_ids)
                    row.update(measured)
                else:
                    row["geometry_comparison_unavailable"] = "Top-k identities/order changed; ranks are not comparable"
                # Persist native Car/Pedestrian/Cyclist names. Existing evaluators
                # apply the product taxonomy identically to predictions and GT.
                for side, candidates, rows in (("pytorch", expected, ref_rows), ("coreml", observed, actual_rows)):
                    selected = rows[filter_masks(candidates, rows)["native_score"]]
                    text = "\n".join(serialize_rows(selected))
                    name = f"{side}/data/{sample_id}.txt"
                    (output / name).write_text(text + ("\n" if text else ""))
                    row["files"][name] = sha256_file(output / name)
                name = f"candidates/{sample_id}.npz"
                np.savez_compressed(output / name, pytorch=expected, coreml=observed,
                                    pytorch_ids=expected_ids, coreml_ids=observed_ids,
                                    calibration=inputs["calibration"], image_size=inputs["image_size"])
                row["files"][name] = sha256_file(output / name)
                row["numerical_gates"] = numerical_gate(row, policy)
                write_json(sample_path, row)  # Commit the sample only after all its files are saved.
            samples.append(row)
            if number % 25 == 0 or number == len(ids):
                report["samples_completed"] = number
                write_json(report_path, report)
                print(f"M59i paired inference {number}/{len(ids)} (saved samples reused after restart)", flush=True)
        for side in ("pytorch", "coreml"):
            exact_prediction_set(output / side / "data", ids)
        metrics = {side: evaluate_predictions(side, dataset, output, trace) for side in ("pytorch", "coreml")}
        gates = {side: preservation_gate_results(values, len(ids)) for side, values in metrics.items()}
        fixed = {row["sample_id"]: numerical_gate(row, policy) for row in samples if row["sample_id"] in fixed_ids}
        fixed_passed = set(fixed) == fixed_ids and all(all(x.values()) for x in fixed.values())
        raw_passed = all(all(x["passed"] for x in row["raw_output_parity"].values()) for row in samples)
        report.update({"complete": True, "samples_completed": len(samples), "metrics": metrics,
                       "coreml_minus_pytorch": {key: metrics["coreml"][key] - metrics["pytorch"][key] for key in metrics["pytorch"]},
                       "preservation_gates": policy["preservation_gates"], "preservation_gate_results": gates,
                       "fixed16_numerical_gates": fixed, "fixed16_numerical_passed": fixed_passed,
                       "full_raw_output_gates_passed": raw_passed,
                       "full_split_numerical_failed_samples": [row["sample_id"] for row in samples if not all(numerical_gate(row, policy).values())],
                       "full_split_identity_order_changed_samples": [row["sample_id"] for row in samples if not row["same_identity_order"]],
                       "prediction_tree_sha256": {side: prediction_tree_sha256(output / side / "data") for side in metrics},
                       "all_validation_gates_passed": fixed_passed and raw_passed and all(all(x.values()) for x in gates.values()),
                       "numerical_scope": policy["numerical_gate_scope"],
                       "next_action": "Review once; do not tune gates or begin another experiment automatically"})
        print(json.dumps({key: report[key] for key in ("metrics", "coreml_minus_pytorch", "all_validation_gates_passed")}, indent=2), flush=True)
    except Exception as error:
        report.update({"complete": False, "samples_completed": len(samples), "error": str(error)})
        raise
    finally:
        write_json(report_path, report)
    if not report["all_validation_gates_passed"]:
        raise SystemExit(f"M59i failed; stop for review: {report_path}")


if __name__ == "__main__":
    main()
