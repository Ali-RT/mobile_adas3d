"""Review downloaded M60 device outputs, preserving the M59i fixed16 policy."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.prepare_monodgp_m60_device_bundle import read_tensor, parity_row, M59I_REPORT_SHA256
from scripts.evaluate_monodgp_m59i_coreml import PACKAGE_SHA256, load_policy
from scripts.collect_monodgp_m59i_validation import reviewed_samples, write_json
from scripts.validate_monodgp_m58_macos_parity import sha256_file, OUTPUT_SHAPES, INPUT_SHAPES
from scripts.diagnose_monodgp_m59h_geometry import upstream_decoder


def timing_stats(values):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Missing, nonfinite or nonpositive timings")
    return {"count": int(values.size), **{f"p{p}_ms": float(np.percentile(values, p)) for p in (50, 90, 95, 99)},
            "max_ms": float(values.max()), "mean_ms": float(values.mean())}


def verify_protocol(manifest, run, manifest_sha256):
    ids = list(reviewed_samples())
    if (manifest.get("complete") is not True or manifest["mlpackage_tree_sha256"] != PACKAGE_SHA256
            or manifest["m59i_report_sha256"] != M59I_REPORT_SHA256 or manifest["policy"] != load_policy()
            or manifest["sample_ids"] != ids
            or [row["sample_id"] for row in manifest["samples"]] != ids
            or (manifest["warmups"], manifest["timed_predictions"], manifest["sustain_seconds"]) != (5, 100, 60)
            or run["manifest_sha256"] != manifest_sha256
            or run["compute_units"] != "ALL" or run.get("physical_device") is not True):
        raise RuntimeError("Device fixture/protocol provenance failed")
    if [row["sample_id"] for row in run["samples"]] != ids:
        raise RuntimeError("Incomplete/extra/duplicate device samples")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--device-run", required=True, type=Path)
    parser.add_argument("--upstream-repo", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {"schema_version": 1, "complete": False, "device_parity_passed": False,
              "model_only_latency_passed": False, "deployment_authorized": False,
              "end_to_end_qualified": False, "product_pedestrian_recall_target_met": False}
    try:
        manifest = json.loads((args.bundle / "manifest.json").read_text())
        run = json.loads((args.device_run / "report.json").read_text())
        verify_protocol(manifest, run, sha256_file(args.bundle / "manifest.json"))
        decode, extract, hashes = upstream_decoder(args.upstream_repo)
        if hashes != manifest["upstream_decoder_hashes"]:
            raise RuntimeError("Decoder provenance changed")
        samples = []
        for fixture, device in zip(manifest["samples"], run["samples"]):
            inputs = {name: read_tensor(args.bundle, fixture["inputs"][name]) for name in INPUT_SHAPES}
            if set(device["outputs"]) != set(OUTPUT_SHAPES):
                raise RuntimeError("Unexpected device output set")
            actual = {name: read_tensor(args.device_run, device["outputs"][name]) for name in OUTPUT_SHAPES}
            if any(list(actual[name].shape) != shape for name, shape in OUTPUT_SHAPES.items()):
                raise RuntimeError("Device output shape changed")
            row = {"sample_id": fixture["sample_id"]}
            for side in ("pytorch_outputs", "mac_outputs"):
                reference = {name: read_tensor(args.bundle, fixture[side][name]) for name in OUTPUT_SHAPES}
                row[side] = parity_row(reference, actual, inputs, extract, decode)
            samples.append(row)
        report["samples"] = samples
        report["device_parity_passed"] = all(row[side]["passed"] for row in samples for side in ("pytorch_outputs", "mac_outputs"))
        report["device_report"] = run
        # A raw-parity stop still produces a useful failed numerical report;
        # never fabricate latency or turn missing timing data into a pass.
        if run.get("complete") is True:
            if (run.get("warmups_completed") != manifest["warmups"]
                    or len(run["prediction_ms"]) != manifest["timed_predictions"]
                    or run["sustain_elapsed_seconds"] < manifest["sustain_seconds"]):
                raise RuntimeError("Incomplete timed/stability protocol")
            report["model_only"] = timing_stats(run["prediction_ms"])
            report["sustained_model_only"] = timing_stats(run["sustain_prediction_ms"])
            report["model_only_latency_passed"] = all(report[key]["p95_ms"] <= 50 for key in ("model_only", "sustained_model_only"))
            report["complete"] = True
        report["next_step"] = "Review device parity, speed, memory and thermal evidence; no automatic camera/quantization approval"
    except Exception as error:
        report["error"] = str(error)
        raise
    finally:
        write_json(args.output, report)
    print(json.dumps({key: value for key, value in report.items() if key not in ("samples", "device_report")}, indent=2))
    if not report["complete"] or not report["device_parity_passed"] or not report["model_only_latency_passed"]:
        raise SystemExit("M60 feasibility not passed; inspect report before proceeding")


if __name__ == "__main__":
    main()
