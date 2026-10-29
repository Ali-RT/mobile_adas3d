"""Development-only V0 error direction audit; no inference, fitting or training.

Recompute the frozen metric, add signed errors at the same aggregation levels,
and join six camera channels from verified original nuScenes metadata. Output
is descriptive: ground-truth object masks, not detection or ultrasonic truth.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common
from scripts import parking_v0_local_io as local
from scripts import parking_v0_v1_metrics as metrics

REVISION = "PARKING-V0-ERROR-AUDIT-2026-10-29-r1"
REVIEWED_RESULT_SHA = "b75bc9e32458e6965080720595f7ead990cca34e806a5ae17f019fb4bd614ee6"
METADATA_SHA = {
    "sample_data": "6dcad49f0b9bd7b1cef04e2a0ff2ac2879b46b938e8d48153f00693cebb4de21",
    "calibrated_sensor": "67781a5dd7b2504b046ef89d6dcb267b12d1cc91af21f1ec5758624588e99865",
    "sensor": "4d5c96570e2d8b09b88ce4c605e40ee43c4909f97bea5741185f18f92eb491ae",
}
CHANNELS = {"CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"}


def verify_result(path, config):
    if common.file_sha(path) != REVIEWED_RESULT_SHA:
        raise ValueError("Use the reviewed completed V0 archive, unchanged")
    _, members = local.read_archive(path)
    index = json.loads(members["checkpoint_inventory.json"])
    registration = json.loads(members["v0_registration.json"])
    local.validate_registration(registration, config)
    run_id = common.identity(registration)
    if (not index["complete"] or index["run_identity"] != run_id
            or index["benchmark_manifest_sha256"] != config["benchmark_manifest_sha256"]
            or set(members) != set(index["files"]) | {"checkpoint_inventory.json"}
            or any(common.sha(members[n]) != digest for n, digest in index["files"].items())):
        raise ValueError("Result inventory, identity or bytes differ")
    if any(common.file_sha(ROOT / name) != digest for name, digest in registration["producer_sha256"].items()):
        raise ValueError("Frozen numerical evaluator sources changed")
    with zipfile.ZipFile(io.BytesIO(members["parking_v0_dev_results.zip"])) as archive:
        if (archive.testzip() is not None or len(archive.namelist()) != len(set(archive.namelist()))
                or set(archive.namelist()) != set(index["files"]) - {"parking_v0_dev_results.zip"}
                or any(archive.read(name) != members[name] for name in archive.namelist())):
            raise ValueError("Nested completion result differs")
    result = json.loads(members["v0_dev_results.json"])
    if (not result["complete"] or result["run_identity"] != run_id or result["role"] != "dev"
            or result["training_performed"] or result["optimizer_steps"] != 0
            or result["heldout_images_opened"] != 0 or result["heldout_evaluated"]):
        raise ValueError("Not a completed development-only V0 result")
    return members, registration, result


def camera_channels(samples, metadata):
    """Stream only metadata; never open original camera or held-out GT files."""
    import ijson
    import numpy as np
    for name, digest in METADATA_SHA.items():
        if common.file_sha(metadata / (name + ".json")) != digest:
            raise ValueError("Original metadata bytes differ: " + name)
    sensors = {r["token"]: r for r in json.loads((metadata / "sensor.json").read_text())}
    calibration = {r["token"]: r for r in json.loads((metadata / "calibrated_sensor.json").read_text())}
    wanted = {s["camera_token"]: s for s in samples}
    if len(wanted) != len(samples) or any(s["role"] != "dev" for s in samples):
        raise ValueError("Camera join accepts unique development views only")
    joined = {}; scanned = 0
    with (metadata / "sample_data.json").open("rb") as stream:
        for row in ijson.items(stream, "item", use_float=True):
            scanned += 1
            token = row["token"]
            if token not in wanted:
                continue
            sample = wanted[token]
            cal = calibration[row["calibrated_sensor_token"]]
            sensor = sensors[cal["sensor_token"]]
            if (token in joined or not row["is_key_frame"] or row["sample_token"] != sample["sample_token"]
                    or (row["width"], row["height"]) != (sample["width"], sample["height"])
                    or sensor["modality"] != "camera" or sensor["channel"] not in CHANNELS
                    or not np.array_equal(cal["camera_intrinsic"], sample["intrinsics"])):
                raise ValueError("Camera token, calibration or metadata differs")
            joined[token] = sensor["channel"]
            if len(joined) == len(wanted):
                break
    if set(joined) != set(wanted):
        raise ValueError("Incomplete camera-channel join")
    return joined, scanned


def observation_rows(sample, points, prediction, channel, protocol):
    import numpy as np
    if sample["role"] != "dev" or channel not in CHANNELS:
        raise ValueError("Only development views and known cameras allowed")
    metrics.validate_gt(points, sample, protocol)
    if prediction.shape != points["z"].shape or prediction.dtype != np.float32:
        raise ValueError("Prediction shape/order/type changed")
    valid = np.isfinite(prediction) & (prediction > 0)
    delta = prediction.astype(np.float64) - points["z"].astype(np.float64)
    rows = []
    for cell, mask in zip(metrics.cell_names(protocol), points["cell_masks"]):
        for index in np.unique(points["object_index"][mask]):
            obj = sample["objects"][int(index)]
            reference = mask & (points["object_index"] == index)
            chosen = reference & valid
            errors = delta[chosen]
            count = int(chosen.sum())
            rows.append({"cell": cell, "camera_token": sample["camera_token"], "channel": channel,
                "recording_log": sample["recording_log"], "scene": sample["scene"],
                "instance_token": obj["instance_token"], "annotation_token": obj["annotation_token"],
                "source_class": obj["source_class"], "reference_points": int(reference.sum()),
                "valid_points": count, "farther_points": int((errors > 0).sum()),
                "closer_points": int((errors < 0).sum()),
                "mae_m": float(np.abs(errors).mean()) if count else None,
                "signed_error_m": float(errors.mean()) if count else None,
                "ground_truth_depth_m": float(points["z"][chosen].astype(np.float64).mean()) if count else None,
                "predicted_depth_m": float(prediction[chosen].astype(np.float64).mean()) if count else None})
    return rows


def summarize(rows):
    """Same cameras -> annotations -> tracks -> logs hierarchy as frozen MAE."""
    import numpy as np
    groups = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for row in rows:
        if row["valid_points"]:
            group = groups[row["recording_log"]][row["instance_token"]][row["annotation_token"]]
            if row["camera_token"] in group:
                raise ValueError("Repeated object-camera observation")
            group[row["camera_token"]] = row
    fields = ["mae_m", "signed_error_m", "ground_truth_depth_m", "predicted_depth_m"]
    balanced = {}
    for field in fields:
        logs = [np.mean([np.mean([np.mean([r[field] for r in cameras.values()])
            for cameras in annotations.values()]) for annotations in tracks.values()]) for tracks in groups.values()]
        balanced[field] = float(np.mean(logs)) if logs else None
    total = sum(r["reference_points"] for r in rows)
    count = sum(r["valid_points"] for r in rows)
    return {"log_track_observation_macro": balanced,
        "reference_points": total, "valid_points": count,
        "prediction_coverage": count / total if total else None,
        "point_fraction_farther_than_truth": sum(r["farther_points"] for r in rows) / count if count else None,
        "point_fraction_closer_than_truth": sum(r["closer_points"] for r in rows) / count if count else None,
        "reference_object_camera_observations": len(rows),
        "reference_annotation_observations": len({r["annotation_token"] for r in rows}),
        "reference_instance_tracks": len({r["instance_token"] for r in rows}),
        "reference_recording_logs": len({r["recording_log"] for r in rows})}


def cell_summary(rows, protocol):
    import numpy as np
    cells = {name: summarize([r for r in rows if r["cell"] == name]) for name in metrics.cell_names(protocol)}
    values = [cells[name]["log_track_observation_macro"]["mae_m"] for name in protocol["primary_cells"]]
    return {"conditional_equal_four_cell_mae_m": float(np.mean(values)) if all(v is not None for v in values) else None,
        "cells": cells}


def diagnose(args):
    import numpy as np
    for source in [args.inputs, args.result, args.metadata]:
        if args.output.is_symlink() or args.output.resolve().is_relative_to(source.resolve()):
            raise ValueError("Use a separate new diagnostic output folder")
    config = local.v0.configuration()
    members, registration, result = verify_result(args.result, config)
    inputs, receipt = local.inspect_capsule(args.inputs, config)
    manifest = json.loads(inputs["benchmark/benchmark_manifest.json"])
    inventory = local.v0.dev_inventory(manifest, config)
    samples = [sample for _, rows in inventory for sample in rows]
    channels, scanned = camera_channels(samples, args.metadata)
    print(f"Verified camera metadata for {len(channels)} development views", flush=True)
    expected = {"predictions/" + Path(shard["file"]).stem + ".zip" for shard, _ in inventory}
    if {name for name in members if name.startswith("predictions/")} != expected:
        raise ValueError("Development prediction shard inventory differs")
    accumulator = metrics.Metrics(manifest["metric_protocol"])
    rows = []
    with tempfile.TemporaryDirectory(prefix="parking-v0-errors-") as temporary:
        for i, (shard, batch) in enumerate(inventory, 1):
            path = Path(temporary) / Path(shard["file"]).name
            common.immutable(path, inputs["benchmark/" + shard["file"]])
            images, truth = local.v0.load_dev_shard(path, shard, batch, manifest["metric_protocol"], manifest["benchmark_identity_sha256"])
            predictions = local.v0.read_prediction_chunk(members["predictions/" + path.name], common.identity(registration), batch, truth)
            for sample in batch:
                token = sample["camera_token"]
                accumulator.add(sample, truth[token], predictions[token])
                rows.extend(observation_rows(sample, truth[token], predictions[token], channels[token], manifest["metric_protocol"]))
            del images, truth, predictions
            path.unlink()
            print(f"Verified/diagnosed development shard {i}/{len(inventory)}", flush=True)
    if accumulator.result() != result["metrics"]:
        raise ValueError("Recomputed frozen V0 metric differs")
    overall = cell_summary(rows, manifest["metric_protocol"])
    for name, cell in overall["cells"].items():
        ref = result["metrics"]["cells"][name]
        own = cell["log_track_observation_macro"]["mae_m"]
        expected_mae = ref["conditional_log_track_observation_macro_mae_m"]
        if ((own is None) != (expected_mae is None)
                or (own is not None and not np.isclose(own, expected_mae, rtol=0, atol=1e-12))
                or cell["prediction_coverage"] != ref["prediction_coverage"]
                or cell["reference_instance_tracks"] != ref["reference_instance_tracks"]
                or cell["reference_recording_logs"] != ref["reference_recording_logs"]):
            raise ValueError("Diagnostic weighting differs from the frozen protocol")
    protocol = manifest["metric_protocol"]
    logs = sorted({s["recording_log"] for s in samples})
    diagnostic = {"complete": True, "revision": REVISION,
        "result_archive_sha256": REVIEWED_RESULT_SHA, "input_capsule": receipt,
        "run_identity": result["run_identity"], "model_checkpoint_sha256": config["checkpoint_sha256"],
        "metadata_sha256": METADATA_SHA, "metadata_records_scanned": scanned,
        "images": len(samples), "role": "dev", "sign_convention": "prediction minus GT; positive means too far",
        "overall": overall,
        "by_camera": {channel: {"images": sum(c == channel for c in channels.values()),
            **cell_summary([r for r in rows if r["channel"] == channel], protocol)} for channel in sorted(CHANNELS)},
        "by_recording_log": {log: {"images": sum(s["recording_log"] == log for s in samples),
            **cell_summary([r for r in rows if r["recording_log"] == log], protocol)} for log in logs},
        "by_source_class": {cls: cell_summary([r for r in rows if r["source_class"] == cls], protocol)
            for cls in sorted({r["source_class"] for r in rows})},
        "leave_one_log_out_sensitivity": {log: cell_summary([r for r in rows if r["recording_log"] != log], protocol)["conditional_equal_four_cell_mae_m"] for log in logs},
        "limitations": ["Descriptive development-only diagnostics, not independent held-out evidence or significance.",
            "Camera and log subsets contain different objects/ranges; differences are not causal camera effects.",
            "Pointwise farther/closer fractions are not the log-balanced signed error and are not detection recall.",
            "Positive camera-axis Z, not radial or ultrasonic nearest-surface distance; no fitted correction.",
            "Overlapping bands reuse points and objects; do not sum their counts.",
            "Sparse 0-1m support cannot establish reliable performance."],
        "frozen_metrics_reproduced_exactly": True, "model_inference_performed": False,
        "scale_or_shift_fitted": False, "training_performed": False, "optimizer_steps": 0,
        "heldout_image_or_gt_bodies_opened": 0, "phone_app_changed": False}
    common.immutable(args.output / "v0_error_diagnostic.json", common.encoded(diagnostic))
    common.immutable(args.output / "object_camera_errors.json", common.encoded({"revision": REVISION, "rows": rows}))
    common.immutable(args.output / "producer.json", common.encoded({"script_sha256": common.file_sha(Path(__file__)),
        "frozen_sources": registration["producer_sha256"], "metadata_sha256": METADATA_SHA}))
    print(json.dumps({"complete": True, "images": len(samples), "primary_mae_m": overall["conditional_equal_four_cell_mae_m"],
        "output": str(args.output)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ["result", "inputs", "metadata", "output"]:
        parser.add_argument("--" + name, type=Path, required=True)
    diagnose(parser.parse_args())


if __name__ == "__main__":
    main()
