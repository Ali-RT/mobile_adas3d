"""Collect epoch-3 camera-pilot checkpoints and inspect them on CPU.

No images, dataset staging, forward inference, optimizer steps, environment
restoration or phone changes. Weight movement is not proof of a useful feature
or the cause of the observed development regression.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path, PurePosixPath
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common

RESULT_SHA = "30df38ae68638ebc00c4f42992d3a5f9c5fba329d85cbd9c7f970ab972ddc3b0"
RUN_ID = "7be069024710debba498a64a844d30cdebbe84fe6be3e3fc86cf5f549ce351ac"
ARMS = ("reference_camera", "actual_camera")
EPOCH = 3
RESULT_NAME = "parking_shared_camera_results.zip"


def safe_archive(raw, maximum=10_000_000):
    archive = zipfile.ZipFile(io.BytesIO(raw))
    names = archive.namelist()
    if len(names) != len(set(names)) or sum(i.file_size for i in archive.infolist()) > maximum:
        archive.close(); raise ValueError("Unexpected archive inventory or size")
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name:
            archive.close(); raise ValueError("Unsafe archive member")
    return archive


def reviewed_result(raw):
    if common.sha(raw) != RESULT_SHA:
        raise ValueError("Use the unchanged reviewed camera-pilot results ZIP")
    with safe_archive(raw, 2_000_000) as z:
        read = lambda name: json.loads(z.read(name))
        identity = read("run_identity.json"); result = read("camera_pilot_results.json")
        if common.identity(identity) != RUN_ID or result["run_identity"] != RUN_ID or not result["complete"]:
            raise ValueError("Camera-pilot identity mismatch")
        for name, digest in identity["source_sha256"].items():
            if common.sha(z.read("source_snapshot/" + name)) != digest:
                raise ValueError("Reviewed producer snapshot changed")
        selections = {}; initial = {}
        for arm in ARMS:
            s = read(f"{arm}/selection.json")
            arm_id = common.identity({"run_identity": RUN_ID, "arm": arm})
            if (s["run_identity"] != arm_id or s["parent_run_identity"] != RUN_ID
                    or s["optimizer_steps"] != 960 or s["selected_epoch"] != 0
                    or s["heldout_used_for_selection"] or [h["epoch"] for h in s["history"]] != list(range(6))
                    or [h["optimizer_steps"] for h in s["history"]] != [192*i for i in range(6)]):
                raise ValueError("Reviewed five-epoch history changed")
            stem = f"{arm}/checkpoints/epoch_000"
            checkpoint = z.read(stem + ".pth")
            if common.sha(checkpoint) != read(stem + ".json")["sha256"]:
                raise ValueError("Initial tensor bytes changed")
            selections[arm], initial[arm] = s, checkpoint
    return identity, result, selections, initial


def archive_bytes(members):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, raw in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, raw)
    return buffer.getvalue()


def collect(run, output):
    """Standard library only. Preserve the completed run; write elsewhere."""
    run, output = Path(run), Path(output)
    if output.is_symlink() or output.resolve().is_relative_to(run.resolve()):
        raise ValueError("Use a separate audit output folder; preserve the training run")
    result_path = run / RESULT_NAME
    if result_path.is_symlink():
        raise ValueError("Preserve symlinked inputs")
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing completed results ZIP: {result_path}. Use the original reviewed ZIP; do not rerun training.")
    result_raw = result_path.read_bytes()
    reviewed_result(result_raw)
    members = {RESULT_NAME: result_raw}
    for arm in ARMS:
        stem = f"{arm}/checkpoints/epoch_{EPOCH:03d}"
        for suffix in [".pth", ".json"]:
            name = stem + suffix; path = run / name
            if path.is_symlink() or not path.resolve().is_relative_to(run.resolve()):
                raise ValueError("Checkpoint path escapes completed run")
            if not path.is_file():
                raise FileNotFoundError(f"Missing trained checkpoint: {path}. Do not rerun training; preserve the completed run.")
            members[name] = path.read_bytes()
        if common.sha(members[stem + ".pth"]) != json.loads(members[stem + ".json"])["sha256"]:
            raise ValueError(f"Trained checkpoint bytes differ from sidecar: {arm}")
    receipt = {"revision": "PARKING-CAMERA-WEIGHT-AUDIT-2026-10-27-r1", "run_identity": RUN_ID,
        "epoch": EPOCH, "members_sha256": {name: common.sha(raw) for name, raw in members.items()},
        "training_performed": False, "optimizer_steps": 0, "images_accessed": False, "phone_app_changed": False}
    members["collection_identity.json"] = common.encoded(receipt)
    path = output / "parking_shared_camera_epoch3_inputs.zip"
    common.immutable(path, archive_bytes(members))
    print(f"CPU collection complete; no training: {path}", flush=True)
    return path


def load_collection(path):
    with safe_archive(Path(path).read_bytes()) as z:
        receipt = json.loads(z.read("collection_identity.json"))
        expected = {RESULT_NAME} | {f"{a}/checkpoints/epoch_{EPOCH:03d}{s}" for a in ARMS for s in [".pth", ".json"]}
        if (set(z.namelist()) != expected | {"collection_identity.json"}
                or set(receipt["members_sha256"]) != expected or receipt["run_identity"] != RUN_ID
                or receipt["epoch"] != EPOCH):
            raise ValueError("Unexpected collection identity")
        members = {name: z.read(name) for name in expected}
        if any(common.sha(raw) != receipt["members_sha256"][name] for name, raw in members.items()):
            raise ValueError("Collection payload changed")
    identity, result, selections, initial = reviewed_result(members[RESULT_NAME])
    for arm in ARMS:
        stem = f"{arm}/checkpoints/epoch_{EPOCH:03d}"
        if common.sha(members[stem + ".pth"]) != json.loads(members[stem + ".json"])["sha256"]:
            raise ValueError("Collected checkpoint sidecar changed")
    return identity, result, selections, initial, members


def tensor_delta(reference, current, names):
    import torch
    names = sorted(names)
    values = changed = tensors = 0; square = reference_square = maximum = 0.
    changed_names = []
    for name in names:
        a, b = reference[name].detach().cpu(), current[name].detach().cpu()
        if a.shape != b.shape or a.dtype != b.dtype or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError(f"Invalid checkpoint tensor: {name}")
        values += a.numel(); tensors += 1
        if not torch.equal(a, b):
            changed += 1; changed_names.append(name)
        delta = b.double() - a.double()
        square += float(delta.square().sum()); reference_square += float(a.double().square().sum())
        maximum = max(maximum, float(delta.abs().max()))
    return {"tensors": tensors, "values": values, "changed_tensors": changed,
        "delta_l2": square**.5, "relative_delta_l2": (square/reference_square)**.5 if reference_square else None,
        "max_abs_delta": maximum, "changed_tensor_names": changed_names}


def state_groups(model):
    parameters = dict(model.named_parameters()); buffers = dict(model.named_buffers())
    return {"existing_depth_parameters": {n for n in parameters if n.startswith("base.")},
        "adapter_parameters": {n for n in parameters if not n.startswith("base.")},
        "existing_depth_bn_running_statistics": {n for n in buffers if n.endswith(("running_mean", "running_var"))},
        "existing_depth_bn_batch_counters": {n for n in buffers if n.endswith("num_batches_tracked")}}


def audit(bundle, output):
    """Safe CPU tensor comparison only; does not run the model forward."""
    identity, result, selections, initial_raw, members = load_collection(bundle)
    for name in ["models/parking_shared.py", "models/parking_shared_camera.py"]:
        if common.file_sha(ROOT/name) != identity["source_sha256"][name]:
            raise ValueError("Local decoder architecture differs from reviewed run")
    # All archive hashes and the reviewed original result are checked first.
    import torch
    from models.parking_shared import ParkingDepthDecoder
    from models.parking_shared_camera import CameraAwareDepthDecoder
    from scripts.screen_parking_shared_model import state_digest
    states = {}; initial_states = {}; arms = {}
    model = CameraAwareDepthDecoder(ParkingDepthDecoder())
    groups = state_groups(model)
    for arm in ARMS:
        arm_id = common.identity({"run_identity": RUN_ID, "arm": arm})
        before = torch.load(io.BytesIO(initial_raw[arm]), map_location="cpu", weights_only=True)
        after = torch.load(io.BytesIO(members[f"{arm}/checkpoints/epoch_{EPOCH:03d}.pth"]), map_location="cpu", weights_only=True)
        if (before["run_identity"] != arm_id or before["arm"] != arm or before["epoch"] != 0 or before["optimizer_steps"] != 0
                or after["run_identity"] != arm_id or after["arm"] != arm or after["epoch"] != EPOCH
                or after["optimizer_steps"] != 192*EPOCH or after["history"] != selections[arm]["history"][:EPOCH+1]):
            raise ValueError("Checkpoint metadata/history does not match reviewed run")
        for checkpoint in [before, after]:
            model.load_state_dict(checkpoint["decoder"], strict=True)
            for name, value in model.state_dict().items():
                if not torch.isfinite(value).all() or (name.endswith("running_var") and (value < 0).any()):
                    raise ValueError("Non-finite tensors or negative BN variance")
            if checkpoint is before and state_digest(model) != identity["initial_adapter_state_sha256"]:
                raise ValueError("Initial adapter state identity differs")
        a, b = before["decoder"], after["decoder"]
        initial_states[arm], states[arm] = a, b
        deltas = {key: tensor_delta(a, b, names) for key, names in groups.items()}
        counters = {name: int(b[name])-int(a[name]) for name in groups["existing_depth_bn_batch_counters"]}
        residual_l2 = float(sum(b[n].double().square().sum() for n in ["residual.weight", "residual.bias"]).sqrt())
        arms[arm] = {"epoch": EPOCH, "training_updates_in_checkpoint": after["optimizer_steps"],
            "recorded_dev": selections[arm]["history"][EPOCH]["dev"], "state_changes": deltas,
            "bn_batch_counter_increments": counters, "residual_output_parameter_l2": residual_l2,
            "adapter_weights_moved": deltas["adapter_parameters"]["changed_tensors"] > 0,
            "existing_depth_weights_moved": deltas["existing_depth_parameters"]["changed_tensors"] > 0,
            "existing_depth_bn_statistics_moved": deltas["existing_depth_bn_running_statistics"]["changed_tensors"] > 0,
            "checkpoint_state_sha256": state_digest(model)}
    if any(not torch.equal(initial_states[ARMS[0]][n], initial_states[ARMS[1]][n]) for n in initial_states[ARMS[0]]):
        raise ValueError("Arms did not share the exact initial tensor state")
    cross = {key: tensor_delta(states[ARMS[0]], states[ARMS[1]], names) for key, names in groups.items()}
    report = {"complete": True, "revision": "PARKING-CAMERA-WEIGHT-AUDIT-2026-10-27-r1", "run_identity": RUN_ID,
        "collection_sha256": common.file_sha(bundle), "reviewed_result_sha256": RESULT_SHA,
        "source_sha256": {str(p.relative_to(ROOT)): common.file_sha(p) for p in
            [Path(__file__), ROOT/"models/parking_shared_camera.py", ROOT/"models/parking_shared.py"]},
        "initial_dev": result["initial_dev"], "arms": arms, "actual_vs_reference_state_delta": cross,
        "interpretation": "Describes tensor movement only. BN drift and adapter updates are not proof of the cause of regression or useful camera conditioning. A fixed-data forward counterfactual is needed to separate their effects.",
        "forward_inference_performed": False, "images_accessed": False, "ground_truth_accessed": False,
        "training_performed": False, "optimizer_steps": 0, "model_weights_updated": False,
        "phone_app_changed": False, "deployment_qualified": False}
    common.immutable(Path(output)/"camera_weight_audit.json", common.encoded(report))
    print(json.dumps(report, indent=2), flush=True)
    return report


def bundle_outputs(inputs, output):
    output = Path(output)
    report = json.loads((output/"camera_weight_audit.json").read_text())
    if not report["complete"] or report["collection_sha256"] != common.file_sha(inputs):
        raise ValueError("Need the matching completed weight audit")
    path = output/"parking_shared_camera_weight_audit_results.zip"
    common.immutable(path, archive_bytes({"camera_weight_audit.json": (output/"camera_weight_audit.json").read_bytes(),
        "parking_shared_camera_epoch3_inputs.zip": Path(inputs).read_bytes()}))
    print(path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["collect", "audit", "bundle"])
    p.add_argument("--run", type=Path); p.add_argument("--inputs", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.stage == "collect":
        if a.run is None: p.error("collect requires --run")
        collect(a.run, a.output)
    else:
        if a.inputs is None: p.error("audit/bundle requires --inputs")
        (audit if a.stage == "audit" else bundle_outputs)(a.inputs, a.output)


if __name__ == "__main__":
    main()
