"""Matched five-epoch fixed-camera versus actual-camera depth-head pilot.

Reuse the reviewed train/dev package and epoch-7 checkpoint. No held-out
evaluation, teacher, automatic export, app replacement or split changes.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ThreadPoolExecutor
import inspect
import io
import json
from pathlib import Path, PurePosixPath
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common
from scripts import parking_shared_near_data as data
from scripts import train_parking_shared_depth as pilot
from scripts import train_parking_shared_near as near
from scripts import parking_shared_body_verification as body_verification

CONFIG = ROOT / "configs/parking_shared_camera_pilot_v1.json"
BUNDLE_SHA = "8239f27a75ba6e06ed8865f3d964426503c61414661cb45b734299769038cd45"
TRAINING_ID = "73549e5976501e20ad8ffcddb1f8fdbf915a1c2ab187392a07cc8590e0ad9897"
CHECKPOINT_SHA = "31bd01a9c1a3383e16bb83e807260ddd534d366ad39fda6d4befc54924ed1214"
INPUT_SHA = "5bc99105a295ef80f4f2c8ed4c3ff71879f993b5cc2afaf54162e0ca907bc85d"
SOURCES = [Path(__file__), CONFIG, ROOT / "models/parking_shared_camera.py",
    ROOT / "models/parking_shared.py", Path(pilot.__file__), Path(near.__file__),
    Path(data.__file__), Path(body_verification.__file__),
    ROOT / "scripts/parking_shared_depth_data.py", ROOT / "scripts/screen_parking_shared_model.py",
    ROOT / "scripts/parking_edge_common.py"]


def configuration():
    return json.loads(CONFIG.read_text())


def read_warmstart(path, load=False):
    """Pin archive/source/selection/weights before any safe tensor loading."""
    if common.file_sha(path) != BUNDLE_SHA:
        raise ValueError("Use the reviewed parking_shared_near_results.zip, unchanged")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if len(names) != len(set(names)) or sum(i.file_size for i in z.infolist()) > 10_000_000:
            raise ValueError("Unexpected warm-start inventory")
        for info in z.infolist():
            p = PurePosixPath(info.filename)
            if p.is_absolute() or ".." in p.parts or "\\" in info.filename or info.file_size > 3_000_000:
                raise ValueError("Unsafe warm-start archive")
        read = lambda name: json.loads(z.read(name))
        identity = read("run_identity.json")
        result = read("near_training_results.json")
        if (common.identity(identity) != TRAINING_ID or not result["complete"]
                or result["run_identity"] != TRAINING_ID or identity["input_manifest_sha256"] != INPUT_SHA):
            raise ValueError("Warm-start training/input identity changed")
        for name, digest in identity["source_sha256"].items():
            local = ROOT / ("models" if name == "parking_shared.py" else "scripts") / name
            if common.file_sha(local) != digest:
                raise ValueError(f"Reviewed warm-start source changed: {name}")
        for name in names:
            if name.startswith("source_snapshot/") and common.sha(z.read(name)) != identity["source_sha256"][PurePosixPath(name).name]:
                raise ValueError("Warm-start snapshot changed")
        selection = read("parking_balanced/selection.json")
        arm_id = common.identity({"run_identity": TRAINING_ID, "arm": "parking_balanced"})
        best = min(selection["history"], key=lambda r: (r["dev"]["equal_four_cell_mae_m"], r["epoch"]))
        if (selection["run_identity"] != arm_id or selection["parent_run_identity"] != TRAINING_ID
                or selection["selected_epoch"] != 7 or best["epoch"] != 7 or selection["heldout_used_for_selection"]):
            raise ValueError("Use the frozen development-selected epoch 7")
        stem = "parking_balanced/checkpoints/epoch_007"
        raw = z.read(stem + ".pth")
        if common.sha(raw) != CHECKPOINT_SHA or read(stem + ".json")["sha256"] != CHECKPOINT_SHA:
            raise ValueError("Warm-start tensor bytes changed")
    if not load:
        return identity, selection, None
    import torch
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if saved["run_identity"] != arm_id or saved["arm"] != "parking_balanced" or saved["epoch"] != 7:
        raise ValueError("Warm-start checkpoint metadata changed")
    return identity, selection, saved


def camera_vector(sample):
    """Pinhole intrinsics after the existing OpenCV resize and letterbox."""
    import numpy as np
    k = np.asarray(sample["intrinsics"], dtype=np.float64)
    w, h = sample["width"], sample["height"]
    if (k.shape != (3, 3) or not np.isfinite(k).all() or w <= 0 or h <= 0
            or not np.allclose(k[2], [0, 0, 1], atol=1e-9, rtol=0)
            or abs(k[0, 1]) > 1e-9 or abs(k[1, 0]) > 1e-9 or min(k[0, 0], k[1, 1]) <= 0):
        raise ValueError("Expected finite, positive-focal, zero-skew pinhole calibration")
    gain = min(640 / w, 384 / h)
    nw, nh = round(w * gain), round(h * gain)
    if min(nw, nh) < 1:
        raise ValueError("Invalid resized image")
    sx, sy = nw / w, nh / h
    left, top = (640 - nw) // 2, (384 - nh) // 2
    return np.asarray([sx*k[0, 0], sy*k[1, 1], sx*(k[0, 2]+.5)+left-.5,
        sy*(k[1, 2]+.5)+top-.5], dtype=np.float32)


def camera_batch(samples, arm, device):
    import numpy as np
    import torch
    config = configuration()
    if arm not in config["arms"]:
        raise ValueError("Unknown camera arm")
    vectors = [camera_vector(s) for s in samples]
    if arm == "reference_camera":
        vectors = [np.asarray(config["reference_camera"], np.float32) for _ in samples]
    return torch.from_numpy(np.stack(vectors)).to(device)


def prepare(a):
    manifest = data.read_manifest(a.inputs)
    identity, selection, _ = read_warmstart(a.warmstart)
    if common.file_sha(a.inputs / "input_manifest.json") != INPUT_SHA:
        raise ValueError("Use the unchanged reviewed train/dev input package")
    if identity["config"] != manifest["config"]:
        raise ValueError("Training recipe/data provenance mismatch")
    for sample in manifest["samples"]:
        camera_vector(sample)
    if a.output.is_symlink() or any(a.output.resolve().is_relative_to(p.resolve()) for p in [a.inputs, a.source, a.weights, a.warmstart.parent]):
        raise ValueError("Keep the pilot output separate from immutable inputs and source")
    receipt = {"complete": True, "input_manifest_sha256": INPUT_SHA, "warmstart_bundle_sha256": BUNDLE_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA, "initial_epoch": 7, "config": configuration(),
        "train_images": sum(s["role"] == "train" for s in manifest["samples"]),
        "dev_images": sum(s["role"] == "dev" for s in manifest["samples"]),
        "heldout_inputs": 0, "optimizer_steps": 0, "initial_selected_dev": selection["selected_dev"]}
    common.immutable(a.output / "prepared_inputs.json", common.encoded(receipt))
    print(json.dumps(receipt, indent=2), flush=True)
    return manifest


def cache_identity(manifest_path, model, model_id):
    from models.parking_shared import SharedParkingNet
    from scripts.screen_parking_shared_model import state_digest, preprocess
    return {"input_manifest_sha256": common.file_sha(manifest_path), "body_sha256": state_digest(model.body),
        "runtime": model_id["runtime"], "producer_sha256": common.sha("\n".join(inspect.getsource(f) for f in
            [pilot.cache_sample, near.pilot_data_point_grid(), preprocess, SharedParkingNet.image_features]).encode())}


def matching_cache(path, identity, samples):
    if path is None or not (path / "cache_identity.json").is_file():
        return False
    if path.is_symlink() or json.loads((path / "cache_identity.json").read_text()) != identity:
        return False
    if any(not (path / f"{s['camera_token']}{suffix}").is_file() for s in samples for suffix in [".npz", ".json"]):
        return False
    with ThreadPoolExecutor(max_workers=4) as pool:
        for _ in pool.map(lambda sample: pilot.read_features(sample, path), samples):
            pass
    return True


def masks_for(rows, valid, mapping, config):
    import numpy as np
    import torch
    masks = []
    for key in config["critical_cells"]:
        padded = np.zeros(tuple(valid.shape), bool)
        for i, row in enumerate(rows):
            padded[i, :len(row["z"])] = data.cell_masks(row, mapping)[key]
        masks.append(torch.from_numpy(padded).to(valid.device))
    return masks


def train_step(decoder, optimizer, rows, samples, arm, device, mapping, config):
    import torch
    features, grid, truth, valid = pilot.batch_arrays(rows, device)
    decoder.train(); optimizer.zero_grad(set_to_none=True)
    prediction = pilot.sample_depth(decoder(features, camera_batch(samples, arm, device)), grid)
    loss = near.balanced_loss(prediction, truth, valid, masks_for(rows, valid, mapping, config), config, "parking_balanced")
    if not torch.isfinite(loss):
        raise ValueError("Non-finite camera pilot loss")
    loss.backward()
    params = list(decoder.parameters())
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
        raise ValueError("Invalid decoder/adapter gradients")
    norm = torch.nn.utils.clip_grad_norm_(params, config["gradient_clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in params):
        raise ValueError("Non-finite updated weights")
    return float(loss.detach().cpu()), float(norm.detach().cpu())


def evaluate(decoder, samples, cache, arm, device, mapping, config):
    import numpy as np
    import torch
    if any(s["role"] != "dev" for s in samples):
        raise ValueError("Checkpoint evaluation accepts development samples only")
    errors = {k: [] for k in config["critical_cells"]}
    fixed = {k: [] for k in errors}; points = {k: 0 for k in errors}
    broad = []; bins = {cls: [] for cls in ["Vehicle", "Pedestrian"]}
    decoder.eval()
    with torch.inference_mode():
        for s in samples:
            row = pilot.read_features(s, cache)
            features, grid, _, _ = pilot.batch_arrays([row], device)
            prediction = pilot.sample_depth(decoder(features, camera_batch([s], arm, device)), grid)[0].cpu().numpy()
            if not np.isfinite(prediction).all() or not (prediction > 0).all():
                raise ValueError("Invalid metric depth")
            absolute = np.abs(prediction.astype(np.float64) - row["z"])
            broad.append(float(absolute[row["conservative"]].mean()))
            for key, mask in data.cell_masks(row, mapping).items():
                if int(mask.sum()) >= config["minimum_points_in_cell_image"]:
                    errors[key].append(float(absolute[mask].mean()))
                    fixed[key].append(float(np.abs(4. - row["z"][mask].astype(np.float64)).mean()))
                    points[key] += int(mask.sum())
            for cls in bins:
                for lo in [2, 3, 4, 5]:
                    upper = row["z"] <= 6 if lo == 5 else row["z"] < lo+1
                    keep = row["conservative"] & np.isin(row["semantic"], mapping[cls]) & (row["z"] >= lo) & upper
                    if int(keep.sum()) >= config["minimum_points_in_cell_image"]:
                        bins[cls].append((float(np.median(row["z"][keep])), float(np.median(prediction[keep]))))
    if not broad or any(not v for v in errors.values()):
        raise ValueError("Missing critical development support")
    cells = {k: {"macro_image_mae_m": float(np.mean(v)), "fixed_4m_mae_m": float(np.mean(fixed[k])),
        "images": len(v), "points": points[k]} for k, v in errors.items()}
    response = {}
    for cls, pairs in bins.items():
        values = np.asarray(pairs)
        slope = None
        if len(values) >= 2 and np.var(values[:, 0]) > 0:
            x, y = values[:, 0], values[:, 1]
            slope = float(np.mean((x-x.mean())*(y-y.mean())) / np.var(x))
        response[cls] = {"image_bin_observations": len(pairs), "descriptive_slope": slope,
            "caution": "Correlated frame/bin medians, not independent targets or a fitted correction"}
    return {"equal_four_cell_mae_m": float(np.mean([v["macro_image_mae_m"] for v in cells.values()])),
        "cells": cells, "broad_macro_image_mae_m": float(np.mean(broad)), "response": response}


def select_best(history):
    if not history or [r["epoch"] for r in history] != list(range(len(history))):
        raise ValueError("Selection needs contiguous history including epoch zero")
    return min(history, key=lambda r: (r["dev"]["equal_four_cell_mae_m"], r["epoch"]))


def validate_initial_replay(baseline, reviewed):
    if (set(baseline["cells"]) != set(reviewed["cells"])
            or any(baseline["cells"][key]["images"] != reviewed["cells"][key]["images"]
                or baseline["cells"][key]["points"] != reviewed["cells"][key]["points"]
                or abs(baseline["cells"][key]["macro_image_mae_m"] - reviewed["cells"][key]["macro_image_mae_m"]) > .005
                for key in baseline["cells"])):
        raise ValueError("Initial epoch-7 development replay differs by more than 5mm or support changed; stop before training")


def research_gates(initial, control, camera, config):
    score = lambda m: m["equal_four_cell_mae_m"]
    limit = config["minimum_relative_dev_gain"]
    tolerance = config["maximum_cell_regression_m"]
    return {"at_least_5_percent_better_than_initial": score(camera) <= (1-limit)*score(initial),
        "at_least_5_percent_better_than_parameter_matched_control": score(camera) <= (1-limit)*score(control),
        "no_cell_more_than_5cm_worse_than_initial_or_control": all(camera["cells"][k]["macro_image_mae_m"] <=
            min(initial["cells"][k]["macro_image_mae_m"], control["cells"][k]["macro_image_mae_m"]) + tolerance
            for k in camera["cells"]),
        "vehicle_2_4_beats_fixed_4m_diagnostic": camera["cells"]["Vehicle:2-4m"]["macro_image_mae_m"] <
            camera["cells"]["Vehicle:2-4m"]["fixed_4m_mae_m"]}


def smoke(model, initial, sample, inputs, device):
    import numpy as np
    import torch
    from PIL import Image
    from models.parking_shared_camera import CameraAwareDepthDecoder
    from scripts.screen_parking_shared_model import preprocess, state_digest
    before = state_digest(model.body)
    if common.file_sha(inputs / sample["image_file"]) != sample["image_sha256"]:
        raise ValueError("Smoke image bytes changed")
    rgb = np.asarray(Image.open(inputs / sample["image_file"]).convert("RGB"))
    tensor, _ = preprocess(rgb, intrinsics=sample["intrinsics"])
    with torch.no_grad():
        features = model.image_features(torch.from_numpy(tensor).to(device))
        native = model.depth_decoder(features).detach()
    adapter = CameraAwareDepthDecoder(copy.deepcopy(model.depth_decoder)).to(device).eval()
    adapter.load_state_dict(initial, strict=True)
    for arm in configuration()["arms"]:
        with torch.no_grad():
            prediction = adapter(features, camera_batch([sample], arm, device))
            if not torch.equal(native, prediction):
                raise ValueError("Zero-initialized adapter changed epoch-7 predictions")
        adapter.zero_grad(set_to_none=True)
        adapter(features, camera_batch([sample], arm, device)).mean().backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in adapter.parameters()):
            raise ValueError("Smoke backward failed")
    if state_digest(model.body) != before or any(p.grad is not None for p in model.body.parameters()):
        raise ValueError("Smoke changed frozen detector")
    return {"complete": True, "both_arms_bit_exact_epoch7_parity": True, "finite_backward": True,
        "optimizer_steps": 0, "frozen_detector_unchanged": True,
        "depth_parameters_per_arm": sum(p.numel() for p in adapter.parameters()),
        "extra_parameters_per_arm": sum(p.numel() for n, p in adapter.named_parameters() if not n.startswith("base."))}


def run(a):
    import numpy as np
    import torch
    from models.parking_shared_camera import CameraAwareDepthDecoder
    from scripts.screen_parking_shared_model import state_digest
    manifest = prepare(a)
    _, warm_selection, saved = read_warmstart(a.warmstart, load=True)
    config = {**manifest["config"], **configuration()}
    (a.output / "runtime_settings").mkdir(parents=True, exist_ok=True)
    model, model_id = pilot.build_model(a, manifest["config"])
    verification = body_verification.verify_warmstart_body(model, model_id, manifest["config"], a.weights)
    device = next(model.parameters()).device
    model.depth_decoder.load_state_dict(saved["decoder"], strict=True); model.eval()
    frozen = state_digest(model.body)
    torch.manual_seed(config["seed"])
    adapter = CameraAwareDepthDecoder(copy.deepcopy(model.depth_decoder), width=config["adapter_width"]).to(device).eval()
    initial = {k: v.detach().cpu().clone() for k, v in adapter.state_dict().items()}
    cache_id = cache_identity(a.inputs / "input_manifest.json", model, model_id)
    identity = {"revision": config["revision"], "config": config, "model": model_id,
        "backbone_verification": verification, "input_manifest_sha256": INPUT_SHA,
        "warmstart_checkpoint_sha256": CHECKPOINT_SHA, "warmstart_bundle_sha256": BUNDLE_SHA,
        "initial_adapter_state_sha256": state_digest(adapter), "cache_identity": cache_id,
        "source_sha256": {str(p.relative_to(ROOT)): common.file_sha(p) for p in SOURCES}}
    common.immutable(a.output / "run_identity.json", common.encoded(identity))
    run_id = common.identity(identity)
    for path in SOURCES:
        common.immutable(a.output / "source_snapshot" / path.relative_to(ROOT), path.read_bytes())
    samples = {role: [s for s in manifest["samples"] if s["role"] == role] for role in ["train", "dev"]}
    report = smoke(model, initial, samples["train"][0], a.inputs, device)
    common.immutable(a.output / "runtime_smoke.json", common.encoded({"run_identity": run_id, **report}))
    print("Runtime smoke: " + json.dumps(report), flush=True)
    if a.stage == "smoke":
        return
    if a.cache.is_symlink() or any(a.cache.resolve().is_relative_to(p.resolve()) for p in [a.inputs, a.source, a.weights, a.output]):
        raise ValueError("Use a separate local feature-cache folder")
    all_samples = manifest["samples"]
    if matching_cache(a.reuse_cache, cache_id, all_samples):
        cache = a.reuse_cache
        print(f"Reusing fully verified frozen features read-only: {cache}", flush=True)
    else:
        cache = a.cache
        common.immutable(cache / "cache_identity.json", common.encoded(cache_id))
        for i, sample in enumerate(all_samples, 1):
            pilot.cache_sample(model, sample, a.inputs, cache, device)
            if i % 64 == 0 or i == len(all_samples):
                print(f"Frozen features cached/verified: {i}/{len(all_samples)}", flush=True)
        if not matching_cache(cache, cache_id, all_samples):
            raise ValueError("Incomplete feature cache")
    if state_digest(model.body) != frozen:
        raise ValueError("Caching changed frozen detector")
    if a.stage == "cache":
        return
    adapter.load_state_dict(initial, strict=True)
    initial_path = a.output / "initial_dev.json"
    if initial_path.exists():
        initial_report = json.loads(initial_path.read_text())
        if initial_report["run_identity"] != run_id:
            raise ValueError("Initial development result identity changed")
        baseline = initial_report["metrics"]
        validate_initial_replay(baseline, warm_selection["selected_dev"])
    else:
        baseline = evaluate(adapter, samples["dev"], cache, "actual_camera", device, manifest["mapping"], config)
        validate_initial_replay(baseline, warm_selection["selected_dev"])
        common.immutable(initial_path, common.encoded({"run_identity": run_id, "metrics": baseline}))
    probe_features = pilot.batch_arrays([pilot.read_features(samples["train"][0], cache)], device)[0]
    with torch.no_grad():
        detection = model.full_urban_candidates(probe_features).detach().cpu().clone()
    selections = {}
    for arm in config["arms"]:
        adapter.load_state_dict(initial, strict=True)
        optimizer = torch.optim.AdamW(adapter.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        folder = a.output / arm
        arm_id = common.identity({"run_identity": run_id, "arm": arm})
        checkpoints = sorted((folder / "checkpoints").glob("epoch_*.pth"))
        if not checkpoints:
            history = [{"epoch": 0, "optimizer_steps": 0, "dev": baseline}]
            pilot.save_checkpoint(folder / "checkpoints/epoch_000.pth", {"run_identity": arm_id,
                "arm": arm, "epoch": 0, "optimizer_steps": 0, "decoder": initial,
                "optimizer": optimizer.state_dict(), "history": history})
            start = steps = 0
        else:
            last = pilot.load_checkpoint(checkpoints[-1], arm_id)
            if last["arm"] != arm:
                raise ValueError("Wrong checkpoint arm")
            adapter.load_state_dict(last["decoder"], strict=True); optimizer.load_state_dict(last["optimizer"])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if torch.is_tensor(v):
                        state[k] = v.to(device)
            history, start, steps = last["history"], last["epoch"], last["optimizer_steps"]
            select_best(history)
            expected_steps = start * ((len(samples["train"]) + config["batch_size"] - 1) // config["batch_size"])
            if history[-1]["epoch"] != start or start > config["epochs"] or steps != expected_steps:
                raise ValueError("Invalid resume history/budget")
        for epoch in range(start + 1, config["epochs"] + 1):
            sampler_epoch = config["sampler_epoch_offset"] + epoch
            torch.manual_seed(config["seed"] + sampler_epoch)
            batches = near.epoch_batches(samples["train"], config, sampler_epoch)
            losses = []; norms = []; began = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                for indices in batches:
                    selected = [samples["train"][i] for i in indices]
                    rows = list(pool.map(lambda s: pilot.read_features(s, cache), selected))
                    loss, norm = train_step(adapter, optimizer, rows, selected, arm, device, manifest["mapping"], config)
                    losses.append(loss); norms.append(norm); steps += 1
            with torch.no_grad():
                current_detection = model.full_urban_candidates(probe_features).detach().cpu()
            if (state_digest(model.body) != frozen or any(p.grad is not None for p in model.body.parameters())
                    or not torch.equal(detection, current_detection)):
                raise ValueError("Frozen body/BN/detection probe changed")
            dev = evaluate(adapter, samples["dev"], cache, arm, device, manifest["mapping"], config)
            row = {"epoch": epoch, "sampler_epoch": sampler_epoch, "optimizer_steps": steps,
                "steps_this_epoch": len(batches), "train_loss": float(np.mean(losses)),
                "max_pre_clip_gradient_norm": max(norms), "seconds": time.perf_counter()-began,
                "frozen_body_bn_detection_unchanged": True, "dev": dev}
            history.append(row)
            pilot.save_checkpoint(folder / "checkpoints" / f"epoch_{epoch:03d}.pth", {"run_identity": arm_id,
                "arm": arm, "epoch": epoch, "optimizer_steps": steps,
                "decoder": {k: v.detach().cpu() for k, v in adapter.state_dict().items()},
                "optimizer": optimizer.state_dict(), "history": history})
            print(json.dumps({"arm": arm, **row}), flush=True)
        if len(history) != config["epochs"] + 1:
            raise ValueError("Incomplete five-epoch arm")
        best = select_best(history)
        selection = {"complete": True, "run_identity": arm_id, "parent_run_identity": run_id,
            "arm": arm, "selected_epoch": best["epoch"], "selected_dev": best["dev"],
            "optimizer_steps": steps, "history": history, "heldout_used_for_selection": False}
        common.immutable(folder / "selection.json", common.encoded(selection)); selections[arm] = selection
    gates = research_gates(baseline, selections["reference_camera"]["selected_dev"], selections["actual_camera"]["selected_dev"], config)
    result = {"complete": True, "run_identity": run_id, "initial_dev": baseline,
        "arms": {k: {key: v[key] for key in ["selected_epoch", "selected_dev", "optimizer_steps"]} for k, v in selections.items()},
        "research_gates": gates, "all_research_gates_passed": all(gates.values()),
        "scope": "Development-only matched camera-conditioning hypothesis test; not independent accuracy qualification",
        "training_performed": True, "knowledge_distillation": False, "heldout_evaluated": False,
        "frozen_detector_unchanged": True, "scale_fitting_performed": False,
        "full_run_authorized": False, "phone_app_changed": False, "deployment_qualified": False,
        "next_step": "Review both selected dev curves/cells, fixed-4m sanity check and response before separately authorizing any frozen-checkpoint held-out evaluation or export"}
    common.immutable(a.output / "camera_pilot_results.json", common.encoded(result))
    print(json.dumps(result, indent=2), flush=True)


def bundle(a):
    identity = json.loads((a.output / "run_identity.json").read_text())
    result = json.loads((a.output / "camera_pilot_results.json").read_text())
    if not result["complete"] or result["run_identity"] != common.identity(identity):
        raise ValueError("Need complete matching camera pilot results")
    members = {name: a.output / name for name in ["run_identity.json", "prepared_inputs.json",
        "runtime_smoke.json", "initial_dev.json", "camera_pilot_results.json"]}
    for arm, value in result["arms"].items():
        selection_path = a.output / arm / "selection.json"
        selection = json.loads(selection_path.read_text())
        if (selection["parent_run_identity"] != result["run_identity"] or selection["selected_epoch"] != value["selected_epoch"]
                or select_best(selection["history"])["epoch"] != value["selected_epoch"]):
            raise ValueError("Selection changed")
        path = a.output / arm / "checkpoints" / f"epoch_{value['selected_epoch']:03d}.pth"
        pilot.load_checkpoint(path, selection["run_identity"])
        for item in [selection_path, path, path.with_suffix(".json")]:
            members[str(item.relative_to(a.output))] = item
    for path in (a.output / "source_snapshot").rglob("*"):
        if path.is_file():
            relative = str(path.relative_to(a.output / "source_snapshot"))
            if common.file_sha(path) != identity["source_sha256"][relative]:
                raise ValueError("Source snapshot changed")
            members[str(path.relative_to(a.output))] = path
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    archive = a.output / "parking_shared_camera_results.zip"
    common.immutable(archive, buffer.getvalue()); print(archive, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "smoke", "cache", "train", "bundle"])
    for key in ["inputs", "warmstart", "cache", "reuse-cache", "source", "weights"]:
        p.add_argument("--" + key, type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    a = p.parse_args()
    if a.stage == "bundle":
        bundle(a)
        return
    required = ["inputs", "warmstart", "source", "weights"] + (["cache"] if a.stage not in ["prepare", "smoke"] else [])
    if any(getattr(a, key) is None for key in required):
        p.error("Need " + " ".join("--" + key for key in required))
    if a.stage == "prepare":
        prepare(a)
    else:
        run(a)


if __name__ == "__main__":
    main()
