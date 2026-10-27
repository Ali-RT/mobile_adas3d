"""Two-arm near-range GT training of the existing shared model's depth head.

Ten epochs per arm, identical initialization/sampler/budget, one loss change.
No teacher, backbone update, held-out input, automatic export, or phone change.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import io
import json
import math
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common
from scripts import parking_shared_near_data as data
from scripts import train_parking_shared_depth as pilot
from scripts import parking_shared_body_verification as body_verification


def cell_masks(row, mapping):
    return data.cell_masks(row, mapping)


def balanced_loss(prediction, truth, valid, masks, config, arm):
    """Equal images within cells, equal cells; never create depth in GT holes."""
    import torch
    broad = pilot.sparse_loss(prediction, truth, valid, config["huber_delta_m"])
    if arm == "control":
        return broad
    if arm != "parking_balanced" or len(masks) != 4:
        raise ValueError("Unknown arm or missing four-cell masks")
    cells = []
    for mask in masks:
        if mask.shape != valid.shape or mask.dtype != torch.bool or (mask & ~valid).any():
            raise ValueError("Critical masks must be subsets of valid sparse references")
        images = []
        for p, t, selected in zip(prediction, truth, mask):
            if int(selected.sum()) >= config["minimum_points_in_cell_image"]:
                images.append(torch.nn.functional.huber_loss(p[selected], t[selected], delta=config["huber_delta_m"], reduction="mean"))
        if images:
            cells.append(torch.stack(images).mean())
    if not cells:
        raise ValueError("A balanced batch needs near-range class references")
    return config["parking_loss_weight"] * torch.stack(cells).mean() + config["broad_loss_weight"] * broad


def epoch_batches(samples, config, epoch):
    """Same deterministic mixed stream for both arms; no dev labels enter."""
    import numpy as np
    if config["batch_size"] != 8 or any(s["role"] != "train" for s in samples):
        raise ValueError("The fixed eight-image sampler accepts training inputs only")
    pools = [[i for i, s in enumerate(samples) if s["counts"][cell] >= config["minimum_points_in_cell_image"]] for cell in config["critical_cells"]]
    if any(not pool for pool in pools):
        raise ValueError("Each critical training pool needs support")
    rng = np.random.default_rng(config["seed"] + epoch)
    broad = list(map(int, rng.permutation(len(samples))))
    cursor = 0; batches = []
    for _ in range(math.ceil(len(samples) / 8)):
        batch = [int(rng.choice(pool)) for pool in pools]
        for _ in range(4):
            if cursor == len(broad):
                broad = list(map(int, rng.permutation(len(samples)))); cursor = 0
            batch.append(broad[cursor]); cursor += 1
        batches.append(batch)
    return batches


def step(decoder, optimizer, rows, device, mapping, config, arm):
    import numpy as np
    import torch
    features, grid, truth, valid = pilot.batch_arrays(rows, device)
    masks = []
    per_row = [cell_masks(row, mapping) for row in rows]
    for key in config["critical_cells"]:
        padded = np.zeros(tuple(valid.shape), bool)
        for i, row in enumerate(rows):
            padded[i, :len(row["z"])] = per_row[i][key]
        masks.append(torch.from_numpy(padded).to(device))
    decoder.train(); optimizer.zero_grad(set_to_none=True)
    prediction = pilot.sample_depth(decoder(features), grid)
    loss = balanced_loss(prediction, truth, valid, masks, config, arm)
    if not torch.isfinite(loss):
        raise ValueError("Non-finite training loss")
    loss.backward()
    parameters = list(decoder.parameters())
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):
        raise ValueError("Invalid decoder gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters):
        raise ValueError("Non-finite decoder parameters")
    return float(loss.detach().cpu()), float(norm.detach().cpu())


def evaluate_dev(decoder, samples, cache, device, mapping, config):
    import numpy as np
    import torch
    decoder.eval(); image_errors = {key: [] for key in config["critical_cells"]}; points = {key: 0 for key in image_errors}
    broad = []
    with torch.inference_mode():
        for sample in samples:
            if sample["role"] != "dev":
                raise ValueError("Checkpoint selection accepts development samples only")
            row = pilot.read_features(sample, cache)
            features, grid, _, _ = pilot.batch_arrays([row], device)
            prediction = pilot.sample_depth(decoder(features), grid)[0].cpu().numpy()
            if not np.isfinite(prediction).all() or not (prediction > 0).all():
                raise ValueError("Invalid metric depth prediction")
            absolute = np.abs(prediction.astype(np.float64) - row["z"])
            broad.append(float(absolute[row["conservative"]].mean()))
            for key, mask in cell_masks(row, mapping).items():
                if int(mask.sum()) >= config["minimum_points_in_cell_image"]:
                    image_errors[key].append(float(absolute[mask].mean())); points[key] += int(mask.sum())
    if any(not v for v in image_errors.values()) or not broad:
        raise ValueError("Cannot select checkpoints using missing critical development cells")
    cells = {key: {"macro_image_mae_m": float(np.mean(value)), "images": len(value), "points": points[key]} for key, value in image_errors.items()}
    return {"equal_four_cell_mae_m": float(np.mean([c["macro_image_mae_m"] for c in cells.values()])),
            "cells": cells, "broad_macro_image_mae_m": float(np.mean(broad))}


def initialize(model, inputs, manifest):
    init = manifest["initialization"]; path = inputs / init["file"]
    if common.file_sha(path) != init["sha256"] or common.file_sha(inputs / init["sidecar_file"]) != init["sidecar_sha256"]:
        raise ValueError("Initial checkpoint bytes changed")
    value = pilot.load_checkpoint(path, init["run_identity"])
    if value["epoch"] != manifest["config"]["initial_epoch"]:
        raise ValueError("Initial checkpoint epoch changed")
    model.depth_decoder.load_state_dict(value["decoder"], strict=True)


def run(a):
    import inspect
    import numpy as np
    import torch
    from models.parking_shared import SharedParkingNet
    from scripts.screen_parking_shared_model import state_digest, preprocess
    manifest = data.read_manifest(a.inputs); config = manifest["config"]
    for path in [a.output, a.cache]:
        if path.is_symlink() or path.resolve().is_relative_to(a.inputs.resolve()) or path.resolve().is_relative_to(a.source.resolve()):
            raise ValueError("Use separate new run and feature-cache folders")
    (a.output / "runtime_settings").mkdir(parents=True, exist_ok=True)
    model, model_id = pilot.build_model(a, config); device = next(model.parameters()).device
    frozen = state_digest(model.body)
    verification = body_verification.verify_warmstart_body(model, model_id, config, a.weights)
    print("Backbone verification: " + json.dumps(verification, sort_keys=True), flush=True)
    initialize(model, a.inputs, manifest); initial = {k: v.detach().cpu().clone() for k, v in model.depth_decoder.state_dict().items()}
    identity = {"model": model_id, "input_manifest_sha256": common.file_sha(a.inputs / "input_manifest.json"),
        "backbone_verification": verification,
        "config": config, "initial_decoder_state_sha256": state_digest(model.depth_decoder),
        "source_sha256": {p.name: common.file_sha(p) for p in [Path(__file__), Path(data.__file__), Path(pilot.__file__), Path(body_verification.__file__),
            ROOT / "scripts/parking_shared_depth_data.py", ROOT / "models/parking_shared.py", ROOT / "scripts/screen_parking_shared_model.py"]}}
    common.immutable(a.output / "run_identity.json", common.encoded(identity)); run_id = common.identity(identity)
    for path in [Path(__file__), Path(data.__file__), Path(pilot.__file__), Path(body_verification.__file__), ROOT / "models/parking_shared.py"]:
        common.immutable(a.output / "source_snapshot" / path.name, path.read_bytes())
    cache_id = {"input_manifest_sha256": identity["input_manifest_sha256"], "body_sha256": frozen,
        "runtime": model_id["runtime"], "producer_sha256": common.sha("\n".join(inspect.getsource(f) for f in
            [pilot.cache_sample, pilot_data_point_grid(), preprocess, SharedParkingNet.image_features]).encode())}
    common.immutable(a.cache / "cache_identity.json", common.encoded(cache_id))
    samples = {role: [s for s in manifest["samples"] if s["role"] == role] for role in ["train", "dev"]}
    for role in ["train", "dev"]:
        for i, sample in enumerate(samples[role], 1):
            pilot.cache_sample(model, sample, a.inputs, a.cache, device)
            if i % 64 == 0 or i == len(samples[role]):
                print(f"Frozen features {role}: {i}/{len(samples[role])}", flush=True)
    if state_digest(model.body) != frozen:
        raise ValueError("Feature caching changed frozen Urban state")
    if a.stage == "cache":
        return
    baseline_path = a.output / "initial_dev.json"
    model.depth_decoder.load_state_dict(initial, strict=True)
    if baseline_path.exists():
        saved_baseline = json.loads(baseline_path.read_text())
        if saved_baseline["run_identity"] != run_id:
            raise ValueError("Initial development result belongs to another run")
        baseline = saved_baseline["metrics"]
    else:
        baseline = evaluate_dev(model.depth_decoder, samples["dev"], a.cache, device, manifest["mapping"], config)
        common.immutable(baseline_path, common.encoded({"run_identity": run_id, "metrics": baseline}))
    probe = pilot.read_features(samples["train"][0], a.cache)
    probe_features = pilot.batch_arrays([probe], device)[0]
    with torch.inference_mode():
        detection_before = model.full_urban_candidates(probe_features).detach().cpu().clone()
    selections = {}
    for arm in config["arms"]:
        model.depth_decoder.load_state_dict(initial, strict=True)
        optimizer = torch.optim.AdamW(model.depth_decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        folder = a.output / arm; history = []; steps = 0; start = 0
        existing = sorted((folder / "checkpoints").glob("epoch_*.pth"))
        arm_identity = common.identity({"run_identity": run_id, "arm": arm})
        if existing:
            saved = pilot.load_checkpoint(existing[-1], arm_identity)
            if saved["arm"] != arm:
                raise ValueError("Arm checkpoint mismatch")
            model.depth_decoder.load_state_dict(saved["decoder"], strict=True); optimizer.load_state_dict(saved["optimizer"])
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(device)
            history, steps, start = saved["history"], saved["optimizer_steps"], saved["epoch"]
        for epoch in range(start + 1, config["epochs"] + 1):
            torch.manual_seed(config["seed"] + epoch)
            batches = epoch_batches(samples["train"], config, epoch); losses = []; norms = []; seen = set(); started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                for batch in batches:
                    rows = list(pool.map(lambda i: pilot.read_features(samples["train"][i], a.cache), batch))
                    loss, norm = step(model.depth_decoder, optimizer, rows, device, manifest["mapping"], config, arm)
                    losses.append(loss); norms.append(norm); steps += 1; seen.update(batch)
            if state_digest(model.body) != frozen or any(p.grad is not None for p in model.body.parameters()):
                raise ValueError("Frozen body/BN protection failed")
            with torch.inference_mode():
                detection_after = model.full_urban_candidates(probe_features).detach().cpu()
            if not torch.equal(detection_before, detection_after):
                raise ValueError("Frozen detection probe changed")
            dev = evaluate_dev(model.depth_decoder, samples["dev"], a.cache, device, manifest["mapping"], config)
            row = {"epoch": epoch, "optimizer_steps": steps, "steps_this_epoch": len(batches),
                "unique_training_images_sampled": len(seen), "image_draws": len(batches) * config["batch_size"],
                "train_loss": float(np.mean(losses)), "max_pre_clip_gradient_norm": max(norms), "dev": dev,
                "frozen_body_bn_and_detection_probe_unchanged": True, "seconds": time.perf_counter() - started}
            history.append(row)
            pilot.save_checkpoint(folder / "checkpoints" / f"epoch_{epoch:03d}.pth", {"run_identity": arm_identity,
                "arm": arm, "epoch": epoch, "optimizer_steps": steps, "decoder": {k: v.detach().cpu() for k, v in model.depth_decoder.state_dict().items()},
                "optimizer": optimizer.state_dict(), "history": history})
            print(json.dumps({"arm": arm, **row}), flush=True)
        if len(history) != config["epochs"]:
            raise ValueError("Incomplete arm budget")
        best = min(history, key=lambda h: (h["dev"]["equal_four_cell_mae_m"], h["epoch"]))
        selection = {"complete": True, "run_identity": arm_identity, "parent_run_identity": run_id, "arm": arm,
            "selected_epoch": best["epoch"], "selected_dev": best["dev"], "optimizer_steps": steps, "history": history,
            "selected_on": "dev_equal_four_cell_macro_image_mae_2_6m", "heldout_used_for_selection": False}
        common.immutable(folder / "selection.json", common.encoded(selection)); selections[arm] = selection
    result = {"complete": True, "run_identity": run_id, "initial_dev": baseline,
        "arms": {k: {"selected_epoch": v["selected_epoch"], "selected_dev": v["selected_dev"], "optimizer_steps": v["optimizer_steps"]} for k, v in selections.items()},
        "comparison_scope": "Development-only two-arm loss comparison; not held-out accuracy or deployment qualification",
        "training_performed": True, "knowledge_distillation": False, "frozen_urban_weights_bn_and_detection_probe_unchanged": True,
        "heldout_evaluation_performed": False, "scale_fitting_performed": False, "phone_app_changed": False, "deployment_qualified": False,
        "next_step": "Review development results and per-cell regressions before a separately authorized frozen-checkpoint held-out comparison or export"}
    common.immutable(a.output / "near_training_results.json", common.encoded(result))
    print(json.dumps(result, indent=2), flush=True)


def pilot_data_point_grid():
    from scripts.parking_shared_depth_data import point_grid
    return point_grid


def bundle(a):
    result = json.loads((a.output / "near_training_results.json").read_text())
    identity = json.loads((a.output / "run_identity.json").read_text())
    if not result["complete"] or result["run_identity"] != common.identity(identity):
        raise ValueError("Complete matching training result is required")
    members = {name: a.output / name for name in ["run_identity.json", "initial_dev.json", "near_training_results.json"]}
    for arm, value in result["arms"].items():
        selection = json.loads((a.output / arm / "selection.json").read_text())
        if selection["selected_epoch"] != value["selected_epoch"] or selection["parent_run_identity"] != result["run_identity"]:
            raise ValueError("Selection/result mismatch")
        checkpoint = a.output / arm / "checkpoints" / f"epoch_{value['selected_epoch']:03d}.pth"
        pilot.load_checkpoint(checkpoint, selection["run_identity"])
        for path in [checkpoint, checkpoint.with_suffix(".json"), a.output / arm / "selection.json"]:
            members[str(path.relative_to(a.output))] = path
    members.update({str(p.relative_to(a.output)): p for p in (a.output / "source_snapshot").glob("*.py")})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    common.immutable(a.output / "parking_shared_near_results.zip", buffer.getvalue())
    print(a.output / "parking_shared_near_results.zip", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["cache", "train", "bundle"])
    p.add_argument("--inputs", type=Path); p.add_argument("--cache", type=Path)
    p.add_argument("--source", type=Path); p.add_argument("--weights", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    a = p.parse_args()
    if a.stage == "bundle":
        bundle(a)
    else:
        if any(getattr(a, key) is None for key in ["inputs", "cache", "source", "weights"]):
            p.error("Model stages need --inputs --cache --source --weights")
        run(a)


if __name__ == "__main__":
    main()
