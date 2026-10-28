"""Full-pool matched adapter-only pilot; no automatic training or deployment.

Use prepare on CPU, then explicitly request smoke/cache/train on a GPU.
Original epoch-7 depth weights and all BN state remain frozen. Five full broad
passes plus critical oversampling per arm; no held-out evaluation or teacher.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import io
import json
from pathlib import Path
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common
from scripts import train_parking_shared_camera as camera
from scripts import sanity_parking_shared_adapter as sanity

CONFIG = ROOT / "configs/parking_shared_adapter_pilot_v1.json"
SOURCES = list(dict.fromkeys([Path(__file__), CONFIG, Path(sanity.__file__), sanity.CONFIG,
    *camera.SOURCES, ROOT / "configs/parking_shared_depth_near_v1.json",
    ROOT / "configs/parking_shared_nano_v1.json", ROOT / "scripts/prepare_parking_lidar_depth.py",
    ROOT / "scripts/screen_parking_edge_candidates.py"]))


def configuration():
    config = json.loads(CONFIG.read_text())
    camera_config = camera.configuration()
    if any(config[key] != camera_config[key] for key in ["arms", "reference_camera", "adapter_width"]):
        raise ValueError("Camera helper and adapter configuration differ")
    return config


def epoch_batches(samples, config, epoch):
    """One full shuffled broad pass plus four balanced draws per update."""
    import numpy as np
    if (not samples or len(samples) % 4 or config["batch_size"] != 8
            or any(s["role"] != "train" for s in samples)
            or len({s["camera_token"] for s in samples}) != len(samples)):
        raise ValueError("Need unique training-only images, batch eight, and count divisible by four")
    pools = [[i for i, s in enumerate(samples)
        if s["counts"][cell] >= config["minimum_points_in_cell_image"]]
        for cell in config["critical_cells"]]
    if len(pools) != 4 or any(not p for p in pools):
        raise ValueError("Every critical training pool needs support")
    rng = np.random.default_rng(config["seed"] + epoch)
    broad = list(map(int, rng.permutation(len(samples))))
    return [[int(rng.choice(pool)) for pool in pools] + broad[i:i+4]
        for i in range(0, len(broad), 4)]


def budget(samples, config):
    batches = epoch_batches(samples, config, 1)
    return {"train_images": len(samples), "epochs_per_arm": config["epochs"],
        "updates_per_epoch": len(batches), "updates_per_arm": len(batches)*config["epochs"],
        "draws_per_epoch": len(batches)*8, "unique_broad_images_per_epoch": len(samples)}


def separate_paths(a):
    protected = [a.inputs, a.source, a.weights, a.warmstart]
    for target in [a.output, a.cache]:
        if target is None:
            continue
        if target.is_symlink() or any(target.resolve().is_relative_to(p.resolve())
                or p.resolve().is_relative_to(target.resolve()) for p in protected):
            raise ValueError("Keep output/cache separate from immutable inputs, source and weights")
    if a.cache is not None and (a.cache.resolve().is_relative_to(a.output.resolve())
            or a.output.resolve().is_relative_to(a.cache.resolve())):
        raise ValueError("Use separate local feature-cache and run folders")


def prepare(a):
    separate_paths(a)
    manifest = camera.data.read_manifest(a.inputs)
    config = configuration()
    _, selection, _ = camera.read_warmstart(a.warmstart)
    if common.file_sha(a.inputs / "input_manifest.json") != camera.INPUT_SHA:
        raise ValueError("Use the unchanged reviewed train/dev package")
    samples = {role: [s for s in manifest["samples"] if s["role"] == role] for role in ["train", "dev"]}
    if ({s["role"] for s in manifest["samples"]} != {"train", "dev"}
            or any(len(samples[r]) != config[r+"_images"] for r in samples)):
        raise ValueError("Use exactly 1536 train and 256 development images; no held-out inputs")
    # The frozen input reader also checks scene/log/keyframe disjointness.
    for sample in manifest["samples"]:
        camera.camera_vector(sample)
    receipt = {"complete": True, "revision": config["revision"], "config": config,
        "input_manifest_sha256": camera.INPUT_SHA, "warmstart_checkpoint_sha256": camera.CHECKPOINT_SHA,
        "warmstart_bundle_sha256": camera.BUNDLE_SHA, "initial_epoch": 7,
        "initial_selected_dev": selection["selected_dev"], "dev_images": len(samples["dev"]),
        "budget": budget(samples["train"], config), "optimizer_steps": 0,
        "prior_tiny_sanity_all_checks_passed": False, "heldout_inputs": 0}
    common.immutable(a.output / "prepared_inputs.json", common.encoded(receipt))
    print(json.dumps(receipt, indent=2), flush=True)
    return manifest


def smoke(native, decoder, sample, inputs, device, mapping, config):
    """Native image forward and GT backward, but zero optimizer updates."""
    import numpy as np
    import torch
    from PIL import Image
    from scripts.screen_parking_shared_model import preprocess, state_digest
    frozen = state_digest(decoder.base)
    body = state_digest(native.body)
    before = state_digest(decoder)
    for key in ["image", "ground_truth"]:
        if common.file_sha(inputs / sample[key+"_file"]) != sample[key+"_sha256"]:
            raise ValueError("Smoke input bytes changed")
    rgb = np.asarray(Image.open(inputs / sample["image_file"]).convert("RGB"))
    tensor, meta = preprocess(rgb, intrinsics=sample["intrinsics"])
    with torch.no_grad():
        features = native.image_features(torch.from_numpy(tensor).to(device))
        original = native.depth_decoder(features)
    with np.load(inputs / sample["ground_truth_file"], allow_pickle=False) as source:
        row = {key: source[key] for key in ["xy", "z", "semantic", "static", "conservative"]}
    row.update({f"p{i+3}": x[0].detach().cpu().numpy() for i, x in enumerate(features)})
    row["grid"] = camera.near.pilot_data_point_grid()(row["xy"], sample["width"], sample["height"], meta)
    for arm in config["arms"]:
        decoder.eval()
        with torch.no_grad():
            if not torch.equal(original, decoder(features, camera.camera_batch([sample], arm, device))):
                raise ValueError("Zero-output adapter changed the original predictions")
        decoder.zero_grad(set_to_none=True)
        decoder.train(); decoder.base.eval()
        prediction, truth, valid = sanity.predict(decoder, [row], [sample], arm, device)
        loss = camera.near.balanced_loss(prediction, truth, valid,
            camera.masks_for([row], valid, mapping, config), config, "parking_balanced")
        loss.backward()
        params = [p for p in decoder.parameters() if p.requires_grad]
        if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params):
            raise ValueError("Native GT backward failed")
        sanity.frozen_check(decoder, frozen)
    decoder.zero_grad(set_to_none=True); decoder.eval()
    if state_digest(decoder) != before or state_digest(native.body) != body:
        raise ValueError("Smoke changed model state")
    return {"complete": True, "both_arms_bit_exact_epoch7_parity": True,
        "finite_gt_backward_adapter_only": True, "base_weights_and_bn_unchanged": True,
        "frozen_detector_unchanged": True, "optimizer_steps": 0, "trainable_parameters": 1553}


def research_gates(initial, control, actual, config):
    score = lambda m: m["equal_four_cell_mae_m"]
    generic = {"at_least_5_percent_better_than_initial":
        score(actual) <= (1-config["minimum_relative_dev_gain"])*score(initial),
        "no_critical_cell_more_than_2cm_worse_than_initial": all(
            actual["cells"][k]["macro_image_mae_m"] <= initial["cells"][k]["macro_image_mae_m"]
            + config["maximum_cell_regression_m"] for k in config["critical_cells"]),
        "broad_mae_no_more_than_10cm_worse_than_initial": actual["broad_macro_image_mae_m"]
            <= initial["broad_macro_image_mae_m"] + config["maximum_broad_regression_m"],
        "vehicle_2_4_beats_fixed_4m_diagnostic": actual["cells"]["Vehicle:2-4m"]["macro_image_mae_m"]
            < actual["cells"]["Vehicle:2-4m"]["fixed_4m_mae_m"]}
    return {"adapter_improvement": generic,
        "camera_specific_gain_at_least_2_percent": score(actual) <= (1-config["minimum_relative_camera_gain"])*score(control)}


def validate_resume(saved, arm, config, samples):
    history = saved["history"]
    camera.select_best(history)
    start = saved["epoch"]
    per_epoch = len(samples)//4
    if (saved["arm"] != arm or not 0 <= start <= config["epochs"]
            or len(history) != start+1 or saved["optimizer_steps"] != start*per_epoch):
        raise ValueError("Resume arm/history/budget changed")
    for row in history:
        epoch = row["epoch"]
        if row["optimizer_steps"] != epoch*per_epoch:
            raise ValueError("Resume history step count changed")
        if epoch and row["sampler_sha256"] != common.identity(epoch_batches(samples, config, epoch)):
            raise ValueError("Resume sampler changed")


def train_arm(a, decoder, initial, baseline, samples, cache, mapping, config, run_id, frozen):
    """Resume only completed immutable epochs, never partially updated weights."""
    import numpy as np
    import torch
    arm = a.arm
    device = next(decoder.parameters()).device
    decoder.load_state_dict(initial, strict=True)
    params = sanity.freeze_base(decoder)
    optimizer = torch.optim.AdamW(params, lr=config["learning_rate"], weight_decay=config["weight_decay"])
    folder = a.output / arm
    arm_id = common.identity({"run_identity": run_id, "arm": arm})
    paths = sorted((folder / "checkpoints").glob("epoch_*.pth"))
    if paths:
        saved = camera.pilot.load_checkpoint(paths[-1], arm_id)
        validate_resume(saved, arm, config, samples["train"])
        if paths[-1].name != f"epoch_{saved['epoch']:03d}.pth":
            raise ValueError("Resume checkpoint filename/epoch mismatch")
        decoder.load_state_dict(saved["decoder"], strict=True)
        sanity.frozen_check(decoder, frozen)
        optimizer.load_state_dict(saved["optimizer"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        history = saved["history"]
    else:
        history = [{"epoch": 0, "optimizer_steps": 0, "dev": baseline}]
        camera.pilot.save_checkpoint(folder / "checkpoints/epoch_000.pth", {"run_identity": arm_id,
            "arm": arm, "epoch": 0, "optimizer_steps": 0, "decoder": initial,
            "optimizer": optimizer.state_dict(), "history": history})
    if {id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for p in params}:
        raise ValueError("Optimizer must contain exactly the adapter parameters")
    steps = history[-1]["optimizer_steps"]
    for epoch in range(history[-1]["epoch"]+1, config["epochs"]+1):
        torch.manual_seed(config["seed"]+epoch)
        batches = epoch_batches(samples["train"], config, epoch)
        losses = []; norms = []; began = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            for update, indices in enumerate(batches, 1):
                chosen = [samples["train"][i] for i in indices]
                rows = list(pool.map(lambda s: camera.pilot.read_features(s, cache), chosen))
                info = sanity.adapter_step(decoder, optimizer, rows, chosen, arm, device, mapping, config, frozen)
                losses.append(info["batch_loss"]); norms.append(info["pre_clip_gradient_norm"]); steps += 1
                if update % 32 == 0 or update == len(batches):
                    print(f"{arm}: epoch {epoch}/{config['epochs']}, update {update}/{len(batches)}", flush=True)
        dev = camera.evaluate(decoder, samples["dev"], cache, arm, device, mapping, config)
        sanity.frozen_check(decoder, frozen)
        history.append({"epoch": epoch, "optimizer_steps": steps, "steps_this_epoch": len(batches),
            "sampler_sha256": common.identity(batches), "train_loss": float(np.mean(losses)),
            "max_pre_clip_gradient_norm": max(norms), "seconds": time.perf_counter()-began,
            "base_weights_and_bn_exactly_frozen": True, "dev": dev})
        camera.pilot.save_checkpoint(folder / "checkpoints" / f"epoch_{epoch:03d}.pth", {
            "run_identity": arm_id, "arm": arm, "epoch": epoch, "optimizer_steps": steps,
            "decoder": {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()},
            "optimizer": optimizer.state_dict(), "history": history})
        print(json.dumps({"arm": arm, **history[-1]}), flush=True)
    best = camera.select_best(history)
    selection = {"complete": True, "run_identity": arm_id, "parent_run_identity": run_id,
        "arm": arm, "selected_epoch": best["epoch"], "selected_dev": best["dev"],
        "optimizer_steps": steps, "history": history, "heldout_used_for_selection": False}
    common.immutable(folder / "selection.json", common.encoded(selection))
    return selection


def run(a):
    import torch
    from models.parking_shared_camera import CameraAwareDepthDecoder
    from scripts.screen_parking_shared_model import state_digest
    manifest = prepare(a); config = configuration()
    _, warm_selection, warm = camera.read_warmstart(a.warmstart, load=True)
    (a.output / "runtime_settings").mkdir(parents=True, exist_ok=True)
    native, model_id = camera.pilot.build_model(a, manifest["config"])
    verified = camera.body_verification.verify_warmstart_body(native, model_id, manifest["config"], a.weights)
    native.depth_decoder.load_state_dict(warm["decoder"], strict=True); native.eval().requires_grad_(False)
    device = next(native.parameters()).device
    frozen_body = state_digest(native.body); frozen = state_digest(native.depth_decoder)
    torch.manual_seed(config["seed"])
    decoder = CameraAwareDepthDecoder(copy.deepcopy(native.depth_decoder), width=config["adapter_width"]).to(device).eval()
    params = sanity.freeze_base(decoder)
    if sum(p.numel() for p in params) != 1553 or sum(p.numel() for p in decoder.base.parameters()) != 57441:
        raise ValueError("Expected 1553 trainable adapter and 57441 frozen original-depth parameters")
    initial = {k: v.detach().cpu().clone() for k, v in decoder.state_dict().items()}
    cache_id = camera.cache_identity(a.inputs / "input_manifest.json", native, model_id)
    samples = {role: [s for s in manifest["samples"] if s["role"] == role] for role in ["train", "dev"]}
    identity = {"revision": config["revision"], "config": config, "model": model_id,
        "backbone_verification": verified, "input_manifest_sha256": camera.INPUT_SHA,
        "warmstart_checkpoint_sha256": camera.CHECKPOINT_SHA, "warmstart_bundle_sha256": camera.BUNDLE_SHA,
        "frozen_body_sha256": frozen_body, "frozen_depth_sha256": frozen,
        "initial_adapter_state_sha256": state_digest(decoder), "cache_identity": cache_id,
        "budget": budget(samples["train"], config),
        "source_sha256": {str(p.relative_to(ROOT)): common.file_sha(p) for p in SOURCES}}
    common.immutable(a.output / "run_identity.json", common.encoded(identity)); run_id = common.identity(identity)
    for path in SOURCES:
        common.immutable(a.output / "source_snapshot" / path.relative_to(ROOT), path.read_bytes())
    probe = next(s for s in samples["train"] if any(s["counts"][k] >= 5 for k in config["critical_cells"]))
    checked = smoke(native, decoder, probe, a.inputs, device, manifest["mapping"], config)
    common.immutable(a.output / "runtime_smoke.json", common.encoded({"run_identity": run_id, **checked}))
    print("Runtime smoke: " + json.dumps(checked), flush=True)
    if a.stage == "smoke":
        return
    cache = next((p for p in a.reuse_cache if camera.matching_cache(p, cache_id, manifest["samples"])), None)
    if cache is None:
        cache = a.cache
        common.immutable(cache / "cache_identity.json", common.encoded(cache_id))
        for i, sample in enumerate(manifest["samples"], 1):
            camera.pilot.cache_sample(native, sample, a.inputs, cache, device)
            if i % 64 == 0 or i == len(manifest["samples"]):
                print(f"Frozen features cached/verified: {i}/{len(manifest['samples'])}", flush=True)
        if not camera.matching_cache(cache, cache_id, manifest["samples"]):
            raise ValueError("Incomplete frozen feature cache")
    print(f"Using fully verified frozen features: {cache}", flush=True)
    if state_digest(native.body) != frozen_body:
        raise ValueError("Caching changed the detector")
    if a.stage == "cache":
        return
    # Recompute the baseline on every resumed train invocation, rather than
    # trusting a stale score file. Cache path is not part of run identity.
    baseline = camera.evaluate(decoder, samples["dev"], cache, "actual_camera", device, manifest["mapping"], config)
    camera.validate_initial_replay(baseline, warm_selection["selected_dev"])
    common.immutable(a.output / "initial_dev.json", common.encoded({"run_identity": run_id, "metrics": baseline}))
    probe_features = camera.pilot.batch_arrays([camera.pilot.read_features(probe, cache)], device)[0]
    with torch.no_grad():
        detection = native.full_urban_candidates(probe_features).detach().cpu().clone()
    selections = {}
    for arm in config["arms"]:
        selections[arm] = train_arm(argparse.Namespace(output=a.output, arm=arm), decoder, initial,
            baseline, samples, cache, manifest["mapping"], config, run_id, frozen)
        with torch.no_grad():
            same = torch.equal(detection, native.full_urban_candidates(probe_features).detach().cpu())
        if not same or state_digest(native.body) != frozen_body or any(p.grad is not None for p in native.parameters()):
            raise ValueError("Frozen detector/head/BN changed")
    gates = research_gates(baseline, selections["reference_camera"]["selected_dev"], selections["actual_camera"]["selected_dev"], config)
    result = {"complete": True, "run_identity": run_id, "initial_dev": baseline, "budget": identity["budget"],
        "arms": {k: {key: v[key] for key in ["selected_epoch", "selected_dev", "optimizer_steps"]} for k, v in selections.items()},
        "research_gates": gates, "adapter_improvement_checks_passed": all(gates["adapter_improvement"].values()),
        "all_research_gates_passed": all(gates["adapter_improvement"].values()) and gates["camera_specific_gain_at_least_2_percent"],
        "prior_tiny_sanity_all_checks_passed": False, "training_performed": True,
        "trainable_parameters": 1553, "frozen_depth_parameters": 57441,
        "frozen_body_depth_bn_detection_unchanged": True, "knowledge_distillation": False,
        "heldout_evaluated": False, "scale_fitting_performed": False,
        "larger_run_authorized": False, "phone_app_changed": False, "deployment_qualified": False,
        "scope": "Previously inspected development data, not fresh blind accuracy or product qualification",
        "next_step": "Return bundle for review; no automatic extra epochs, held-out evaluation, export or app replacement"}
    common.immutable(a.output / "adapter_pilot_results.json", common.encoded(result))
    print(json.dumps(result, indent=2), flush=True)


def bundle(a):
    identity = json.loads((a.output / "run_identity.json").read_text())
    result = json.loads((a.output / "adapter_pilot_results.json").read_text())
    if not result["complete"] or result["run_identity"] != common.identity(identity):
        raise ValueError("Need complete matching adapter results")
    members = {name: a.output / name for name in ["run_identity.json", "prepared_inputs.json",
        "runtime_smoke.json", "initial_dev.json", "adapter_pilot_results.json"]}
    for arm, summary in result["arms"].items():
        selection_path = a.output / arm / "selection.json"
        selection = json.loads(selection_path.read_text())
        arm_id = common.identity({"run_identity": result["run_identity"], "arm": arm})
        if (selection["parent_run_identity"] != result["run_identity"] or selection["run_identity"] != arm_id
                or selection["selected_epoch"] != summary["selected_epoch"]
                or camera.select_best(selection["history"])["epoch"] != summary["selected_epoch"]
                or selection["selected_dev"] != summary["selected_dev"]):
            raise ValueError("Selection changed")
        members[str(selection_path.relative_to(a.output))] = selection_path
        # Include trained endpoint even if initialization wins selection.
        for epoch in {0, summary["selected_epoch"], identity["config"]["epochs"]}:
            path = a.output / arm / "checkpoints" / f"epoch_{epoch:03d}.pth"
            saved = camera.pilot.load_checkpoint(path, arm_id)
            if saved["epoch"] != epoch or saved["arm"] != arm or saved["history"] != selection["history"][:epoch+1]:
                raise ValueError("Packaged checkpoint does not match selection history")
            for item in [path, path.with_suffix(".json")]:
                members[str(item.relative_to(a.output))] = item
    for relative, digest in identity["source_sha256"].items():
        path = a.output / "source_snapshot" / relative
        if common.file_sha(path) != digest:
            raise ValueError("Source snapshot changed")
        members[str(path.relative_to(a.output))] = path
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(members.items()):
            info = zipfile.ZipInfo(name, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, path.read_bytes())
    path = a.output / "parking_shared_adapter_results.zip"
    common.immutable(path, buffer.getvalue()); print(path, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "smoke", "cache", "train", "bundle"])
    for name in ["inputs", "warmstart", "source", "weights", "cache"]:
        p.add_argument("--"+name, type=Path)
    p.add_argument("--reuse-cache", type=Path, action="append", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cuda")
    a = p.parse_args()
    if a.stage == "bundle":
        bundle(a)
    else:
        required = ["inputs", "warmstart", "source", "weights"] + (["cache"] if a.stage in ["cache", "train"] else [])
        if any(getattr(a, name) is None for name in required):
            p.error("Need "+" ".join("--"+name for name in required))
        prepare(a) if a.stage == "prepare" else run(a)


if __name__ == "__main__":
    main()
