"""Decoder-only GT pilot; frozen Urban weights/BN, no teacher or phone edits.

Cache frozen image features once on the compute device, train a small FP32 head,
select on scene-separated dev only, then open the fixed held-out screen once.
No per-image depth scaling and no claims of dense/10cm ground-truth accuracy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common
from scripts import parking_shared_depth_data as data


def build_model(a, config):
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ["YOLO_CONFIG_DIR"] = str((a.output / "runtime_settings").resolve())
    import torch
    from models.parking_shared import SharedParkingNet
    from scripts.screen_parking_edge_candidates import urban_source, download
    from scripts.screen_parking_shared_model import state_digest
    architecture = json.loads((ROOT / config["architecture_config"]).read_text())
    if importlib.metadata.version("ultralytics") != architecture["urban_ultralytics_version"]:
        raise ValueError("Use pinned ultralytics==8.4.61; do not alter the reviewed head")
    urban_source(a.source, {"source_url": architecture["urban_source_url"], "source_commit": architecture["urban_source_commit"]})
    checkpoint = download(architecture["urban_checkpoint"], a.weights)
    from ultralytics import YOLO
    torch.manual_seed(config["seed"])
    native = YOLO(str(checkpoint), task="pose").model.float().cpu().eval(); native.fuse(verbose=False)
    model = SharedParkingNet(native, architecture["shared_features"]["channels"], architecture["depth_decoder_width"])
    device = torch.device(a.device)
    if device.type == "cuda" and not torch.cuda.is_available(): raise ValueError("CUDA GPU is not available")
    if device.type == "mps" and not torch.backends.mps.is_available(): raise ValueError("MPS is not available")
    if device.type not in ["cpu", "cuda", "mps"]: raise ValueError("Use cpu, cuda or mps")
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(config["seed"])
    torch.set_num_threads(2)
    identity = {"revision": config["revision"], "config": config, "architecture": architecture,
                "source_checkpoint_sha256": common.file_sha(checkpoint), "frozen_body_state_sha256": state_digest(model.body),
                "initial_decoder_state_sha256": state_digest(model.depth_decoder),
                "implementation_sha256": {p.name: common.file_sha(p) for p in [Path(__file__), Path(data.__file__), ROOT / "models/parking_shared.py", ROOT / "scripts/screen_parking_shared_model.py"]},
                "runtime": {"python": platform.python_version(), "torch": str(torch.__version__),
                            "ultralytics": importlib.metadata.version("ultralytics"), "numpy": importlib.metadata.version("numpy"),
                            "pillow": importlib.metadata.version("pillow"), "opencv_imported": __import__("cv2").__version__,
                            "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.machine()}}
    return model.to(device), identity


def npz_bytes(**arrays):
    import numpy as np
    buffer = io.BytesIO(); np.savez_compressed(buffer, **arrays); return buffer.getvalue()


def cache_sample(model, sample, inputs, cache, device):
    import numpy as np
    import torch
    from PIL import Image
    from scripts.screen_parking_shared_model import preprocess
    token = sample["camera_token"]; path = cache / f"{token}.npz"; receipt = path.with_suffix(".json")
    if path.exists():
        saved = json.loads(receipt.read_text())
        if saved["sample"] != sample or common.file_sha(path) != saved["sha256"]:
            raise ValueError("Preserve changed feature cache")
        return path
    for name in ["image", "ground_truth"]:
        if common.file_sha(inputs / sample[f"{name}_file"]) != sample[f"{name}_sha256"]:
            raise ValueError("Input bytes changed before feature extraction")
    rgb = np.asarray(Image.open(inputs / sample["image_file"]).convert("RGB"))
    if rgb.shape != (sample["height"], sample["width"], 3): raise ValueError("Camera image shape changed")
    tensor, meta = preprocess(rgb, intrinsics=sample["intrinsics"])
    with torch.no_grad(): features = model.image_features(torch.from_numpy(tensor).to(device))
    arrays = {f"p{i+3}": x.detach().cpu().numpy()[0] for i, x in enumerate(features)}
    with np.load(inputs / sample["ground_truth_file"], allow_pickle=False) as source:
        for key in ["xy", "z", "semantic", "static", "conservative"]: arrays[key] = source[key]
    arrays["grid"] = data.point_grid(arrays["xy"], sample["width"], sample["height"], meta)
    raw = npz_bytes(**arrays); common.immutable(path, raw)
    common.immutable(receipt, common.encoded({"sample": sample, "sha256": common.sha(raw)}))
    return path


def read_features(sample, cache):
    import numpy as np
    path = cache / f"{sample['camera_token']}.npz"
    record = json.loads(path.with_suffix(".json").read_text())
    if record["sample"] != sample or common.file_sha(path) != record["sha256"]:
        raise ValueError("Cached feature identity changed")
    with np.load(path, allow_pickle=False) as source: arrays = {k: source[k] for k in source.files}
    expected = {"p3": (64, 48, 80), "p4": (128, 24, 40), "p5": (256, 12, 20)}
    for key, shape in expected.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Feature cache shape/type changed")
    n = len(arrays["z"])
    if (arrays["grid"].shape != (n, 2) or not np.isfinite(arrays["grid"]).all()
            or (np.abs(arrays["grid"]) > 1).any() or not np.isfinite(arrays["z"]).all()
            or not (arrays["z"] > 0).all() or arrays["conservative"].shape != (n,)
            or arrays["conservative"].dtype != bool): raise ValueError("Malformed sparse GT cache")
    return arrays


def batch_arrays(rows, device):
    import numpy as np
    import torch
    features = [torch.from_numpy(np.stack([r[f"p{i}"] for r in rows])).to(device) for i in [3, 4, 5]]
    n = max(len(r["z"]) for r in rows); b = len(rows)
    grid = np.zeros((b, n, 1, 2), np.float32); truth = np.ones((b, n), np.float32); valid = np.zeros((b, n), bool)
    for i, row in enumerate(rows):
        size = len(row["z"]); grid[i, :size, 0] = row["grid"]; truth[i, :size] = row["z"]; valid[i, :size] = row["conservative"]
    return features, torch.from_numpy(grid).to(device), torch.from_numpy(truth).to(device), torch.from_numpy(valid).to(device)


def sample_depth(depth, grid):
    """Bilinear/border/half-pixel sampling using portable gather arithmetic.

    Equivalent to grid_sample(..., align_corners=False, padding_mode='border')
    for a single depth channel. MPS lacks grid_sample's backward kernel; this
    avoids a silent CPU fallback without changing the supervision locations.
    """
    if depth.ndim != 4 or depth.shape[1] != 1 or grid.ndim != 4 or grid.shape[2:] != (1, 2) or depth.shape[0] != grid.shape[0]:
        raise ValueError("Expected Bx1xHxW depth and BxNx1x2 sampling grid")
    h, w = depth.shape[-2:]
    x = (((grid[:, :, 0, 0] + 1) * w - 1) / 2).clamp(0, w - 1)
    y = (((grid[:, :, 0, 1] + 1) * h - 1) / 2).clamp(0, h - 1)
    x0, y0 = x.floor().long(), y.floor().long()
    x1, y1 = (x0 + 1).clamp(max=w-1), (y0 + 1).clamp(max=h-1)
    dx, dy = x - x0, y - y0
    flat = depth[:, 0].flatten(1)
    at = lambda xx, yy: flat.gather(1, yy * w + xx)
    return at(x0, y0) * (1-dx) * (1-dy) + at(x1, y0) * dx * (1-dy) + at(x0, y1) * (1-dx) * dy + at(x1, y1) * dx * dy


def sparse_loss(prediction, truth, valid, delta):
    from models.parking_shared import metric_depth_loss
    if prediction.shape != truth.shape or valid.shape != truth.shape or not valid.any(dim=1).all():
        raise ValueError("Each image needs valid sparse references")
    # Equal image weights: a near-dense road image must not dominate a sparse
    # image simply because it contains more LiDAR returns.
    return sum(metric_depth_loss(p, t, v, delta=delta) for p, t, v in zip(prediction, truth, valid)) / len(prediction)


def step(decoder, optimizer, rows, device, config):
    import torch
    features, grid, truth, valid = batch_arrays(rows, device)
    decoder.train(); optimizer.zero_grad(set_to_none=True)
    prediction = sample_depth(decoder(features), grid)
    loss = sparse_loss(prediction, truth, valid, config["huber_delta_m"])
    if not torch.isfinite(loss): raise ValueError("Non-finite GT loss")
    loss.backward()
    parameters = list(decoder.parameters())
    if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters): raise ValueError("Invalid depth-decoder gradients")
    norm = torch.nn.utils.clip_grad_norm_(parameters, config["gradient_clip_norm"], error_if_nonfinite=True)
    optimizer.step()
    if any(not torch.isfinite(p).all() for p in parameters): raise ValueError("Non-finite decoder weights")
    return float(loss.detach().cpu()), float(norm.detach().cpu())


def evaluate_decoder(decoder, rows, cache, device, mapping, detailed=False):
    import numpy as np
    import torch
    from scripts.evaluate_parking_lidar_depth import metrics, mask
    decoder.eval(); image_mae = []; cells = {}
    scopes = ["conservative", "conservative_static_objects"] if detailed else ["conservative"]
    with torch.no_grad():
        for sample in rows:
            arrays = read_features(sample, cache); features, grid, truth, valid = batch_arrays([arrays], device)
            prediction = sample_depth(decoder(features), grid)[0].cpu().numpy()
            gt = arrays["z"]; keep = arrays["conservative"]
            if not np.isfinite(prediction).all() or not (prediction > 0).all(): raise ValueError("Invalid metric predictions")
            if keep.any(): image_mae.append(float(np.abs(prediction[keep] - gt[keep]).mean()))
            if detailed:
                for scope in scopes:
                    for cls in ["All", "Vehicle", "Pedestrian"]:
                        for band, lo, hi in [("2-4", 2, 4), ("4-6", 4, 6), ("2-6", 2, 6), ("2-80", 2, 80)]:
                            m = mask(arrays, mapping, scope, cls, lo, hi, hi in [6, 80])
                            if m.any(): cells.setdefault(f"{scope}/{cls}/{band}", []).append((gt[m], prediction[m]))
    if not image_mae: raise ValueError("No valid evaluation references")
    result = {"images": len(image_mae), "macro_image_mae_m": float(np.mean(image_mae))}
    if detailed:
        result["cells"] = {}
        for scope in scopes:
            for cls in ["All", "Vehicle", "Pedestrian"]:
                for band in ["2-4", "4-6", "2-6", "2-80"]:
                    key = f"{scope}/{cls}/{band}"; values = cells.get(key, [])
                    reference = np.concatenate([x[0] for x in values]) if values else np.empty(0)
                    pred = np.concatenate([x[1] for x in values]) if values else np.empty(0)
                    result["cells"][key] = {"images": len(values), "points": len(reference), "metrics": metrics(reference, pred)}
    return result


def save_checkpoint(path, value):
    import torch
    buffer = io.BytesIO(); torch.save(value, buffer); raw = buffer.getvalue()
    common.immutable(path, raw); common.immutable(path.with_suffix(".json"), common.encoded({"sha256": common.sha(raw)}))


def load_checkpoint(path, run_identity):
    import torch
    record = json.loads(path.with_suffix(".json").read_text())
    if common.file_sha(path) != record["sha256"]: raise ValueError("Checkpoint bytes changed")
    value = torch.load(path, map_location="cpu", weights_only=True)
    if value["run_identity"] != run_identity: raise ValueError("Preserve checkpoint from a different run")
    return value


def run(a):
    import numpy as np
    import torch
    from scripts.screen_parking_shared_model import state_digest
    manifest = data.read_manifest(a.inputs); config = manifest["config"]
    if config != json.loads(data.CONFIG.read_text()):
        raise ValueError("Prepared training recipe differs from the reviewed config")
    if (a.output.is_symlink() or a.cache.is_symlink() or a.output.resolve().is_relative_to(a.inputs.resolve())
            or a.cache.resolve().is_relative_to(a.inputs.resolve()) or a.output.resolve().is_relative_to(a.source.resolve())):
        raise ValueError("Keep output separate from immutable input data")
    model, identity = build_model(a, config); device = next(model.parameters()).device
    identity["input_manifest_sha256"] = common.file_sha(a.inputs / "input_manifest.json")
    common.immutable(a.output / "run_identity.json", common.encoded(identity)); run_id = common.identity(identity)
    # Cache binds feature-producing maths, data and runtime, not the optimizer
    # loop. Future training-only changes need a fresh run, not feature recopy.
    import inspect
    from models.parking_shared import SharedParkingNet
    from scripts.screen_parking_shared_model import preprocess
    cache_identity = {"input_manifest_sha256": identity["input_manifest_sha256"],
                      "frozen_body_state_sha256": identity["frozen_body_state_sha256"], "runtime": identity["runtime"],
                      "producer_functions_sha256": common.sha("\n".join(inspect.getsource(f) for f in
                          [cache_sample, data.point_grid, preprocess, SharedParkingNet.image_features]).encode())}
    common.immutable(a.cache / "cache_identity.json", common.encoded(cache_identity))
    roles = {role: [s for s in manifest["samples"] if s["role"] == role] for role in ["train", "dev", "test"]}
    frozen_sha = state_digest(model.body); initial_decoder = {k: v.detach().cpu().clone() for k, v in model.depth_decoder.state_dict().items()}
    for role in ["train", "dev"]:
        for i, sample in enumerate(roles[role], 1):
            cache_sample(model, sample, a.inputs, a.cache, device)
            if i % 32 == 0 or i == len(roles[role]): print(f"Frozen features {role}: {i}/{len(roles[role])}", flush=True)
    if state_digest(model.body) != frozen_sha: raise ValueError("Caching changed frozen body/BN")
    if a.stage == "cache": return
    optimizer = torch.optim.AdamW(model.depth_decoder.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    checkpoints = sorted((a.output / "checkpoints").glob("epoch_*.pth")); history = []; steps = 0; start_epoch = 0
    if checkpoints:
        last = load_checkpoint(checkpoints[-1], run_id); model.depth_decoder.load_state_dict(last["decoder"], strict=True)
        optimizer.load_state_dict(last["optimizer"]); history = last["history"]; steps = last["optimizer_steps"]; start_epoch = last["epoch"]
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value): state[key] = value.to(device)
    if a.stage == "train":
        if not history:
            baseline = evaluate_decoder(model.depth_decoder, roles["dev"], a.cache, device, manifest["mapping"])
            common.immutable(a.output / "initial_dev.json", common.encoded(baseline))
        probe = read_features(roles["train"][0], a.cache)
        probe_features = batch_arrays([probe], device)[0]
        with torch.no_grad(): detection_before = model.full_urban_candidates(probe_features).detach().cpu().clone()
        for epoch in range(start_epoch + 1, config["epochs"] + 1):
            # Epoch-local seeds make a checkpoint-boundary resume independent
            # of how many cache hits or validation calls occurred previously.
            torch.manual_seed(config["seed"] + epoch)
            order = np.random.default_rng(config["seed"] + epoch).permutation(len(roles["train"]))
            losses = []; norms = []; began = time.perf_counter()
            with ThreadPoolExecutor(max_workers=4) as pool:
                for offset in range(0, len(order), config["batch_size"]):
                    indices = order[offset:offset+config["batch_size"]]
                    arrays = list(pool.map(lambda i: read_features(roles["train"][int(i)], a.cache), indices))
                    loss, norm = step(model.depth_decoder, optimizer, arrays, device, config)
                    losses.append(loss); norms.append(norm); steps += 1
            if state_digest(model.body) != frozen_sha or any(p.grad is not None for p in model.body.parameters()):
                raise ValueError("Frozen Urban weights/BN or gradient protection failed")
            with torch.no_grad(): detection_after = model.full_urban_candidates(probe_features).detach().cpu()
            if not torch.equal(detection_before, detection_after): raise ValueError("Pretrained detection branch changed")
            dev = evaluate_decoder(model.depth_decoder, roles["dev"], a.cache, device, manifest["mapping"])
            row = {"epoch": epoch, "optimizer_steps": steps, "train_huber_m": float(np.mean(losses)),
                   "maximum_pre_clip_gradient_norm": max(norms), "dev": dev, "seconds": time.perf_counter()-began,
                   "frozen_body_and_bn_unchanged": True, "detection_probe_exactly_preserved": True}
            history.append(row)
            save_checkpoint(a.output / "checkpoints" / f"epoch_{epoch:03d}.pth", {"run_identity": run_id, "epoch": epoch,
                "optimizer_steps": steps, "decoder": {k: v.detach().cpu() for k, v in model.depth_decoder.state_dict().items()},
                "optimizer": optimizer.state_dict(), "history": history})
            print(json.dumps(row), flush=True)
        if not history or history[-1]["epoch"] != config["epochs"]: raise ValueError("Training budget incomplete")
        best = min(history, key=lambda row: (row["dev"]["macro_image_mae_m"], row["epoch"]))
        selection = {"complete": True, "run_identity": run_id, "selected_epoch": best["epoch"], "selected_on": "dev_only_macro_image_mae_m",
                     "optimizer_steps": steps, "history": history, "frozen_body_state_sha256": frozen_sha,
                     "frozen_body_and_batchnorm_unchanged": True, "knowledge_distillation": False, "test_used_for_selection": False}
        common.immutable(a.output / "selection.json", common.encoded(selection)); return
    selection = json.loads((a.output / "selection.json").read_text())
    if selection["run_identity"] != run_id or not selection["complete"] or selection["test_used_for_selection"]:
        raise ValueError("Complete dev-only selection is required before held-out evaluation")
    report = a.output / "depth_pilot_results.json"
    if report.exists():
        saved = json.loads(report.read_text())
        if saved["run_identity"] != run_id: raise ValueError("Preserve different completed evaluation")
        print("Completed held-out result verified; no repeated selection", flush=True); return
    for i, sample in enumerate(roles["test"], 1):
        cache_sample(model, sample, a.inputs, a.cache, device)
        if i % 16 == 0: print(f"Held-out features: {i}/{len(roles['test'])}", flush=True)
    model.depth_decoder.load_state_dict(initial_decoder, strict=True)
    baseline = evaluate_decoder(model.depth_decoder, roles["test"], a.cache, device, manifest["mapping"], detailed=True)
    chosen = load_checkpoint(a.output / "checkpoints" / f"epoch_{selection['selected_epoch']:03d}.pth", run_id)
    model.depth_decoder.load_state_dict(chosen["decoder"], strict=True)
    trained = evaluate_decoder(model.depth_decoder, roles["test"], a.cache, device, manifest["mapping"], detailed=True)
    if state_digest(model.body) != frozen_sha: raise ValueError("Frozen body changed during evaluation")
    coverage_gates = {key: cell["images"] >= config["minimum_final_cell_images"] and cell["points"] >= config["minimum_final_cell_points"]
                      for key, cell in trained["cells"].items() if key.startswith("conservative/") and "/All/" not in key and key.endswith(("/2-4", "/4-6"))}
    result = {"schema_version": 1, "complete": True, "run_identity": run_id, "revision": config["revision"],
              "selected_epoch": selection["selected_epoch"], "optimizer_steps": selection["optimizer_steps"],
              "untrained_initialization": baseline, "trained_decoder": trained, "coverage": manifest["coverage"],
              "critical_cell_coverage_gates": coverage_gates, "all_critical_cells_have_descriptive_coverage": all(coverage_gates.values()),
              "frozen_urban_weights_and_bn_unchanged": True, "decoder_weights_changed": state_digest(model.depth_decoder) != identity["initial_decoder_state_sha256"],
              "training_performed": True, "knowledge_distillation": False, "test_used_for_selection": False,
              "per_image_scale_fitting": False, "GT_scope": manifest["GT_scope"], "phone_app_changed": False,
              "accuracy_pass_or_fail_threshold_declared": False, "deployment_qualified": False,
              "next_step": "Review raw metre errors and coverage before more training or a new export. No automatic phone replacement."}
    common.immutable(report, common.encoded(result)); print(json.dumps({k: v for k, v in result.items() if k not in ["untrained_initialization", "trained_decoder", "coverage"]}, indent=2))


def bundle(a):
    paths = [a.output / n for n in ["run_identity.json", "initial_dev.json", "selection.json", "depth_pilot_results.json"]]
    result = json.loads(paths[-1].read_text())
    identity = json.loads(paths[0].read_text()); selection = json.loads(paths[2].read_text())
    if (not result["complete"] or result["run_identity"] != common.identity(identity)
            or result["run_identity"] != selection["run_identity"]
            or result["selected_epoch"] != selection["selected_epoch"]): raise ValueError("Incomplete or mismatched evaluation")
    selected = a.output / "checkpoints" / f"epoch_{result['selected_epoch']:03d}.pth"
    load_checkpoint(selected, result["run_identity"])
    archive = a.output / "parking_shared_depth_results.zip"
    members = {path.name: path for path in paths}
    members.update({f"checkpoints/{path.name}": path for path in [selected, selected.with_suffix(".json")]})
    members.update({f"source_snapshot/{path.name}": path for path in (a.output / "source_snapshot").glob("*.py")})
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as target:
        for member, path in sorted(members.items()):
            info = zipfile.ZipInfo(member, (1980, 1, 1, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(info, path.read_bytes())
    common.immutable(archive, buffer.getvalue())
    common.immutable(a.output / "bundle_receipt.json", common.encoded({"file": archive.name, "sha256": common.file_sha(archive),
        "members_sha256": {member: common.file_sha(path) for member, path in members.items()},
        "packaging_helper_sha256": common.file_sha(Path(__file__)), "optimizer_steps_taken_by_packaging": 0}))
    print(archive)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("stage", choices=["cache", "train", "evaluate", "bundle"])
    p.add_argument("--inputs", type=Path); p.add_argument("--cache", type=Path); p.add_argument("--source", type=Path)
    p.add_argument("--weights", type=Path); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    a = p.parse_args()
    if a.stage == "bundle": bundle(a)
    else:
        if any(getattr(a, k) is None for k in ["inputs", "cache", "source", "weights"]): p.error("Model stages require --inputs --cache --source --weights")
        run(a)


if __name__ == "__main__": main()
