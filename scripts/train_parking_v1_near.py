"""Bounded matched V1 adapter pilot. No held-out evaluation or app changes.

CPU stage first (parking_v1_pilot_data.py), then explicit GPU smoke/cache/train.
Completed blocks are immutable/resumable; partial blocks restart. Frozen image
features are cached locally once and reused in both arms. No external teacher.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import functools
import importlib.metadata
import io
import json
import math
from pathlib import Path
import platform
import random
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_v1_pilot_data as data
from scripts import parking_edge_common as common
from scripts import parking_v0_v1_metrics as metrics
from scripts import parking_v0_local_io as local


def sources():
    names = ["scripts/train_parking_v1_near.py", "scripts/parking_v1_pilot_data.py",
        "configs/parking_v1_matched_pilot_v1.json", "configs/parking_v1_near_field_proposal_v1.json",
        "models/parking_shared_camera.py", "models/parking_shared.py",
        "scripts/train_parking_shared_depth.py", "scripts/train_parking_shared_camera.py",
        "scripts/screen_parking_shared_model.py", "scripts/parking_shared_depth_data.py",
        "scripts/parking_shared_body_verification.py", "scripts/parking_v0_v1_metrics.py",
        "scripts/parking_near_field_geometry.py", "scripts/evaluate_parking_v0_dev.py",
        "scripts/parking_edge_common.py", "scripts/prepare_parking_v1_training.py",
        "configs/parking_v1_training_inputs_v1.json", "scripts/parking_v0_local_io.py"]
    return {n: common.file_sha(ROOT / n) for n in names}


def runtime(device):
    import torch
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    versions = {n: importlib.metadata.version(n) for n in ["numpy", "pillow", "scipy", "ultralytics", "opencv-python-headless"]}
    expected = {"numpy": "2.1.3", "pillow": "11.3.0", "scipy": "1.16.3", "ultralytics": "8.4.61", "opencv-python-headless": "5.0.0.93"}
    if versions != expected: raise ValueError("Run pinned task-package setup in the notebook: " + json.dumps(versions))
    if device == "cuda" and not torch.cuda.is_available(): raise ValueError("Select a GPU runtime before smoke/cache/train")
    return {"python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "CPU",
        "packages": versions, "opencv_imported": __import__("cv2").__version__,
        "deterministic": True, "tf32": False, "cpu_threads": torch.get_num_threads()}


def freeze_adapter(decoder):
    decoder.eval().requires_grad_(True)
    decoder.base.eval().requires_grad_(False)
    return [p for p in decoder.parameters() if p.requires_grad]


def build(a, c, receipt):
    import torch
    from models.parking_shared_camera import CameraAwareDepthDecoder
    from scripts import train_parking_shared_depth as original
    from scripts import parking_shared_body_verification as verification
    from scripts.screen_parking_shared_model import state_digest
    run_runtime = runtime(a.device)
    parent, saved = local.v0.read_v0_bundle(a.inputs / "dev/model/parking_shared_adapter_results.zip", local.v0.configuration(), load=True)
    native, model_id = original.build_model(a, parent["model"]["config"])
    checked = verification.verify_warmstart_body(native, model_id, parent["model"]["config"], a.weights)
    native.eval().requires_grad_(False)
    decoder = CameraAwareDepthDecoder(native.depth_decoder, width=c["adapter_width"]).to(a.device)
    decoder.load_state_dict(saved["decoder"], strict=True)
    params = freeze_adapter(decoder)
    if sum(p.numel() for p in params) != c["trainable_parameters"]:
        raise ValueError("Only the existing 1553 adapter parameters may train")
    architecture = common.identity({"shared": model_id["architecture"], "adapter": "CameraAwareDepthDecoder", "width": c["adapter_width"]})
    proposal = json.loads((ROOT / "configs/parking_v1_near_field_proposal_v1.json").read_text())
    if architecture != proposal["unchanged"]["architecture_sha256"]: raise ValueError("V1 architecture changed")
    identity = {"config": c, "input_receipt": receipt, "runtime": run_runtime,
        "v0_checkpoint_sha256": local.v0.configuration()["checkpoint_sha256"], "architecture_sha256": architecture,
        "initial_decoder_state_sha256": state_digest(decoder), "frozen_base_sha256": state_digest(decoder.base),
        "frozen_body_sha256": state_digest(native.body), "backbone_verification": checked, "source_sha256": sources()}
    common.immutable(a.output / "run_identity.json", common.encoded(identity))
    for name in identity["source_sha256"]:
        common.immutable(a.output / "source_snapshot" / name, (ROOT / name).read_bytes())
    return native, decoder, identity


def frozen_check(native, decoder, identity):
    from scripts.screen_parking_shared_model import state_digest
    if (state_digest(native.body) != identity["frozen_body_sha256"]
            or state_digest(decoder.base) != identity["frozen_base_sha256"]
            or any(p.requires_grad or p.grad is not None for p in native.body.parameters())
            or any(p.requires_grad or p.grad is not None for p in decoder.base.parameters())
            or any(m.training for m in native.body.modules()) or any(m.training for m in decoder.base.modules())):
        raise ValueError("Frozen image body, detector, base-depth weights or BN state changed")


def feature_payload(arrays, sample, cache_id):
    import numpy as np
    members = {}
    for name, array in arrays.items():
        stream = io.BytesIO(); np.save(stream, array, allow_pickle=False)
        members[name+".npy"] = stream.getvalue()
    members["cache_manifest.json"] = common.encoded({"cache_identity": cache_id,
        "sample_sha256": common.identity(sample), "files": {n: common.sha(v) for n, v in members.items()}})
    return local.v0.zip_bytes(members)


def read_features(path, sample, cache_id):
    import numpy as np
    with zipfile.ZipFile(path) as z:
        record = json.loads(z.read("cache_manifest.json"))
        if (record["cache_identity"] != cache_id or record["sample_sha256"] != common.identity(sample)
                or set(z.namelist()) != set(record["files"]) | {"cache_manifest.json"}
                or len(z.namelist()) != len(record["files"])+1 or z.testzip() is not None):
            raise ValueError("Frozen feature cache identity/inventory differs")
        arrays = {}
        for name, digest in record["files"].items():
            raw = z.read(name)
            if common.sha(raw) != digest: raise ValueError("Cached feature bytes differ")
            arrays[Path(name).stem] = np.load(io.BytesIO(raw), allow_pickle=False)
    expected = {"p3": (64, 48, 80), "p4": (128, 24, 40), "p5": (256, 12, 20)}
    for key, shape in expected.items():
        if arrays[key].shape != shape or arrays[key].dtype != np.float32 or not np.isfinite(arrays[key]).all():
            raise ValueError("Cached feature shape/dtype/values changed")
    n = len(arrays["z"])
    if (arrays["grid"].shape != (n,2) or arrays["object_index"].shape != (n,)
            or arrays["cell_masks"].shape != (len(sample.get("cell_names", metrics.cell_names(data.prepare.reference.load_protocol()))), n)):
        raise ValueError("Cached near reference arrays differ")
    for prefix in ["", "broad_"]:
        z, grid = arrays[prefix+"z"], arrays[prefix+"grid"]
        if (z.dtype != np.float32 or grid.dtype != np.float32 or grid.shape != (len(z),2)
                or not np.isfinite(z).all() or not (z > 0).all() or not np.isfinite(grid).all()
                or (abs(grid) > 1).any()): raise ValueError("Cached sampling coordinates/metric truth changed")
    return arrays


def cache_one(native, rgb, points, broad, sample, device):
    import numpy as np
    import torch
    from scripts.screen_parking_shared_model import preprocess
    from scripts.parking_shared_depth_data import point_grid
    tensor, meta = preprocess(rgb, intrinsics=sample["intrinsics"])
    with torch.no_grad(): features = native.image_features(torch.from_numpy(tensor).to(device))
    arrays = {f"p{i+3}": f[0].detach().cpu().numpy().copy() for i, f in enumerate(features)}
    arrays.update({k: points[k] for k in ["z", "object_index", "cell_masks"]})
    arrays["grid"] = point_grid(points["xy"], sample["width"], sample["height"], meta)
    arrays["broad_z"] = broad["z"]
    arrays["broad_grid"] = point_grid(broad["xy"], sample["width"], sample["height"], meta)
    return arrays


def index_observations(index, sample, points, c):
    import numpy as np
    if sample["role"] != "train": raise ValueError("Sampler may contain only training observations")
    names = sample["cell_names"]
    for cell in c["near_cells"]:
        mask = points["cell_masks"][names.index(cell)]
        for owner in np.unique(points["object_index"][mask]):
            obj = sample["objects"][int(owner)]
            if int((mask & (points["object_index"] == owner)).sum()) < 5: raise ValueError("Ineligible sampler object")
            by_log = index["near"][cell].setdefault(sample["recording_log"], {})
            by_ann = by_log.setdefault(obj["instance_token"], {}).setdefault(obj["annotation_token"], [])
            if sample["camera_token"] in by_ann: raise ValueError("Repeated sampler view")
            by_ann.append(sample["camera_token"])
    if sample["broad_reference_points"]:
        index["broad"].setdefault(sample["recording_log"], []).append(sample["camera_token"])


def cache(a, native, decoder, identity, manifest, benchmark, inventory):
    import numpy as np
    cache_id = common.identity({k: identity[k] for k in ["input_receipt", "runtime", "frozen_body_sha256", "source_sha256"]})
    common.immutable(a.cache / "identity.json", common.encoded({"cache_identity": cache_id}))
    index = {"near": {cell: {} for cell in identity["config"]["near_cells"]}, "broad": {}}
    completed = 0
    for name, samples in data.training_shards(manifest):
        records = data.load_train_shard(a.inputs, name, samples, manifest["near_mask_protocol"])
        for sample in samples:
            token = sample["camera_token"]; record = records[token]; path = a.cache / (token+".zip")
            index_observations(index, sample, record["ground_truth"], identity["config"])
            if not path.exists():
                arrays = cache_one(native, record["rgb"], record["ground_truth"], record["broad"], sample, a.device)
                common.immutable(path, feature_payload(arrays, sample, cache_id))
            read_features(path, sample, cache_id); completed += 1
        print(f"Train features cached/verified {completed}/{len(manifest['samples'])}", flush=True)
    for shard, samples in inventory:
        images, points = local.v0.load_dev_shard(a.inputs / "dev/benchmark" / shard["file"], shard, samples,
            benchmark["metric_protocol"], benchmark["benchmark_identity_sha256"])
        for sample in samples:
            token = sample["camera_token"]; path = a.cache / (token+".zip")
            if not path.exists():
                empty = {"xy": np.empty((0,2),np.int32), "z": np.empty(0,np.float32)}
                arrays = cache_one(native, images[token], points[token], empty, sample, a.device)
                common.immutable(path, feature_payload(arrays, sample, cache_id))
            read_features(path, sample, cache_id)
        print("Dev features cached/verified: " + shard["file"], flush=True)
    if any(not index["near"][cell] for cell in identity["config"]["near_cells"]) or not index["broad"]:
        raise ValueError("Every training component needs support")
    frozen_check(native, decoder, identity)
    common.immutable(a.cache / "sampling_index.json", common.encoded(index))
    common.immutable(a.cache / "complete.json", common.encoded({"cache_identity": cache_id,
        "train_images": len(manifest["samples"]), "dev_images": sum(len(s) for _, s in inventory),
        "sampler_index_sha256": common.identity(index)}))
    return cache_id, index


def batches(index, c, block):
    if not 1 <= block <= c["blocks"]: raise ValueError("Block outside the approved budget")
    rng = random.Random(c["seed"]+block)
    choose = lambda values: rng.choice(sorted(values))
    result = []
    for _ in range(c["updates_per_block"]):
        draws = []
        for cell in c["near_cells"]:
            logs = index["near"][cell]; log = choose(logs); tracks = logs[log]; track = choose(tracks)
            anns = tracks[track]; ann = choose(anns); token = choose(anns[ann])
            draws.append({"camera_token": token, "cell": cell, "log": log, "track": track, "annotation": ann})
        for _ in range(2):
            log = choose(index["broad"]); token = choose(index["broad"][log])
            draws.append({"camera_token": token, "cell": "broad", "log": log})
        result.append(draws)
    return result


def predict_points(decoder, rows, samples, device, prefix=""):
    import numpy as np
    import torch
    from scripts.train_parking_shared_camera import camera_vector
    from scripts.train_parking_shared_depth import sample_depth
    features = [torch.from_numpy(np.stack([row[f"p{i}"] for row in rows])).to(device) for i in [3,4,5]]
    cameras = torch.from_numpy(np.stack([camera_vector(s) for s in samples])).to(device)
    depth = decoder(features, cameras)
    if tuple(depth.shape) != (len(rows),1,96,160): raise ValueError("Depth output contract changed")
    result = []
    for i, row in enumerate(rows):
        grid = torch.from_numpy(row[prefix+"grid"][None,:,None,:]).to(device)
        result.append(sample_depth(depth[i:i+1], grid)[0])
    return result, depth


def loss_components(predictions, broad_predictions, rows, samples, c, arm):
    import numpy as np
    import torch
    from torch.nn import functional as F
    if arm not in c["arms"]: raise ValueError("Unknown training arm")
    cells = c["control_cells"] if arm == c["arms"][0] else c["near_cells"]
    near = {}
    for cell in cells:
        images = []
        for pred, row, sample in zip(predictions, rows, samples):
            mask = row["cell_masks"][sample["cell_names"].index(cell)]
            objects = []
            for owner in np.unique(row["object_index"][mask]):
                selected = torch.from_numpy(mask & (row["object_index"] == owner)).to(pred.device)
                truth = torch.from_numpy(row["z"]).to(pred.device)
                objects.append(F.huber_loss(pred[selected], truth[selected], delta=c["huber_delta_m"]))
            if objects: images.append(torch.stack(objects).mean())
        if not images: raise ValueError("A declared near loss component is missing: " + cell)
        near[cell] = torch.stack(images).mean()
    broad = []
    for pred, row in zip(broad_predictions, rows):
        if len(row["broad_z"]):
            truth = torch.from_numpy(row["broad_z"]).to(pred.device)
            broad.append(F.huber_loss(pred, truth, delta=c["huber_delta_m"]))
    if not broad: raise ValueError("Missing broad supervision")
    if any(not torch.isfinite(p).all() or not (p > 0).all() for p in predictions+broad_predictions):
        raise ValueError("Nonpositive/nonfinite predictions; never hide them from the training loss")
    near_loss = torch.stack(list(near.values())).mean(); broad_loss = torch.stack(broad).mean()
    loss = c["near_loss_weight"]*near_loss + c["broad_loss_weight"]*broad_loss
    return loss, {**near, "near_mean": near_loss, "broad": broad_loss}


def adapter_step(decoder, optimizer, rows, samples, c, arm, device, update=True):
    import torch
    from scripts.train_parking_shared_depth import sample_depth
    decoder.eval(); optimizer.zero_grad(set_to_none=True)
    prediction, depth = predict_points(decoder, rows, samples, device)
    broad = [sample_depth(depth[i:i+1], torch.from_numpy(r["broad_grid"][None,:,None,:]).to(device))[0] for i,r in enumerate(rows)]
    loss, parts = loss_components(prediction, broad, rows, samples, c, arm)
    loss.backward()
    params = [p for p in decoder.parameters() if p.requires_grad]
    if ({id(p) for g in optimizer.param_groups for p in g["params"]} != {id(p) for p in params}
            or any(p.grad is None or not torch.isfinite(p.grad).all() for p in params)):
        raise ValueError("Optimizer/finite gradient scope differs")
    norm = torch.nn.utils.clip_grad_norm_(params, c["gradient_clip_norm"], error_if_nonfinite=True)
    if update: optimizer.step()
    return {"loss": float(loss.detach()), "pre_clip_gradient_norm": float(norm),
        "components": {k: float(v.detach()) for k,v in parts.items()}}


def signed_bias(samples, points, predictions, protocol):
    import numpy as np
    result = {}
    for ci, cell in enumerate(metrics.cell_names(protocol)):
        groups = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        for sample in samples:
            token = sample["camera_token"]; row = points[token]; pred = predictions[token]
            mask = row["cell_masks"][ci] & np.isfinite(pred) & (pred > 0)
            for owner in np.unique(row["object_index"][mask]):
                obj = sample["objects"][int(owner)]; chosen = mask & (row["object_index"] == owner)
                groups[sample["recording_log"]][obj["instance_token"]][obj["annotation_token"]][token] = float(
                    (pred[chosen].astype(np.float64)-row["z"][chosen].astype(np.float64)).mean())
        logs = [np.mean([np.mean([np.mean(list(cameras.values())) for cameras in anns.values()])
            for anns in tracks.values()]) for tracks in groups.values()]
        result[cell] = float(np.mean(logs)) if logs else None
    return result


def evaluate(a, decoder, identity, benchmark, inventory, cache_id, arm, block):
    import numpy as np
    import torch
    from scripts.screen_parking_shared_model import state_digest
    decoder.eval()
    folder = a.output / "endpoints" / arm / f"block_{block:03d}"
    binding = {"run_identity": common.identity(identity), "arm": arm, "block": block, "decoder_sha256": state_digest(decoder)}
    common.immutable(folder / "identity.json", common.encoded(binding)); pred_id = common.identity(binding)
    all_points = {}; predictions = {}; samples = []; accumulator = metrics.Metrics(benchmark["metric_protocol"])
    for shard, batch in inventory:
        _, points = local.v0.load_dev_shard(a.inputs / "dev/benchmark" / shard["file"], shard, batch,
            benchmark["metric_protocol"], benchmark["benchmark_identity_sha256"])
        path = folder / "predictions" / Path(shard["file"]).name
        if path.exists():
            values = local.v0.read_prediction_chunk(path.read_bytes(), pred_id, batch, points)
        else:
            values = {}
            with torch.no_grad():
                for sample in batch:
                    token = sample["camera_token"]
                    row = read_features(a.cache / (token+".zip"), sample, cache_id)
                    value, _ = predict_points(decoder, [row], [sample], a.device)
                    values[token] = value[0].cpu().numpy().astype(np.float32, copy=True)
            common.immutable(path, local.v0.prediction_chunk(pred_id, batch, values))
        for sample in batch:
            token = sample["camera_token"]; accumulator.add(sample, points[token], values[token])
        samples.extend(batch); all_points.update(points); predictions.update(values)
    result = accumulator.result(); result["signed_bias_m"] = signed_bias(samples, all_points, predictions, benchmark["metric_protocol"])
    common.immutable(folder / "metrics.json", common.encoded(result))
    return result, samples, all_points, predictions


def read_endpoint(a, identity, benchmark, inventory, arm, block):
    folder = a.output / "endpoints" / arm / f"block_{block:03d}"
    binding = json.loads((folder / "identity.json").read_text())
    if binding["run_identity"] != common.identity(identity) or binding["arm"] != arm or binding["block"] != block:
        raise ValueError("Endpoint prediction identity differs")
    predictions = {}; points = {}; samples = []
    for shard, batch in inventory:
        _, truth = local.v0.load_dev_shard(a.inputs / "dev/benchmark" / shard["file"], shard, batch,
            benchmark["metric_protocol"], benchmark["benchmark_identity_sha256"])
        raw = (folder / "predictions" / Path(shard["file"]).name).read_bytes()
        predictions.update(local.v0.read_prediction_chunk(raw, common.identity(binding), batch, truth))
        points.update(truth); samples.extend(batch)
    return samples, points, predictions


def replay(a, decoder, identity, benchmark, inventory, cache_id):
    import numpy as np
    from scripts.diagnose_parking_v0_errors import verify_result
    members, registration, reviewed = verify_result(a.inputs / "archives/v0_complete.zip", local.v0.configuration())
    current, _, _, _ = evaluate(a, decoder, identity, benchmark, inventory, cache_id, "V0", 0)
    maximum = 0.
    for shard, rows in inventory:
        _, truth = local.v0.load_dev_shard(a.inputs / "dev/benchmark" / shard["file"], shard, rows,
            benchmark["metric_protocol"], benchmark["benchmark_identity_sha256"])
        old = local.v0.read_prediction_chunk(members["predictions/"+Path(shard["file"]).name], common.identity(registration), rows, truth)
        folder = a.output / "endpoints/V0/block_000"
        binding = json.loads((folder/"identity.json").read_text())
        now = local.v0.read_prediction_chunk((folder/"predictions"/Path(shard["file"]).name).read_bytes(), common.identity(binding), rows, truth)
        for token in old:
            if not np.allclose(now[token], old[token], atol=identity["config"]["v0_replay_atol_m"], rtol=identity["config"]["v0_replay_rtol"]):
                raise ValueError("Unchanged V0 failed pointwise CPU/GPU replay; no training authorized on this runtime")
            maximum = max(maximum, float(np.max(np.abs(now[token]-old[token]))))
    receipt = {"complete": True, "pointwise_v0_replay_passed": True, "maximum_abs_delta_m": maximum,
        "reviewed_cpu_primary_mae_m": reviewed["metrics"]["conditional_equal_four_cell_mae_m"],
        "current_runtime_primary_mae_m": current["conditional_equal_four_cell_mae_m"],
        "optimizer_steps": 0, "heldout_bodies_opened": 0}
    common.immutable(a.output / "v0_replay.json", common.encoded(receipt))
    return current


def checkpoint(path, value):
    import torch
    stream = io.BytesIO(); torch.save(value, stream); raw = stream.getvalue()
    common.immutable(path, local.v0.zip_bytes({"state.pt": raw, "inventory.json": common.encoded({"sha256": common.sha(raw)})}))


def read_checkpoint(path, run_id):
    import torch
    with zipfile.ZipFile(path) as z:
        if set(z.namelist()) != {"state.pt", "inventory.json"} or len(z.namelist()) != 2 or z.testzip() is not None:
            raise ValueError("Checkpoint inventory/CRC differs")
        raw = z.read("state.pt")
        if len(raw) > 10_000_000 or common.sha(raw) != json.loads(z.read("inventory.json"))["sha256"]:
            raise ValueError("Checkpoint digest differs")
    saved = torch.load(io.BytesIO(raw), weights_only=True, map_location="cpu")
    if saved["run_identity"] != run_id: raise ValueError("Checkpoint from another run")
    return saved


def best_endpoint(history, baseline):
    def eligible(row):
        score = row["dev"]["conditional_equal_four_cell_mae_m"]
        return (score is not None and math.isfinite(score) and all(
            cell["prediction_coverage"] is None or (base["prediction_coverage"] is not None
                and cell["prediction_coverage"] >= base["prediction_coverage"])
            for cell, base in [(v, baseline["cells"][k]) for k, v in row["dev"]["cells"].items()]))
    valid = [r for r in history if eligible(r)]
    if not valid: raise ValueError("No finite endpoint with preserved coverage")
    return min(valid, key=lambda r: (r["dev"]["conditional_equal_four_cell_mae_m"], r["block"]))


def validate_resume(saved, arm, run_id, index, c):
    block = saved["block"]; history = saved["history"]
    if (saved["run_identity"] != run_id or saved["arm"] != arm or not 0 <= block <= c["blocks"]
            or saved["optimizer_steps"] != block*c["updates_per_block"] or len(history) != block+1):
        raise ValueError("Resume arm/block/budget differs")
    for number, row in enumerate(history):
        if row["block"] != number or row["optimizer_steps"] != number*c["updates_per_block"]:
            raise ValueError("Resume history is not contiguous")
        if number and row["sampler_sha256"] != common.identity(batches(index, c, number)):
            raise ValueError("Resume sampler differs")


def progress(a, identity, label):
    """Only local output state, never RGB/features, in one portable archive."""
    members = {}
    for path in sorted(a.output.rglob("*")):
        if path.is_file() and not any(k in path.relative_to(a.output).parts for k in ["portable", "runtime_settings", "colab_logs"]):
            if path.suffix == ".tmp": continue
            members[str(path.relative_to(a.output))] = path.read_bytes()
    members["progress_inventory.json"] = common.encoded({"run_identity": common.identity(identity),
        "files": {n: common.sha(v) for n,v in members.items()}, "heldout_bodies_included": 0})
    target = a.output / "portable" / (label+".zip")
    common.immutable(target, local.stored_zip(members))
    if a.publish is not None:
        try:
            data.transfer(target, a.publish / target.name, common.file_sha(target))
            print("Saved one durable archive: " + str(a.publish / target.name), flush=True)
        except (OSError, RuntimeError) as error:
            print(f"WARNING: durable archive save failed: {error}. Local progress is safe; DOWNLOAD {target} before ending this runtime.", flush=True)
    return target


def restore_progress(path, output):
    if path.stat().st_size > 250_000_000: raise ValueError("Oversized progress archive")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        if (len(names) != len(set(names)) or len(names) > 1024
                or sum(i.file_size for i in z.infolist()) > 250_000_000 or z.testzip() is not None):
            raise ValueError("Progress archive inventory/CRC/size differs")
        for name in names: local.v0.safe_name(name)
        members = {n: z.read(n) for n in names}
    record = json.loads(members["progress_inventory.json"])
    if (set(members) != set(record["files"]) | {"progress_inventory.json"}
            or record["heldout_bodies_included"] or record["run_identity"] != common.identity(json.loads(members["run_identity.json"]))):
        raise ValueError("Progress archive identity/inventory differs")
    for name, digest in record["files"].items():
        local.v0.safe_name(name)
        if name.startswith(("portable/", "runtime_settings/", "colab_logs/")) or common.sha(members[name]) != digest:
            raise ValueError("Progress member changed")
        common.immutable(output / name, members[name])


def train_arm(a, native, decoder, identity, index, cache_id, manifest, benchmark, inventory, initial, baseline, arm):
    import numpy as np
    import torch
    c = identity["config"]; run_id = common.identity(identity)
    decoder.load_state_dict(initial, strict=True); params = freeze_adapter(decoder)
    optimizer = torch.optim.AdamW(params, lr=c["learning_rate"], weight_decay=c["weight_decay"])
    folder = a.output / "checkpoints" / arm
    paths = sorted(folder.glob("block_*.zip"))
    if paths:
        saved = read_checkpoint(paths[-1], run_id); validate_resume(saved, arm, run_id, index, c)
        if paths[-1].name != f"block_{saved['block']:03d}.zip": raise ValueError("Checkpoint filename differs")
        decoder.load_state_dict(saved["decoder"], strict=True); optimizer.load_state_dict(saved["optimizer"])
        history = saved["history"]
    else:
        history = [{"block": 0, "optimizer_steps": 0, "dev": baseline}]
        checkpoint(folder/"block_000.zip", {"run_identity": run_id, "arm": arm, "block": 0,
            "optimizer_steps": 0, "decoder": initial, "optimizer": optimizer.state_dict(), "history": history})
    frozen_check(native, decoder, identity)
    samples_by_token = {s["camera_token"]: s for s in manifest["samples"]}
    @functools.lru_cache(maxsize=32)
    def cached(token): return read_features(a.cache / (token+".zip"), samples_by_token[token], cache_id)
    for block in range(history[-1]["block"]+1, c["blocks"]+1):
        torch.manual_seed(c["seed"]+block)
        stream = batches(index, c, block); losses = []; norms = []; components = defaultdict(list)
        with ThreadPoolExecutor(max_workers=c["workers"]) as pool:
            for update, draws in enumerate(stream, 1):
                samples = [samples_by_token[d["camera_token"]] for d in draws]
                rows = list(pool.map(lambda s: cached(s["camera_token"]), samples))
                info = adapter_step(decoder, optimizer, rows, samples, c, arm, a.device)
                losses.append(info["loss"]); norms.append(info["pre_clip_gradient_norm"])
                for key, value in info["components"].items(): components[key].append(value)
                if update % 32 == 0: print(f"{arm}: block {block}/5 update {update}/384 loss={info['loss']:.6f}", flush=True)
        frozen_check(native, decoder, identity)
        dev, _, _, _ = evaluate(a, decoder, identity, benchmark, inventory, cache_id, arm, block)
        history.append({"block": block, "optimizer_steps": block*c["updates_per_block"],
            "sampler_sha256": common.identity(stream), "unique_image_draws": len({d["camera_token"] for b in stream for d in b}),
            "draws": len(stream)*8, "draw_log_counts": dict(Counter(d["log"] for b in stream for d in b)),
            "mean_training_loss": float(np.mean(losses)), "max_gradient_norm_before_clip": max(norms),
            "mean_components": {k: float(np.mean(v)) for k,v in components.items()}, "dev": dev})
        checkpoint(folder / f"block_{block:03d}.zip", {"run_identity": run_id, "arm": arm, "block": block,
            "optimizer_steps": block*c["updates_per_block"], "decoder": {k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()},
            "optimizer": optimizer.state_dict(), "history": history})
        progress(a, identity, f"{arm}_block_{block:03d}")
        print(json.dumps({"arm": arm, "block": block, "primary_mae_m": dev["conditional_equal_four_cell_mae_m"]}), flush=True)
    best = best_endpoint(history, baseline)
    selection = {"complete": True, "run_identity": run_id, "arm": arm, "selected_block": best["block"],
        "selected_dev": best["dev"], "fixed_block_5_dev": history[-1]["dev"], "history": history,
        "optimizer_steps": c["updates_per_arm"], "heldout_used": False, "promotion_authorized": False}
    common.immutable(a.output / arm / "selection.json", common.encoded(selection))
    return selection


def final_report(a, identity, benchmark, inventory, baseline, selections):
    samples, points, v0 = read_endpoint(a, identity, benchmark, inventory, "V0", 0)
    pairs = {}; dev_results = {}
    for arm, selected in selections.items():
        if selected["selected_block"]:
            _, _, predictions = read_endpoint(a, identity, benchmark, inventory, arm, selected["selected_block"])
        else: predictions = v0
        dev_results[arm] = predictions
        pairs[arm+"_vs_V0"] = metrics.paired_metrics(samples, points, v0, predictions, benchmark["metric_protocol"])
    arms = identity["config"]["arms"]
    pairs["candidate_vs_control"] = metrics.paired_metrics(samples, points, dev_results[arms[0]], dev_results[arms[1]], benchmark["metric_protocol"])
    for arm in arms:
        for other in arms:
            if [r.get("sampler_sha256") for r in selections[arm]["history"]] != [r.get("sampler_sha256") for r in selections[other]["history"]]:
                raise ValueError("Matched arms consumed different declared sampling streams")
    result = {"complete": True, "revision": identity["config"]["revision"], "run_identity": common.identity(identity),
        "baseline": baseline, "arms": selections, "paired": pairs,
        "updates_per_arm": identity["config"]["updates_per_arm"], "matched_sampling_streams": True,
        "architecture_unchanged": True, "frozen_body_base_depth_bn_unchanged": True,
        "trainable_parameters": 1553, "external_teacher_used": False, "heldout_bodies_opened": 0,
        "heldout_evaluated": False, "phone_app_changed": False, "accuracy_pass_fail_enabled": False,
        "promotion_authorized": False, "deployment_qualified": False,
        "limitations": ["Sparse GT-object-owned camera-Z depth errors, not detector recall/mAP or ultrasonic surface range.",
            "Front-camera Vehicle 0-2m lacks training examples; sub-1m remains exploratory.",
            "Development-selected results require review; no automatic held-out scoring, extra blocks or phone replacement."]}
    common.immutable(a.output / "v1_pilot_results.json", common.encoded(result))
    return progress(a, identity, "parking_v1_matched_pilot_results")


def run(a):
    import torch
    c, receipt, manifest, benchmark, inventory = data.load_prepared(a.inputs)
    for path in [a.output, a.cache, a.source, a.weights]: data.local_path(path)
    data.paths_separate([a.inputs, a.output, a.cache, a.source, a.weights])
    if a.resume_archive is not None: restore_progress(a.resume_archive, a.output)
    native, decoder, identity = build(a, c, receipt)
    if a.stage == "bundle":
        if not (a.output / "v1_pilot_results.json").exists(): raise ValueError("Complete both pilot arms before final bundle")
        print(progress(a, identity, "parking_v1_matched_pilot_results")); return
    initial = {k:v.detach().cpu().clone() for k,v in decoder.state_dict().items()}
    if a.stage == "cache":
        cache(a, native, decoder, identity, manifest, benchmark, inventory); return
    cache_id = json.loads((a.cache / "identity.json").read_text())["cache_identity"]
    complete = json.loads((a.cache / "complete.json").read_text()); index = json.loads((a.cache / "sampling_index.json").read_text())
    expected_cache = common.identity({k: identity[k] for k in ["input_receipt", "runtime", "frozen_body_sha256", "source_sha256"]})
    if (cache_id != expected_cache or complete["cache_identity"] != cache_id
            or complete["sampler_index_sha256"] != common.identity(index)
            or complete["train_images"] != c["train_images"] or complete["dev_images"] != c["dev_images"]):
        raise ValueError("Finish cache stage in this exact runtime before smoke/train")
    baseline = replay(a, decoder, identity, benchmark, inventory, cache_id)
    probe = batches(index, c, 1)[0]; by_token = {s["camera_token"]:s for s in manifest["samples"]}
    chosen = [by_token[d["camera_token"]] for d in probe]
    rows = [read_features(a.cache / (s["camera_token"]+".zip"), s, cache_id) for s in chosen]
    for arm in c["arms"]:
        decoder.load_state_dict(initial, strict=True); params = freeze_adapter(decoder)
        optimizer = torch.optim.AdamW(params, lr=c["learning_rate"], weight_decay=c["weight_decay"])
        adapter_step(decoder, optimizer, rows, chosen, c, arm, a.device, update=False)
        from scripts.screen_parking_shared_model import state_digest
        if state_digest(decoder) != identity["initial_decoder_state_sha256"]: raise ValueError("Smoke changed model weights")
        decoder.zero_grad(set_to_none=True); frozen_check(native, decoder, identity)
    common.immutable(a.output / "runtime_smoke.json", common.encoded({"complete": True,
        "both_arms_finite_backward": True, "optimizer_steps": 0, "v0_replay_passed": True, "frozen_state_unchanged": True}))
    if a.stage == "smoke": print("Smoke passed: V0 replay and both-arm adapter gradients, zero optimizer updates", flush=True); return
    selections = {}
    for arm in c["arms"]:
        selections[arm] = train_arm(a, native, decoder, identity, index, cache_id, manifest, benchmark, inventory, initial, baseline, arm)
    frozen_check(native, decoder, identity)
    print("Return result archive: " + str(final_report(a, identity, benchmark, inventory, baseline, selections)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["cache", "smoke", "train", "bundle"])
    for name in ["inputs", "cache", "source", "weights", "output"]: p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--publish", type=Path)
    p.add_argument("--resume-archive", type=Path)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    run(p.parse_args())


if __name__ == "__main__": main()
