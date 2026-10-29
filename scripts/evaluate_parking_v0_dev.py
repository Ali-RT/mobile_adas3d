"""Freeze the reviewed epoch-4 V0 and score ONLY frozen development shards.

CPU inference, no CUDA extensions, optimizer, model selection or phone edits.
One atomic prediction ZIP per dev data shard makes completed work resumable.
The original benchmark and training artifacts are read-only.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import io
import json
from pathlib import Path, PurePosixPath
import platform
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common
from scripts import parking_v0_v1_metrics as metrics

CONFIG = ROOT / "configs/parking_v0_dev_evaluation_v1.json"


def configuration():
    return json.loads(CONFIG.read_text())


def check_signed(value):
    if value.get("manifest_sha256") != common.identity({k: v for k, v in value.items() if k != "manifest_sha256"}):
        raise ValueError("Frozen record digest differs")
    return value


def safe_name(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError("Unsafe artifact member")
    return path


def dev_inventory(manifest, config):
    check_signed(manifest)
    if (not manifest.get("complete") or manifest["manifest_sha256"] != config["benchmark_manifest_sha256"]
            or manifest["benchmark_identity_sha256"] != config["benchmark_identity_sha256"]
            or manifest["metric_protocol"] != json.loads((ROOT / "configs/parking_v0_v1_benchmark_v1.json").read_text())
            or config["role"] != "dev" or config["heldout_scoring_authorized"] or config["training_authorized"]):
        raise ValueError("Use the reviewed frozen benchmark and development-only contract")
    samples = [s for s in manifest["samples"] if s["role"] == "dev"]
    if len(samples) != config["expected_images"] or len({s["camera_token"] for s in samples}) != len(samples):
        raise ValueError("Development image inventory differs")
    by_shard = defaultdict(list)
    index = {s["file"]: s for s in manifest["shards"]}
    if len(index) != len(manifest["shards"]):
        raise ValueError("Duplicate frozen shard")
    for sample in samples:
        name = sample["shard_file"]; safe_name(name)
        if not name.startswith("shards/dev/") or sample["shard_sha256"] != index[name]["sha256"]:
            raise ValueError("Development sample points at another role or shard")
        by_shard[name].append(sample)
    return [(index[name], rows) for name, rows in sorted(by_shard.items())]


def read_v0_bundle(path, config, load=False):
    if common.file_sha(path) != config["adapter_bundle_sha256"]:
        raise ValueError("Use the unchanged reviewed parking_shared_adapter_results.zip")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or sum(x.file_size for x in archive.infolist()) > 15_000_000:
            raise ValueError("Unexpected V0 bundle inventory")
        for name in names: safe_name(name)
        if archive.testzip() is not None: raise ValueError("V0 bundle CRC failed")
        read = lambda name: json.loads(archive.read(name))
        identity = read("run_identity.json"); result = read("adapter_pilot_results.json")
        if (common.identity(identity) != config["adapter_run_identity"] or not result["complete"]
                or result["run_identity"] != config["adapter_run_identity"]):
            raise ValueError("Reviewed V0 training provenance changed")
        for relative, digest in identity["source_sha256"].items():
            safe_name(relative)
            if common.file_sha(ROOT / relative) != digest or common.sha(archive.read("source_snapshot/"+relative)) != digest:
                raise ValueError(f"Reviewed V0 implementation changed: {relative}")
        selection = read(config["arm"] + "/selection.json")
        arm_id = common.identity({"run_identity": config["adapter_run_identity"], "arm": config["arm"]})
        best = min(selection["history"], key=lambda r: (r["dev"]["equal_four_cell_mae_m"], r["epoch"]))
        if (selection["selected_epoch"] != config["epoch"] or best["epoch"] != config["epoch"]
                or selection["parent_run_identity"] != config["adapter_run_identity"]
                or selection["run_identity"] != arm_id or selection["heldout_used_for_selection"]
                or result["arms"][config["arm"]]["selected_dev"] != selection["selected_dev"]):
            raise ValueError("The original development-selected V0 changed")
        raw = archive.read(config["checkpoint_member"])
        if (common.sha(raw) != config["checkpoint_sha256"]
                or read(str(PurePosixPath(config["checkpoint_member"]).with_suffix(".json")))["sha256"] != config["checkpoint_sha256"]):
            raise ValueError("V0 checkpoint bytes differ")
    if not load: return identity, None
    import torch
    saved = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if (saved["run_identity"] != arm_id or saved["epoch"] != config["epoch"] or saved["arm"] != config["arm"]
            or saved["history"] != selection["history"][:config["epoch"]+1]):
        raise ValueError("V0 tensor checkpoint metadata differs")
    return identity, saved


def zip_bytes(members):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, raw in sorted(members.items()):
            safe_name(name)
            info = zipfile.ZipInfo(name, (2026, 10, 29, 0, 0, 0)); info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, raw)
    return stream.getvalue()


def prediction_chunk(run_id, rows, predictions):
    import numpy as np
    expected = {s["camera_token"] for s in rows}
    if set(predictions) != expected: raise ValueError("No dropped or additional prediction images allowed")
    records = []; members = {}
    for sample in rows:
        token = sample["camera_token"]; array = predictions[token]
        if array.dtype != np.float32 or array.ndim != 1: raise ValueError("Expected FP32 point vectors")
        stream = io.BytesIO(); np.save(stream, array, allow_pickle=False)
        name = token + ".npy"; members[name] = stream.getvalue()
        records.append({"camera_token": token, "ground_truth_sha256": sample["ground_truth_sha256"],
            "points": len(array), "member": name, "sha256": common.sha(members[name])})
    members["prediction_manifest.json"] = common.encoded({"run_identity": run_id, "samples": records})
    return zip_bytes(members)


def read_prediction_chunk(raw, run_id, rows, points):
    import numpy as np
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        value = json.loads(archive.read("prediction_manifest.json"))
        if value["run_identity"] != run_id or [r["camera_token"] for r in value["samples"]] != [s["camera_token"] for s in rows]:
            raise ValueError("Prediction cache identity/image order differs")
        expected = {"prediction_manifest.json"}; result = {}
        for sample, record in zip(rows, value["samples"]):
            token = sample["camera_token"]; name = token + ".npy"; expected.add(name)
            payload = archive.read(name)
            array = np.load(io.BytesIO(payload), allow_pickle=False)
            if (record["member"] != name or record["sha256"] != common.sha(payload)
                    or record["ground_truth_sha256"] != sample["ground_truth_sha256"]
                    or record["points"] != len(points[token]["z"])
                    or array.shape != points[token]["z"].shape or array.dtype != np.float32):
                raise ValueError("Cached point predictions changed or lost GT order")
            result[token] = array
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected) or archive.testzip() is not None:
            raise ValueError("Prediction cache inventory/CRC differs")
    return result


def load_dev_shard(path, shard, rows, protocol, benchmark_id):
    import numpy as np
    from PIL import Image
    if any(s["role"] != "dev" for s in rows) or not shard["file"].startswith("shards/dev/"):
        raise ValueError("Never open held-out shard bodies in the V0 dev workflow")
    raw = common.retry_read(path)
    if len(raw) != shard["size_bytes"] or common.sha(raw) != shard["sha256"]:
        raise ValueError("Frozen dev shard size/hash differs")
    images = {}; points = {}; expected = {"shard_manifest.json"}
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        value = check_signed(json.loads(archive.read("shard_manifest.json")))
        if value["benchmark_identity_sha256"] != benchmark_id or value["samples"] != [{k: v for k, v in s.items() if k not in ["shard_file", "shard_sha256"]} for s in rows]:
            raise ValueError("Frozen dev shard metadata differs")
        for sample in rows:
            token = sample["camera_token"]
            for kind in ["image", "ground_truth"]:
                name = sample[kind+"_member"]; safe_name(name); expected.add(name)
                if common.sha(archive.read(name)) != sample[kind+"_sha256"]:
                    raise ValueError("Frozen image/GT bytes differ")
            with Image.open(io.BytesIO(archive.read(sample["image_member"]))) as image:
                if image.mode != "RGB" or image.size != (sample["width"], sample["height"]):
                    raise ValueError("Frozen RGB shape/mode differs")
                images[token] = np.asarray(image).copy()
            with np.load(io.BytesIO(archive.read(sample["ground_truth_member"])), allow_pickle=False) as values:
                points[token] = {k: values[k] for k in values.files}
            metrics.validate_gt(points[token], sample, protocol)
        if set(archive.namelist()) != expected or len(archive.namelist()) != len(expected) or archive.testzip() is not None:
            raise ValueError("Unexpected dev shard inventory/CRC")
    return images, points


def local_shard(benchmark, local, shard):
    safe_name(shard["file"])
    if not shard["file"].startswith("shards/dev/"): raise ValueError("Only development data can be staged")
    destination = local / "dev_shards" / Path(shard["file"]).name
    if destination.exists():
        if destination.stat().st_size != shard["size_bytes"] or common.file_sha(destination) != shard["sha256"]:
            raise ValueError("Preserve changed local shard; choose a fresh local cache")
        return destination
    raw = common.retry_read(benchmark / shard["file"])
    if len(raw) != shard["size_bytes"] or common.sha(raw) != shard["sha256"]:
        raise ValueError("Source dev shard differs before staging")
    common.immutable(destination, raw)
    return destination


class V0Predictor:
    def __init__(self, args, parent, saved, config):
        import torch
        from models.parking_shared_camera import CameraAwareDepthDecoder
        from scripts import train_parking_shared_depth as training
        from scripts import parking_shared_body_verification as verification
        from scripts.screen_parking_shared_model import state_digest
        args.output.joinpath("runtime_settings").mkdir(parents=True, exist_ok=True)
        self.native, model_id = training.build_model(args, parent["model"]["config"])
        verified = verification.verify_warmstart_body(self.native, model_id, parent["model"]["config"], args.weights)
        self.decoder = CameraAwareDepthDecoder(self.native.depth_decoder, width=parent["config"]["adapter_width"])
        self.decoder.load_state_dict(saved["decoder"], strict=True)
        self.native.eval().requires_grad_(False); self.decoder.eval().requires_grad_(False)
        torch.set_num_threads(config["cpu_threads"])
        self.initial = {"body": state_digest(self.native.body), "decoder": state_digest(self.decoder)}
        self.registration = {"model_role": "V0_unchanged_baseline", "arm": config["arm"], "epoch": config["epoch"],
            "artifact_sha256": config["checkpoint_sha256"], "source_urban_checkpoint_sha256": model_id["source_checkpoint_sha256"],
            "architecture_sha256": common.identity({"shared": model_id["architecture"], "adapter": "CameraAwareDepthDecoder", "width": parent["config"]["adapter_width"]}),
            "state_sha256": self.initial, "backbone_verification": verified,
            "runtime": {**model_id["runtime"], "cpu_threads": torch.get_num_threads(), "dtype": "float32", "platform": platform.platform()},
            "preprocessing_sha256": common.file_sha(ROOT / "scripts/screen_parking_shared_model.py"),
            "calibration_sha256": common.file_sha(ROOT / "scripts/train_parking_shared_camera.py"),
            "output_sampling_sha256": common.identity({name: common.file_sha(ROOT / name) for name in ["scripts/parking_shared_depth_data.py", "scripts/train_parking_shared_depth.py"]})}

    def predict(self, sample, rgb, points):
        import numpy as np
        import torch
        from scripts.screen_parking_shared_model import preprocess
        from scripts.parking_shared_depth_data import point_grid
        from scripts.train_parking_shared_depth import sample_depth
        from scripts.train_parking_shared_camera import camera_vector
        tensor, meta = preprocess(rgb, intrinsics=sample["intrinsics"])
        grid = point_grid(points["xy"], sample["width"], sample["height"], meta)
        camera = torch.from_numpy(camera_vector(sample)[None])
        with torch.inference_mode():
            features = self.native.image_features(torch.from_numpy(tensor))
            depth = self.decoder(features, camera)
            if tuple(depth.shape) != (1, 1, 96, 160): raise ValueError("Depth output shape changed")
            result = sample_depth(depth, torch.from_numpy(grid[None, :, None, :]))[0].cpu().numpy().copy()
        if result.dtype != np.float32 or result.shape != points["z"].shape: raise ValueError("Invalid point output contract")
        # NaN/nonpositive outputs are kept for the metric's coverage denominator.
        return result

    def assert_unchanged(self):
        from scripts.screen_parking_shared_model import state_digest
        if ({"body": state_digest(self.native.body), "decoder": state_digest(self.decoder)} != self.initial
                or any(p.grad is not None for p in self.native.parameters())
                or any(m.training for m in self.native.modules()) or any(m.training for m in self.decoder.modules())):
            raise ValueError("V0 weights, BatchNorm buffers, gradients or eval mode changed")


def score_dev(manifest, inventory, benchmark, output, local, predictor, identity):
    run_id = common.identity(identity)
    common.immutable(output / "v0_registration.json", common.encoded(identity))
    accumulator = metrics.Metrics(manifest["metric_protocol"]); receipts = []; completed = 0
    for shard, rows in inventory:
        staged = local_shard(benchmark, local, shard)
        images, points = load_dev_shard(staged, shard, rows, manifest["metric_protocol"], manifest["benchmark_identity_sha256"])
        destination = output / "predictions" / (Path(shard["file"]).stem + ".zip")
        reused = destination.exists()
        if reused:
            predictions = read_prediction_chunk(common.retry_read(destination), run_id, rows, points)
        else:
            predictions = {s["camera_token"]: predictor.predict(s, images[s["camera_token"]], points[s["camera_token"]]) for s in rows}
            predictor.assert_unchanged()
            raw = prediction_chunk(run_id, rows, predictions)
            read_prediction_chunk(raw, run_id, rows, points)
            common.immutable(destination, raw)
        for sample in rows:
            token = sample["camera_token"]; accumulator.add(sample, points[token], predictions[token])
        receipts.append({"file": str(destination.relative_to(output)), "sha256": common.file_sha(destination), "images": len(rows)})
        completed += len(rows)
        print(f"V0 dev cached/verified {completed}/{sum(len(r) for _, r in inventory)} images ({'reused' if reused else 'new'} shard)", flush=True)
    predictor.assert_unchanged()
    result = {"complete": True, "revision": identity["config"]["revision"], "run_identity": run_id,
        "role": "dev", "model": "V0_actual_camera_epoch4", "metrics": accumulator.result(), "prediction_shards": receipts,
        "model_weights_and_bn_unchanged": True, "optimizer_steps": 0, "training_performed": False,
        "heldout_images_opened": 0, "heldout_evaluated": False, "v1_trained": False, "phone_app_changed": False,
        "accuracy_pass_fail_enabled": False, "deployment_qualified": False,
        "limitations": [manifest["metric_protocol"]["scope"], manifest["metric_protocol"]["confidence"],
            "Native CPU inference is not an iPhone latency measurement; no per-image scale fitting or GT-band clamping.",
            "0-2m support is limited; 0-1m remains exploratory and missing cells are null."],
        "next_step": "Review development errors and coverage before defining V1. Do not open held-out data for tuning."}
    if completed != identity["config"]["expected_images"]: raise ValueError("Incomplete development inventory")
    common.immutable(output / "v0_dev_results.json", common.encoded(result))
    return result


def evaluate(args):
    config = configuration()
    for source in [args.benchmark, args.bundle, args.source, args.weights]:
        if any(target.resolve().is_relative_to(source.resolve()) or source.resolve().is_relative_to(target.resolve())
               for target in [args.output, args.local]):
            raise ValueError("Keep V0 outputs separate from immutable inputs/source/weights")
    if args.device != "cpu": raise ValueError("This workflow freezes a CPU baseline; no GPU is needed")
    manifest = json.loads(common.retry_read(args.benchmark / "benchmark_manifest.json"))
    inventory = dev_inventory(manifest, config)
    parent, saved = read_v0_bundle(args.bundle, config, load=True)
    predictor = V0Predictor(args, parent, saved, config)
    source_files = [Path(__file__), CONFIG, ROOT / "scripts/parking_v0_v1_metrics.py", ROOT / "scripts/parking_near_field_geometry.py"]
    identity = {"config": config, "benchmark_manifest_sha256": manifest["manifest_sha256"],
        "model_registration": predictor.registration, "producer_sha256": {str(p.relative_to(ROOT)): common.file_sha(p) for p in source_files},
        "training_performed": False, "heldout_scoring_authorized": False}
    result = score_dev(manifest, inventory, args.benchmark, args.output, args.local, predictor, identity)
    members = {"v0_registration.json": (args.output / "v0_registration.json").read_bytes(),
        "v0_dev_results.json": (args.output / "v0_dev_results.json").read_bytes()}
    for shard in result["prediction_shards"]:
        raw = common.retry_read(args.output / shard["file"])
        if common.sha(raw) != shard["sha256"]: raise ValueError("Final prediction artifact changed")
        members[shard["file"]] = raw
    common.immutable(args.output / "parking_v0_dev_results.zip", zip_bytes(members))
    print(json.dumps(result, indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["benchmark", "bundle", "output", "local", "source", "weights"]:
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--device", choices=["cpu"], default="cpu")
    evaluate(p.parse_args())


if __name__ == "__main__":
    main()
