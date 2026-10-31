"""CPU-only trained V0/V1 export, frozen-development parity and phone package.

No training, held-out bodies, app installation, Core ML or calibration fitting.
Read two local archives, not thousands of mounted-Drive files. Complete export
and prediction chunks are immutable and resumable on the same code/runtime.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import parking_edge_common as common
from scripts import parking_v0_local_io as local
from scripts import evaluate_parking_v0_dev as v0
from scripts import parking_v0_v1_metrics as metrics

CONFIG = ROOT / "configs/parking_v0_v1_onnx_v1.json"
ROLES = ("V0", "V1")
STREAMS = ("V0_native", "V0_onnx", "V1_native", "V1_onnx")


def configuration():
    return json.loads(CONFIG.read_text())


def tensor_digest(state):
    from types import SimpleNamespace
    from scripts.screen_parking_shared_model import state_digest
    return state_digest(SimpleNamespace(state_dict=lambda: state))


def read_zip(path, maximum=250_000_000, count=1024):
    if Path(path).stat().st_size > maximum: raise ValueError("Oversized input archive")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if (len(names) != len(set(names)) or len(names) > count
                or sum(i.file_size for i in archive.infolist()) > maximum):
            raise ValueError("Oversized/duplicate archive inventory")
        for name in names: v0.safe_name(name)
        if archive.testzip() is not None: raise ValueError("Archive CRC failed")
        return {name: archive.read(name) for name in names}


def copy_published_archive(source, destination, digest):
    """One bounded sequential transfer; no repeated small Drive reads/retries."""
    source, destination = Path(source), Path(destination)
    if destination.is_symlink(): raise ValueError("Preserve symlinked archive destination")
    if destination.exists():
        if common.file_sha(destination) != digest: raise ValueError("Preserve conflicting published archive")
        return
    if source.stat().st_size > configuration()["maximum_phone_package_bytes"]:
        raise ValueError("Oversized phone archive; no transfer started")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with source.open("rb") as incoming, tempfile.NamedTemporaryFile(dir=destination.parent, prefix=".onnx_transfer_", delete=False) as outgoing:
            temporary = Path(outgoing.name); actual = hashlib.sha256()
            while block := incoming.read(4*1024*1024):
                outgoing.write(block); actual.update(block)
        if actual.hexdigest() != digest: raise ValueError("Archive transfer digest differs")
        if destination.exists(): raise ValueError("Another writer published this archive; stop concurrent notebooks")
        temporary.replace(destination)
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)


def verify_pilot(path, c, original_state):
    """Authenticate bytes and selection before weights-only loading a checkpoint."""
    import torch
    from scripts import train_parking_v1_near as training
    if common.file_sha(path) != c["pilot_sha256"]:
        raise ValueError("Use the unchanged reviewed matched-pilot result ZIP")
    members = read_zip(path)
    read = lambda n: json.loads(members[n])
    identity = read("run_identity.json"); result = read("v1_pilot_results.json")
    inventory = read("progress_inventory.json"); run_id = common.identity(identity)
    if (run_id != c["pilot_run_identity"] or inventory["run_identity"] != run_id
            or result["run_identity"] != run_id or inventory["heldout_bodies_included"] != 0
            or set(members) != set(inventory["files"]) | {"progress_inventory.json"}):
        raise ValueError("Matched-pilot identity/inventory differs")
    for name, digest in inventory["files"].items():
        if common.sha(members[name]) != digest: raise ValueError("Matched-pilot member hash differs")
    if identity["source_sha256"] != training.sources(): raise ValueError("Frozen pilot implementation changed")
    for name, digest in identity["source_sha256"].items():
        v0.safe_name(name)
        if common.sha(members["source_snapshot/"+name]) != digest:
            raise ValueError("Frozen pilot source snapshot changed")
    selection = read(c["v1_arm"]+"/selection.json")
    if (not result["complete"] or result["heldout_evaluated"] or result["heldout_bodies_opened"]
            or not result["architecture_unchanged"] or not result["frozen_body_base_depth_bn_unchanged"]
            or selection != result["arms"][c["v1_arm"]] or not selection["complete"]
            or selection["heldout_used"] or selection["selected_block"] != c["v1_block"]
            or training.best_endpoint(selection["history"], result["baseline"])["block"] != c["v1_block"]
            or common.sha(members[c["v1_checkpoint_member"]]) != c["v1_checkpoint_sha256"]):
        raise ValueError("Reviewed V1 development selection/checkpoint differs")
    saved = training.read_checkpoint(io.BytesIO(members[c["v1_checkpoint_member"]]), run_id)
    if (saved["arm"] != c["v1_arm"] or saved["block"] != c["v1_block"] or saved["optimizer_steps"] != 1920
            or saved["history"] != selection["history"][:c["v1_block"]+1]):
        raise ValueError("Selected V1 tensor checkpoint metadata differs")
    current = saved["decoder"]
    if (current.keys() != original_state.keys() or tensor_digest(original_state) != c["v0_decoder_sha256"]
            or tensor_digest(current) != c["v1_decoder_sha256"]):
        raise ValueError("V0/V1 tensor state identity differs")
    if any(not torch.isfinite(v).all() or (k.startswith("base.") and not torch.equal(v, original_state[k]))
           for k, v in current.items()):
        raise ValueError("Frozen base-depth/BN changed or non-finite V1 tensor")
    for role, folder in endpoint_folders(c).items():
        binding = read(folder+"/identity.json")
        if (binding["run_identity"] != run_id or binding["decoder_sha256"] != c[role.lower()+"_decoder_sha256"]
                or read(folder+"/metrics.json")["conditional_equal_four_cell_mae_m"] != c["reviewed_primary_mae_m"][role]):
            raise ValueError("Reviewed endpoint binding differs")
    return members, saved


def endpoint_folders(c):
    return {"V0": "endpoints/V0/block_000", "V1": f"endpoints/{c['v1_arm']}/block_{c['v1_block']:03d}"}


def runtime(c):
    import torch
    versions = {n: importlib.metadata.version(n) for n in c["packages"]}
    if versions != c["packages"]:
        raise ValueError("Use the notebook's pinned task-local packages: " + json.dumps(versions))
    if (str(torch.__version__).split("+")[0] != c["torch_base_version"]
            or importlib.metadata.version("torchvision").split("+")[0] != c["torchvision_base_version"]):
        raise ValueError("Use the notebook's task-local PyTorch/torchvision CPU release; preserve this run")
    torch.set_num_threads(c["threads"])
    torch.use_deterministic_algorithms(True)
    return {"python": platform.python_version(), "torch": str(torch.__version__),
        "platform": platform.platform(), "packages": versions, "opencv_imported": __import__("cv2").__version__,
        "device": "cpu", "threads": torch.get_num_threads(), "deterministic": True}


def prepare(a, c):
    """Materialize only authenticated development bodies on local disk."""
    paths = [a.output, a.capsule, a.pilot, a.source, a.weights]
    if any(str(p.resolve()).startswith("/content/drive/") for p in paths):
        raise ValueError("Stage archives on /content first; no mounted-Drive inference or source tree")
    for i, left in enumerate(paths):
        for right in paths[i+1:]:
            if left.resolve().is_relative_to(right.resolve()) or right.resolve().is_relative_to(left.resolve()):
                raise ValueError("Keep archives, outputs, source and weights in separate directories")
    if common.file_sha(a.capsule) != c["capsule_sha256"]: raise ValueError("Frozen dev-only input capsule differs")
    members, receipt = local.inspect_capsule(a.capsule, v0.configuration())
    for name, raw in members.items(): common.immutable(a.output / "inputs" / name, raw)
    parent, saved = v0.read_v0_bundle(a.output / "inputs/model/parking_shared_adapter_results.zip", v0.configuration(), load=True)
    pilot, selected = verify_pilot(a.pilot, c, saved["decoder"])
    manifest = json.loads(members["benchmark/benchmark_manifest.json"])
    return parent, saved, selected, pilot, manifest, v0.dev_inventory(manifest, v0.configuration()), receipt


def build(a, c, parent, original, selected, receipt):
    from models.parking_shared_trained_export import TrainedParkingExport
    predictor = v0.V0Predictor(a, parent, original, v0.configuration())
    if predictor.registration["architecture_sha256"] != c["architecture_sha256"]:
        raise ValueError("Trained architecture differs")
    decoder = copy.deepcopy(predictor.decoder)
    decoder.load_state_dict(selected["decoder"], strict=True)
    wrappers = {"V0": TrainedParkingExport(predictor.native, predictor.decoder),
                "V1": TrainedParkingExport(predictor.native, decoder)}
    from scripts.screen_parking_shared_model import state_digest
    hashes = {role: state_digest(model) for role, model in wrappers.items()}
    source_names = set(__import__("scripts.train_parking_v1_near", fromlist=["sources"]).sources()) | {
        "scripts/export_parking_v0_v1_onnx.py", "models/parking_shared_trained_export.py",
        "configs/parking_v0_v1_onnx_v1.json", "scripts/parking_v0_v1_metrics.py"}
    identity = {"config": c, "runtime": runtime(c), "input_receipt": receipt,
        "v0_registration": predictor.registration, "model_states": hashes,
        "producer_sha256": {n: common.file_sha(ROOT/n) for n in sorted(source_names)},
        "pilot_sha256": c["pilot_sha256"], "training_performed": False, "heldout_scoring_authorized": False}
    common.immutable(a.output / "run_identity.json", common.encoded(identity))
    return wrappers, identity


def assert_unchanged(wrappers, identity):
    from scripts.screen_parking_shared_model import state_digest
    if any(state_digest(model) != identity["model_states"][role]
           or any(m.training for m in model.modules()) or any(p.grad is not None or p.requires_grad for p in model.parameters())
           for role, model in wrappers.items()):
        raise ValueError("Model weights, BN, eval mode or gradient state changed")


def input_arrays(sample, rgb, points):
    import numpy as np
    from scripts.screen_parking_shared_model import preprocess
    from scripts.train_parking_shared_camera import camera_vector
    from scripts.parking_shared_depth_data import point_grid
    image, metadata = preprocess(rgb, intrinsics=sample["intrinsics"])
    camera = camera_vector(sample)[None]
    grid = point_grid(points["xy"], sample["width"], sample["height"], metadata)
    if (image.dtype != np.float32 or camera.dtype != np.float32 or not np.isfinite(camera).all()
            or not (camera[0, :2] > 0).all()): raise ValueError("Invalid calibrated FP32 inputs")
    return {"image": image, "camera": camera}, grid, metadata


def session(graph, c):
    import onnx
    import onnxruntime as ort
    model = onnx.load_model_from_string(graph); onnx.checker.check_model(model, full_check=True)
    if (any(n.domain not in ("", "ai.onnx") for n in model.graph.node)
            or any(t.data_location == onnx.TensorProto.EXTERNAL for t in model.graph.initializer)):
        raise ValueError("Only self-contained standard-operator ONNX graphs are allowed")
    options = ort.SessionOptions(); options.intra_op_num_threads = c["threads"]; options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    for key in ("intra_op", "inter_op"): options.add_session_config_entry(f"session.{key}.allow_spinning", "0")
    value = ort.InferenceSession(graph, options, providers=["CPUExecutionProvider"])
    expected = lambda values: [(n, shape, "tensor(float)") for n, shape in values.items()]
    if (value.get_providers() != ["CPUExecutionProvider"]
            or [(v.name, v.shape, v.type) for v in value.get_inputs()] != expected(c["input_shapes"])
            or [(v.name, v.shape, v.type) for v in value.get_outputs()] != expected(c["output_shapes"])):
        raise ValueError("ONNX inputs/outputs differ; calibrated camera input must not be baked away")
    return value


def export_models(a, c, wrappers, identity, example):
    import torch
    graphs = {}; sessions = {}; run_id = common.identity(identity)
    for role, model in wrappers.items():
        # Populate Urban's non-persistent, fixed-shape anchor cache before both
        # traces. Otherwise the first graph builds anchors while the second
        # embeds the cached anchors, despite identical frozen detector weights.
        with torch.inference_mode():
            model(*(torch.from_numpy(example[n]) for n in c["input_shapes"]))
        destination = a.output / "exports" / (role+".zip")
        if destination.exists():
            members = read_zip(destination, count=2); record = json.loads(members["export.json"])
            graph = members["model.onnx"]
            if (set(members) != {"model.onnx", "export.json"} or record != {
                    "run_identity": run_id, "role": role, "onnx_sha256": common.sha(graph)}):
                raise ValueError("Existing export identity differs; preserve it and use a new run directory")
        else:
            a.output.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(dir=a.output, prefix=".onnx_") as temporary:
                path = Path(temporary) / "model.onnx"
                with torch.inference_mode():
                    torch.onnx.export(model, tuple(torch.from_numpy(example[n]) for n in c["input_shapes"]), str(path),
                        input_names=list(c["input_shapes"]), output_names=list(c["output_shapes"]),
                        opset_version=c["opset"], dynamo=False, do_constant_folding=True, export_params=True)
                graph = path.read_bytes()
            session(graph, c)
            record = {"run_identity": run_id, "role": role, "onnx_sha256": common.sha(graph)}
            common.immutable(destination, v0.zip_bytes({"model.onnx": graph, "export.json": common.encoded(record)}))
        graphs[role] = graph; sessions[role] = session(graph, c)
        print(f"{role}: verified two-input FP32 ONNX ({len(graph):,} bytes)", flush=True)
    assert_unchanged(wrappers, identity)
    return graphs, sessions


def compare(reference, actual, atol, rtol):
    from scripts.screen_parking_shared_model import compare as check
    return check(reference, actual, atol, rtol)


def sample_points(depth, grid):
    import numpy as np
    import torch
    from scripts.train_parking_shared_depth import sample_depth
    return sample_depth(torch.from_numpy(depth), torch.from_numpy(grid[None, :, None, :]))[0].numpy().astype(np.float32, copy=True)


def output_checks(reference, actual, c):
    import numpy as np
    checks = {}
    for name, expected in c["output_shapes"].items():
        value = actual[name]
        if list(value.shape) != expected or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError("Invalid ONNX output: " + name)
        if name == "depth_m" and not (value > 0).all(): raise ValueError("Invalid metric depth; no clamping or point dropping")
        prefix = "candidate" if name == "candidates" else "depth"
        checks[name] = compare(reference[name], value, c["parity"][prefix+"_atol"+("_m" if prefix == "depth" else "")],
                               c["parity"][prefix+"_rtol"])
    return checks


def reviewed_predictions(pilot, c, role, shard, rows, truth):
    folder = endpoint_folders(c)[role]
    binding = json.loads(pilot[folder+"/identity.json"])
    return v0.read_prediction_chunk(pilot[folder+"/predictions/"+Path(shard["file"]).name],
                                    common.identity(binding), rows, truth)


def chunk_bytes(run_id, rows, values, checks):
    members = {key+".zip": v0.prediction_chunk(run_id, rows, values[key]) for key in STREAMS}
    members["checks.json"] = common.encoded(checks)
    members["inventory.json"] = common.encoded({"run_identity": run_id,
        "files": {n: common.sha(v) for n, v in members.items()}})
    return v0.zip_bytes(members)


def read_chunk(path, run_id, rows, truth):
    members = read_zip(path, maximum=20_000_000, count=6)
    index = json.loads(members["inventory.json"])
    expected = {k+".zip" for k in STREAMS} | {"checks.json"}
    if (index["run_identity"] != run_id or set(members) != expected | {"inventory.json"}
            or set(index["files"]) != expected
            or any(common.sha(members[n]) != digest for n, digest in index["files"].items())):
        raise ValueError("Completed comparison chunk changed")
    checks = json.loads(members["checks.json"])
    if set(checks) != {s["camera_token"] for s in rows}: raise ValueError("Incomplete per-image parity records")
    return {k: v0.read_prediction_chunk(members[k+".zip"], run_id, rows, truth) for k in STREAMS}, checks


def check_metrics(actual, reviewed, c):
    """Numerical conversion gate, not a product accuracy requirement."""
    if (actual["images"] != reviewed["images"] or actual["images_with_invalid_predictions"] != 0
            or actual["conditional_equal_four_cell_mae_m"] is None):
        raise ValueError("Incomplete/invalid development predictions")
    largest = abs(actual["conditional_equal_four_cell_mae_m"]-reviewed["conditional_equal_four_cell_mae_m"])
    for name, cell in actual["cells"].items():
        old = reviewed["cells"][name]
        for key in ("reference_points", "valid_predicted_points", "prediction_coverage",
                    "reference_annotation_observations", "reference_instance_tracks", "reference_recording_logs"):
            if cell[key] != old[key]: raise ValueError("Conversion changed coverage/denominators: " + name)
        key = "conditional_log_track_observation_macro_mae_m"
        if (cell[key] is None) != (old[key] is None): raise ValueError("Missing metric cell after conversion")
        if cell[key] is not None: largest = max(largest, abs(cell[key]-old[key]))
    if largest > c["parity"]["metric_atol_m"]: raise ValueError("Conversion changed frozen benchmark metrics")
    return {"passed": True, "maximum_cell_or_primary_mae_delta_m": largest,
        "absolute_tolerance_m": c["parity"]["metric_atol_m"], "coverage_and_denominators_preserved": True}


def validate(a, c, wrappers, identity, graphs, sessions, pilot, manifest, inventory):
    import numpy as np
    import torch
    run_id = common.identity({"run": common.identity(identity), "graphs": {r: common.sha(v) for r,v in graphs.items()}})
    accumulators = {k: metrics.Metrics(manifest["metric_protocol"]) for k in STREAMS}
    predictions = {k: {} for k in STREAMS}; all_truth = {}; all_rows = []; all_checks = {}
    for number, (shard, rows) in enumerate(inventory, 1):
        images, truth = v0.load_dev_shard(a.output / "inputs/benchmark" / shard["file"], shard, rows,
            manifest["metric_protocol"], manifest["benchmark_identity_sha256"])
        path = a.output / "comparisons" / Path(shard["file"]).name
        if not path.exists():
            values = {k: {} for k in STREAMS}; checks = {}
            reviewed = {r: reviewed_predictions(pilot, c, r, shard, rows, truth) for r in ROLES}
            for sample in rows:
                token = sample["camera_token"]; inputs, grid, meta = input_arrays(sample, images[token], truth[token])
                refs = {}; actuals = {}; record = {"preprocessed_image_sha256": common.sha(inputs["image"].astype("<f4").tobytes()),
                    "camera": inputs["camera"].tolist(), "grid_sha256": common.sha(grid.astype("<f4").tobytes()), "preprocessing": meta}
                with torch.inference_mode():
                    for role in ROLES:
                        native = wrappers[role](*(torch.from_numpy(inputs[n]) for n in c["input_shapes"]))
                        refs[role] = {n: v.numpy().copy() for n,v in zip(c["output_shapes"], native)}
                        actuals[role] = dict(zip(c["output_shapes"], sessions[role].run(None, inputs)))
                        record[role] = output_checks(refs[role], actuals[role], c)
                        for suffix, source in (("native", refs[role]), ("onnx", actuals[role])):
                            value = sample_points(source["depth_m"], grid)
                            try:
                                record[role][suffix+"_vs_reviewed_points"] = compare(reviewed[role][token], value,
                                    c["parity"]["reviewed_point_atol_m"], c["parity"]["reviewed_point_rtol"])
                            except (ValueError, RuntimeError) as error:
                                raise RuntimeError(f"{role} {suffix} differs from the reviewed Linux endpoint at {token}. "
                                    "No phone package is authorized. Preserve the run and compare native runtime/preprocessing; "
                                    "do not widen tolerances or refit depth scale. " + str(error)) from error
                            values[role+"_"+suffix][token] = value
                if not np.array_equal(refs["V0"]["candidates"], refs["V1"]["candidates"]):
                    raise ValueError("Native detector branch changed between V0 and V1")
                if not np.array_equal(actuals["V0"]["candidates"], actuals["V1"]["candidates"]):
                    raise ValueError("ONNX detector branch changed between V0 and V1")
                checks[token] = record
            assert_unchanged(wrappers, identity)
            common.immutable(path, chunk_bytes(run_id, rows, values, checks))
        values, checks = read_chunk(path, run_id, rows, truth)
        for key in STREAMS:
            for sample in rows:
                token = sample["camera_token"]; accumulators[key].add(sample, truth[token], values[key][token])
            predictions[key].update(values[key])
        all_truth.update(truth); all_rows.extend(rows); all_checks.update(checks)
        print(f"Parity/dev: verified {number}/{len(inventory)} shards, {len(all_rows)}/{c['development_images']} images", flush=True)
    if len(all_rows) != c["development_images"]: raise ValueError("Incomplete frozen development replay")
    measured = {k: v.result() for k,v in accumulators.items()}; gates = {}
    for role in ROLES:
        reviewed = json.loads(pilot[endpoint_folders(c)[role]+"/metrics.json"])
        gates[role] = {suffix: check_metrics(measured[role+"_"+suffix], reviewed, c) for suffix in ("native", "onnx")}
    assert_unchanged(wrappers, identity)
    report = {"complete": True, "revision": c["revision"], "run_identity": common.identity(identity),
        "comparison_identity": run_id, "models": {r: {"sha256": common.sha(g), "bytes": len(g)} for r,g in graphs.items()},
        "development_images": len(all_rows), "full_output_parity_passed": True, "reviewed_point_parity_passed": True,
        "frozen_dev_metrics_preserved": gates, "metrics": measured,
        "paired_onnx": metrics.paired_metrics(all_rows, all_truth, predictions["V0_onnx"], predictions["V1_onnx"], manifest["metric_protocol"]),
        "detector_branch_identical_between_versions": True, "weights_and_bn_unchanged": True,
        "training_performed": False, "optimizer_steps": 0, "heldout_bodies_opened": 0,
        "iphone_performance_measured": False, "phone_app_changed": False, "deployment_qualified": False,
        "product_accuracy_pass_fail_enabled": False,
        "scope": c["depth_definition"], "limitations": ["Development-only numerical conversion validation, not a new held-out score.",
            "No detector mAP/recall evaluation, metric 3D cuboid validation, ultrasonic distance or iPhone speed measurement.",
            "0-2m support is limited; below 1m remains exploratory. No per-image scale fitting or band clamping."]}
    common.immutable(a.output / "onnx_comparison.json", common.encoded(report))
    return report, all_checks


def choose_fixtures(samples, maximum):
    """Deterministic metadata-only coverage; never choose by model errors."""
    def features(sample):
        return {"calibration:"+common.identity(sample["intrinsics"])} | {
            "cell:"+name for name in sample["cell_names"] if sample["eligible_reference_counts"].get(name, 0)} | {
            "class:"+obj["class_name"] for obj in sample["objects"]}
    remaining = sorted(samples, key=lambda s: s["camera_token"]); chosen = []; seen = set()
    while remaining and len(chosen) < maximum:
        best = min(remaining, key=lambda s: (-len(features(s)-seen), s["camera_token"]))
        chosen.append(best); seen |= features(best); remaining.remove(best)
    return chosen


def float_table():
    """Exact 256 FP32 values used by the reviewed uint8 -> /255 input path."""
    import numpy as np
    return np.arange(256, dtype=np.float32) / np.float32(255)


def lossless_input_indices(image):
    """A storage codec only: recover the original FP32 input bit-for-bit.

    Avoid platform JPEG/resize differences in camera-OFF device accuracy replay.
    This is NOT INT8 model quantization; the ONNX model still receives FP32.
    """
    import numpy as np
    indices = np.rint(image * np.float32(255)).astype(np.uint8)
    if image.dtype != np.float32 or not np.array_equal(float_table()[indices], image):
        raise ValueError("Cannot store this input losslessly with the declared codec")
    return indices


def binary_asset(members, name, array, dtype):
    import numpy as np
    value = np.ascontiguousarray(array, dtype=dtype); raw = value.tobytes()
    members[name] = raw
    return {"file": name, "shape": list(value.shape), "dtype": value.dtype.str,
        "bytes": len(raw), "sha256": common.sha(raw)}


def accuracy_shard(rows, images, truth, values, checks, c):
    """Portable preprocessed inputs + original sparse owners/masks, no JPEG decode.

    The fixed arrays preserve every denominator and object/log identity. There
    is no densification, scale fit, GT-band clamp or new eligibility decision.
    """
    import numpy as np
    members = {}; records = []
    for sample in rows:
        token = sample["camera_token"]; point = truth[token]
        inputs, grid, _ = input_arrays(sample, images[token], point)
        if (common.sha(inputs["image"].astype("<f4").tobytes()) != checks[token]["preprocessed_image_sha256"]
                or common.sha(grid.astype("<f4").tobytes()) != checks[token]["grid_sha256"]
                or inputs["camera"].tolist() != checks[token]["camera"]):
            raise ValueError("Preprocessed input changed between validation and packaging")
        at = lambda name, array, dtype: binary_asset(members, token+"/"+name, array, dtype)
        records.append({"sample": sample,
            "image_indices": at("image.u8", lossless_input_indices(inputs["image"]), "u1"),
            "reconstructed_image_fp32_sha256": checks[token]["preprocessed_image_sha256"],
            "camera": at("camera.bin", inputs["camera"], "<f4"), "grid": at("grid.bin", grid, "<f4"),
            "truth_z_m": at("z.bin", point["z"], "<f4"),
            "object_index": at("owners.bin", point["object_index"], "<i4"),
            "cell_masks": at("masks.u8", point["cell_masks"].astype(np.uint8), "u1"),
            "native_point_references": {role: at(role+"_points.bin", values[role+"_native"][token], "<f4") for role in ROLES}})
    members["shard_manifest.json"] = common.encoded({"codec": c["phone_accuracy_input_codec"], "samples": records,
        "files": {n: common.sha(raw) for n,raw in members.items()}, "heldout_bodies_included": 0})
    return v0.zip_bytes(members)


def package(a, c, wrappers, identity, graphs, sessions, report, checks, manifest, inventory):
    """Bounded runtime fixtures plus the complete dev-only accuracy replay data."""
    import numpy as np
    import torch
    if (not report.get("complete") or report.get("run_identity") != common.identity(identity)
            or report.get("development_images") != c["development_images"]
            or not report.get("full_output_parity_passed") or not report.get("reviewed_point_parity_passed")
            or report.get("heldout_bodies_opened") != 0
            or not all(report.get("frozen_dev_metrics_preserved", {}).get(r, {}).get(s, {}).get("passed", False)
                       for r in ROLES for s in ("native", "onnx"))
            or any(report["models"][r]["sha256"] != common.sha(raw) for r,raw in graphs.items())):
        raise ValueError("No phone package until every frozen-development conversion gate passes")
    members = {f"models/{r}.onnx": raw for r,raw in graphs.items()}
    members["onnx_comparison.json"] = common.encoded(report)
    members["run_identity.json"] = common.encoded(identity)
    chosen = choose_fixtures([s for _, rows in inventory for s in rows], c["fixed_phone_fixtures"])
    chosen_tokens = {s["camera_token"] for s in chosen}; fixtures = []; calibration_checks = {}; accuracy_shards = []
    def store(name, value):
        array = np.asarray(value, dtype="<f4", order="C"); raw = array.tobytes()
        members[name] = raw
        return {"file": name, "shape": list(array.shape), "dtype": "float32_le", "bytes": len(raw), "sha256": common.sha(raw)}
    for shard, rows in inventory:
        images, truth = v0.load_dev_shard(a.output / "inputs/benchmark" / shard["file"], shard, rows,
            manifest["metric_protocol"], manifest["benchmark_identity_sha256"])
        values, _ = read_chunk(a.output / "comparisons" / Path(shard["file"]).name,
            report["comparison_identity"], rows, truth)
        name = "accuracy/shards/"+Path(shard["file"]).name
        raw = accuracy_shard(rows, images, truth, values, checks, c); members[name] = raw
        accuracy_shards.append({"file": name, "sha256": common.sha(raw), "bytes": len(raw), "images": len(rows),
            "source_frozen_shard_sha256": shard["sha256"]})
        for sample in rows:
            token = sample["camera_token"]
            if token not in chosen_tokens: continue
            inputs, grid, _ = input_arrays(sample, images[token], truth[token]); prefix = "fixtures/"+token
            entry = {"camera_token": token, "inputs": {n: store(prefix+"/"+n+".bin", value) for n,value in inputs.items()},
                "grid": store(prefix+"/grid.bin", grid), "truth_z_m": store(prefix+"/truth_z_m.bin", truth[token]["z"]), "outputs": {}}
            for role in ROLES:
                outputs = dict(zip(c["output_shapes"], sessions[role].run(None, inputs)))
                entry["outputs"][role] = {n: store(prefix+"/"+role+"/"+n+".bin", value) for n,value in outputs.items()}
                if role not in calibration_checks:
                    alternate = {n: v.copy() for n,v in inputs.items()}
                    alternate["camera"][0, :2] *= np.float32(1.1)
                    perturbed = dict(zip(c["output_shapes"], sessions[role].run(None, alternate)))
                    with torch.inference_mode():
                        native = wrappers[role](*(torch.from_numpy(alternate[n]) for n in c["input_shapes"]))
                    parity = output_checks({n: v.numpy() for n,v in zip(c["output_shapes"],native)}, perturbed, c)
                    delta = float(np.max(np.abs(perturbed["depth_m"]-outputs["depth_m"])))
                    if delta <= 1e-6 or not np.array_equal(perturbed["candidates"], outputs["candidates"]):
                        raise ValueError("Camera input was baked away or incorrectly affects the detector")
                    calibration_checks[role] = {"passed": True, "focal_length_multiplier": 1.1,
                        "maximum_depth_change_m": delta, "detector_unchanged": True, "native_onnx_parity": parity}
            fixtures.append(entry)
    if len(fixtures) != c["fixed_phone_fixtures"]: raise ValueError("Incomplete phone fixture inventory")
    table = store("accuracy/float32_lookup.bin", float_table())
    dev = {"source_benchmark_manifest_sha256": manifest["manifest_sha256"], "metric_protocol": manifest["metric_protocol"],
        "shards": accuracy_shards, "images": sum(s["images"] for s in accuracy_shards), "role": "dev",
        "input_codec": c["phone_accuracy_input_codec"], "float32_lookup": table,
        "input_reconstruction": "Float32 tensor[i] = lookup[image_indices[i]] in NCHW order. Verify reconstructed-image SHA before inference; no JPEG or resizing on the device for this fixed-input replay.",
        "model_quantization_performed": False, "heldout_shard_bodies_included": 0}
    members["accuracy/dev_manifest.json"] = common.encoded(dev)
    for path in sorted((a.output / "comparisons").glob("*.zip")):
        members["accuracy/comparisons/"+path.name] = path.read_bytes()
    phone = {"revision": c["revision"], "run_identity": common.identity(identity), "complete": True,
        "models": {r: {"file": f"models/{r}.onnx", "sha256": common.sha(raw), "bytes": len(raw)} for r,raw in graphs.items()},
        "inputs": c["input_shapes"], "outputs": c["output_shapes"], "camera_layout": c["camera_layout"],
        "candidate_layout": json.loads((ROOT / "configs/parking_shared_nano_v1.json").read_text())["candidate_layout"],
        "depth_definition": c["depth_definition"], "fixtures": fixtures, "calibration_perturbation_checks": calibration_checks,
        "provider": "CPUExecutionProvider", "intra_op_threads": c["threads"], "inter_op_threads": 1,
        "spinning": False, "execution_mode": "sequential", "parity": c["parity"],
        "runtime_protocol": {"warmups": 5, "timed_runs": 50, "camera_enabled": False,
            "thermal_stop_states": ["serious", "critical"], "report_both_outputs_fetched": True,
            "timing_scope": "ORT run with both outputs; report preprocessing/point sampling separately"},
        "fixed_fixture_scope": "16 metadata-selected real dev views for runtime/parity; not the full512 metric",
        "accuracy_replay_scope": "All 512 dev preprocessed input tensors, camera/grid arrays, original sparse GT owners/masks and native point references. Lossless storage codec reconstructs exact FP32 inputs; no on-phone JPEG/resize variability.",
        "permitted_next_step": "Integrate in the existing app for camera-OFF fixed-input V0/V1 CPU replay; no live/production qualification implied.",
        "training_performed": False, "heldout_bodies_included": 0, "iphone_performance_measured": False,
        "phone_app_changed": False, "deployment_qualified": False,
        "license_note": "Urban/Ultralytics components retain upstream AGPL-3.0 terms; no independent ownership claim."}
    members["phone_manifest.json"] = common.encoded(phone)
    members["package_inventory.json"] = common.encoded({"revision": c["revision"],
        "files": {n: {"sha256": common.sha(raw), "bytes": len(raw)} for n,raw in members.items()}, "heldout_bodies_included": 0})
    payload = v0.zip_bytes(members)
    if len(payload) > c["maximum_phone_package_bytes"]: raise ValueError("Phone package exceeds bounded transfer size")
    destination = a.output / "portable/parking_v0_v1_phone.zip"
    common.immutable(destination, payload)
    small = {n: raw for n,raw in members.items() if n in ("run_identity.json", "onnx_comparison.json", "phone_manifest.json")}
    small["phone_package_receipt.json"] = common.encoded({"file": destination.name, "sha256": common.sha(payload),
        "bytes": len(payload), "complete": True, "phone_app_changed": False})
    common.immutable(a.output / "portable/parking_v0_v1_onnx_results.zip", v0.zip_bytes(small))
    assert_unchanged(wrappers, identity)
    print(f"Ready for review: {destination} ({len(payload):,} bytes). iPhone app is unchanged.", flush=True)


def run(a):
    c = configuration(); runtime(c)
    parent, original, selected, pilot, manifest, inventory, receipt = prepare(a, c)
    wrappers, identity = build(a, c, parent, original, selected, receipt)
    shard, rows = inventory[0]
    images, truth = v0.load_dev_shard(a.output / "inputs/benchmark" / shard["file"], shard, rows,
        manifest["metric_protocol"], manifest["benchmark_identity_sha256"])
    sample = rows[0]; example, _, _ = input_arrays(sample, images[sample["camera_token"]], truth[sample["camera_token"]])
    graphs, sessions = export_models(a, c, wrappers, identity, example)
    if a.stage == "export": return
    report, checks = validate(a, c, wrappers, identity, graphs, sessions, pilot, manifest, inventory)
    if a.stage == "bundle": package(a, c, wrappers, identity, graphs, sessions, report, checks, manifest, inventory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["export", "validate", "bundle"])
    for name in ("capsule", "pilot", "output", "source", "weights"):
        parser.add_argument("--"+name, type=Path, required=True)
    a = parser.parse_args(); a.device = "cpu"
    try: run(a)
    except Exception as error:
        import datetime
        import traceback
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        failure = {"complete": False, "stage": a.stage, "error": str(error), "traceback": traceback.format_exc(),
            "training_performed": False, "optimizer_steps": 0, "phone_app_changed": False}
        if not str(a.output.resolve()).startswith("/content/drive/"):
            common.immutable(a.output / "errors" / (stamp+".json"), common.encoded(failure))
        raise


if __name__ == "__main__": main()
