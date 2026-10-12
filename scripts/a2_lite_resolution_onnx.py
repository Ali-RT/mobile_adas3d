"""Two lower-resolution A2-Lite speed screens; same pruned weights, no training.

Use the exact reviewed A2-Lite checkpoint/fixtures on Drive, not a historical
runtime. Resize only the normalized image, preserving its field of view and
original-image calibration/size convention. Update every portable attention
grid, then check native/ONNX raw AND decoded outputs at each new resolution.
These resized fixtures are for speed/parity only, not KITTI accuracy scoring.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np
import torch
import torch.nn.functional as F

import a2_lite_onnx as base
from compact_a2_teacher import decoded_fixture_comparison
from m68_onnx_common import (INPUT_NAMES, INPUT_SHAPES, OUTPUT_NAMES, OUTPUT_SHAPES,
                             RAW_LIMITS, compare_outputs, read_f32, sha256, signature,
                             write_f32, write_json)

REVISION = "A2-LITE-RESOLUTION-ONNX-CPU-SCREEN-2026-10-12-r1"
SCREEN_SHA = "64d8865e15cd1593ebcebd694d7266ca450882516b788dcc25b6a0af793241df"
PACKAGE_SHA = "bf7580b5a57d782a6e0ced8b4fe02a7181b35edc61800b02371b0d1f8b727cb0"
CHECKPOINT_SHA = "43e0e84d95de649a5cee645d7826baf6bf7ba338895940160ad94bcdd5d3281d"
ONNX_SHA = "8e9c8472b03fcece8ed90e873f4b484d10c2e77bc6a2dcecd2f3473a870819a4"
VARIANTS = {
    "r960x288": dict(height=288, width=960, resource="A2_LITE_960_ONNX"),
    "r640x192": dict(height=192, width=640, resource="A2_LITE_640_ONNX"),
}
IDS = ["000001", "000002", "000004", "000005", "000006", "000008", "000015", "000019",
       "000020", "000021", "000023", "000024", "000025", "000027", "000028", "000031"]
PREPROCESSING = "bilinear resize of reviewed normalized full-resolution tensor; align_corners=False; antialias=True; no crop/padding; original calibration and image_size unchanged"


def implementation():
    return {**base.implementation(), "scripts/a2_lite_resolution_onnx.py": sha256(__file__)}


def signed(value, field, expected):
    if value.get(field) != expected or signature({k: v for k, v in value.items() if k != field}) != expected:
        raise RuntimeError("Changed reviewed A2-Lite metadata; preserve the baseline")
    return value


def input_shapes(variant):
    v = VARIANTS[variant]
    return {**INPUT_SHAPES, "image": (1, 3, v["height"], v["width"])}


def feature_shapes(height, width):
    if (height, width) not in {(384, 1280), (288, 960), (192, 640)}:
        raise ValueError("Only the reviewed baseline and two fixed resolution variants are supported")
    return tuple(((height + stride - 1)//stride, (width + stride - 1)//stride)
                 for stride in (8, 16, 32, 64))


def configure_attention(model, height, width):
    shapes = feature_shapes(height, width)
    modules = [m for m in model.modules() if hasattr(m, "coreml_spatial_shapes")]
    if len(modules) != 2 or any(m.n_levels != 4 or not m.coreml_export for m in modules):
        raise RuntimeError("Expected one visual and one decoder portable four-level attention module")
    for module in modules:
        module.coreml_spatial_shapes = shapes
    return shapes


def state_signature(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode()); digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
    return digest.hexdigest()


def resize_inputs(inputs, variant):
    if set(inputs) != set(INPUT_NAMES): raise ValueError("Unexpected source inputs")
    for name in INPUT_NAMES:
        value = inputs[name]
        if value.shape != INPUT_SHAPES[name] or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"Invalid source tensor: {name}")
    v = VARIANTS[variant]
    with torch.inference_mode():
        image = F.interpolate(torch.from_numpy(inputs["image"]), size=(v["height"], v["width"]),
                              mode="bilinear", align_corners=False, antialias=True).numpy().copy()
    result = dict(image=image, calibration=inputs["calibration"].copy(), image_size=inputs["image_size"].copy())
    for name in ("calibration", "image_size"):
        if not np.array_equal(result[name], inputs[name]): raise RuntimeError("Original coordinate convention changed")
    return result


def baseline(root):
    root = Path(root)
    m = signed(json.loads((root/"a2_lite_manifest.json").read_text()), "signature_sha256", SCREEN_SHA)
    package_root = root/"export/A2_LITE_ONNX"
    p = signed(json.loads((package_root/"manifest.json").read_text()), "manifest_sha256", PACKAGE_SHA)
    checkpoint = root/"a2_lite_initialization.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing {checkpoint}. Use the original A2-Lite Drive output folder; its checkpoint was deliberately excluded from the download ZIP.")
    if (m["revision"] != base.REVISION or m["implementation"] != base.implementation()
            or m["checkpoint_sha256"] != base.q.A2_SHA
            or p["checkpoint_sha256"] != CHECKPOINT_SHA
            or p["model"]["sha256"] != ONNX_SHA
            or p["sample_ids"] != IDS or [f["sample_id"] for f in p["fixtures"]] != IDS
            or p["accuracy_qualified"] or p["training_performed"] or p["optimizer_steps"] != 0
            or p["source_checkpoint_sha256"] != base.q.A2_SHA
            or sha256(checkpoint) != CHECKPOINT_SHA
            or sha256(package_root/p["model"]["file"]) != ONNX_SHA):
        raise RuntimeError("Use the exact reviewed full-resolution A2-Lite screen, checkpoint and fixtures")
    return m, p, checkpoint, package_root


def cpu_session(path, shapes):
    import onnxruntime as ort
    if ort.__version__ != "1.30.0": raise RuntimeError("Use onnxruntime==1.30.0 to match the phone")
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    if (session.get_providers() != ["CPUExecutionProvider"]
            or {v.name: tuple(v.shape) for v in session.get_inputs()} != shapes
            or {v.name: tuple(v.shape) for v in session.get_outputs()} != OUTPUT_SHAPES):
        raise RuntimeError("CPU input/output contract differs")
    return session


def prepare(args):
    if args.output.is_relative_to(args.baseline): raise RuntimeError("Use a separate output; never edit the baseline")
    m, p, checkpoint, root = baseline(args.baseline)
    base.prepare_repo(args.repo)
    if base.q.source_hash(args.repo) != m["source_sha256"]: raise RuntimeError("Fresh portable source differs from baseline")
    session = cpu_session(root/p["model"]["file"], INPUT_SHAPES)
    rows = []
    for fixture in p["fixtures"]:
        inputs = {n: read_f32(root, fixture["inputs"][n]) for n in INPUT_NAMES}
        expected = {n: read_f32(root, fixture["cpu_reference_outputs"][n]) for n in OUTPUT_NAMES}
        check = compare_outputs(dict(zip(OUTPUT_NAMES, session.run(list(OUTPUT_NAMES), inputs))), expected)
        if not check["passed"]: raise RuntimeError("Original A2-Lite CPU fixture parity failed")
        rows.append(dict(sample_id=fixture["sample_id"], comparison=check))
    value = dict(revision=REVISION, baseline=str(args.baseline), repo=str(args.repo), output=str(args.output),
                 checkpoint_sha256=CHECKPOINT_SHA, parent_package_sha256=PACKAGE_SHA,
                 source_sha256=m["source_sha256"], implementation=implementation(), variants=VARIANTS,
                 original_config=m["original_config"], baseline_parity_checks=rows,
                 preprocessing=PREPROCESSING, training_performed=False, optimizer_steps=0,
                 accuracy_evaluated=False, deployment_qualified=False)
    value["signature_sha256"] = signature(value)
    target = args.output/"a2_lite_resolution_manifest.json"
    if target.exists() and json.loads(target.read_text()) != value:
        raise RuntimeError("Preserve changed inputs and use a fresh RUN_ID")
    write_json(target, value)
    print("Exact A2-Lite baseline verified. Two resolution screens prepared; no training.", flush=True)


def load(args):
    m = json.loads((args.output/"a2_lite_resolution_manifest.json").read_text())
    signed(m, "signature_sha256", m["signature_sha256"])
    if (m["revision"] != REVISION or m["implementation"] != implementation()
            or base.q.source_hash(Path(m["repo"])) != m["source_sha256"]):
        raise RuntimeError("Resolution-screen code/source changed; use a fresh RUN_ID")
    baseline(m["baseline"])
    return m


def load_model(repo, cfg, checkpoint, package):
    model = base.inference_model(repo, cfg["model"])
    inventory = base.prune_a2_layers(model)
    if inventory != package["architecture"]: raise RuntimeError("Pruned architecture differs from baseline")
    payload = base.q.safe_payload(checkpoint)
    if (payload.get("epoch") != 0 or payload.get("source_epoch") != 130
            or payload.get("source_checkpoint_sha256") != base.q.A2_SHA
            or payload.get("training_performed") is not False or payload.get("optimizer_steps") != 0
            or payload.get("pruning") != inventory or payload.get("optimizer_state") is not None):
        raise RuntimeError("Not the original layer-pruned A2 initialization")
    model.load_state_dict(payload["model_state"], strict=True)
    return model, inventory


def graph_export(wrapper, first, path, variant):
    import onnx
    with torch.inference_mode():
        torch.onnx.export(wrapper, tuple(torch.from_numpy(first[n]) for n in INPUT_NAMES), str(path),
                          export_params=True, opset_version=17, do_constant_folding=True, dynamo=False,
                          input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES))
    graph = onnx.load(str(path)); onnx.checker.check_model(graph, full_check=True)
    domains = sorted({n.domain for n in graph.graph.node if n.domain not in ("", "ai.onnx")})
    if domains or any(i.data_location == onnx.TensorProto.EXTERNAL for i in graph.graph.initializer):
        raise RuntimeError("Custom domains or external weights are not allowed")
    return cpu_session(path, input_shapes(variant)), dict(opset=17, ir_version=graph.ir_version,
                                                         nodes=len(graph.graph.node), custom_operator_domains=[])


def export(args):
    m = load(args); variant = args.variant; v = VARIANTS[variant]
    destination = args.output/"exports"/variant
    if destination.exists() and any(destination.iterdir()):
        raise RuntimeError("Preserve partial/completed exports and choose a fresh RUN_ID")
    _, package, checkpoint, root = baseline(m["baseline"])
    torch.set_num_threads(2); torch.manual_seed(123)
    model, inventory = load_model(Path(m["repo"]), m["original_config"], checkpoint, package)
    state_before = state_signature(model)
    wrapper = base.Wrapper(model).eval()
    source_inputs = [{n: read_f32(root, f["inputs"][n]) for n in INPUT_NAMES} for f in package["fixtures"]]
    # Confirm the reconstructed pruned checkpoint at its original resolution first.
    for f, data in zip(package["fixtures"], source_inputs):
        with torch.inference_mode(): values = base.tensors_to_arrays(wrapper(*(torch.from_numpy(data[n]) for n in INPUT_NAMES)))
        refs = {n: read_f32(root, f["cpu_reference_outputs"][n]) for n in OUTPUT_NAMES}
        if not compare_outputs(values, refs)["passed"]: raise RuntimeError("Reconstructed original A2-Lite differs")
    shapes = configure_attention(model, v["height"], v["width"])
    if state_signature(model) != state_before: raise RuntimeError("Resolution configuration changed weights")
    observed = []
    def check_features(module, inputs):
        got = tuple(tuple(int(d) for d in value.shape[-2:]) for value in inputs[0])
        if got != shapes: raise RuntimeError(f"Actual feature grids {got} differ from portable attention {shapes}")
        observed.append(got)
    hook = model.depthaware_transformer.register_forward_pre_hook(check_features)
    inputs = [resize_inputs(data, variant) for data in source_inputs]
    try:
        with torch.inference_mode(): wrapper(*(torch.from_numpy(inputs[0][n]) for n in INPUT_NAMES))
    finally:
        hook.remove()
    if not observed: raise RuntimeError("Feature-grid validation did not execute")
    phone_root = destination/v["resource"]; phone_root.mkdir(parents=True)
    shutil.copyfile(Path(m["repo"])/"LICENSE.txt", phone_root/"LICENSE_MONODETR.txt")
    filename = f"A2_Lite_E1_D1_{v['width']}x{v['height']}_FP32.onnx"
    session, graph = graph_export(wrapper, inputs[0], phone_root/filename, variant)
    rows, fixtures = [], []
    for f, data in zip(package["fixtures"], inputs):
        with torch.inference_mode(): native = base.tensors_to_arrays(wrapper(*(torch.from_numpy(data[n]) for n in INPUT_NAMES)))
        actual = dict(zip(OUTPUT_NAMES, session.run(list(OUTPUT_NAMES), data)))
        raw = compare_outputs(actual, native)
        if not raw["passed"]: raise RuntimeError(f"Candidate export raw parity failed: {f['sample_id']}")
        decoded = decoded_fixture_comparison(actual, native, data["calibration"][0], data["image_size"][0])
        rows.append(dict(sample_id=f["sample_id"], raw_comparison=raw, decoded_comparison=decoded))
        def save_tensors(values, kind):
            records = {}
            for name, value in values.items():
                path = phone_root/"fixtures"/f"{f['sample_id']}_{kind}_{name}.bin"
                record = write_f32(path, value); record["file"] = str(path.relative_to(phone_root)); records[name] = record
            return records
        fixtures.append(dict(sample_id=f["sample_id"], inputs=save_tensors(data, "input"),
                             cpu_reference_outputs=save_tensors(actual, "reference")))
        print(f"{variant} export/decoded parity {len(rows)}/16: passed", flush=True)
    state_after = state_signature(model)
    if state_before != state_after: raise RuntimeError("Model tensors mutated during export")
    timing = base.timing(session, inputs)
    phone = dict(schema_version=1, revision=REVISION, variant_id=variant, complete=True,
                 model=dict(file=filename, sha256=sha256(phone_root/filename),
                            size_bytes=(phone_root/filename).stat().st_size, **graph),
                 checkpoint_sha256=CHECKPOINT_SHA, source_checkpoint_sha256=base.q.A2_SHA,
                 parent_package_sha256=PACKAGE_SHA, model_state_sha256=state_before,
                 checkpoint_kind="layer_pruned_A2_initialization", weights_inherited_from_A2=True,
                 model_role="layer_pruned_resolution_speed_screen", resource_name=v["resource"],
                 precision="FP32", providers=["CPUExecutionProvider"], onnxruntime_version="1.30.0",
                 inputs={n:dict(shape=list(s), dtype="float32") for n,s in input_shapes(variant).items()},
                 outputs={n:dict(shape=list(s), dtype="float32") for n,s in OUTPUT_SHAPES.items()},
                 sample_ids=IDS, fixtures=fixtures, warmups=5, timed_predictions=100, sustain_seconds=60,
                 raw_limits=RAW_LIMITS, architecture={**inventory, "input_resolution_changed": True},
                 feature_spatial_shapes=shapes, preprocessing=PREPROCESSING,
                 calibration_convention="original image P2 unchanged", image_size_convention="original [width,height] unchanged",
                 image_height=v["height"], image_width=v["width"],
                 reference_scope="candidate ONNX CPU versus candidate native PyTorch at identical reduced-resolution inputs",
                 reference_environment=dict(torch=torch.__version__, timm=importlib.metadata.version("timm"),
                                            onnxruntime="1.30.0", numpy=np.__version__),
                 training_performed=False, optimizer_steps=0, recovery_training_performed=False,
                 accuracy_evaluated=False, accuracy_qualified=False, iphone_performance_measured=False,
                 coreml_or_neural_engine_used=False, camera_pipeline_included=False, deployment_qualified=False,
                 must_not_replace_original_A2_assets=True)
    phone["manifest_sha256"] = signature(phone); write_json(phone_root/"manifest.json", phone)
    report = dict(complete=True, revision=REVISION, variant_id=variant, checkpoint_sha256=CHECKPOINT_SHA,
                  parent_package_sha256=PACKAGE_SHA, package_manifest_sha256=phone["manifest_sha256"],
                  onnx_sha256=phone["model"]["sha256"], onnx_size_bytes=phone["model"]["size_bytes"],
                  feature_grids_verified=True, feature_spatial_shapes=shapes, model_weights_unchanged=True,
                  model_state_sha256=state_before, checks=rows, host_cpu_timing=timing,
                  training_performed=False, optimizer_steps=0, accuracy_evaluated=False,
                  iphone_performance_measured=False, deployment_qualified=False)
    write_json(destination/"a2_lite_resolution_export.json", report)
    print(json.dumps({k:report[k] for k in ("variant_id", "feature_spatial_shapes", "model_weights_unchanged", "host_cpu_timing")},indent=2),flush=True)


def bundle(args):
    target = args.output/"a2_lite_resolution_results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(args.output.rglob("*")):
            if path.is_file() and path != target and (path.suffix in (".json", ".log", ".onnx", ".bin") or path.name.startswith("LICENSE")):
                z.write(path, path.relative_to(args.output))
    print(target, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "export", "bundle"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--baseline", type=Path)
    p.add_argument("--repo", type=Path)
    p.add_argument("--variant", choices=tuple(VARIANTS))
    args = p.parse_args()
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    for key in ("baseline", "repo"):
        if getattr(args,key) is not None: setattr(args,key,getattr(args,key).resolve())
    if args.stage == "prepare" and (args.baseline is None or args.repo is None): p.error("prepare requires --baseline and --repo")
    if args.stage == "export" and args.variant is None: p.error("export requires --variant")
    try: dict(prepare=prepare, export=export, bundle=bundle)[args.stage](args)
    except Exception as error:
        import traceback
        write_json(args.output/"failures"/f"{args.stage}_{time.time_ns()}.json",
                   dict(complete=False, error=str(error), traceback=traceback.format_exc(), training_performed=False))
        raise


if __name__ == "__main__": main()
