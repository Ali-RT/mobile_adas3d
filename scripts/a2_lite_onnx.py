"""A2-Lite layer-pruning speed screen, using CPU tensor ops only.

This is not a trained replacement for A2. Load full A2 strictly, retain the
first visual encoder/decoder and their matching stage-0 heads, leave the
depth predictor/backbone untouched, then export and validate a fixed ONNX
graph. Use signed M68 input fixtures: no KITTI restore, CUDA extension,
historical virtual environment, Core ML converter, or optimization steps.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]

import numpy as np
import torch

import m64_teacher_qualification as q
from audit_m67_a2_coreml import validate_original
from compact_a2_teacher import A2Teacher, materialize_export, decoded_fixture_comparison
from m68_onnx_common import (INPUT_NAMES, INPUT_SHAPES, OUTPUT_KEYS, OUTPUT_NAMES,
                             OUTPUT_SHAPES, RAW_LIMITS, compare_outputs, read_f32,
                             sha256, signature, write_f32, write_json)

REVISION = "A2-LITE-E1-D1-ONNX-CPU-SCREEN-2026-10-12-r1"
HEADS = ("class_embed", "bbox_embed", "dim_embed_3d", "angle_embed", "depth_embed")
PATCHES = ("patch_monodetr_colab_compat.py", "patch_monodetr_product_taxonomy.py",
           "patch_monodetr_mobilenetv4.py", "patch_monodetr_coreml_export.py")


def run(command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), cwd=cwd, check=True)


def implementation():
    files = ["scripts/a2_lite_onnx.py", "scripts/compact_a2_teacher.py", "scripts/m68_onnx_common.py",
             "scripts/m64_teacher_qualification.py", "scripts/audit_m67_a2_coreml.py"]
    files += ["scripts/" + name for name in PATCHES]
    files += ["third_party/monodetr/coreml_export.patch", "third_party/monodetr/mobilenetv4_backbone.patch"]
    return {name: sha256(ROOT / name) for name in files}


def prepare_repo(repo):
    """Dedicated disposable checkout; never reset or overwrite existing edits."""
    if not repo.name.startswith("MonoDETR_A2LITE_"):
        raise RuntimeError("Use a dedicated MonoDETR_A2LITE_<RUN_ID> checkout")
    receipt = repo / "a2_lite_source_receipt.json"
    if not repo.exists():
        run(["git", "clone", "--no-checkout", "https://github.com/ZrrSkywalker/MonoDETR.git", repo])
        run(["git", "checkout", "--detach", q.A2_COMMIT], cwd=repo)
    if not (repo / ".git").is_dir(): raise RuntimeError(f"Preserve non-Git directory: {repo}")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if head != q.A2_COMMIT: raise RuntimeError("Wrong A2-Lite upstream commit; use a new checkout")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        if saved != dict(implementation=implementation(), source_sha256=q.source_hash(repo)):
            raise RuntimeError("A2-Lite source/patches changed; preserve checkout and choose a new RUN_ID")
        return
    dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], cwd=repo, text=True)
    if dirty.strip(): raise RuntimeError("Preserve tracked edits and use a fresh A2-Lite checkout")
    for script in PATCHES:
        run([sys.executable, ROOT / "scripts" / script, "--monodetr-repo", repo])
    write_json(receipt, dict(implementation=implementation(), source_sha256=q.source_hash(repo)))


def prune_a2_layers(model):
    """Retain trained stage 0; rebind refinement aliases as well as layer lists."""
    encoder, decoder = model.depthaware_transformer.encoder, model.depthaware_transformer.decoder
    if (encoder.num_layers != 3 or decoder.num_layers != 3
            or len(encoder.layers) != 3 or len(decoder.layers) != 3
            or any(len(getattr(model, name)) != 3 for name in HEADS)):
        raise RuntimeError("Expected full A2's three encoders, decoders and stage heads")
    if model.two_stage or model.use_dab or model.two_stage_dino or not model.with_box_refine:
        raise RuntimeError("Unexpected A2 query/refinement configuration")
    total = sum(p.numel() for p in model.parameters())
    depth = model.depth_predictor
    encoder.layers = torch.nn.ModuleList([encoder.layers[0]]); encoder.num_layers = 1
    decoder.layers = torch.nn.ModuleList([decoder.layers[0]]); decoder.num_layers = 1
    for name in HEADS:
        setattr(model, name, torch.nn.ModuleList([getattr(model, name)[0]]))
    decoder.bbox_embed = model.bbox_embed
    decoder.dim_embed = model.dim_embed_3d
    if model.depth_predictor is not depth: raise RuntimeError("Depth predictor was replaced")
    after = sum(p.numel() for p in model.parameters())
    return dict(full_parameters=total, lite_parameters=after, removed_parameters=total-after,
                visual_encoder_layers_before=3, visual_encoder_layers_after=1,
                decoder_layers_before=3, decoder_layers_after=1, retained_encoder_indices=[0],
                retained_decoder_indices=[0], retained_head_indices=[0],
                depth_module_changed=False, backbone_changed=False, width_changed=False,
                queries_changed=False, input_resolution_changed=False)


def inference_model(repo, model_config):
    sys.path.insert(0, str(repo))
    import lib.models.monodetr.monodetr as native
    if not Path(native.__file__).resolve().is_relative_to(repo.resolve()):
        raise RuntimeError("Wrong MonoDETR package resolved; run this stage in a fresh subprocess")
    c = copy.deepcopy(model_config)
    c.update(device="cpu", backbone_pretrained=False)
    # Native build() also creates DDNLoss, which calls CUDA even in a CPU
    # process. Construct exactly its inference modules, not its unused loss.
    model = native.MonoDETR(native.build_backbone(c), native.build_depthaware_transformer(c),
                           native.DepthPredictor(c), num_classes=c["num_classes"],
                           num_queries=c["num_queries"], aux_loss=c["aux_loss"],
                           num_feature_levels=c["num_feature_levels"], with_box_refine=c["with_box_refine"],
                           two_stage=c["two_stage"], init_box=c["init_box"], use_dab=c["use_dab"],
                           two_stage_dino=c["two_stage_dino"])
    count = 0
    for module in model.modules():
        if hasattr(module, "coreml_export"):
            module.coreml_export = True; count += 1
    if not count: raise RuntimeError("Portable attention adapters are missing")
    # This patch makes native MSDA optional. Fail if an extension was loaded:
    # all inference must use the portable tensor path in this screen.
    if "MultiScaleDeformableAttention" in sys.modules:
        raise RuntimeError("Native attention extension loaded; use a clean CPU-only subprocess")
    return model.cpu().eval().requires_grad_(False)


class Wrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__(); self.base = model

    def forward(self, image, calibration, image_size):
        result = self.base(image, calibration, None, image_size)
        return tuple(result[key] for key in OUTPUT_KEYS)


def tensors_to_arrays(values):
    return {name: v.detach().cpu().numpy().astype(np.float32, copy=True)
            for name, v in zip(OUTPUT_NAMES, values)}


def prepare(args):
    record = validate_original(json.loads(args.a2_record.read_text()))
    checkpoint = args.checkpoint if args.checkpoint is not None else Path(record["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Frozen A2 checkpoint missing: {checkpoint}. Use --checkpoint for a relocated copy; SHA256 must still match original A2.")
    if args.output.is_relative_to(checkpoint.parent.resolve()):
        raise RuntimeError("Use a separate output, never the original A2 checkpoint directory")
    if sha256(checkpoint) != q.A2_SHA: raise RuntimeError("Frozen A2 checkpoint changed")
    prepare_repo(args.repo)
    teacher_root = materialize_export(args.teacher_export, args.output / "teacher_export")
    if args.teacher_export.is_dir() and args.output.is_relative_to(args.teacher_export):
        raise RuntimeError("Use a separate output, never the original M68 teacher export directory")
    teacher = A2Teacher(teacher_root)
    if len(teacher.package["fixtures"]) != 16: raise RuntimeError("Need all16 signed M68 fixtures")
    for f in teacher.package["fixtures"]:
        inputs = {name: read_f32(teacher.root, f["inputs"][name]) for name in INPUT_NAMES}
        references = {name: read_f32(teacher.root, f["cpu_reference_outputs"][name]) for name in OUTPUT_NAMES}
        if not compare_outputs(teacher.raw(inputs), references)["passed"]:
            raise RuntimeError("Original A2 exact-input ONNX Runtime parity failed")
    value = dict(revision=REVISION, repo=str(args.repo), source_sha256=q.source_hash(args.repo),
                 a2_record=str(args.a2_record), a2_record_sha256=sha256(args.a2_record),
                 checkpoint=str(checkpoint), checkpoint_sha256=q.A2_SHA, original_epoch=130,
                 original_config=record["config"], teacher_export=str(teacher_root),
                 implementation=implementation(), output=str(args.output), original_ort_fixture_parity=True,
                 training_performed=False, optimizer_steps=0, accuracy_qualified=False,
                 phone_targets=dict(model_p95_ms=50., capture_to_result_p95_ms=100., sustained_processed_fps=10.))
    value["signature_sha256"] = signature(value)
    path = args.output / "a2_lite_manifest.json"
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError("Preserve changed screen inputs and use a new RUN_ID")
    write_json(path, value)
    print("Prepared original A2, portable CPU-only source and all16 saved inputs; no training.", flush=True)


def load(args):
    value = json.loads((args.output / "a2_lite_manifest.json").read_text())
    saved = value.pop("signature_sha256")
    if signature(value) != saved: raise RuntimeError("Changed A2-Lite manifest")
    value["signature_sha256"] = saved
    if (value["revision"] != REVISION or value["implementation"] != implementation()
            or q.source_hash(Path(value["repo"])) != value["source_sha256"]
            or sha256(value["a2_record"]) != value["a2_record_sha256"]
            or sha256(value["checkpoint"]) != q.A2_SHA):
        raise RuntimeError("Screen code/source/checkpoint changed; use a new RUN_ID")
    return value


def export_graph(wrapper, first, path):
    import onnx
    import onnxruntime as ort
    with torch.inference_mode():
        torch.onnx.export(wrapper, tuple(torch.from_numpy(first[n]) for n in INPUT_NAMES), str(path),
                          export_params=True, opset_version=17, do_constant_folding=True, dynamo=False,
                          input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES))
    graph = onnx.load(str(path)); onnx.checker.check_model(graph, full_check=True)
    domains = sorted({n.domain for n in graph.graph.node if n.domain not in ("", "ai.onnx")})
    if domains: raise RuntimeError(f"Custom ONNX operator domains: {domains}")
    shapes = {v.name: tuple(d.dim_value for d in v.type.tensor_type.shape.dim) for v in graph.graph.input}
    if shapes != INPUT_SHAPES: raise RuntimeError(f"Changed input contract: {shapes}")
    options = ort.SessionOptions(); options.intra_op_num_threads = 2
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]: raise RuntimeError("Use ORT CPU only")
    if [v.name for v in session.get_outputs()] != list(OUTPUT_NAMES): raise RuntimeError("Changed output names")
    return session, dict(nodes=len(graph.graph.node), custom_operator_domains=domains,
                         opset=17, ir_version=graph.ir_version)


def timing(session, inputs, repetitions=12):
    for i in range(3): session.run(list(OUTPUT_NAMES), inputs[i % len(inputs)])
    times = []
    for i in range(repetitions):
        start = time.perf_counter(); session.run(list(OUTPUT_NAMES), inputs[i % len(inputs)])
        times.append((time.perf_counter()-start)*1000)
    return dict(count=len(times), mean_ms=float(np.mean(times)), p50_ms=float(np.median(times)),
                p95_ms=float(np.quantile(times, .95)), threads=2,
                scope="ORTSession.run only on this host; NOT iPhone or camera-to-label performance")


def export(args):
    import onnxruntime as ort
    if ort.__version__ != "1.30.0": raise RuntimeError("Use onnxruntime==1.30.0, matching the existing iPhone runtime")
    m = load(args); out = args.output / "export"
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("Preserve existing export; choose a fresh RUN_ID instead of overwriting it")
    out.mkdir(parents=True)
    teacher = A2Teacher(m["teacher_export"])
    fixtures = teacher.package["fixtures"]
    inputs = [{n: read_f32(teacher.root, f["inputs"][n]) for n in INPUT_NAMES} for f in fixtures]
    torch.set_num_threads(2); torch.manual_seed(123)
    model = inference_model(Path(m["repo"]), m["original_config"]["model"])
    payload = q.safe_payload(Path(m["checkpoint"]))
    if int(payload.get("epoch", -1)) != 130: raise RuntimeError("Wrong source checkpoint epoch")
    model.load_state_dict(payload["model_state"], strict=True); del payload
    original = Wrapper(model).eval()
    baseline_checks = []
    for f, data in zip(fixtures, inputs):
        with torch.inference_mode(): actual = tensors_to_arrays(original(*(torch.from_numpy(data[n]) for n in INPUT_NAMES)))
        expected = {n: read_f32(teacher.root, f["cpu_reference_outputs"][n]) for n in OUTPUT_NAMES}
        check = decoded_fixture_comparison(actual, expected, data["calibration"][0], data["image_size"][0])
        baseline_checks.append(dict(sample_id=f["sample_id"], comparison=check))
        print(f"Full A2 source/decoded fixture verified: {f['sample_id']}", flush=True)
    inventory = prune_a2_layers(model)
    wrapper = Wrapper(model).eval()
    initialized = args.output / "a2_lite_initialization.pth"
    torch.save(dict(epoch=0, model_state=model.state_dict(), optimizer_state=None,
                    source_checkpoint_sha256=q.A2_SHA, source_epoch=130, pruning=inventory,
                    training_performed=False, optimizer_steps=0), initialized)
    package = out / "A2_LITE_ONNX"; package.mkdir()
    shutil.copyfile(Path(m["repo"]) / "LICENSE.txt", package / "LICENSE_MONODETR.txt")
    model_path = package / "A2_Lite_E1_D1_FP32.onnx"
    session, graph = export_graph(wrapper, inputs[0], model_path)
    rows, exported_fixtures = [], []
    for f, data in zip(fixtures, inputs):
        with torch.inference_mode(): expected = tensors_to_arrays(wrapper(*(torch.from_numpy(data[n]) for n in INPUT_NAMES)))
        actual = dict(zip(OUTPUT_NAMES, session.run(list(OUTPUT_NAMES), data)))
        parity = compare_outputs(actual, expected)
        rows.append(dict(sample_id=f["sample_id"], comparison=parity))
        if not parity["passed"]:
            write_json(out / "a2_lite_export_failure.json", dict(complete=False, rows=rows, training_performed=False))
            raise RuntimeError("A2-Lite ONNX/CPU PyTorch parity failed; no phone-ready package")
        def save_tensors(values, kind):
            records = {}
            for name, value in values.items():
                path = package / "fixtures" / f"{f['sample_id']}_{kind}_{name}.bin"
                record = write_f32(path, value); record["file"] = str(path.relative_to(package)); records[name] = record
            return records
        exported_fixtures.append(dict(sample_id=f["sample_id"], inputs=save_tensors(data, "input"),
                                     cpu_reference_outputs=save_tensors(actual, "reference")))
        print(f"A2-Lite export parity: {f['sample_id']} pass=True", flush=True)
    phone = dict(schema_version=1, revision=REVISION, complete=True,
                 model=dict(file=model_path.name, sha256=sha256(model_path), size_bytes=model_path.stat().st_size, **graph),
                 checkpoint_sha256=sha256(initialized), source_checkpoint_sha256=q.A2_SHA,
                 checkpoint_kind="layer_pruned_A2_initialization", model_role="layer_pruned_architecture_speed_screen",
                 precision="FP32", providers=session.get_providers(), onnxruntime_version=ort.__version__,
                 inputs={n:dict(shape=list(s), dtype="float32") for n,s in INPUT_SHAPES.items()},
                 outputs={n:dict(shape=list(s), dtype="float32") for n,s in OUTPUT_SHAPES.items()},
                 sample_ids=[f["sample_id"] for f in fixtures], fixtures=exported_fixtures,
                 warmups=5, timed_predictions=100, sustain_seconds=60, raw_limits=RAW_LIMITS,
                 architecture=inventory, training_performed=False, optimizer_steps=0,
                 weights_inherited_from_A2=True, accuracy_evaluated=False, accuracy_qualified=False,
                 iphone_performance_measured=False, deployment_qualified=False,
                 coreml_or_neural_engine_used=False, camera_pipeline_included=False,
                 resource_name="A2_LITE_ONNX", must_not_replace_original_A2_assets=True)
    phone["manifest_sha256"] = signature(phone); write_json(package / "manifest.json", phone)
    print("Paired host CPU timing: original A2, then A2-Lite; fixed order, not randomized.", flush=True)
    baseline_timing, lite_timing = timing(teacher.session, inputs), timing(session, inputs)
    report = dict(complete=True, revision=REVISION, manifest_sha256=m["signature_sha256"],
                  pruning=inventory, source_decoded_fixture_checks=baseline_checks, lite_export_checks=rows,
                  onnx_sha256=sha256(model_path), onnx_size_bytes=model_path.stat().st_size,
                  initialization_sha256=sha256(initialized), graph=graph,
                  host_cpu_original=baseline_timing, host_cpu_lite=lite_timing,
                  host_cpu_p50_speedup=baseline_timing["p50_ms"]/lite_timing["p50_ms"],
                  timing_is_fixed_order_not_randomized=True,
                  environment=dict(python=sys.version.split()[0], torch=torch.__version__,
                                   timm=importlib.metadata.version("timm"), onnxruntime=ort.__version__),
                  training_performed=False, optimizer_steps=0, recovery_training_performed=False,
                  accuracy_evaluated=False, iphone_performance_measured=False, deployment_qualified=False,
                  next_step="Review artifacts, integrate separate A2-Lite resource in existing phone app, then measure actual phone latency before training")
    write_json(out / "a2_lite_export_report.json", report)
    print(json.dumps({k: report[k] for k in ("pruning", "host_cpu_original", "host_cpu_lite", "host_cpu_p50_speedup")}, indent=2), flush=True)


def bundle(args):
    target = args.output / "a2_lite_screen_results.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for p in sorted(args.output.rglob("*")):
            if not p.is_file() or p == target or "teacher_export" in p.parts: continue
            if p.suffix in (".json", ".log", ".onnx", ".bin") or p.name.startswith("LICENSE"):
                z.write(p, p.relative_to(args.output))
    print(target, flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("prepare", "export", "bundle"))
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--a2-record", type=Path)
    p.add_argument("--teacher-export", type=Path)
    p.add_argument("--repo", type=Path)
    p.add_argument("--checkpoint", type=Path, help="Only for a relocated copy of the same frozen A2 bytes")
    args = p.parse_args()
    args.output = args.output.resolve(); args.output.mkdir(parents=True, exist_ok=True)
    for key in ("a2_record", "teacher_export", "repo", "checkpoint"):
        if getattr(args,key) is not None: setattr(args,key,getattr(args,key).resolve())
    if args.stage == "prepare" and any(getattr(args,k) is None for k in ("a2_record","teacher_export","repo")):
        p.error("prepare requires a2-record, teacher-export and repo")
    try: dict(prepare=prepare, export=export, bundle=bundle)[args.stage](args)
    except Exception as error:
        import traceback
        write_json(args.output / "failures" / f"{args.stage}_{time.time_ns()}.json",
                   dict(complete=False, error=str(error), traceback=traceback.format_exc(), training_performed=False))
        raise


if __name__ == "__main__": main()
