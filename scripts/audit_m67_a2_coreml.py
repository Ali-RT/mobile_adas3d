"""M67: unchanged trained A2 export feasibility, with a separate Mac handoff."""
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
sys.path.insert(0, str(ROOT / "scripts"))
import m64_teacher_qualification as q

REVISION = "M67-A2-EXPORT-RUNTIME-AUDIT-2026-10-06-r1"
INPUT_MANIFEST_SHA = "68c2247a9ab28603e10f7f9748fc17a43e5bdefeedc57eebe87e2c62f430d968"
NATIVE_SOURCE_SHA = "f1ce39fa8877e76c3fc1c456de98ec57a8a9db288fdd5a0d68d9ba34f92dfb82"
OUTPUT_NAMES = ("logits", "boxes", "dimensions", "depth", "angle")
OUTPUT_KEYS = ("pred_logits", "pred_boxes", "pred_3d_dim", "pred_depth", "pred_angle")
OUTPUT_SHAPES = dict(logits=(1, 50, 3), boxes=(1, 50, 6), dimensions=(1, 50, 3),
                     depth=(1, 50, 2), angle=(1, 50, 24))
# Prospective FP32 raw-output diagnostic limits, not an AP/deployment gate.
# 1 cm is intentionally a strict geometry smoke bound, not a product metric.
RAW_LIMITS = dict(logits=0.001, boxes=0.0001, dimensions=0.001, depth=0.01, angle=0.001)
PHONE_TARGETS = dict(model_p95_ms=50.0, capture_to_result_p95_ms=100.0,
                     sustained_processed_fps=10.0)
SAMPLES = 16


def signed(value):
    value = copy.deepcopy(value)
    value["signature_sha256"] = q.signature(value)
    return value


def verify_signed(value):
    if q.signature({k: v for k, v in value.items() if k != "signature_sha256"}) != value.get("signature_sha256"):
        raise RuntimeError("M67 signed record changed")


def implementation_hash():
    files = [
        Path(__file__),
        ROOT / "scripts/patch_monodetr_colab_compat.py",
        ROOT / "scripts/patch_monodetr_product_taxonomy.py",
        ROOT / "scripts/patch_monodetr_mobilenetv4.py",
        ROOT / "scripts/patch_m64_inference_sources.py",
        ROOT / "scripts/build_m64_attention.py",
        ROOT / "scripts/setup_m64_runtime.py",
        ROOT / "scripts/patch_monodetr_coreml_export.py",
        ROOT / "scripts/probe_coreml_full_monodetr.py",
        ROOT / "third_party/monodetr/coreml_export.patch",
        ROOT / "scripts/m64_teacher_qualification.py",
        ROOT / "scripts/setup_m67_runtime.py",
    ]
    missing = [str(path) for path in files if not path.is_file()]
    if missing:
        raise RuntimeError(f"Incomplete M67 implementation inventory: {missing}")
    return q.signature({str(p.relative_to(ROOT)): q.sha(p) for p in files})


def tree_hash(path):
    path = Path(path)
    files = {str(p.relative_to(path)): q.sha(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not files:
        raise RuntimeError(f"Empty package: {path}")
    return q.signature(files)


def validate_original(value):
    if (value.get("revision") != "M66-R0-A2-VEHICLE-FEATURE-KD-2026-10-06-r1"
            or value.get("manifest_sha256") != INPUT_MANIFEST_SHA
            or q.signature({k: v for k, v in value.items() if k != "manifest_sha256"}) != INPUT_MANIFEST_SHA):
        raise RuntimeError("Use the exact reviewed M66 manifest as an original-A2 configuration record")
    entry = value["models"]["a2"]
    cfg = entry["config"]
    required = dict(backbone="mobilenetv4_conv_medium.e500_r256_in1k", backbone_source="timm",
                    num_classes=3, num_queries=50, num_feature_levels=4, hidden_dim=256,
                    enc_layers=3, dec_layers=3, nheads=8, with_box_refine=True,
                    two_stage=False, use_dab=False, two_stage_dino=False)
    if (entry["checkpoint_sha256"] != q.A2_SHA or entry["patched_source_sha256"] != NATIVE_SOURCE_SHA
            or entry["upstream_commit"] != q.A2_COMMIT
            or any(cfg["model"].get(k) != v for k, v in required.items())
            or cfg["dataset"].get("meanshape") is not False
            or cfg["dataset"].get("class_mapping") != q.CLASS_MAPPING
            or cfg["model"].get("backbone_pretrained") is not False):
        raise RuntimeError("Original A2 architecture, taxonomy or checkpoint identity differs")
    return copy.deepcopy(entry)


def restore_inputs(dataset, splits, candidates):
    """Restore only the 16 required validation cases, without downloads or resets."""
    ids = q.split_ids(splits / "val.txt", "val")[:SAMPLES]
    sources = {}
    for folder, suffix in (("image_2", ".png"), ("calib", ".txt"), ("label_2", ".txt")):
        aliases = [folder] + ([folder[:-1] + "02"] if folder in {"image_2", "label_2"} else [])
        choices = [root / "training" / alias for root in [dataset, *candidates] for alias in aliases]
        source = next((p for p in choices if all((p / (i + suffix)).is_file() for i in ids)), None)
        if source is None:
            raise RuntimeError(f"Missing fixed16 {folder}. Restore KITTI locally or add --candidate; checked {choices}")
        sources[folder] = source.resolve()
        dest = dataset / "training" / folder
        if dest.exists() and dest.resolve() != source.resolve():
            raise RuntimeError(f"Preserve conflicting dataset pointer: {dest}")
    dest_split = dataset / "ImageSets/val.txt"
    if dest_split.exists() and dest_split.read_bytes() != (splits / "val.txt").read_bytes():
        raise RuntimeError(f"Preserve changed split: {dest_split}")
    # All source/split checks precede local pointer changes.
    (dataset / "training").mkdir(parents=True, exist_ok=True)
    dest_split.parent.mkdir(parents=True, exist_ok=True)
    for folder, source in sources.items():
        dest = dataset / "training" / folder
        if dest.is_symlink() and not dest.exists():
            dest.unlink()  # Only the dangling local pointer, never source data.
        if not dest.exists():
            dest.symlink_to(source, target_is_directory=True)
    if not dest_split.exists():
        shutil.copy2(splits / "val.txt", dest_split)
    return ids


def fixture_inventory(dataset, ids):
    return {f"{folder}/{i}{suffix}": q.sha(dataset / "training" / folder / (i + suffix))
            for i in ids for folder, suffix in (("image_2", ".png"), ("calib", ".txt"), ("label_2", ".txt"))}


def prepare(args):
    import torch
    from patch_monodetr_coreml_export import patch_monodetr
    old = q.read_json(args.a2_manifest)
    entry = validate_original(old)
    runtime_receipt_path = args.runtime_receipt.resolve()
    runtime_receipt = q.read_json(runtime_receipt_path)
    runtime_signature = runtime_receipt.get("signature_sha256")
    if (runtime_receipt.get("complete") is not True
            or runtime_receipt.get("revision") != REVISION
            or runtime_receipt.get("coremltools") != "9.0"
            or Path(runtime_receipt.get("python", "")).resolve() != Path(sys.executable).resolve()
            or q.signature({k: v for k, v in runtime_receipt.items() if k != "signature_sha256"}) != runtime_signature
            or importlib.metadata.version("coremltools") != "9.0"):
        raise RuntimeError("M67 isolated CUDA13/Core ML runtime receipt is missing, changed or from another interpreter")
    repo, output, dataset = args.repo.resolve(), args.output.resolve(), args.dataset_root.resolve()
    if not repo.name.startswith("MonoDETR_M67"):
        raise RuntimeError("Use a new MonoDETR_M67 checkout, never a historical checkout")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip() != q.A2_COMMIT:
        raise RuntimeError("Unexpected upstream source commit")
    if q.sha(entry["checkpoint"]) != q.A2_SHA:
        raise RuntimeError("Original A2 epoch130 checkpoint is missing or changed")
    receipt = repo / "lib/models/monodetr/ops/m64_build_receipt.json"
    build = q.read_json(receipt)
    binary = Path(build["binary"])
    if not build.get("import_verified") or not binary.resolve().is_relative_to(repo) or q.sha(binary) != build["binary_sha256"]:
        raise RuntimeError("Build the attention binary in this new checkout first")
    if not torch.cuda.is_available() or torch.version.cuda != "13.0":
        raise RuntimeError("Native parity requires the prospective CUDA13 GPU runtime")
    if importlib.metadata.version("coremltools") != "9.0":
        raise RuntimeError("Use the frozen coremltools==9.0 converter for this audit")
    ids = restore_inputs(dataset, args.split_dir.resolve(), [p.resolve() for p in args.candidate])
    path = output / "m67_manifest.json"
    source = q.source_hash(repo)
    previous = q.read_json(path) if path.exists() else None
    if previous:
        verify_signed(previous)
        if source != previous["export_source_sha256"]:
            raise RuntimeError("Existing M67 source differs; preserve the run and use a new RUN_ID")
    elif source != NATIVE_SOURCE_SHA:
        raise RuntimeError("Fresh native A2 source does not match the reviewed inference baseline")
    patch_monodetr(repo)
    cfg = entry["config"]
    cfg["dataset"]["root_dir"] = str(dataset)
    manifest = dict(schema_version=1, revision=REVISION, repo=str(repo), output=str(output),
                    checkpoint=entry["checkpoint"], checkpoint_sha256=q.A2_SHA, checkpoint_epoch=130,
                    config=cfg, native_source_sha256=NATIVE_SOURCE_SHA, export_source_sha256=q.source_hash(repo),
                    original_manifest_sha256=INPUT_MANIFEST_SHA, original_manifest_file_sha256=q.sha(args.a2_manifest),
                    runtime_receipt=str(runtime_receipt_path), runtime_receipt_sha256=q.sha(runtime_receipt_path),
                    environment=q.environment(),
                    coremltools_version="9.0", implementation_sha256=implementation_hash(),
                    attention_binary=str(binary), attention_binary_sha256=q.sha(binary),
                    dataset_root=str(dataset), sample_ids=ids, input_files=fixture_inventory(dataset, ids),
                    raw_limits=RAW_LIMITS, phone_targets=PHONE_TARGETS, input_shape=[1, 3, 384, 1280],
                    native_classes={"Pedestrian": 0, "Vehicle": 1, "unused_Cyclist": 2},
                    precision="FP32", optimizer_steps=0, training_authorized=False,
                    quantization_authorized=False, deployment_authorized=False,
                    full_validation_performed=False, iphone_performance_measured=False)
    manifest = signed(manifest)
    if previous and previous != manifest:
        raise RuntimeError("Run inputs/environment changed; preserve existing files and use a new RUN_ID")
    q.write_json(path, manifest)
    print(f"{REVISION}: exact A2/config, fixed16 inputs and prospective source frozen", flush=True)


def load_manifest(path, native=True):
    m = q.read_json(path)
    verify_signed(m)
    if (m.get("revision") != REVISION or m.get("checkpoint_sha256") != q.A2_SHA
            or m.get("raw_limits") != RAW_LIMITS or m.get("phone_targets") != PHONE_TARGETS
            or m.get("implementation_sha256") != implementation_hash()):
        raise RuntimeError("M67 protocol or implementation changed")
    if native:
        if q.sha(m["checkpoint"]) != q.A2_SHA or q.source_hash(Path(m["repo"])) != m["export_source_sha256"]:
            raise RuntimeError("A2 weights or prospective source changed")
        if q.sha(m["attention_binary"]) != m["attention_binary_sha256"] or q.environment() != m["environment"]:
            raise RuntimeError("Native binary or runtime changed; choose a fresh RUN_ID")
        if (q.sha(m["runtime_receipt"]) != m["runtime_receipt_sha256"]
                or importlib.metadata.version("coremltools") != "9.0"):
            raise RuntimeError("Core ML converter/runtime receipt changed; choose a fresh RUN_ID")
        if fixture_inventory(Path(m["dataset_root"]), m["sample_ids"]) != m["input_files"]:
            raise RuntimeError("Fixed16 source images/calibration/labels changed")
    return m


def compare_outputs(actual, expected):
    import numpy as np
    if set(actual) != set(OUTPUT_NAMES) or set(expected) != set(OUTPUT_NAMES):
        raise RuntimeError("Missing or unexpected raw-output family")
    rows = {}
    for name in OUTPUT_NAMES:
        a, b = np.asarray(actual[name]), np.asarray(expected[name])
        if a.shape != OUTPUT_SHAPES[name] or b.shape != OUTPUT_SHAPES[name]:
            raise RuntimeError(f"Changed {name} shape: {a.shape}/{b.shape}")
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise RuntimeError(f"Nonfinite {name} output")
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        rows[name] = dict(max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                          limit=RAW_LIMITS[name], passed=bool(delta.max() <= RAW_LIMITS[name]))
    return dict(outputs=rows, passed=all(row["passed"] for row in rows.values()))


def set_export(model, enabled):
    count = 0
    for module in model.modules():
        if hasattr(module, "coreml_export"):
            module.coreml_export = enabled
            count += 1
    if not count:
        raise RuntimeError("No export adapters found")
    return count


def arrays(values):
    return {name: value.detach().cpu().numpy() for name, value in zip(OUTPUT_NAMES, values)}


def latency_summary(samples):
    import numpy as np
    if not samples or not np.isfinite(samples).all() or min(samples) <= 0:
        raise RuntimeError("Invalid latency samples")
    return dict(timed_runs=len(samples), mean_ms=float(np.mean(samples)),
                p50_ms=float(np.percentile(samples, 50)), p95_ms=float(np.percentile(samples, 95)),
                min_ms=float(min(samples)), max_ms=float(max(samples)))


def verify_artifacts(output, report):
    verify_signed(report)
    for relative, digest in report["artifacts"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise RuntimeError(f"Artifact path escapes audit folder: {relative}")
        path = output / relative
        if not path.resolve().is_relative_to(output.resolve()):
            raise RuntimeError(f"Artifact symlink escapes audit folder: {relative}")
        if (tree_hash(path) if path.is_dir() else q.sha(path)) != digest:
            raise RuntimeError(f"Export artifact changed: {relative}")


def audit(args):
    import numpy as np
    import torch
    import coremltools as ct
    from torch import nn
    m = load_manifest(args.manifest)
    output, repo = Path(m["output"]), Path(m["repo"])
    report_path = output / "m67_export_audit.json"
    if report_path.exists():
        old = q.read_json(report_path)
        verify_artifacts(output, old)
        if old["manifest_signature"] != m["signature_sha256"] or not old["export_gate_passed"]:
            raise RuntimeError("Prior audit differs/failed; preserve it and choose a fresh RUN_ID")
        print("Verified completed M67 export reused. No model changes or training.", flush=True)
        return
    # Never mix an interrupted conversion with a fresh conversion's outputs.
    for name in ("A2_M67_FP32.pt", "A2_M67_FP32.mlpackage", "fixtures"):
        if (output / name).exists():
            raise RuntimeError(f"Partial {name} exists; preserve it and choose a fresh RUN_ID")
    q.seed()
    sys.path[:0] = [str(repo), str(repo / "lib/models/monodetr/ops")]
    import MultiScaleDeformableAttention as extension
    if Path(extension.__file__).resolve() != Path(m["attention_binary"]).resolve():
        raise RuntimeError("Attention import resolved outside the frozen checkout")
    from lib.helpers.model_helper import build_model
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    if any("rotate_iou" in k or "kitti_eval_python.eval" in k for k in sys.modules):
        raise RuntimeError("AP evaluator must not initialize during export")
    cfg = copy.deepcopy(m["config"])
    dataset = KITTI_Dataset("val", cfg["dataset"])
    dataset.data_augmentation = False
    if dataset.cls2id != {"Pedestrian": 0, "Car": 1, "Cyclist": 2}:
        raise RuntimeError("Native class order changed")
    model, _ = build_model(cfg["model"])
    payload = q.safe_payload(Path(m["checkpoint"]))
    if int(payload.get("epoch", -1)) != 130:
        raise RuntimeError("Expected original A2 epoch130")
    model.load_state_dict(payload["model_state"], strict=True)
    del payload
    model.cuda().eval().requires_grad_(False)
    state_before = q.tensor_hash(model.state_dict())

    class Wrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, image, calibration, image_size):
            values = self.base(image, calibration, None, image_size)
            return tuple(values[key] for key in OUTPUT_KEYS)

    wrapper = Wrapper(model).eval()
    fixture_values, comparisons = [], []
    with torch.no_grad():
        for index, sample in enumerate(m["sample_ids"]):
            image, calibration, _, info = dataset[index]
            inputs = (torch.from_numpy(image).unsqueeze(0).float().cuda(),
                      torch.from_numpy(calibration).unsqueeze(0).float().cuda(),
                      torch.as_tensor(info["img_size"]).unsqueeze(0).float().cuda())
            if tuple(inputs[0].shape) != (1, 3, 384, 1280) or f'{int(info["img_id"]):06d}' != sample:
                raise RuntimeError("Input geometry or fixed16 order changed")
            set_export(model, False)
            native = arrays(wrapper(*inputs))
            set_export(model, True)
            portable = arrays(wrapper(*inputs))
            comparison = compare_outputs(portable, native)
            comparisons.append(dict(sample_id=sample, cuda_export_vs_native=comparison))
            fixture_values.append((tuple(t.cpu() for t in inputs), native))
            print(f"fixed16 CUDA {index + 1}/{SAMPLES}: {sample}, parity={comparison['passed']}", flush=True)
        if not all(row["cuda_export_vs_native"]["passed"] for row in comparisons):
            q.write_json(output / "m67_parity_failure.json", signed(dict(revision=REVISION, rows=comparisons,
                         stage="CUDA export versus native", optimizer_steps=0, export_gate_passed=False)))
            raise RuntimeError("Native/export parity failed. Do not convert, change limits or test phone speed")
        # Native GPU timing is separate from Core ML/device performance.
        set_export(model, False)
        first_gpu = tuple(t.cuda() for t in fixture_values[0][0])
        for _ in range(5):
            wrapper(*first_gpu)
        torch.cuda.synchronize()
        latencies = []
        for _ in range(20):
            start = time.perf_counter()
            wrapper(*first_gpu)
            torch.cuda.synchronize()
            latencies.append((time.perf_counter() - start) * 1000)
        native_timing = latency_summary(latencies)
        native_timing.update(scope="batch-one native CUDA model-only, fixed sample, 5 warmups",
                             gpu=torch.cuda.get_device_name(0), iphone_latency_claimed=False)
        set_export(model, True)
        model.cpu()
        start = time.perf_counter()
        traced = torch.jit.trace(wrapper, fixture_values[0][0], strict=True, check_trace=False)
        trace_seconds = time.perf_counter() - start
        for row, (inputs, native) in zip(comparisons, fixture_values):
            row["cpu_export_vs_native"] = compare_outputs(arrays(wrapper(*inputs)), native)
            row["trace_vs_native"] = compare_outputs(arrays(traced(*inputs)), native)
            print(f"fixed16 CPU/trace: {row['sample_id']}, parity={row['trace_vs_native']['passed']}", flush=True)
        if not all(row[key]["passed"] for row in comparisons for key in ("cpu_export_vs_native", "trace_vs_native")):
            q.write_json(output / "m67_parity_failure.json", signed(dict(revision=REVISION, rows=comparisons,
                         stage="CPU export and traced graph versus native", optimizer_steps=0, export_gate_passed=False)))
            raise RuntimeError("CPU/trace parity failed. Preserve the report; stop before Core ML conversion")
    if q.tensor_hash(model.state_dict()) != state_before or any(p.grad is not None for p in model.parameters()):
        raise RuntimeError("Model state or gradient buffers changed during the zero-update audit")
    output.mkdir(parents=True, exist_ok=True)
    trace = output / "A2_M67_FP32.pt"
    traced.save(str(trace))
    artifact_hashes = {trace.name: q.sha(trace)}
    for sample, (inputs, native) in zip(m["sample_ids"], fixture_values):
        path = output / "fixtures" / (sample + ".npz")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            np.savez_compressed(stream, image=inputs[0].numpy(), calibration=inputs[1].numpy(),
                                image_size=inputs[2].numpy(), **native)
        artifact_hashes[str(path.relative_to(output))] = q.sha(path)
    start = time.perf_counter()
    mlmodel = ct.convert(traced, convert_to="mlprogram", minimum_deployment_target=ct.target.iOS17,
                         compute_precision=ct.precision.FLOAT32, compute_units=ct.ComputeUnit.CPU_ONLY,
                         skip_model_load=True,
                         inputs=[ct.TensorType(name=name, shape=t.shape, dtype=np.float32)
                                 for name, t in zip(("image", "calibration", "image_size"), fixture_values[0][0])],
                         outputs=[ct.TensorType(name=name) for name in OUTPUT_NAMES])
    conversion_seconds = time.perf_counter() - start
    from probe_coreml_full_monodetr import operation_counts
    counts = dict(sorted(operation_counts(mlmodel._mil_program).items()))
    package = output / "A2_M67_FP32.mlpackage"
    mlmodel.save(str(package))
    artifact_hashes[package.name] = tree_hash(package)
    report = signed(dict(schema_version=1, revision=REVISION, complete=True,
        manifest_signature=m["signature_sha256"], checkpoint_sha256=q.A2_SHA,
        fixed_validation_images=SAMPLES, comparisons=comparisons, raw_limits=RAW_LIMITS,
        native_gpu_latency=native_timing, trace_seconds=trace_seconds, conversion_seconds=conversion_seconds,
        parameters=sum(p.numel() for p in model.parameters()),
        parameter_bytes=sum(p.numel() * p.element_size() for p in model.parameters()),
        fp32_package_bytes=sum(p.stat().st_size for p in package.rglob("*") if p.is_file()),
        mil_operation_counts=counts, mil_custom_operations=counts.get("custom", 0),
        artifacts=artifact_hashes, export_gate_passed=counts.get("custom", 0) == 0,
        model_state_unchanged=True, optimizer_steps=0, precision="FP32",
        decoded_detection_parity_performed=False, full_validation_performed=False,
        iphone_performance_measured=False, deployment_authorized=False,
        next_action="Review exact Core ML package on Mac, then authorize physical-phone feasibility"))
    q.write_json(report_path, report)
    if q.sha(m["checkpoint"]) != q.A2_SHA:
        raise RuntimeError("Original checkpoint bytes changed")
    if not report["export_gate_passed"]:
        raise RuntimeError("Custom Core ML operations remain; stop before device testing")
    print(json.dumps({k: report[k] for k in ("export_gate_passed", "parameters", "fp32_package_bytes", "native_gpu_latency")}, indent=2))
    print("STOP: this is an export audit, not measured iPhone performance or deployment qualification.", flush=True)


def review_mac(args):
    import numpy as np
    import coremltools as ct
    if sys.platform != "darwin":
        raise RuntimeError("Core ML execution requires a Mac; Linux only exports")
    output = args.output.resolve()
    path = output / f"m67_mac_{args.compute_units}.json"
    if path.exists():
        raise RuntimeError(f"Preserve existing Mac report: {path}; choose a new extracted audit folder")
    m = load_manifest(output / "m67_manifest.json", native=False)
    audit_report = q.read_json(output / "m67_export_audit.json")
    verify_artifacts(output, audit_report)
    if not audit_report["export_gate_passed"] or audit_report["manifest_signature"] != m["signature_sha256"]:
        raise RuntimeError("No passing M67 export with matching identity")
    units = {"all": ct.ComputeUnit.ALL, "cpu_only": ct.ComputeUnit.CPU_ONLY,
             "cpu_and_neural_engine": ct.ComputeUnit.CPU_AND_NE}[args.compute_units]
    started = time.perf_counter()
    model = ct.models.MLModel(str(output / "A2_M67_FP32.mlpackage"), compute_units=units)
    load_seconds = time.perf_counter() - started
    rows = []
    first = None
    for sample in m["sample_ids"]:
        with np.load(output / "fixtures" / (sample + ".npz"), allow_pickle=False) as values:
            inputs = {k: values[k] for k in ("image", "calibration", "image_size")}
            reference = {k: values[k] for k in OUTPUT_NAMES}
        actual = model.predict(inputs)
        result = compare_outputs(actual, reference)
        rows.append(dict(sample_id=sample, **result))
        if first is None:
            first = inputs
        print(f"Mac fixed16 {sample}: parity={result['passed']}", flush=True)
    latencies = []
    if all(row["passed"] for row in rows):
        for _ in range(5):
            model.predict(first)
        for _ in range(20):
            start = time.perf_counter()
            model.predict(first)
            latencies.append((time.perf_counter() - start) * 1000)
    report = signed(dict(revision=REVISION, complete=True, manifest_signature=m["signature_sha256"],
        export_report_signature=audit_report["signature_sha256"], compute_units=args.compute_units,
        coremltools_version=ct.__version__, load_seconds=load_seconds, rows=rows,
        fixed16_raw_parity_passed=all(row["passed"] for row in rows),
        mac_model_latency=latency_summary(latencies) if latencies else None,
        iphone_performance_measured=False, decoded_detection_parity_performed=False,
        deployment_authorized=False, optimizer_steps=0))
    q.write_json(path, report)
    if not report["fixed16_raw_parity_passed"]:
        raise RuntimeError("Mac raw parity failed; stop before phone timing")
    print("Mac raw parity passed. Phone timing, decoded parity and full AP qualification remain unmeasured.", flush=True)


def bundle(args):
    output = args.output.resolve()
    destination = output / "m67_a2_export_results.zip"
    if destination.exists():
        print(f"Preserving existing result bundle: {destination}", flush=True)
        return
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(output.rglob("*")):
            if p.is_file() and p != destination and p.suffix != ".zip":
                archive.write(p, str(p.relative_to(output)))
    print(destination, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="stage", required=True)
    p = subs.add_parser("prepare")
    for name in ("a2-manifest", "repo", "output", "dataset-root", "split-dir", "runtime-receipt"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--candidate", type=Path, action="append", default=[])
    subs.add_parser("audit").add_argument("--manifest", type=Path, required=True)
    p = subs.add_parser("review-mac")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compute-units", choices=("all", "cpu_only", "cpu_and_neural_engine"), default="all")
    subs.add_parser("bundle").add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dict(prepare=prepare, audit=audit, **{"review-mac": review_mac}, bundle=bundle)[args.stage](args)


if __name__ == "__main__":
    main()
