"""Export the frozen A2 checkpoint to ONNX and audit ONNX Runtime CPU outputs.

The exporter uses the reviewed M67 portable attention path and compares the
result against the original CUDA model. It does not train, quantize, or alter
the A2 checkpoint. A CPU raw-output mismatch is reported, not hidden; the
separate full-validation step determines whether decoded KITTI quality holds.
"""
from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_m67_a2_coreml as m67
import m64_teacher_qualification as q
from m68_onnx_common import (
    INPUT_NAMES, INPUT_SHAPES, OPSET_VERSION, OUTPUT_KEYS, OUTPUT_NAMES,
    OUTPUT_SHAPES, RAW_LIMITS, REVISION, compare_outputs, sha256, signature,
    write_f32, write_json,
)


def tensor_dict(values) -> dict:
    return {name: value.detach().cpu().numpy().astype(np.float32, copy=True)
            for name, value in zip(OUTPUT_NAMES, values)}


def build_model(manifest: dict, dataset_root: Path | None = None):
    repo = Path(manifest["repo"]).resolve()
    ops = repo / "lib/models/monodetr/ops"
    sys.path[:0] = [str(repo), str(ops)]
    import MultiScaleDeformableAttention as extension
    if Path(extension.__file__).resolve() != Path(manifest["attention_binary"]).resolve():
        raise RuntimeError("Wrong attention extension resolved for the frozen A2 checkout")
    from lib.helpers.model_helper import build_model as native_build_model
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset

    cfg = copy.deepcopy(manifest["config"])
    if dataset_root is not None:
        cfg["dataset"]["root_dir"] = str(Path(dataset_root).resolve())
    dataset = KITTI_Dataset("val", cfg["dataset"])
    dataset.data_augmentation = False
    if dataset.cls2id != {"Pedestrian": 0, "Car": 1, "Cyclist": 2}:
        raise RuntimeError("A2 native class order changed")
    model, _ = native_build_model(cfg["model"])
    payload = q.safe_payload(Path(manifest["checkpoint"]))
    if int(payload.get("epoch", -1)) != 130:
        raise RuntimeError("M68 must use the exact A2 epoch-130 checkpoint")
    model.load_state_dict(payload["model_state"], strict=True)
    del payload
    model.cuda().eval().requires_grad_(False)
    return model, dataset


def input_tensors(dataset, index: int, expected_id: str):
    image, calibration, _, info = dataset[index]
    sample_id = f'{int(info["img_id"]):06d}'
    if sample_id != expected_id:
        raise RuntimeError(f"Fixed validation order changed: expected {expected_id}, got {sample_id}")
    values = {
        "image": image.unsqueeze(0).float().cuda(),
        "calibration": calibration.unsqueeze(0).float().cuda(),
        "image_size": torch.as_tensor(info["img_size"]).unsqueeze(0).float().cuda(),
    }
    for name, value in values.items():
        if tuple(value.shape) != INPUT_SHAPES[name] or not bool(torch.isfinite(value).all()):
            raise RuntimeError(f"Unexpected {name} input shape or values: {tuple(value.shape)}")
    return values


def export(args) -> None:
    try:
        import onnx
        import onnxruntime as ort
    except ImportError as exc:
        raise RuntimeError("Install `onnx` and `onnxruntime` in the M67 isolated environment first") from exc

    manifest_path = args.m67_manifest.resolve()
    manifest = m67.load_manifest(manifest_path)
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Preserve existing M68 output and use a fresh RUN_ID: {output}")
    if not torch.cuda.is_available() or torch.version.cuda != "13.0":
        raise RuntimeError("M68 export references must be generated on the reviewed CUDA 13 runtime")
    if q.sha(Path(manifest["checkpoint"])) != q.A2_SHA:
        raise RuntimeError("Original A2 checkpoint SHA256 changed")
    expected_source = manifest["export_source_sha256"]
    if q.source_hash(Path(manifest["repo"])) != expected_source:
        raise RuntimeError("Reviewed M67 portable export source changed")

    output.mkdir(parents=True, exist_ok=True)
    q.seed()
    model, dataset = build_model(manifest)

    class Wrapper(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, image, calibration, image_size):
            result = self.base(image, calibration, None, image_size)
            return tuple(result[key] for key in OUTPUT_KEYS)

    wrapper = Wrapper(model).eval()
    fixture_inputs, native_outputs, portable_outputs = [], [], []
    for index, sample_id in enumerate(manifest["sample_ids"]):
        inputs = input_tensors(dataset, index, sample_id)
        ordered = tuple(inputs[name] for name in INPUT_NAMES)
        m67.set_export(model, False)
        with torch.inference_mode():
            native = tensor_dict(wrapper(*ordered))
        m67.set_export(model, True)
        with torch.inference_mode():
            portable = tensor_dict(wrapper(*ordered))
        for name in INPUT_NAMES:
            inputs[name] = inputs[name].detach().cpu().numpy().astype(np.float32, copy=True)
        fixture_inputs.append(inputs)
        native_outputs.append(native)
        portable_outputs.append(portable)
        parity = compare_outputs(portable, native)
        print(f"CUDA portable/native {index + 1}/{len(manifest['sample_ids'])}: {sample_id} pass={parity['passed']}", flush=True)

    if not all(compare_outputs(portable, native)["passed"]
               for portable, native in zip(portable_outputs, native_outputs)):
        report = {"revision": REVISION, "complete": False, "stage": "CUDA portable graph versus native A2",
                  "optimizer_steps": 0, "training_performed": False,
                  "rows": [{"sample_id": sample, "comparison": compare_outputs(portable, native)}
                           for sample, portable, native in zip(manifest["sample_ids"], portable_outputs, native_outputs)]}
        write_json(output / "m68_export_failure.json", report)
        raise RuntimeError("M67 portable graph no longer matches native A2; stop before ONNX export")

    model_path = output / "A2_M68_FP32.onnx"
    first = tuple(torch.from_numpy(fixture_inputs[0][name]).cuda() for name in INPUT_NAMES)
    m67.set_export(model, True)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper, first, str(model_path), export_params=True, opset_version=OPSET_VERSION,
            do_constant_folding=True, input_names=list(INPUT_NAMES), output_names=list(OUTPUT_NAMES),
            dynamo=False, training=torch.onnx.TrainingMode.EVAL,
        )
    graph = onnx.load(str(model_path), load_external_data=True)
    onnx.checker.check_model(graph, full_check=True)
    domains = sorted({node.domain for node in graph.graph.node if node.domain not in ("", "ai.onnx")})
    if domains:
        raise RuntimeError(f"Unexpected custom ONNX operator domain(s): {domains}")
    graph_inputs = {value.name: value for value in graph.graph.input}
    if set(graph_inputs) != set(INPUT_NAMES):
        raise RuntimeError(f"Unexpected ONNX input names: {sorted(graph_inputs)}")
    for name in INPUT_NAMES:
        shape = tuple(dim.dim_value for dim in graph_inputs[name].type.tensor_type.shape.dim)
        if shape != INPUT_SHAPES[name]:
            raise RuntimeError(f"Unexpected fixed {name} shape: {shape}")
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise RuntimeError(f"M68 must run CPUExecutionProvider only: {session.get_providers()}")
    if [value.name for value in session.get_inputs()] != list(INPUT_NAMES):
        raise RuntimeError("ONNX Runtime input contract changed")
    if [value.name for value in session.get_outputs()] != list(OUTPUT_NAMES):
        raise RuntimeError("ONNX Runtime output contract changed")

    output_rows, fixture_rows = [], []
    for index, sample_id in enumerate(manifest["sample_ids"]):
        actual_values = session.run(list(OUTPUT_NAMES), fixture_inputs[index])
        actual = {name: np.asarray(value, dtype=np.float32) for name, value in zip(OUTPUT_NAMES, actual_values)}
        portable = portable_outputs[index]
        native = native_outputs[index]
        portable_comparison = compare_outputs(actual, portable)
        native_comparison = compare_outputs(actual, native)
        output_rows.append({"sample_id": sample_id, "onnx_vs_cuda_portable": portable_comparison,
                            "onnx_vs_native_cuda": native_comparison})
        tensors = {}
        fixture_dir = output / "A2_ONNX" / "fixtures"
        for name in INPUT_NAMES:
            tensor_path = fixture_dir / f"{sample_id}_input_{name}.bin"
            tensors[name] = write_f32(tensor_path, fixture_inputs[index][name])
            tensors[name]["file"] = str(tensor_path.relative_to(output / "A2_ONNX"))
        references = {}
        for name in OUTPUT_NAMES:
            tensor_path = fixture_dir / f"{sample_id}_reference_{name}.bin"
            references[name] = write_f32(tensor_path, actual[name])
            references[name]["file"] = str(tensor_path.relative_to(output / "A2_ONNX"))
        fixture_rows.append({"sample_id": sample_id, "inputs": tensors, "cpu_reference_outputs": references})
        print(f"ONNX Runtime CPU {index + 1}/{len(manifest['sample_ids'])}: {sample_id} "
              f"native-pass={native_comparison['passed']}", flush=True)

    portable_native_passed = all(compare_outputs(p, n)["passed"] for p, n in zip(portable_outputs, native_outputs))
    cpu_native_passed = all(row["onnx_vs_native_cuda"]["passed"] for row in output_rows)
    cpu_portable_passed = all(row["onnx_vs_cuda_portable"]["passed"] for row in output_rows)
    onnx_relative = Path("A2_ONNX/A2_M68_FP32.onnx")
    package_model = output / onnx_relative
    package_model.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_path, package_model)
    package_manifest = {
        "schema_version": 1, "revision": REVISION, "complete": True,
        "model": {"file": "A2_M68_FP32.onnx", "sha256": sha256(model_path),
                  "size_bytes": model_path.stat().st_size, "opset": OPSET_VERSION,
                  "ir_version": graph.ir_version, "custom_operator_domains": domains},
        "checkpoint_sha256": q.A2_SHA, "checkpoint_epoch": 130,
        "original_m67_manifest_signature": manifest["signature_sha256"],
        "onnx_version": onnx.__version__, "onnxruntime_version": ort.__version__,
        "providers": session.get_providers(), "precision": "FP32",
        "inputs": {name: {"shape": list(INPUT_SHAPES[name]), "dtype": "float32"} for name in INPUT_NAMES},
        "outputs": {name: {"shape": list(OUTPUT_SHAPES[name]), "dtype": "float32"} for name in OUTPUT_NAMES},
        "sample_ids": list(manifest["sample_ids"]), "warmups": 5, "timed_predictions": 100,
        "sustain_seconds": 60, "optimizer_steps": 0, "training_performed": False,
        "quantization_performed": False, "camera_pipeline_included": False,
        "coreml_or_neural_engine_used": False,
        "raw_limits": RAW_LIMITS,
        "fixtures": fixture_rows,
    }
    package_manifest["manifest_sha256"] = signature(package_manifest)
    write_json(output / "A2_ONNX" / "manifest.json", package_manifest)
    report = {
        "schema_version": 1, "revision": REVISION, "complete": True,
        "checkpoint_sha256": q.A2_SHA, "checkpoint_epoch": 130,
        "m67_manifest_signature": manifest["signature_sha256"],
        "onnx_sha256": sha256(model_path), "onnx_size_bytes": model_path.stat().st_size,
        "onnx_opset": OPSET_VERSION, "onnx_ir_version": graph.ir_version,
        "onnx_custom_operator_domains": domains, "onnx_node_count": len(graph.graph.node),
        "onnx_version": onnx.__version__, "onnxruntime_version": ort.__version__,
        "onnxruntime_provider": session.get_providers(), "comparisons": output_rows,
        "fixed16_portable_vs_native_passed": portable_native_passed,
        "fixed16_ort_cpu_vs_portable_passed": cpu_portable_passed,
        "fixed16_ort_cpu_vs_native_passed": cpu_native_passed,
        "full_validation_performed": False, "iphone_performance_measured": False,
        "deployment_qualified": False, "optimizer_steps": 0, "training_performed": False,
        "quantization_performed": False,
        "next_step": "run full Chen-3769 validation for ONNX Runtime CPU, then only phone-test if decoded quality is acceptable",
    }
    write_json(output / "m68_onnx_export.json", report)
    package_path = output / "m68_a2_onnx_phone_bundle.zip"
    with zipfile.ZipFile(package_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted((output / "A2_ONNX").rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output))
    print(json.dumps({key: report[key] for key in (
        "fixed16_portable_vs_native_passed", "fixed16_ort_cpu_vs_portable_passed",
        "fixed16_ort_cpu_vs_native_passed", "onnx_size_bytes", "onnxruntime_provider")}, indent=2), flush=True)
    print(f"Phone bundle (not yet quality-qualified): {package_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m67-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    export(parser.parse_args())


if __name__ == "__main__":
    main()
