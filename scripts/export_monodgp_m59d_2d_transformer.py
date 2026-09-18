"""Export a layer-by-layer FP32 Core ML diagnostic for MonoDGP's 2D transformer.

M59d reuses the frozen M59b checkpoint and provenance.  It captures the region
features, depth context, every 2D encoder/decoder layer, and the final 2D query
state.  It performs no training and does not authorize deployment or precision
compression.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np


PROBE_LIMIT = 1.0e-3
INPUT_SHAPES = {
    "image": [1, 3, 384, 1280],
    "calibration": [1, 3, 4],
    "image_size": [1, 2],
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def tree_size(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def operation_counts(program) -> dict[str, int]:
    counts: dict[str, int] = {}

    def visit(block) -> None:
        for operation in block.operations:
            counts[operation.op_type] = counts.get(operation.op_type, 0) + 1
            for child in operation.blocks:
                visit(child)

    visit(program.functions["main"])
    return counts


class TapStore:
    def __init__(self, torch_module):
        self.torch = torch_module
        self.values: dict[str, Any] = {}

    def clear(self) -> None:
        self.values.clear()

    def put(self, name: str, value: Any) -> None:
        if not self.torch.is_tensor(value):
            raise RuntimeError(f"Diagnostic tap {name} is not a tensor")
        self.values[name] = value.contiguous()


def _as_tensor(value: Any, torch_module):
    if torch_module.is_tensor(value):
        return value
    if isinstance(value, dict):
        for key in ("tgt", "output", "memory", "hs"):
            if key in value:
                candidate = _as_tensor(value[key], torch_module)
                if candidate is not None:
                    return candidate
    if isinstance(value, (tuple, list)):
        for candidate in value:
            tensor = _as_tensor(candidate, torch_module)
            if tensor is not None:
                return tensor
    return None


def _feature_tensor(value):
    return value.tensors if hasattr(value, "tensors") else value


def install_taps(model, torch_module):
    taps = TapStore(torch_module)
    handles = []

    def hook(module, callback: Callable[[Any], None]):
        handles.append(module.register_forward_hook(lambda _m, _i, out: callback(out)))

    def capture_region(output):
        enhanced, _regions, _seg_embed = output
        for index, value in enumerate(enhanced):
            taps.put(f"enhanced_src_{index}", value)

    hook(model.region_head, capture_region)

    def capture_depth(output):
        _depth_logits, depth_pos, weighted_depth = output
        taps.put("depth_pos_embed", depth_pos)
        taps.put("weighted_depth", weighted_depth)

    hook(model.depth_predictor, capture_depth)

    transformer = model.det2d_transformer
    encoder_layers = list(transformer.encoder.layers)
    decoder_layers = list(transformer.decoder.layers)
    if not encoder_layers or not decoder_layers:
        raise RuntimeError("M59d requires non-empty 2D encoder and decoder layer lists")
    for index, layer in enumerate(encoder_layers):
        hook(
            layer,
            lambda output, index=index: taps.put(
                f"det2d_encoder_layer_{index}", _as_tensor(output, torch_module)
            ),
        )
    for index, layer in enumerate(decoder_layers):
        hook(
            layer,
            lambda output, index=index: taps.put(
                f"det2d_decoder_layer_{index}", _as_tensor(output, torch_module)
            ),
        )

    def capture_transformer(output):
        if not isinstance(output, dict) or "hs" not in output:
            raise RuntimeError("M59d 2D transformer output has no hs sequence")
        hs = output["hs"]
        taps.put("det2d_hs_last", hs[-1])

    hook(transformer, capture_transformer)
    return taps, handles, len(encoder_layers), len(decoder_layers)


def compare_tensors(reference, candidate, names):
    report = {}
    for name in names:
        expected = reference.get(name)
        actual = candidate.get(name)
        if expected is None or actual is None:
            report[name] = {"passed": False, "error": "missing tensor", "limit": PROBE_LIMIT}
            continue
        expected = np.asarray(expected, dtype=np.float32)
        actual = np.asarray(actual, dtype=np.float32)
        if expected.shape != actual.shape:
            report[name] = {
                "passed": False,
                "error": "shape mismatch",
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
                "limit": PROBE_LIMIT,
            }
            continue
        delta = np.abs(actual - expected)
        report[name] = {
            "passed": bool(np.isfinite(actual).all() and (not delta.size or delta.max() <= PROBE_LIMIT)),
            "shape": list(actual.shape),
            "max_abs_delta": float(delta.max()) if delta.size else 0.0,
            "mean_abs_delta": float(delta.mean()) if delta.size else 0.0,
            "limit": PROBE_LIMIT,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export the M59d 2D-transformer diagnostic.")
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--m59b-export-gate", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    import torch
    import yaml

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "m59d_2d_transformer_export_gate.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": False,
        "experiment": "M59d 2D-transformer layer diagnostic export",
        "training_performed": False,
        "weights_changed": False,
        "probe_limit_max_abs": PROBE_LIMIT,
        "physical_device_testing_authorized": False,
        "fp16_or_quantization_authorized": False,
        "product_safety_qualified": False,
    }
    try:
        import coremltools as ct

        gate = json.loads(args.m59b_export_gate.resolve().read_text(encoding="utf-8"))
        if gate.get("complete") is not True or gate.get("all_export_gates_passed") is not True:
            raise RuntimeError("M59b export gate is incomplete or failed")
        if gate.get("diagnostic_runtime_authorized") is not True:
            raise RuntimeError("M59b gate did not authorize diagnostic reuse")
        repo = args.monodgp_repo.resolve()
        checkpoint = Path(gate["source_checkpoint"]).resolve()
        runtime_config = Path(gate["source_runtime_config"]).resolve()
        split = args.split_dir.resolve() / "val.txt"
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
        if gate.get("source_commit") and commit != gate["source_commit"]:
            raise RuntimeError(f"M59b source commit changed: expected {gate['source_commit']}, found {commit}")
        if not checkpoint.is_file() or not runtime_config.is_file() or not split.is_file():
            raise FileNotFoundError("M59b checkpoint, runtime config, or Chen val split is missing")
        if len(split.read_text(encoding="utf-8").splitlines()) != 3769:
            raise RuntimeError("M59d requires the frozen Chen 3769-image validation split")

        os.environ["MONODGP_PORTABLE_DEFORM_ATTN"] = "1"
        sys.path.insert(0, str(repo))
        from lib.helpers.dataloader_helper import build_dataloader
        from lib.helpers.model_helper import build_model
        from lib.helpers.save_helper import load_checkpoint
        import lib.models.monodgp.ops.modules.ms_deform_attn as attention_module

        config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
        config["dataset"]["batch_size"] = 1
        config["model"]["device"] = "cpu"
        _, validation_loader = build_dataloader(config["dataset"], workers=0)
        model, _ = build_model(config["model"])
        device = torch.device("cpu")
        model = model.to(device).eval()
        epoch, _, _ = load_checkpoint(model, None, checkpoint, device, type("L", (), {"info": print})())
        if int(epoch) != 100:
            raise RuntimeError(f"Expected frozen checkpoint epoch 100, found {epoch}")
        for module in model.modules():
            if isinstance(module, attention_module.MSDeformAttn):
                module.use_portable_deform_attn = True
        if not hasattr(model, "m58_coreml_export_compat"):
            raise RuntimeError("M58 Core ML source patch marker is missing")
        model.m58_coreml_export_compat = True
        model.det2d_transformer.decoder.m58_coreml_export_compat = True
        model.det3d_transformer.decoder.m58_coreml_export_compat = True

        native_calls = 0

        class ForbiddenNative:
            @staticmethod
            def apply(*_arguments, **_kwargs):
                nonlocal native_calls
                native_calls += 1
                raise RuntimeError("M59d export path called the native CUDA operator")

        attention_module.MSDeformAttnFunction = ForbiddenNative
        inputs, calibs, _, info = next(iter(validation_loader))
        image = inputs.float().cpu()
        calibration = calibs.float().cpu()
        image_size = info["img_size"].float().cpu()
        if {"image": list(image.shape), "calibration": list(calibration.shape), "image_size": list(image_size.shape)} != INPUT_SHAPES:
            raise RuntimeError("M59d fixed input signature changed")

        class DiagnosticWrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
                self.taps, self.handles, self.encoder_count, self.decoder_count = install_taps(wrapped, torch)
                self.output_names: list[str] = []

            def forward(self, image, calibration, image_size):
                self.taps.clear()
                self.wrapped(image, calibration, None, image_size, dn_args=0)
                if not self.output_names:
                    self.output_names = list(self.taps.values.keys())
                missing = [name for name in self.output_names if name not in self.taps.values]
                if missing:
                    raise RuntimeError(f"M59d taps disappeared: {missing}")
                return tuple(self.taps.values[name] for name in self.output_names)

        wrapper = DiagnosticWrapper(model).eval()
        with torch.inference_mode():
            reference_tensors = wrapper(image, calibration, image_size)
            output_names = list(wrapper.output_names)
            traced = torch.jit.trace(wrapper, (image, calibration, image_size), strict=False, check_trace=False)
            traced_tensors = traced(image, calibration, image_size)
        reference = {name: value.detach().cpu().numpy() for name, value in zip(output_names, reference_tensors)}
        traced_arrays = {name: value.detach().cpu().numpy() for name, value in zip(output_names, traced_tensors)}
        trace_parity = compare_tensors(reference, traced_arrays, output_names)
        if not all(row["passed"] for row in trace_parity.values()):
            raise RuntimeError(f"M59d TorchScript trace parity failed: {trace_parity}")

        trace_path = output / "MonoDGP_M59d_2d_transformer_fp32.pt"
        traced.save(str(trace_path))
        reference_path = output / "m59d_2d_transformer_reference_io.npz"
        np.savez_compressed(reference_path, image=image.numpy(), calibration=calibration.numpy(), image_size=image_size.numpy(), **reference)
        conversion_start = time.perf_counter()
        mlmodel = ct.convert(
            traced,
            source="pytorch",
            convert_to="mlprogram",
            minimum_deployment_target=ct.target.iOS17,
            compute_precision=ct.precision.FLOAT32,
            compute_units=ct.ComputeUnit.CPU_ONLY,
            skip_model_load=True,
            inputs=[
                ct.TensorType(name="image", shape=image.shape, dtype=np.float32),
                ct.TensorType(name="calibration", shape=calibration.shape, dtype=np.float32),
                ct.TensorType(name="image_size", shape=image_size.shape, dtype=np.float32),
            ],
            outputs=[ct.TensorType(name=name) for name in output_names],
        )
        conversion_seconds = time.perf_counter() - conversion_start
        counts = operation_counts(mlmodel._mil_program)
        package_path = output / "MonoDGP_M59d_2d_transformer_fp32.mlpackage"
        if package_path.exists():
            shutil.rmtree(package_path)
        mlmodel.save(str(package_path))
        archive_path = Path(shutil.make_archive(str(output / "MonoDGP_M59d_2d_transformer_fp32"), "zip", package_path.parent, package_path.name))
        trace_graph = str(traced.inlined_graph)
        spec_type = mlmodel.get_spec().WhichOneof("Type")
        gate_results = {
            "m59b_export_gate_passed": True,
            "fixed_input_signature": {"image": list(image.shape), "calibration": list(calibration.shape), "image_size": list(image_size.shape)} == INPUT_SHAPES,
            "native_attention_not_called": native_calls == 0,
            "all_reference_outputs_finite": all(np.isfinite(value).all() for value in reference.values()),
            "trace_parity_within_probe_limit": all(row["passed"] for row in trace_parity.values()),
            "trace_has_no_custom_attention": "MSDeformAttnFunction" not in trace_graph and "MultiScaleDeformableAttention" not in trace_graph,
            "mlprogram_created": spec_type == "mlProgram",
            "mil_has_no_custom_op": counts.get("custom", 0) == 0,
            "package_saved": package_path.is_dir() and tree_size(package_path) > 0,
            "reference_io_saved": reference_path.is_file(),
            "no_training_or_weight_change": True,
        }
        report.update({
            "complete": True,
            "source_commit": commit,
            "source_checkpoint": str(checkpoint),
            "source_runtime_config": str(runtime_config),
            "m59b_export_gate": str(args.m59b_export_gate.resolve()),
            "encoder_layer_count": wrapper.encoder_count,
            "decoder_layer_count": wrapper.decoder_count,
            "output_names": output_names,
            "output_shapes": {name: list(value.shape) for name, value in reference.items()},
            "trace_parity": trace_parity,
            "conversion_seconds": conversion_seconds,
            "coreml_spec_type": spec_type,
            "mil_operation_counts": dict(sorted(counts.items())),
            "artifacts": {
                "torchscript": str(trace_path),
                "torchscript_sha256": sha256_file(trace_path),
                "reference_io": str(reference_path),
                "reference_io_sha256": sha256_file(reference_path),
                "mlpackage": str(package_path),
                "mlpackage_tree_sha256": tree_sha256(package_path),
                "mlpackage_size_bytes": tree_size(package_path),
                "mlpackage_zip": str(archive_path),
                "mlpackage_zip_sha256": sha256_file(archive_path),
            },
            "gate_results": gate_results,
            "all_export_gates_passed": all(gate_results.values()),
        })
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not report["all_export_gates_passed"]:
            raise RuntimeError(f"M59d export gate failed; see {report_path}")
    except Exception as error:
        report["conversion_error"] = {"type": type(error).__name__, "message": str(error)}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
