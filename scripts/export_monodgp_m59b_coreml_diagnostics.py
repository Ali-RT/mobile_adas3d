"""Export an intermediate-tensor Core ML diagnostic for the M58 graph.

M59b is intentionally separate from the M58 product artifact.  It reuses the
exact M56d/M57/M58 provenance, but returns causal intermediate tensors so a
macOS run can identify the first Core ML runtime divergence.  It does not train,
change weights, or authorize a device or deployment experiment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

try:
    from scripts.export_monodgp_m58_coreml import (
        COREMLTOOLS_VERSION,
        M58_COREML_PATCH_MARKER,
        validate_m57_complete_evidence,
    )
    from scripts.prepare_monodgp_m56_fp16_storage import sha256_file
except ModuleNotFoundError:
    from export_monodgp_m58_coreml import (
        COREMLTOOLS_VERSION,
        M58_COREML_PATCH_MARKER,
        validate_m57_complete_evidence,
    )
    from prepare_monodgp_m56_fp16_storage import sha256_file


PROBE_LIMIT = 1.0e-3
INPUT_SHAPES = {
    "image": [1, 3, 384, 1280],
    "calibration": [1, 3, 4],
    "image_size": [1, 2],
}
FINAL_OUTPUT_KEYS = (
    "pred_logits",
    "pred_boxes",
    "pred_3d_dim",
    "pred_depth",
    "pred_angle",
    "pred_depth_map_logits",
)


def tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
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


def validate_m58_gate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    gate = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "complete": True,
        "all_export_gates_passed": True,
        "experimental_coreml_conversion_performed": True,
        "training_performed": False,
        "weights_changed": False,
    }
    gate_results = gate.get("gate_results", {})
    if any(gate.get(key) != value for key, value in required.items()) or gate_results.get(
        "no_training_or_weight_change"
    ) is not True:
        raise RuntimeError("M58 export gate is incomplete or failed")
    if gate.get("macos_coreml_prediction_parity_authorized") is not True:
        raise RuntimeError("M58 export gate did not authorize the diagnostic runtime")
    if gate.get("output_names") is None or gate.get("expected_output_shapes") is None:
        raise RuntimeError("M58 export gate is missing its frozen interface")
    return gate


class Logger:
    def info(self, message):
        print(message, flush=True)


class TapStore:
    """Keep traced tensor taps in deterministic insertion order."""

    def __init__(self, torch_module):
        self.torch = torch_module
        self.values: dict[str, Any] = {}

    def clear(self) -> None:
        self.values.clear()

    def put(self, name: str, value: Any) -> None:
        if not self.torch.is_tensor(value):
            raise RuntimeError(f"Diagnostic tap {name} is not a tensor")
        self.values[name] = value.contiguous()


def _feature_tensor(value):
    return value.tensors if hasattr(value, "tensors") else value


def install_taps(model, torch_module):
    """Attach hooks to the exact modules used by MonoDGP.forward.

    Hooks are only used to return graph values from the export wrapper; no
    hooks are active in the M58 product package.
    """

    taps = TapStore(torch_module)
    handles = []

    def hook(module, callback: Callable[[Any], None]):
        handles.append(module.register_forward_hook(lambda _m, _i, out: callback(out)))

    def capture_backbone(output):
        features, positions = output
        for index, feature in enumerate(features):
            taps.put(f"backbone_feat_{index}", _feature_tensor(feature))
        for index, position in enumerate(positions):
            taps.put(f"backbone_pos_{index}", _feature_tensor(position))

    hook(model.backbone, capture_backbone)
    for index, module in enumerate(model.input_proj):
        hook(module, lambda output, index=index: taps.put(f"input_proj_{index}", output))

    def capture_region(output):
        enhanced, regions, seg_embed = output
        for index, value in enumerate(enhanced):
            taps.put(f"enhanced_src_{index}", value)
        for index, value in enumerate(seg_embed):
            taps.put(f"seg_embed_{index}", value)
        for index, value in enumerate(regions):
            taps.put(f"region_prob_{index}", value)

    hook(model.region_head, capture_region)

    def capture_depth(output):
        depth_logits, depth_pos, weighted_depth = output
        taps.put("depth_map_logits", depth_logits)
        taps.put("depth_pos_embed", depth_pos)
        taps.put("weighted_depth", weighted_depth)

    hook(model.depth_predictor, capture_depth)

    def capture_det2d(output):
        taps.put("det2d_hs_last", output["hs"][-1])

    hook(model.det2d_transformer, capture_det2d)

    def capture_det3d(output):
        taps.put("det3d_hs_last", output[0][-1])

    hook(model.det3d_transformer, capture_det3d)

    # These modules are called by both the 2D and 3D paths when box refinement
    # is disabled.  The final call is the final 3D head, which is the raw value
    # immediately before MonoDGP's reference/activation fusion.
    for key, modules in (
        ("raw_logits", model.class_embed),
        ("raw_bbox", model.bbox_embed),
        ("raw_dim", model.dim_embed_3d),
        ("raw_depth", model.depth_embed),
        ("raw_angle", model.angle_embed),
    ):
        hook(modules[-1], lambda output, key=key: taps.put(key, output))

    return taps, handles


def compare_tensors(reference: dict[str, np.ndarray], candidate: dict[str, np.ndarray], names: list[str]) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name in names:
        expected = reference.get(name)
        actual = candidate.get(name)
        if expected is None or actual is None:
            report[name] = {
                "passed": False,
                "error": "missing tensor",
                "limit": PROBE_LIMIT,
            }
            continue
        expected = np.asarray(expected, dtype=np.float32)
        actual = np.asarray(actual, dtype=np.float32)
        if expected.shape != actual.shape:
            report[name] = {
                "passed": False,
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
                "limit": PROBE_LIMIT,
            }
            continue
        delta = np.abs(actual - expected)
        max_abs = float(delta.max()) if delta.size else 0.0
        mean_abs = float(delta.mean()) if delta.size else 0.0
        report[name] = {
            "passed": bool(np.isfinite(actual).all() and max_abs <= PROBE_LIMIT),
            "shape": list(actual.shape),
            "max_abs_delta": max_abs,
            "mean_abs_delta": mean_abs,
            "limit": PROBE_LIMIT,
        }
    return report


def first_diverging_tensor(comparison: dict[str, dict[str, Any]]) -> str | None:
    for name, row in comparison.items():
        if row.get("passed") is not True:
            return name
    return None


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(
        description="Export M59b causal intermediate tensors for macOS Core ML drift diagnosis."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--m58-export-gate", type=Path, required=True)
    parser.add_argument("--m57-manifest", type=Path, required=True)
    parser.add_argument("--m57-smoke", type=Path, required=True)
    parser.add_argument("--m57-gate", type=Path, required=True)
    parser.add_argument("--m57-comparison", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "m59b_coreml_diagnostic_export_gate.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": False,
        "experiment": "M59b Core ML intermediate-tensor diagnostic export",
        "training_performed": False,
        "weights_changed": False,
        "probe_limit_max_abs": PROBE_LIMIT,
        "diagnostic_runtime_authorized": False,
        "physical_device_testing_authorized": False,
        "fp16_or_quantization_authorized": False,
        "product_safety_qualified": False,
    }
    try:
        import coremltools as ct

        validate_m58_gate(args.m58_export_gate.resolve())
        manifest, _, m57_gate, _ = validate_m57_complete_evidence(
            args.m57_manifest.resolve(),
            args.m57_smoke.resolve(),
            args.m57_gate.resolve(),
            args.m57_comparison.resolve(),
        )
        if ct.__version__ != COREMLTOOLS_VERSION:
            raise RuntimeError(f"Expected coremltools {COREMLTOOLS_VERSION}, found {ct.__version__}")

        repo = args.monodgp_repo.resolve()
        runtime_config = Path(manifest["runtime_config"]).resolve()
        checkpoint = Path(manifest["checkpoint"]).resolve()
        dataset = args.dataset_root.resolve()
        val_split = args.split_dir.resolve() / "val.txt"
        coreml_sources = (
            repo / "lib/models/monodgp/det2d_transformer.py",
            repo / "lib/models/monodgp/det3d_transformer.py",
            repo / "lib/models/monodgp/monodgp.py",
        )
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
        ).stdout.strip()
        if (
            any(M58_COREML_PATCH_MARKER not in path.read_text(encoding="utf-8") for path in coreml_sources)
            or not val_split.is_file()
            or len(val_split.read_text(encoding="utf-8").splitlines()) != 3769
            or dataset != Path(manifest["dataset_root"]).resolve()
        ):
            raise RuntimeError("M59b source, dataset, or split provenance changed")

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
        epoch, _, _ = load_checkpoint(model, None, checkpoint, device, Logger())
        if int(epoch) != 100:
            raise RuntimeError(f"Expected checkpoint epoch 100, found {epoch}")
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
                raise RuntimeError("M59b export path called the native CUDA operator")

        attention_module.MSDeformAttnFunction = ForbiddenNative

        inputs, calibs, _, info = next(iter(validation_loader))
        image = inputs.float().cpu()
        calibration = calibs.float().cpu()
        image_size = info["img_size"].float().cpu()
        if {
            "image": list(image.shape),
            "calibration": list(calibration.shape),
            "image_size": list(image_size.shape),
        } != INPUT_SHAPES:
            raise RuntimeError("M59b fixed input signature changed")

        class DiagnosticWrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped
                self.taps, self.handles = install_taps(wrapped, torch)
                self.output_names: list[str] = []

            def forward(self, image, calibration, image_size):
                self.taps.clear()
                values = self.wrapped(image, calibration, None, image_size, dn_args=0)
                for key in FINAL_OUTPUT_KEYS:
                    self.taps.put(f"final_{key}", values[key])
                for index, value in enumerate(values["pred_region_prob"]):
                    self.taps.put(f"final_region_prob_{index}", value)
                if not self.output_names:
                    self.output_names = list(self.taps.values.keys())
                missing = [name for name in self.output_names if name not in self.taps.values]
                if missing:
                    raise RuntimeError(f"Diagnostic taps disappeared: {missing}")
                return tuple(self.taps.values[name] for name in self.output_names)

        wrapper = DiagnosticWrapper(model).eval()
        with torch.inference_mode():
            reference_tensors = wrapper(image, calibration, image_size)
            output_names = list(wrapper.output_names)
            if len(output_names) < 10:
                raise RuntimeError(f"Too few diagnostic outputs: {output_names}")
            traced = torch.jit.trace(
                wrapper,
                (image, calibration, image_size),
                strict=False,
                check_trace=False,
            )
            traced_tensors = traced(image, calibration, image_size)
        reference = {
            name: value.detach().cpu().numpy()
            for name, value in zip(output_names, reference_tensors)
        }
        traced_arrays = {
            name: value.detach().cpu().numpy()
            for name, value in zip(output_names, traced_tensors)
        }
        trace_comparison = compare_tensors(reference, traced_arrays, output_names)
        if not all(row["passed"] for row in trace_comparison.values()):
            raise RuntimeError(f"M59b TorchScript trace parity failed: {trace_comparison}")

        trace_path = output / "MonoDGP_M59b_intermediate_diagnostic_fp32.pt"
        traced.save(str(trace_path))
        reference_path = output / "m59b_diagnostic_reference_io.npz"
        np.savez_compressed(
            reference_path,
            image=image.numpy(),
            calibration=calibration.numpy(),
            image_size=image_size.numpy(),
            **reference,
        )

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
        package_path = output / "MonoDGP_M59b_intermediate_diagnostic_fp32.mlpackage"
        if package_path.exists():
            shutil.rmtree(package_path)
        mlmodel.save(str(package_path))
        archive_path = Path(
            shutil.make_archive(
                str(output / "MonoDGP_M59b_intermediate_diagnostic_fp32"),
                "zip",
                package_path.parent,
                package_path.name,
            )
        )
        trace_graph = str(traced.inlined_graph)
        spec_type = mlmodel.get_spec().WhichOneof("Type")
        gates = {
            "m58_export_gate_passed": True,
            "coremltools_version": ct.__version__ == COREMLTOOLS_VERSION,
            "fixed_input_signature": {
                "image": list(image.shape),
                "calibration": list(calibration.shape),
                "image_size": list(image_size.shape),
            } == INPUT_SHAPES,
            "native_attention_not_called": native_calls == 0,
            "all_reference_outputs_finite": all(np.isfinite(value).all() for value in reference.values()),
            "trace_parity_within_probe_limit": all(row["passed"] for row in trace_comparison.values()),
            "trace_has_no_custom_attention": "MSDeformAttnFunction" not in trace_graph and "MultiScaleDeformableAttention" not in trace_graph,
            "mlprogram_created": spec_type == "mlProgram",
            "mil_has_no_custom_op": counts.get("custom", 0) == 0,
            "package_saved": package_path.is_dir() and tree_size(package_path) > 0,
            "reference_io_saved": reference_path.is_file(),
            "no_training_or_weight_change": True,
        }
        report.update(
            {
                "complete": True,
                "source_commit": commit,
                "source_checkpoint": str(checkpoint),
                "source_runtime_config": str(runtime_config),
                "m58_export_gate": str(args.m58_export_gate.resolve()),
                "m58_export_gate_sha256": sha256_file(args.m58_export_gate.resolve()),
                "m57_portable_metrics": m57_gate["portable_metrics"],
                "output_names": output_names,
                "output_shapes": {name: list(value.shape) for name, value in reference.items()},
                "trace_parity": trace_comparison,
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
                "gate_results": gates,
                "all_export_gates_passed": all(gates.values()),
                "diagnostic_runtime_authorized": all(gates.values()),
            }
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not report["all_export_gates_passed"]:
            raise RuntimeError(f"M59b diagnostic export gate failed; see {report_path}")
    except Exception as error:
        report["conversion_error"] = {"type": type(error).__name__, "message": str(error)}
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
