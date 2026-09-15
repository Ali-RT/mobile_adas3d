from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scripts.evaluate_monodgp_m57_deformable_attention import (
        M57_MANIFEST_SHA256,
        M57_SMOKE_SHA256,
        validate_reviewed_pair,
    )
    from scripts.prepare_monodgp_m56_fp16_storage import sha256_file
    from scripts.prepare_monodgp_m57_deformable_attention import (
        EXPECTED_MODULES,
        EXPECTED_M56D_METRICS,
        M56D_CANDIDATE_SHA256,
        M56D_RUNTIME_CONFIG_SHA256,
        M57_PATCHED_SOURCE_SHA256,
        PINNED_COMMIT,
        PORTABLE_ENV,
        same_numbers,
    )
    from scripts.smoke_test_monodgp_m56_fp16_storage import PARITY_LIMITS
except ModuleNotFoundError:
    from evaluate_monodgp_m57_deformable_attention import (
        M57_MANIFEST_SHA256,
        M57_SMOKE_SHA256,
        validate_reviewed_pair,
    )
    from prepare_monodgp_m56_fp16_storage import sha256_file
    from prepare_monodgp_m57_deformable_attention import (
        EXPECTED_MODULES,
        EXPECTED_M56D_METRICS,
        M56D_CANDIDATE_SHA256,
        M56D_RUNTIME_CONFIG_SHA256,
        M57_PATCHED_SOURCE_SHA256,
        PINNED_COMMIT,
        PORTABLE_ENV,
        same_numbers,
    )
    from smoke_test_monodgp_m56_fp16_storage import PARITY_LIMITS


M57_GATE_SHA256 = "63f2b71d1f5be69bad31907db229f79e39fb17c033ba43999c6fdfe1df3b97a7"
M57_COMPARISON_SHA256 = "687b15cd4a044981f4e00fa038f2fc5e7053cee692e13bfdb85fe2f90656412d"
M57_PREDICTION_TREE_SHA256 = (
    "b550362a6f0ba77667b69cf2a15a457c419c1e0db4751e2704228a762841ca05"
)
M58_COREML_PATCH_MARKER = "self.m58_coreml_export_compat"
COREMLTOOLS_VERSION = "9.0"
MINIMUM_DEPLOYMENT_TARGET = "iOS17"
COMPUTE_PRECISION = "FLOAT32"
INPUT_SHAPES = {
    "image": [1, 3, 384, 1280],
    "calibration": [1, 3, 4],
    "image_size": [1, 2],
}
SEMANTIC_OUTPUT_NAMES = (
    "pred_logits",
    "pred_boxes",
    "pred_3d_dim",
    "pred_depth",
    "pred_angle",
    "pred_depth_map_logits",
    "pred_region_prob",
)
REGION_OUTPUT_NAMES = tuple(f"pred_region_prob_{index}" for index in range(4))
OUTPUT_NAMES = SEMANTIC_OUTPUT_NAMES[:-1] + REGION_OUTPUT_NAMES
EXPECTED_OUTPUT_SHAPES = {
    "pred_logits": [1, 50, 3],
    "pred_boxes": [1, 50, 6],
    "pred_3d_dim": [1, 50, 3],
    "pred_depth": [1, 50, 2],
    "pred_angle": [1, 50, 24],
    "pred_depth_map_logits": [1, 81, 24, 80],
    "pred_region_prob_0": [1, 1, 48, 160],
    "pred_region_prob_1": [1, 1, 24, 80],
    "pred_region_prob_2": [1, 1, 12, 40],
    "pred_region_prob_3": [1, 1, 6, 20],
}


class Logger:
    def info(self, message):
        print(message, flush=True)


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


def operation_counts(program) -> Counter:
    counts: Counter = Counter()

    def visit(block) -> None:
        for operation in block.operations:
            counts[operation.op_type] += 1
            for child in operation.blocks:
                visit(child)

    visit(program.functions["main"])
    return counts


def validate_m57_complete_evidence(
    manifest_path: Path,
    smoke_path: Path,
    gate_path: Path,
    comparison_path: Path,
) -> tuple[dict, dict, dict, list[dict[str, str]]]:
    manifest, smoke = validate_reviewed_pair(manifest_path, smoke_path)
    for path, expected in (
        (gate_path, M57_GATE_SHA256),
        (comparison_path, M57_COMPARISON_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Reviewed M57 complete evidence is missing or changed: {path}")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    with comparison_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    row_by_metric = {row["metric"]: row for row in rows}
    expected_metrics = set(EXPECTED_M56D_METRICS) | {"prediction_files"}
    csv_matches = (
        set(row_by_metric) == expected_metrics
        and all(row["passed"] == "True" for row in rows)
        and all(float(row["delta"]) == 0.0 for row in rows)
        and all(
            math.isclose(
                float(row_by_metric[name]["portable"]),
                float(value),
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for name, value in EXPECTED_M56D_METRICS.items()
        )
        and int(float(row_by_metric["prediction_files"]["portable"])) == 3769
    )
    if (
        gate.get("complete") is not True
        or gate.get("training_performed") is not False
        or gate.get("weights_changed") is not False
        or gate.get("checkpoint_sha256") != M56D_CANDIDATE_SHA256
        or gate.get("m57_manifest_sha256") != M57_MANIFEST_SHA256
        or gate.get("m57_smoke_sha256") != M57_SMOKE_SHA256
        or gate.get("portable_environment_variable") != f"{PORTABLE_ENV}=1"
        or gate.get("prediction_files") != 3769
        or gate.get("prediction_tree_sha256") != M57_PREDICTION_TREE_SHA256
        or not same_numbers(gate.get("portable_metrics", {}), EXPECTED_M56D_METRICS)
        or gate.get("portable_minus_m56d") != {
            key: 0.0 for key in EXPECTED_M56D_METRICS
        }
        or not gate.get("preservation_gate_results")
        or not all(gate["preservation_gate_results"].values())
        or gate.get("all_preservation_gates_passed") is not True
        or gate.get("portable_operator_candidate_selected") is not True
        or gate.get("next_coreml_conversion_gate_authorized") is not True
        or gate.get("direct_coreml_conversion_authorized") is not False
        or gate.get("deployment_authorized") is not False
        or gate.get("product_safety_qualified") is not False
        or not csv_matches
    ):
        raise RuntimeError("Exact M57 evidence did not authorize M58 experimental conversion")
    return manifest, smoke, gate, rows


def flatten_export_outputs(values) -> tuple:
    primary = tuple(values[name] for name in SEMANTIC_OUTPUT_NAMES[:-1])
    region_probabilities = values["pred_region_prob"]
    regions = tuple(
        region_probabilities[index] for index in range(len(REGION_OUTPUT_NAMES))
    )
    return primary + regions


def output_family(name: str) -> str:
    if name in REGION_OUTPUT_NAMES:
        return "pred_region_prob"
    return name


def validate_export_outputs(values, torch_module) -> tuple:
    region_probabilities = values.get("pred_region_prob")
    if not isinstance(region_probabilities, (list, tuple)):
        raise RuntimeError("pred_region_prob must be a four-tensor sequence")
    if len(region_probabilities) != len(REGION_OUTPUT_NAMES):
        raise RuntimeError(
            "Expected four pred_region_prob tensors, found "
            f"{len(region_probabilities)}"
        )
    flattened = flatten_export_outputs(values)
    for name, value in zip(OUTPUT_NAMES, flattened):
        if not isinstance(value, torch_module.Tensor):
            raise RuntimeError(f"M58 output {name} is not a tensor")
        shape = list(value.shape)
        if shape != EXPECTED_OUTPUT_SHAPES[name]:
            raise RuntimeError(
                f"M58 output {name} shape changed: expected "
                f"{EXPECTED_OUTPUT_SHAPES[name]}, found {shape}"
            )
    return flattened


def compare_outputs(reference, candidate) -> dict[str, dict[str, Any]]:
    if len(reference) != len(OUTPUT_NAMES) or len(candidate) != len(OUTPUT_NAMES):
        raise RuntimeError(
            f"Expected {len(OUTPUT_NAMES)} flattened outputs; found "
            f"{len(reference)} reference and {len(candidate)} candidate outputs"
        )
    comparison: dict[str, dict[str, Any]] = {}
    for name, expected, actual in zip(OUTPUT_NAMES, reference, candidate):
        if actual.shape != expected.shape:
            raise RuntimeError(
                f"M58 traced output {name} shape changed: "
                f"{list(expected.shape)} versus {list(actual.shape)}"
            )
        difference = (actual.float() - expected.float()).abs()
        maximum = float(difference.max().item()) if difference.numel() else 0.0
        mean = float(difference.mean().item()) if difference.numel() else 0.0
        family = output_family(name)
        comparison[name] = {
            "semantic_family": family,
            "shape": list(expected.shape),
            "max_abs": maximum,
            "mean_abs": mean,
            "limit_max_abs": PARITY_LIMITS[family],
            "finite": bool(actual.isfinite().all().item()),
            "passed": math.isfinite(maximum) and maximum <= PARITY_LIMITS[family],
        }
    return comparison


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(
        description="Create the fixed-shape M58 FP32 Core ML conversion candidate."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
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
    report_path = output / "m58_coreml_export_gate.json"
    manifest_path = args.m57_manifest.resolve()
    smoke_path = args.m57_smoke.resolve()
    gate_path = args.m57_gate.resolve()
    comparison_path = args.m57_comparison.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "complete": False,
        "experiment": "M58 fixed-shape FP32 Core ML experimental conversion",
        "training_performed": False,
        "weights_changed": False,
        "source_checkpoint_sha256": M56D_CANDIDATE_SHA256,
        "m57_manifest_sha256": M57_MANIFEST_SHA256,
        "m57_smoke_sha256": M57_SMOKE_SHA256,
        "m57_gate_sha256": M57_GATE_SHA256,
        "m57_comparison_sha256": M57_COMPARISON_SHA256,
        "input_shapes": INPUT_SHAPES,
        "semantic_output_names": list(SEMANTIC_OUTPUT_NAMES),
        "output_names": list(OUTPUT_NAMES),
        "expected_output_shapes": EXPECTED_OUTPUT_SHAPES,
        "minimum_deployment_target": MINIMUM_DEPLOYMENT_TARGET,
        "compute_precision": COMPUTE_PRECISION,
        "experimental_coreml_conversion_performed": False,
        "macos_coreml_prediction_parity_authorized": False,
        "physical_device_gate_authorized": False,
        "deployment_authorized": False,
        "product_safety_qualified": False,
    }
    try:
        import coremltools as ct

        manifest, _, m57_gate, _ = validate_m57_complete_evidence(
            manifest_path, smoke_path, gate_path, comparison_path
        )
        repo = args.monodgp_repo.resolve()
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        attention_source = repo / "lib/models/monodgp/ops/modules/ms_deform_attn.py"
        coreml_sources = (
            repo / "lib/models/monodgp/det2d_transformer.py",
            repo / "lib/models/monodgp/det3d_transformer.py",
            repo / "lib/models/monodgp/monodgp.py",
        )
        checkpoint = Path(manifest["checkpoint"]).resolve()
        runtime_config = Path(manifest["runtime_config"]).resolve()
        val_split = args.split_dir.resolve() / "val.txt"
        dataset = args.dataset_root.resolve()
        if (
            commit != PINNED_COMMIT
            or sha256_file(attention_source) != M57_PATCHED_SOURCE_SHA256
            or any(
                M58_COREML_PATCH_MARKER
                not in path.read_text(encoding="utf-8")
                for path in coreml_sources
            )
            or sha256_file(checkpoint) != M56D_CANDIDATE_SHA256
            or sha256_file(runtime_config) != M56D_RUNTIME_CONFIG_SHA256
            or dataset != Path(manifest["dataset_root"]).resolve()
            or not val_split.is_file()
            or len(val_split.read_text(encoding="utf-8").splitlines()) != 3769
        ):
            raise RuntimeError("M58 source, checkpoint, runtime, dataset, or split changed")
        if ct.__version__ != COREMLTOOLS_VERSION:
            raise RuntimeError(
                f"Expected coremltools {COREMLTOOLS_VERSION}, found {ct.__version__}"
            )

        os.environ[PORTABLE_ENV] = "1"
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
        modules = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, attention_module.MSDeformAttn)
        }
        if tuple(modules) != EXPECTED_MODULES:
            raise RuntimeError(f"Unexpected M58 attention module set: {tuple(modules)}")
        for module in modules.values():
            module.use_portable_deform_attn = True
        if not hasattr(model, "m58_coreml_export_compat"):
            raise RuntimeError("M58 Core ML source patch marker is missing")
        model.m58_coreml_export_compat = True
        model.det2d_transformer.decoder.m58_coreml_export_compat = True
        model.det3d_transformer.decoder.m58_coreml_export_compat = True

        native_calls = 0

        class ForbiddenNative:
            @staticmethod
            def apply(*arguments, **kwargs):
                nonlocal native_calls
                native_calls += 1
                raise RuntimeError("M58 export path called the native CUDA operator")

        attention_module.MSDeformAttnFunction = ForbiddenNative

        class ExportWrapper(torch.nn.Module):
            def __init__(self, wrapped):
                super().__init__()
                self.wrapped = wrapped

            def forward(self, image, calibration, image_size):
                values = self.wrapped(
                    image, calibration, None, image_size, dn_args=0
                )
                return flatten_export_outputs(values)

        inputs, calibs, _, info = next(iter(validation_loader))
        image = inputs.float().cpu()
        calibration = calibs.float().cpu()
        image_size = info["img_size"].float().cpu()
        actual_shapes = {
            "image": list(image.shape),
            "calibration": list(calibration.shape),
            "image_size": list(image_size.shape),
        }
        if actual_shapes != INPUT_SHAPES:
            raise RuntimeError(f"Fixed M58 input signature changed: {actual_shapes}")
        sample_value = info["img_id"][0]
        sample_id = int(sample_value.item()) if hasattr(sample_value, "item") else int(sample_value)
        wrapper = ExportWrapper(model).eval()
        with torch.inference_mode():
            source_outputs = model(
                image, calibration, None, image_size, dn_args=0
            )
            reference = validate_export_outputs(source_outputs, torch)
            traced = torch.jit.trace(
                wrapper,
                (image, calibration, image_size),
                strict=False,
                check_trace=False,
            )
            traced_outputs = traced(image, calibration, image_size)
        trace_parity = compare_outputs(reference, traced_outputs)
        trace_graph = str(traced.inlined_graph)
        trace_path = output / "MonoDGP_M58_fixed_fp32.pt"
        traced.save(str(trace_path))
        reference_path = output / "m58_reference_io.npz"
        np.savez_compressed(
            reference_path,
            image=image.numpy(),
            calibration=calibration.numpy(),
            image_size=image_size.numpy(),
            **{
                name: value.detach().cpu().numpy()
                for name, value in zip(OUTPUT_NAMES, reference)
            },
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
                ct.TensorType(
                    name="calibration", shape=calibration.shape, dtype=np.float32
                ),
                ct.TensorType(
                    name="image_size", shape=image_size.shape, dtype=np.float32
                ),
            ],
            outputs=[ct.TensorType(name=name) for name in OUTPUT_NAMES],
        )
        conversion_seconds = time.perf_counter() - conversion_start
        counts = operation_counts(mlmodel._mil_program)
        package_path = output / "MonoDGP_M58_fixed_fp32.mlpackage"
        if package_path.exists():
            shutil.rmtree(package_path)
        mlmodel.save(str(package_path))
        archive_base = output / "MonoDGP_M58_fixed_fp32"
        archive_path = Path(
            shutil.make_archive(
                str(archive_base), "zip", package_path.parent, package_path.name
            )
        )
        spec_type = mlmodel.get_spec().WhichOneof("Type")
        gates = {
            "reviewed_m57_evidence": True,
            "pinned_monodgp_commit": commit == PINNED_COMMIT,
            "checkpoint_sha256": sha256_file(checkpoint) == M56D_CANDIDATE_SHA256,
            "runtime_config_sha256": (
                sha256_file(runtime_config) == M56D_RUNTIME_CONFIG_SHA256
            ),
            "portable_attention_modules_exact": tuple(modules) == EXPECTED_MODULES,
            "coreml_inplace_update_patch": all(
                M58_COREML_PATCH_MARKER
                in path.read_text(encoding="utf-8")
                for path in coreml_sources
            ),
            "portable_path_avoids_native_cuda": native_calls == 0,
            "fixed_input_signature": actual_shapes == INPUT_SHAPES,
            "all_reference_outputs_finite": all(
                bool(value.isfinite().all().item()) for value in reference
            ),
            "trace_uses_grid_sample": "aten::grid_sampler" in trace_graph,
            "trace_has_no_custom_attention": (
                "MultiScaleDeformableAttention" not in trace_graph
                and "MSDeformAttnFunction" not in trace_graph
            ),
            "trace_parity_within_frozen_limits": all(
                row["passed"] for row in trace_parity.values()
            ),
            "mlprogram_created": spec_type == "mlProgram",
            "mil_contains_resample": counts["resample"] > 0,
            "mil_has_no_custom_op": counts["custom"] == 0,
            "package_saved": package_path.is_dir() and tree_size(package_path) > 0,
            "reference_io_saved": reference_path.is_file(),
            "no_training_or_weight_change": True,
        }
        all_passed = all(gates.values())
        report.update(
            {
                "complete": True,
                "sample_id": f"{sample_id:06d}",
                "software": {
                    "python": platform.python_version(),
                    "torch": torch.__version__,
                    "coremltools": ct.__version__,
                },
                "source_runtime_config_sha256": sha256_file(runtime_config),
                "source_checkpoint": str(checkpoint),
                "m57_portable_metrics": m57_gate["portable_metrics"],
                "trace_parity": trace_parity,
                "trace_graph_uses_grid_sample": "aten::grid_sampler" in trace_graph,
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
                "all_export_gates_passed": all_passed,
                "experimental_coreml_conversion_performed": True,
                "macos_coreml_prediction_parity_authorized": all_passed,
            }
        )
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        if not all_passed:
            raise RuntimeError(f"M58 export gate failed; see {report_path}")
    except Exception as error:
        report["conversion_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
