from __future__ import annotations

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.patch_monodgp_m57_deformable_attention import (
        MODE_ENVIRONMENT_VARIABLE,
    )
    from scripts.prepare_monodgp_m56_fp16_storage import sha256_file
    from scripts.prepare_monodgp_m57_deformable_attention import (
        EXPECTED_MODULES,
        M56D_CANDIDATE_SHA256,
        M57_PATCHED_SOURCE_SHA256,
        M57_POLICY_ID,
        PINNED_COMMIT,
    )
    from scripts.smoke_test_monodgp_m56_fp16_storage import (
        PARITY_LIMITS,
        flatten_tensors,
        output_kind,
        percentile,
        summarize_depth_channels,
    )
except ModuleNotFoundError:
    from patch_monodgp_m57_deformable_attention import (
        MODE_ENVIRONMENT_VARIABLE,
    )
    from prepare_monodgp_m56_fp16_storage import sha256_file
    from prepare_monodgp_m57_deformable_attention import (
        EXPECTED_MODULES,
        M56D_CANDIDATE_SHA256,
        M57_PATCHED_SOURCE_SHA256,
        M57_POLICY_ID,
        PINNED_COMMIT,
    )
    from smoke_test_monodgp_m56_fp16_storage import (
        PARITY_LIMITS,
        flatten_tensors,
        output_kind,
        percentile,
        summarize_depth_channels,
    )


MODULE_OUTPUT_MAX_ABS_LIMIT = 1e-3
EXPECTED_TRACE_SIGNATURES = {
    "q10200_ref2_standard",
    "q50_ref2_standard",
    "q50_ref6_standard",
}


class Logger:
    def info(self, message):
        print(message, flush=True)


def clone_argument(value, torch_module):
    if isinstance(value, torch_module.Tensor):
        return value.detach().clone()
    if value is None:
        return None
    raise TypeError(f"Unexpected attention argument type: {type(value)}")


def compare_flat_outputs(baseline: dict, candidate: dict, torch_module) -> dict:
    same_structure = set(baseline) == set(candidate) and all(
        baseline[name].shape == candidate[name].shape
        for name in set(baseline) & set(candidate)
    )
    parity_by_tensor: dict[str, dict[str, Any]] = {}
    maxima = {name: 0.0 for name in PARITY_LIMITS}
    means: dict[str, list[float]] = {name: [] for name in PARITY_LIMITS}
    if same_structure:
        for name in sorted(baseline):
            difference = (candidate[name].float() - baseline[name].float()).abs()
            maximum = float(difference.max().item()) if difference.numel() else 0.0
            mean = float(difference.mean().item()) if difference.numel() else 0.0
            kind = output_kind(name)
            parity_by_tensor[name] = {
                "shape": list(difference.shape),
                "kind": kind,
                "max_abs": maximum,
                "mean_abs": mean,
            }
            if kind is not None:
                maxima[kind] = max(maxima[kind], maximum)
                means[kind].append(mean)
    summary = {
        name: {
            "limit_max_abs": PARITY_LIMITS[name],
            "max_abs": maxima[name],
            "mean_abs_across_tensors": (
                statistics.fmean(means[name]) if means[name] else None
            ),
            "passed": bool(means[name])
            and math.isfinite(maxima[name])
            and maxima[name] <= PARITY_LIMITS[name],
        }
        for name in PARITY_LIMITS
    }
    return {
        "same_structure": same_structure,
        "parity_by_tensor": parity_by_tensor,
        "parity_summary": summary,
        "all_within_limits": same_structure and all(row["passed"] for row in summary.values()),
    }


def set_portable_mode(modules: dict, enabled: bool) -> None:
    for module in modules.values():
        module.use_portable_deform_attn = enabled


def profile_model(
    model,
    inputs,
    calibs,
    image_sizes,
    modules: dict,
    portable: bool,
    attention_module,
    native_function,
    torch_module,
) -> tuple[dict, dict, int]:
    forbidden_calls = 0

    class ForbiddenNative:
        @staticmethod
        def apply(*args, **kwargs):
            nonlocal forbidden_calls
            forbidden_calls += 1
            raise RuntimeError("M57 portable path called the native CUDA operator")

    set_portable_mode(modules, portable)
    attention_module.MSDeformAttnFunction = (
        ForbiddenNative if portable else native_function
    )
    device = inputs.device
    torch_module.cuda.synchronize(device)
    torch_module.cuda.reset_peak_memory_stats(device)
    with torch_module.inference_mode():
        for _ in range(5):
            model(inputs, calibs, None, image_sizes, dn_args=0)
    torch_module.cuda.synchronize(device)
    timings: list[float] = []
    with torch_module.inference_mode():
        for index in range(100):
            start = torch_module.cuda.Event(enable_timing=True)
            end = torch_module.cuda.Event(enable_timing=True)
            start.record()
            model(inputs, calibs, None, image_sizes, dn_args=0)
            end.record()
            end.synchronize()
            elapsed = float(start.elapsed_time(end))
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError(f"Invalid M57 CUDA latency at run {index}: {elapsed}")
            timings.append(elapsed)
            if (index + 1) % 10 == 0:
                label = "portable" if portable else "native"
                print(
                    f"{label} predictions: {index + 1}/100 latest={elapsed:.3f} ms",
                    flush=True,
                )
    latency = {
        "clock": "CUDA events with synchronization",
        "warmup_runs": 5,
        "timed_runs": len(timings),
        "mean_ms": statistics.fmean(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": percentile(timings, 0.95),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }
    memory = {
        "peak_allocated_bytes": torch_module.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch_module.cuda.max_memory_reserved(device),
    }
    return latency, memory, forbidden_calls


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare native and portable MonoDGP deformable attention on CUDA."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("M57 smoke requires a CUDA GPU for the native reference")
    repo = args.monodgp_repo.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP {PINNED_COMMIT}, found {commit}")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("complete") is not True
        or manifest.get("smoke_authorized") is not True
        or not manifest.get("preparation_gate_results")
        or not all(manifest["preparation_gate_results"].values())
        or manifest.get("replacement_policy", {}).get("policy_id") != M57_POLICY_ID
        or manifest.get("checkpoint_sha256") != M56D_CANDIDATE_SHA256
        or manifest.get("training_performed") is not False
        or manifest.get("weights_changed") is not False
        or manifest.get("direct_coreml_conversion_authorized") is not False
    ):
        raise RuntimeError("Invalid or unauthorized M57 manifest")
    checkpoint = Path(manifest["checkpoint"]).resolve()
    runtime_config = Path(manifest["runtime_config"]).resolve()
    attention_source = Path(manifest["patched_source"]).resolve()
    if sha256_file(checkpoint) != M56D_CANDIDATE_SHA256:
        raise RuntimeError("M57 checkpoint changed")
    if sha256_file(runtime_config) != manifest["runtime_config_sha256"]:
        raise RuntimeError("M57 runtime config changed")
    if sha256_file(attention_source) != M57_PATCHED_SOURCE_SHA256:
        raise RuntimeError("M57 patched attention source changed")

    os.environ.pop(MODE_ENVIRONMENT_VARIABLE, None)
    sys.path.insert(0, str(repo))
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint
    import lib.models.monodgp.ops.modules.ms_deform_attn as attention_module

    config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    config["dataset"]["batch_size"] = 1
    _, validation_loader = build_dataloader(config["dataset"], workers=0)
    model, _ = build_model(config["model"])
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    epoch, _, _ = load_checkpoint(model, None, checkpoint, device, Logger())
    if int(epoch) != 100:
        raise RuntimeError("M57 checkpoint epoch changed")
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, attention_module.MSDeformAttn)
    }
    if tuple(modules) != EXPECTED_MODULES:
        raise RuntimeError(f"Unexpected M57 module set: {tuple(modules)}")
    if any(module.use_portable_deform_attn for module in modules.values()):
        raise RuntimeError("M57 native reference did not start in native mode")

    inputs, calibs, _, info = next(iter(validation_loader))
    inputs = inputs.to(device)
    calibs = calibs.to(device)
    image_sizes = info["img_size"].to(device)
    sample_value = info["img_id"][0]
    sample_id = int(sample_value.item()) if hasattr(sample_value, "item") else int(sample_value)

    captured: dict[str, dict[str, Any]] = {}
    handles = []
    for name, module in modules.items():
        def capture(current_module, arguments, output, module_name=name):
            if module_name in captured:
                raise RuntimeError(f"M57 attention module called twice: {module_name}")
            captured[module_name] = {
                "arguments": tuple(clone_argument(value, torch) for value in arguments),
                "output": output.detach().clone(),
            }
        handles.append(module.register_forward_hook(capture))

    native_function = attention_module.MSDeformAttnFunction
    native_calls = 0

    class CountingNative:
        @staticmethod
        def apply(*arguments, **kwargs):
            nonlocal native_calls
            native_calls += 1
            return native_function.apply(*arguments, **kwargs)

    attention_module.MSDeformAttnFunction = CountingNative
    try:
        with torch.inference_mode():
            native_outputs = model(inputs, calibs, None, image_sizes, dn_args=0)
    finally:
        for handle in handles:
            handle.remove()
        attention_module.MSDeformAttnFunction = native_function
    if tuple(captured) != EXPECTED_MODULES or native_calls != len(EXPECTED_MODULES):
        raise RuntimeError(
            f"M57 native coverage changed: modules={tuple(captured)}, calls={native_calls}"
        )
    native_flat = {
        name: tensor.detach().cpu()
        for name, tensor in flatten_tensors(native_outputs, torch).items()
    }

    forbidden_calls = 0

    class ForbiddenNative:
        @staticmethod
        def apply(*arguments, **kwargs):
            nonlocal forbidden_calls
            forbidden_calls += 1
            raise RuntimeError("M57 portable path called the native CUDA operator")

    attention_module.MSDeformAttnFunction = ForbiddenNative
    module_parity: dict[str, dict[str, Any]] = {}
    trace_representatives: dict[str, tuple[Any, tuple]] = {}
    try:
        with torch.inference_mode():
            for name, module in modules.items():
                module.use_portable_deform_attn = True
                arguments = captured[name]["arguments"]
                portable_output = module(*arguments)
                difference = (portable_output.float() - captured[name]["output"].float()).abs()
                maximum = float(difference.max().item())
                mean = float(difference.mean().item())
                reference_dim = int(arguments[1].shape[-1])
                query_count = int(arguments[0].shape[1])
                conditional = "conditional" if module.conditional else "standard"
                signature = f"q{query_count}_ref{reference_dim}_{conditional}"
                module_parity[name] = {
                    "signature": signature,
                    "shape": list(portable_output.shape),
                    "max_abs": maximum,
                    "mean_abs": mean,
                    "limit_max_abs": MODULE_OUTPUT_MAX_ABS_LIMIT,
                    "passed": math.isfinite(maximum)
                    and maximum <= MODULE_OUTPUT_MAX_ABS_LIMIT,
                }
                trace_representatives.setdefault(signature, (module, arguments))

        trace_results: dict[str, dict[str, Any]] = {}
        for signature, (module, arguments) in trace_representatives.items():
            with torch.inference_mode():
                traced = torch.jit.trace(
                    module, arguments, strict=False, check_trace=False
                )
            graph = str(traced.inlined_graph)
            trace_results[signature] = {
                "has_grid_sampler": "aten::grid_sampler" in graph,
                "has_custom_extension": "MultiScaleDeformableAttention" in graph,
                "has_native_autograd_function": "MSDeformAttnFunction" in graph,
                "passed": (
                    "aten::grid_sampler" in graph
                    and "MultiScaleDeformableAttention" not in graph
                    and "MSDeformAttnFunction" not in graph
                ),
            }

        set_portable_mode(modules, True)
        with torch.inference_mode():
            portable_outputs = model(inputs, calibs, None, image_sizes, dn_args=0)
    finally:
        attention_module.MSDeformAttnFunction = native_function
    portable_flat = {
        name: tensor.detach().cpu()
        for name, tensor in flatten_tensors(portable_outputs, torch).items()
    }
    output_parity = compare_flat_outputs(native_flat, portable_flat, torch)
    depth_channels = summarize_depth_channels(native_flat, portable_flat)
    finite_outputs = bool(portable_flat) and all(
        torch.isfinite(tensor).all().item() for tensor in portable_flat.values()
    )
    runtime_fp32 = all(
        not parameter.is_floating_point() or parameter.dtype == torch.float32
        for parameter in model.parameters()
    )

    native_latency, native_memory, native_profile_forbidden = profile_model(
        model, inputs, calibs, image_sizes, modules, False,
        attention_module, native_function, torch,
    )
    portable_latency, portable_memory, portable_profile_forbidden = profile_model(
        model, inputs, calibs, image_sizes, modules, True,
        attention_module, native_function, torch,
    )
    attention_module.MSDeformAttnFunction = native_function
    runtime_comparison = {
        "comparable_environment": True,
        "native_mean_ms": native_latency["mean_ms"],
        "portable_mean_ms": portable_latency["mean_ms"],
        "mean_ratio": portable_latency["mean_ms"] / native_latency["mean_ms"],
        "native_p95_ms": native_latency["p95_ms"],
        "portable_p95_ms": portable_latency["p95_ms"],
        "p95_ratio": portable_latency["p95_ms"] / native_latency["p95_ms"],
        "claim": "same model, sample, device, software, and timing method",
    }

    gates = {
        "manifest_authorized": True,
        "checkpoint_sha256": sha256_file(checkpoint) == M56D_CANDIDATE_SHA256,
        "patched_source_sha256": sha256_file(attention_source) == M57_PATCHED_SOURCE_SHA256,
        "nine_native_modules_observed": tuple(captured) == EXPECTED_MODULES,
        "nine_native_cuda_calls_observed": native_calls == len(EXPECTED_MODULES),
        "all_module_outputs_within_limit": all(row["passed"] for row in module_parity.values()),
        "all_observed_signatures_traced": set(trace_results) == EXPECTED_TRACE_SIGNATURES,
        "traces_use_grid_sample_without_custom_op": all(row["passed"] for row in trace_results.values()),
        "portable_full_model_avoids_native_cuda": forbidden_calls == 0,
        "output_structure_unchanged": output_parity["same_structure"],
        "finite_portable_outputs": finite_outputs,
        "raw_output_parity_within_frozen_limits": output_parity["all_within_limits"],
        "runtime_parameters_fp32": runtime_fp32,
        "native_profile_complete": native_latency["timed_runs"] == 100 and native_profile_forbidden == 0,
        "portable_profile_complete": portable_latency["timed_runs"] == 100 and portable_profile_forbidden == 0,
        "no_training_or_weight_change": (
            manifest["training_performed"] is False
            and manifest["weights_changed"] is False
        ),
        "coreml_and_product_claims_remain_false": (
            manifest["direct_coreml_conversion_authorized"] is False
            and manifest["product_safety_qualified"] is False
        ),
    }
    all_passed = all(gates.values())
    report = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M57 native-vs-portable deformable-attention CUDA parity smoke",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": M56D_CANDIDATE_SHA256,
        "sample_id": f"{sample_id:06d}",
        "device": {"name": torch.cuda.get_device_name(device)},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "target_modules": list(EXPECTED_MODULES),
        "native_cuda_calls": native_calls,
        "portable_native_cuda_calls": forbidden_calls,
        "module_output_max_abs_limit": MODULE_OUTPUT_MAX_ABS_LIMIT,
        "module_parity": module_parity,
        "trace_signatures": trace_results,
        "raw_output_parity_limits": PARITY_LIMITS,
        "raw_output_parity": output_parity["parity_summary"],
        "raw_output_parity_by_tensor": output_parity["parity_by_tensor"],
        "pred_depth_channel_summary": depth_channels,
        "native_latency": native_latency,
        "portable_latency": portable_latency,
        "native_memory": native_memory,
        "portable_memory": portable_memory,
        "runtime_comparison": runtime_comparison,
        "gate_results": gates,
        "all_smoke_gates_passed": all_passed,
        "full_evaluation_authorized": all_passed,
        "coreml_microkernel_conversion_authorized": all_passed,
        "direct_coreml_conversion_authorized": False,
        "product_safety_qualified": False,
        "training_performed": False,
    }
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not all_passed:
        raise RuntimeError(f"M57 smoke gate failed; see {output_path}")


if __name__ == "__main__":
    main()
