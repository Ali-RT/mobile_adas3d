from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PINNED_COMMIT = "aa059a18214aebf644510e7f0793971b403f9d14"
PARENT_CHECKPOINT_SHA256 = (
    "8e79f3921d96e1de70cbb4219245e3fcc3fa1fb67ae675468b4ebca90e579847"
)
REQUIRED_OUTPUTS = {
    "pred_logits",
    "pred_boxes",
    "pred_3d_dim",
    "pred_depth",
    "pred_angle",
    "pred_depth_map_logits",
    "pred_region_prob",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def tensor_summary(value: Any, prefix: str = "") -> tuple[dict[str, dict], bool]:
    import torch

    summaries: dict[str, dict] = {}
    finite = True
    if isinstance(value, torch.Tensor):
        key = prefix or "tensor"
        is_finite = bool(torch.isfinite(value).all().item())
        summaries[key] = {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "finite": is_finite,
        }
        return summaries, is_finite
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            child_summaries, child_finite = tensor_summary(child, child_prefix)
            summaries.update(child_summaries)
            finite = finite and child_finite
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            child_summaries, child_finite = tensor_summary(child, child_prefix)
            summaries.update(child_summaries)
            finite = finite and child_finite
    return summaries, finite


def parameter_inventory(model) -> tuple[dict, dict]:
    import torch

    parameters = list(model.named_parameters())
    total_parameters = sum(parameter.numel() for _, parameter in parameters)
    trainable_parameters = sum(
        parameter.numel() for _, parameter in parameters if parameter.requires_grad
    )
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for _, parameter in parameters
    )
    state_bytes = sum(
        value.numel() * value.element_size()
        for value in model.state_dict().values()
        if isinstance(value, torch.Tensor)
    )
    dtype_bytes: defaultdict[str, int] = defaultdict(int)
    component_bytes: defaultdict[str, int] = defaultdict(int)
    for name, parameter in parameters:
        size = parameter.numel() * parameter.element_size()
        dtype_bytes[str(parameter.dtype)] += size
        component_bytes[name.split(".", 1)[0]] += size

    module_counts: Counter[str] = Counter()
    custom_attention_modules = []
    eligible_parameter_ids: set[int] = set()
    eligible_bytes_by_family: defaultdict[str, int] = defaultdict(int)
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        module_counts[class_name] += 1
        normalized = class_name.lower()
        if "deformattn" in normalized or "deformableattention" in normalized:
            custom_attention_modules.append(
                {"name": name, "class_name": class_name, "module": module.__class__.__module__}
            )
        family = None
        if isinstance(module, torch.nn.Conv2d):
            family = "Conv2d"
        elif isinstance(module, torch.nn.Linear):
            family = "Linear"
        if family:
            for parameter in module.parameters(recurse=False):
                identity = id(parameter)
                if identity not in eligible_parameter_ids:
                    eligible_parameter_ids.add(identity)
                    eligible_bytes_by_family[family] += parameter.numel() * parameter.element_size()
    eligible_bytes = sum(eligible_bytes_by_family.values())
    inventory = {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "parameter_bytes": parameter_bytes,
        "state_dict_tensor_bytes": state_bytes,
        "parameter_megabytes_decimal": parameter_bytes / 1_000_000,
        "state_dict_megabytes_decimal": state_bytes / 1_000_000,
        "parameter_bytes_by_dtype": dict(sorted(dtype_bytes.items())),
        "parameter_bytes_by_top_level_component": dict(
            sorted(component_bytes.items(), key=lambda item: (-item[1], item[0]))
        ),
    }
    audit = {
        "module_type_counts": dict(sorted(module_counts.items())),
        "ordinary_weight_operator_families": ["Conv2d", "Linear"],
        "eligible_parameter_bytes_by_family": dict(sorted(eligible_bytes_by_family.items())),
        "eligible_parameter_bytes": eligible_bytes,
        "eligible_parameter_fraction": eligible_bytes / parameter_bytes if parameter_bytes else 0.0,
        "custom_deformable_attention_modules": custom_attention_modules,
    }
    return inventory, audit


class Logger:
    def info(self, message):
        print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile the untouched M54 parent for M55.")
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("M55 native baseline profiling requires a CUDA GPU")
    repo = args.monodgp_repo.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP commit {PINNED_COMMIT}, found {commit}")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not manifest.get("complete")
        or manifest.get("profile_authorized") is not True
        or manifest.get("compression_performed") is not False
        or manifest.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("Invalid M55 feasibility manifest")
    checkpoint = Path(manifest["parent_checkpoint"]).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("M55 parent checkpoint is missing or has the wrong hash")
    runtime_config = Path(manifest["runtime_config"]).resolve()
    if sha256_file(runtime_config) != manifest["runtime_config_sha256"]:
        raise RuntimeError("M55 runtime configuration hash mismatch")
    settings = manifest["profile_settings"]
    if settings != {
        "batch_size": 1,
        "warmup_runs": 5,
        "timed_runs": 100,
        "sample_policy": "first Chen-validation image",
        "latency_clock": "CUDA events with synchronization",
    }:
        raise RuntimeError("M55 profile settings changed")

    sys.path.insert(0, str(repo))
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint

    config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    config["dataset"]["batch_size"] = 1
    _, validation_loader = build_dataloader(config["dataset"], workers=0)
    model, _ = build_model(config["model"])
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    checkpoint_epoch, _, _ = load_checkpoint(model, None, checkpoint, device, Logger())
    if int(checkpoint_epoch) != 100:
        raise RuntimeError(f"M55 expected checkpoint payload epoch 100, found {checkpoint_epoch}")
    inventory, audit = parameter_inventory(model)

    inputs, calibs, _, info = next(iter(validation_loader))
    inputs = inputs.to(device)
    calibs = calibs.to(device)
    image_sizes = info["img_size"].to(device)
    sample_id_value = info["img_id"][0]
    sample_id = int(sample_id_value.item()) if hasattr(sample_id_value, "item") else int(sample_id_value)

    torch.cuda.synchronize()
    loaded_allocated = torch.cuda.memory_allocated(device)
    loaded_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(settings["warmup_runs"]):
            outputs = model(inputs, calibs, None, image_sizes, dn_args=0)
    torch.cuda.synchronize()

    timings = []
    with torch.inference_mode():
        for run_index in range(settings["timed_runs"]):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            outputs = model(inputs, calibs, None, image_sizes, dn_args=0)
            end.record()
            end.synchronize()
            elapsed = float(start.elapsed_time(end))
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError(f"Invalid CUDA latency at run {run_index}: {elapsed}")
            timings.append(elapsed)
            if (run_index + 1) % 10 == 0 or run_index + 1 == settings["timed_runs"]:
                print(
                    f"Timed predictions: {run_index + 1}/{settings['timed_runs']} "
                    f"latest={elapsed:.3f} ms",
                    flush=True,
                )

    torch.cuda.synchronize()
    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    output_tensors, finite_outputs = tensor_summary(outputs)
    missing_outputs = sorted(REQUIRED_OUTPUTS - set(outputs))
    if missing_outputs or not finite_outputs:
        raise RuntimeError(
            f"M55 parent produced missing/non-finite outputs: missing={missing_outputs} finite={finite_outputs}"
        )

    flop_profile = {
        "complete": False,
        "partial_lower_bound": True,
        "profiled_flops": None,
        "profiled_gflops": None,
        "note": "Custom deformable-attention CUDA kernels are not FLOP-accounted.",
    }
    try:
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
            with torch.inference_mode():
                model(inputs, calibs, None, image_sizes, dn_args=0)
        torch.cuda.synchronize()
        profiled_flops = int(sum(int(event.flops or 0) for event in profiler.key_averages()))
        flop_profile.update(
            {
                "complete": True,
                "profiled_flops": profiled_flops,
                "profiled_gflops": profiled_flops / 1_000_000_000,
            }
        )
    except Exception as error:
        flop_profile["error"] = f"{type(error).__name__}: {error}"

    latency_summary = {
        "clock": settings["latency_clock"],
        "warmup_runs": settings["warmup_runs"],
        "timed_runs": settings["timed_runs"],
        "mean_ms": statistics.fmean(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": percentile(timings, 0.95),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }
    memory = {
        "loaded_allocated_bytes": loaded_allocated,
        "loaded_reserved_bytes": loaded_reserved,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
    }
    properties = torch.cuda.get_device_properties(device)
    profile = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M55 untouched M54 native CUDA baseline",
        "training_performed": False,
        "compression_performed": False,
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "upstream_commit": commit,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "checkpoint_epoch": int(checkpoint_epoch),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": sha256_file(runtime_config),
        "sample_id": f"{sample_id:06d}",
        "batch_size": 1,
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "device": {
            "name": torch.cuda.get_device_name(device),
            "capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
        },
        "inputs": {
            "image_shape": list(inputs.shape),
            "image_dtype": str(inputs.dtype),
            "calibration_shape": list(calibs.shape),
            "image_size_shape": list(image_sizes.shape),
        },
        "outputs": output_tensors,
        "finite_outputs": finite_outputs,
        "parameter_inventory": inventory,
        "latency": latency_summary,
        "memory": memory,
        "compute": flop_profile,
    }
    audit.update(
        {
            "schema_version": 1,
            "complete": True,
            "experiment": "M55 MonoDGP weight-compression and export operator audit",
            "checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
            "static_graph_unchanged": True,
            "custom_cuda_extension_required": bool(audit["custom_deformable_attention_modules"]),
            "direct_coreml_export_ready": not bool(audit["custom_deformable_attention_modules"]),
            "direct_coreml_conversion_authorized": False,
            "offline_weight_compression_scope_available": audit["eligible_parameter_bytes"] > 0,
            "known_export_blockers": [
                "custom MultiScaleDeformableAttention CUDA autograd operator requires decomposition or replacement",
                "calibration matrix and image size are dynamic model inputs and require explicit parity coverage",
                "KITTI decoding and product post-processing are outside the raw network graph",
            ],
            "required_remediation": (
                "replace/decompose custom deformable attention, then prove raw tensor parity before Core ML conversion"
            ),
            "product_safety_qualified": False,
        }
    )

    profile_path = output / "m55_native_baseline_profile.json"
    latency_path = output / "m55_native_baseline_latency.csv"
    audit_path = output / "m55_operator_export_audit.json"
    profile_path.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    with latency_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run", "cuda_event_ms"])
        writer.writeheader()
        writer.writerows(
            {"run": index + 1, "cuda_event_ms": value}
            for index, value in enumerate(timings)
        )
    print(json.dumps(profile, indent=2))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
