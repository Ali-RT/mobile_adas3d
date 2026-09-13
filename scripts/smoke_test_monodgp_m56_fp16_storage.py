from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.prepare_monodgp_m55_feasibility import PARENT_CHECKPOINT_SHA256, PINNED_COMMIT
    from scripts.prepare_monodgp_m56_fp16_storage import (
        MAX_MODEL_ONLY_SIZE_RATIO,
        M55_PROFILE_SHA256,
        collect_eligible_parameter_names,
        sha256_file,
    )
except ModuleNotFoundError:
    from prepare_monodgp_m55_feasibility import PARENT_CHECKPOINT_SHA256, PINNED_COMMIT
    from prepare_monodgp_m56_fp16_storage import (
        MAX_MODEL_ONLY_SIZE_RATIO,
        M55_PROFILE_SHA256,
        collect_eligible_parameter_names,
        sha256_file,
    )


PARITY_LIMITS = {
    "pred_logits": 0.10,
    "pred_boxes": 0.01,
    "pred_3d_dim": 0.10,
    "pred_depth": 0.50,
    "pred_angle": 0.10,
    "pred_depth_map_logits": 0.10,
    "pred_region_prob": 0.01,
}


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def flatten_tensors(value: Any, torch_module, prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    if isinstance(value, torch_module.Tensor):
        flattened[prefix or "tensor"] = value
    elif isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(flatten_tensors(child, torch_module, child_prefix))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            flattened.update(flatten_tensors(child, torch_module, child_prefix))
    return flattened


def output_kind(path: str) -> str | None:
    for name in sorted(PARITY_LIMITS, key=len, reverse=True):
        if name in path:
            return name
    return None


class Logger:
    def info(self, message):
        print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M56 FP16-storage raw-output parity and CUDA profile smoke."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--m55-profile", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("M56 smoke requires a CUDA GPU")
    repo = args.monodgp_repo.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP commit {PINNED_COMMIT}, found {commit}")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("complete") is not True
        or manifest.get("smoke_authorized") is not True
        or not all(manifest.get("preparation_gate_results", {}).values())
        or manifest.get("training_performed") is not False
        or manifest.get("graph_changed") is not False
        or manifest.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("Invalid or unauthorized M56 manifest")

    m55_profile_path = args.m55_profile.resolve()
    if (
        not m55_profile_path.is_file()
        or sha256_file(m55_profile_path) != M55_PROFILE_SHA256
        or sha256_file(m55_profile_path) != manifest.get("m55_profile_sha256")
    ):
        raise RuntimeError("M56 smoke is not bound to the frozen M55 profile")
    m55_profile = json.loads(m55_profile_path.read_text(encoding="utf-8"))

    source = Path(manifest["parent_checkpoint"]).resolve()
    candidate = Path(manifest["candidate_checkpoint"]).resolve()
    runtime_config = Path(manifest["runtime_config"]).resolve()
    if sha256_file(source) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("M56 source checkpoint changed")
    if sha256_file(candidate) != manifest["candidate_checkpoint_sha256"]:
        raise RuntimeError("M56 compressed checkpoint changed")
    if sha256_file(runtime_config) != manifest["runtime_config_sha256"]:
        raise RuntimeError("M56 runtime config changed")

    sys.path.insert(0, str(repo))
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint, load_checkpoint_safely

    config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    config["dataset"]["batch_size"] = 1
    _, validation_loader = build_dataloader(config["dataset"], workers=0)
    model, _ = build_model(config["model"])
    device = torch.device("cuda:0")
    model = model.to(device).eval()
    eligible_names, _ = collect_eligible_parameter_names(model, torch)

    candidate_payload = load_checkpoint_safely(candidate, torch.device("cpu"))
    candidate_state = candidate_payload.get("model_state", {})
    source_payload = load_checkpoint_safely(source, torch.device("cpu"))
    source_state = source_payload.get("model_state", {})
    stored_fp16 = [
        name for name in eligible_names
        if isinstance(candidate_state.get(name), torch.Tensor)
        and candidate_state[name].dtype == torch.float16
    ]
    noneligible_equal = all(
        name in candidate_state
        and isinstance(value, torch.Tensor)
        and torch.equal(value, candidate_state[name])
        for name, value in source_state.items()
        if isinstance(value, torch.Tensor) and name not in eligible_names
    )

    inputs, calibs, _, info = next(iter(validation_loader))
    inputs = inputs.to(device)
    calibs = calibs.to(device)
    image_sizes = info["img_size"].to(device)
    sample_value = info["img_id"][0]
    sample_id = int(sample_value.item()) if hasattr(sample_value, "item") else int(sample_value)

    baseline_epoch, _, _ = load_checkpoint(model, None, source, device, Logger())
    if int(baseline_epoch) != 100:
        raise RuntimeError("M56 source payload epoch changed")
    with torch.inference_mode():
        baseline_outputs = model(inputs, calibs, None, image_sizes, dn_args=0)
    baseline_flat = {
        name: tensor.detach().cpu()
        for name, tensor in flatten_tensors(baseline_outputs, torch).items()
    }

    candidate_epoch, _, _ = load_checkpoint(model, None, candidate, device, Logger())
    runtime_fp32 = all(
        not parameter.is_floating_point() or parameter.dtype == torch.float32
        for parameter in model.parameters()
    )
    with torch.inference_mode():
        candidate_outputs = model(inputs, calibs, None, image_sizes, dn_args=0)
    candidate_flat = {
        name: tensor.detach().cpu()
        for name, tensor in flatten_tensors(candidate_outputs, torch).items()
    }

    same_structure = set(baseline_flat) == set(candidate_flat) and all(
        baseline_flat[name].shape == candidate_flat[name].shape
        for name in set(baseline_flat) & set(candidate_flat)
    )
    finite_outputs = bool(candidate_flat) and all(
        torch.isfinite(value).all().item() for value in candidate_flat.values()
    )
    parity_by_tensor: dict[str, dict] = {}
    maxima_by_kind = {name: 0.0 for name in PARITY_LIMITS}
    means_by_kind: dict[str, list[float]] = {name: [] for name in PARITY_LIMITS}
    if same_structure:
        for name in sorted(baseline_flat):
            difference = (candidate_flat[name].float() - baseline_flat[name].float()).abs()
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
                maxima_by_kind[kind] = max(maxima_by_kind[kind], maximum)
                means_by_kind[kind].append(mean)
    parity_summary = {
        name: {
            "limit_max_abs": PARITY_LIMITS[name],
            "max_abs": maxima_by_kind[name],
            "mean_abs_across_tensors": (
                statistics.fmean(means_by_kind[name]) if means_by_kind[name] else None
            ),
            "passed": bool(means_by_kind[name])
            and math.isfinite(maxima_by_kind[name])
            and maxima_by_kind[name] <= PARITY_LIMITS[name],
        }
        for name in PARITY_LIMITS
    }
    raw_output_parity = same_structure and all(row["passed"] for row in parity_summary.values())

    torch.cuda.synchronize()
    loaded_allocated = torch.cuda.memory_allocated(device)
    loaded_reserved = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        for _ in range(5):
            candidate_outputs = model(inputs, calibs, None, image_sizes, dn_args=0)
    torch.cuda.synchronize()
    timings: list[float] = []
    with torch.inference_mode():
        for run_index in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            candidate_outputs = model(inputs, calibs, None, image_sizes, dn_args=0)
            end.record()
            end.synchronize()
            elapsed = float(start.elapsed_time(end))
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise RuntimeError(f"Invalid M56 CUDA latency at run {run_index}: {elapsed}")
            timings.append(elapsed)
            if (run_index + 1) % 10 == 0:
                print(f"Timed predictions: {run_index + 1}/100 latest={elapsed:.3f} ms", flush=True)

    latency = {
        "clock": "CUDA events with synchronization",
        "warmup_runs": 5,
        "timed_runs": 100,
        "mean_ms": statistics.fmean(timings),
        "median_ms": statistics.median(timings),
        "p95_ms": percentile(timings, 0.95),
        "min_ms": min(timings),
        "max_ms": max(timings),
    }
    memory = {
        "loaded_allocated_bytes": torch.cuda.memory_allocated(device),
        "loaded_reserved_bytes": torch.cuda.memory_reserved(device),
        "profile_start_allocated_bytes": loaded_allocated,
        "profile_start_reserved_bytes": loaded_reserved,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    software = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }
    device_name = torch.cuda.get_device_name(device)
    runtime_comparable = (
        device_name == m55_profile.get("device", {}).get("name")
        and software["torch"] == m55_profile.get("software", {}).get("torch")
        and software["cuda_runtime"] == m55_profile.get("software", {}).get("cuda_runtime")
        and f"{sample_id:06d}" == m55_profile.get("sample_id")
    )
    baseline_latency = m55_profile["latency"]
    latency_comparison = {
        "comparable_environment": runtime_comparable,
        "baseline_mean_ms": baseline_latency["mean_ms"],
        "candidate_mean_ms": latency["mean_ms"],
        "mean_ratio": latency["mean_ms"] / baseline_latency["mean_ms"],
        "baseline_p95_ms": baseline_latency["p95_ms"],
        "candidate_p95_ms": latency["p95_ms"],
        "p95_ratio": latency["p95_ms"] / baseline_latency["p95_ms"],
        "claim": (
            "same-environment model-only comparison"
            if runtime_comparable
            else "informational only; M55 device/software/sample did not all match"
        ),
    }

    gates = {
        "manifest_authorized": True,
        "source_checkpoint_sha256": sha256_file(source) == PARENT_CHECKPOINT_SHA256,
        "candidate_checkpoint_sha256": (
            sha256_file(candidate) == manifest["candidate_checkpoint_sha256"]
        ),
        "candidate_payload_epoch": int(candidate_epoch) == 100,
        "all_eligible_parameters_stored_fp16": len(stored_fp16) == len(eligible_names),
        "noneligible_tensors_bitwise_unchanged": noneligible_equal,
        "candidate_runtime_parameters_fp32": runtime_fp32,
        "output_structure_unchanged": same_structure,
        "finite_candidate_outputs": finite_outputs,
        "raw_output_parity_within_limits": raw_output_parity,
        "model_only_size_ratio_le_0_60": (
            float(manifest["model_only_checkpoint_size_ratio"])
            <= MAX_MODEL_ONLY_SIZE_RATIO
        ),
        "five_warmups_100_predictions": len(timings) == 100,
        "no_training_or_graph_change": (
            manifest["training_performed"] is False
            and manifest["graph_changed"] is False
        ),
        "coreml_and_product_claims_remain_false": (
            manifest["direct_coreml_conversion_authorized"] is False
            and manifest["product_safety_qualified"] is False
        ),
    }
    report = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M56 FP16 parameter-storage CUDA parity and profile smoke",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "candidate_checkpoint": str(candidate),
        "candidate_checkpoint_sha256": manifest["candidate_checkpoint_sha256"],
        "sample_id": f"{sample_id:06d}",
        "device": {"name": device_name},
        "software": software,
        "stored_eligible_parameter_count": len(stored_fp16),
        "expected_eligible_parameter_count": len(eligible_names),
        "runtime_precision": "FP32 after checkpoint load",
        "parity_limits": PARITY_LIMITS,
        "parity_summary": parity_summary,
        "parity_by_tensor": parity_by_tensor,
        "latency": latency,
        "memory": memory,
        "m55_latency_comparison": latency_comparison,
        "gate_results": gates,
        "all_smoke_gates_passed": all(gates.values()),
        "full_evaluation_authorized": all(gates.values()),
        "training_performed": False,
        "direct_coreml_conversion_authorized": False,
        "product_safety_qualified": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_smoke_gates_passed"]:
        raise RuntimeError("M56 smoke gate failed; do not run full validation")


if __name__ == "__main__":
    main()
