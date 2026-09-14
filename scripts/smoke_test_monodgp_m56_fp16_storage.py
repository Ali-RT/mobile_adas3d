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


def validate_candidate_storage(
    source_state: dict,
    candidate_state: dict,
    fp16_names: set[str],
    fp32_preserved_names: set[str],
    torch_module,
) -> dict[str, Any]:
    exact_keys = set(source_state) == set(candidate_state)
    disjoint = not (fp16_names & fp32_preserved_names)
    known_policy_names = fp16_names | fp32_preserved_names
    policy_names_present = known_policy_names <= set(source_state)
    stored_fp16 = [
        name
        for name in fp16_names
        if isinstance(candidate_state.get(name), torch_module.Tensor)
        and candidate_state[name].dtype == torch_module.float16
    ]
    preserved_fp32_exact = [
        name
        for name in fp32_preserved_names
        if isinstance(source_state.get(name), torch_module.Tensor)
        and isinstance(candidate_state.get(name), torch_module.Tensor)
        and candidate_state[name].dtype == torch_module.float32
        and torch_module.equal(source_state[name], candidate_state[name])
    ]
    unchanged_names = set(source_state) - fp16_names
    unchanged_equal = exact_keys and all(
        (
            isinstance(source_state[name], torch_module.Tensor)
            and isinstance(candidate_state.get(name), torch_module.Tensor)
            and torch_module.equal(source_state[name], candidate_state[name])
        )
        or (
            not isinstance(source_state[name], torch_module.Tensor)
            and source_state[name] == candidate_state.get(name)
        )
        for name in unchanged_names
    )
    return {
        "exact_state_dict_keys": exact_keys,
        "policy_sets_disjoint": disjoint,
        "policy_names_present": policy_names_present,
        "stored_fp16_names": stored_fp16,
        "all_policy_fp16_names_stored_fp16": len(stored_fp16) == len(fp16_names),
        "preserved_fp32_exact_names": preserved_fp32_exact,
        "all_policy_fp32_names_exact": (
            len(preserved_fp32_exact) == len(fp32_preserved_names)
        ),
        "all_uncompressed_state_unchanged": unchanged_equal,
    }


def reproduce_explicit_storage_policy(model, manifest: dict, torch_module):
    policy_id = manifest.get("compression_policy", {}).get("policy_id")
    if policy_id == "m56d_det2d_transformer_fp32":
        try:
            from scripts.prepare_monodgp_m56c_group_sensitivity import (
                collect_grouped_storage_policy,
            )
        except ModuleNotFoundError:
            from prepare_monodgp_m56c_group_sensitivity import (
                collect_grouped_storage_policy,
            )
        grouped, _, _ = collect_grouped_storage_policy(model, torch_module)
        expected_fp32 = set(grouped["det2d_transformer"])
        expected_fp16 = {
            alias
            for group_name, aliases in grouped.items()
            if group_name != "det2d_transformer"
            for alias in aliases
        }
        mode = "explicit alias-consistent det2d-transformer-FP32 storage"
    elif policy_id in (None, "m56b_geometry_heads_fp32"):
        try:
            from scripts.prepare_monodgp_m56b_selective_fp16_storage import (
                collect_alias_aware_storage_policy,
            )
        except ModuleNotFoundError:
            from prepare_monodgp_m56b_selective_fp16_storage import (
                collect_alias_aware_storage_policy,
            )
        fp16, fp32, _ = collect_alias_aware_storage_policy(model, torch_module)
        expected_fp16 = set(fp16)
        expected_fp32 = set(fp32)
        mode = "explicit alias-consistent selective storage"
    else:
        raise RuntimeError(f"Unsupported explicit M56 storage policy: {policy_id}")
    return expected_fp16, expected_fp32, mode


def summarize_depth_channels(baseline: dict, candidate: dict) -> dict:
    channel_maxima = [0.0, 0.0]
    channel_means: list[list[float]] = [[], []]
    tensors = 0
    for name in sorted(set(baseline) & set(candidate)):
        if output_kind(name) != "pred_depth" or baseline[name].shape[-1] != 2:
            continue
        difference = (candidate[name].float() - baseline[name].float()).abs()
        tensors += 1
        for index in range(2):
            values = difference[..., index]
            channel_maxima[index] = max(channel_maxima[index], float(values.max().item()))
            channel_means[index].append(float(values.mean().item()))
    labels = ("depth_m", "log_variance")
    summary = {
        labels[index]: {
            "max_abs": channel_maxima[index],
            "mean_abs_across_tensors": (
                statistics.fmean(channel_means[index]) if channel_means[index] else None
            ),
        }
        for index in range(2)
    }
    summary["tensors"] = tensors
    return summary


class Logger:
    def info(self, message):
        print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run M56-family FP16-storage raw-output parity and CUDA profile smoke."
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
        raise RuntimeError("Invalid or unauthorized M56-family manifest")

    m55_profile_path = args.m55_profile.resolve()
    if (
        not m55_profile_path.is_file()
        or sha256_file(m55_profile_path) != M55_PROFILE_SHA256
        or sha256_file(m55_profile_path) != manifest.get("m55_profile_sha256")
    ):
        raise RuntimeError("Smoke test is not bound to the frozen M55 profile")
    m55_profile = json.loads(m55_profile_path.read_text(encoding="utf-8"))

    source = Path(manifest["parent_checkpoint"]).resolve()
    candidate = Path(manifest["candidate_checkpoint"]).resolve()
    runtime_config = Path(manifest["runtime_config"]).resolve()
    if sha256_file(source) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("Source checkpoint changed")
    if sha256_file(candidate) != manifest["candidate_checkpoint_sha256"]:
        raise RuntimeError("Compressed checkpoint changed")
    if sha256_file(runtime_config) != manifest["runtime_config_sha256"]:
        raise RuntimeError("Runtime config changed")

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
    explicit_fp16_names = manifest.get("fp16_parameter_state_names")
    if explicit_fp16_names is None:
        explicit_policy_reproduced = True
        policy_name_hashes_match = True
        fp16_names = set(eligible_names)
        fp32_preserved_names: set[str] = set()
        storage_policy_mode = "M56 canonical eligible parameter names"
    else:
        explicit_fp32_names = manifest.get("fp32_preserved_parameter_state_names")
        if not isinstance(explicit_fp16_names, list) or not isinstance(
            explicit_fp32_names, list
        ):
            raise RuntimeError("Explicit FP16 and FP32 parameter policies must be lists")
        fp16_names = set(explicit_fp16_names)
        fp32_preserved_names = set(explicit_fp32_names)
        expected_fp16, expected_fp32, storage_policy_mode = (
            reproduce_explicit_storage_policy(model, manifest, torch)
        )
        explicit_policy_reproduced = (
            fp16_names == expected_fp16
            and fp32_preserved_names == expected_fp32
        )
        policy_name_hashes_match = (
            manifest.get("fp16_parameter_state_names_sha256")
            == hashlib.sha256("\n".join(sorted(fp16_names)).encode("utf-8")).hexdigest()
            and manifest.get("fp32_preserved_parameter_state_names_sha256")
            == hashlib.sha256("\n".join(sorted(fp32_preserved_names)).encode("utf-8")).hexdigest()
        )
    storage_checks = validate_candidate_storage(
        source_state,
        candidate_state,
        fp16_names,
        fp32_preserved_names,
        torch,
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
    depth_channel_summary = summarize_depth_channels(baseline_flat, candidate_flat)
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
        "exact_state_dict_keys": storage_checks["exact_state_dict_keys"],
        "storage_policy_sets_disjoint": storage_checks["policy_sets_disjoint"],
        "storage_policy_names_present": storage_checks["policy_names_present"],
        "explicit_storage_policy_reproduced": explicit_policy_reproduced,
        "storage_policy_name_hashes_match": policy_name_hashes_match,
        "all_policy_fp16_parameters_stored_fp16": storage_checks[
            "all_policy_fp16_names_stored_fp16"
        ],
        "all_policy_fp32_parameters_exact": storage_checks[
            "all_policy_fp32_names_exact"
        ],
        "all_uncompressed_tensors_bitwise_unchanged": storage_checks[
            "all_uncompressed_state_unchanged"
        ],
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
        "experiment": f"{manifest.get('experiment', 'M56-family')} CUDA parity and profile smoke",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "candidate_checkpoint": str(candidate),
        "candidate_checkpoint_sha256": manifest["candidate_checkpoint_sha256"],
        "sample_id": f"{sample_id:06d}",
        "device": {"name": device_name},
        "software": software,
        "storage_policy_mode": storage_policy_mode,
        "stored_fp16_parameter_state_name_count": len(
            storage_checks["stored_fp16_names"]
        ),
        "expected_fp16_parameter_state_name_count": len(fp16_names),
        "preserved_fp32_parameter_state_name_count": len(
            storage_checks["preserved_fp32_exact_names"]
        ),
        "expected_preserved_fp32_parameter_state_name_count": len(
            fp32_preserved_names
        ),
        "runtime_precision": "FP32 after checkpoint load",
        "parity_limits": PARITY_LIMITS,
        "parity_summary": parity_summary,
        "pred_depth_channel_summary": depth_channel_summary,
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
        raise RuntimeError("FP16-storage smoke gate failed; do not run full validation")


if __name__ == "__main__":
    main()
