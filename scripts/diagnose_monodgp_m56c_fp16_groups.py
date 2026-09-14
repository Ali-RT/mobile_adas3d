from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import platform
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

try:
    from scripts.prepare_monodgp_m55_feasibility import (
        PARENT_CHECKPOINT_SHA256,
        PINNED_COMMIT,
    )
    from scripts.prepare_monodgp_m56_fp16_storage import sha256_file
    from scripts.prepare_monodgp_m56c_group_sensitivity import (
        GROUP_ORDER,
        M56B_FAILED_DEPTH_MAX_ABS,
        M56B_MANIFEST_SHA256,
        M56B_SMOKE_SHA256,
        build_candidate_specs,
        candidate_fp16_names,
        collect_grouped_storage_policy,
        names_sha256,
    )
    from scripts.smoke_test_monodgp_m56_fp16_storage import (
        PARITY_LIMITS,
        flatten_tensors,
        output_kind,
        summarize_depth_channels,
        validate_candidate_storage,
    )
except ModuleNotFoundError:
    from prepare_monodgp_m55_feasibility import (
        PARENT_CHECKPOINT_SHA256,
        PINNED_COMMIT,
    )
    from prepare_monodgp_m56_fp16_storage import sha256_file
    from prepare_monodgp_m56c_group_sensitivity import (
        GROUP_ORDER,
        M56B_FAILED_DEPTH_MAX_ABS,
        M56B_MANIFEST_SHA256,
        M56B_SMOKE_SHA256,
        build_candidate_specs,
        candidate_fp16_names,
        collect_grouped_storage_policy,
        names_sha256,
    )
    from smoke_test_monodgp_m56_fp16_storage import (
        PARITY_LIMITS,
        flatten_tensors,
        output_kind,
        summarize_depth_channels,
        validate_candidate_storage,
    )


REFERENCE_DEPTH_TOLERANCE = 0.01


def rounded_state_dict(
    source_state: dict[str, Any], fp16_names: set[str], torch_module
) -> OrderedDict:
    missing = sorted(fp16_names - set(source_state))
    if missing:
        raise RuntimeError(f"M56c FP16 aliases missing from source state: {missing[:10]}")
    candidate = OrderedDict()
    for name, value in source_state.items():
        if not isinstance(value, torch_module.Tensor):
            candidate[name] = value
            continue
        source = value.detach().cpu()
        candidate[name] = (
            source.to(dtype=torch_module.float16)
            if name in fp16_names
            else source.clone()
        )
    metadata = getattr(source_state, "_metadata", None)
    if metadata is not None:
        candidate._metadata = metadata
    return candidate


def compare_outputs(baseline: dict, candidate: dict, torch_module) -> dict:
    same_structure = set(baseline) == set(candidate) and all(
        baseline[name].shape == candidate[name].shape
        for name in set(baseline) & set(candidate)
    )
    finite = bool(candidate) and all(
        torch_module.isfinite(value).all().item() for value in candidate.values()
    )
    maxima = {name: 0.0 for name in PARITY_LIMITS}
    sums = {name: 0.0 for name in PARITY_LIMITS}
    counts = {name: 0 for name in PARITY_LIMITS}
    by_tensor = {}
    if same_structure:
        for name in sorted(baseline):
            difference = (candidate[name].float() - baseline[name].float()).abs()
            maximum = float(difference.max().item()) if difference.numel() else 0.0
            mean = float(difference.mean().item()) if difference.numel() else 0.0
            kind = output_kind(name)
            by_tensor[name] = {
                "shape": list(difference.shape),
                "kind": kind,
                "max_abs": maximum,
                "mean_abs": mean,
            }
            if kind is not None:
                maxima[kind] = max(maxima[kind], maximum)
                sums[kind] += mean
                counts[kind] += 1
    summary = {
        name: {
            "limit_max_abs": limit,
            "max_abs": maxima[name],
            "mean_abs_across_tensors": (
                sums[name] / counts[name] if counts[name] else None
            ),
            "passed": bool(counts[name])
            and math.isfinite(maxima[name])
            and maxima[name] <= limit,
        }
        for name, limit in PARITY_LIMITS.items()
    }
    return {
        "output_structure_unchanged": same_structure,
        "finite_outputs": finite,
        "all_parity_limits_passed": same_structure
        and finite
        and all(row["passed"] for row in summary.values()),
        "failed_output_families": sorted(
            name for name, row in summary.items() if not row["passed"]
        ),
        "parity_summary": summary,
        "pred_depth_channel_summary": summarize_depth_channels(
            baseline, candidate
        ),
        "parity_by_tensor": by_tensor,
    }


def write_csv(path: Path, results: list[dict]) -> None:
    fieldnames = [
        "candidate",
        "mode",
        "fp32_held_group",
        "fp16_groups",
        "fp16_unique_parameter_bytes",
        "projected_parameter_size_ratio",
        "all_parity_limits_passed",
        "failed_output_families",
        "pred_depth_max_abs",
        "depth_m_max_abs",
        *[f"{name}_max_abs" for name in PARITY_LIMITS if name != "pred_depth"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            spec = result["spec"]
            parity = result["parity"]
            row = {
                "candidate": spec["name"],
                "mode": spec["mode"],
                "fp32_held_group": spec.get("fp32_held_group", ""),
                "fp16_groups": ";".join(spec["fp16_groups"]),
                "fp16_unique_parameter_bytes": spec[
                    "fp16_unique_parameter_bytes"
                ],
                "projected_parameter_size_ratio": spec[
                    "projected_parameter_size_ratio"
                ],
                "all_parity_limits_passed": parity["all_parity_limits_passed"],
                "failed_output_families": ";".join(
                    parity["failed_output_families"]
                ),
                "pred_depth_max_abs": parity["parity_summary"]["pred_depth"][
                    "max_abs"
                ],
                "depth_m_max_abs": parity["pred_depth_channel_summary"]["depth_m"][
                    "max_abs"
                ],
            }
            for name in PARITY_LIMITS:
                if name != "pred_depth":
                    row[f"{name}_max_abs"] = parity["parity_summary"][name][
                        "max_abs"
                    ]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the M56c grouped FP16-storage CUDA sensitivity matrix."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("M56c grouped sensitivity requires a CUDA GPU")
    repo = args.monodgp_repo.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP commit {PINNED_COMMIT}, found {commit}")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("complete") is not True
        or manifest.get("experiment")
        != "M56c grouped FP16-storage sensitivity isolation"
        or manifest.get("diagnostic_authorized") is not True
        or not all(manifest.get("preparation_gate_results", {}).values())
        or manifest.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or manifest.get("m56b_manifest_sha256") != M56B_MANIFEST_SHA256
        or manifest.get("m56b_smoke_sha256") != M56B_SMOKE_SHA256
        or manifest.get("training_performed") is not False
        or manifest.get("graph_changed") is not False
        or manifest.get("full_evaluation_authorized") is not False
    ):
        raise RuntimeError("Invalid or unauthorized M56c manifest")

    source = Path(manifest["parent_checkpoint"]).resolve()
    runtime_config = Path(manifest["runtime_config"]).resolve()
    if sha256_file(source) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("M56c source checkpoint changed")
    if sha256_file(runtime_config) != manifest["runtime_config_sha256"]:
        raise RuntimeError("M56c runtime config changed")

    sys.path.insert(0, str(repo))
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint_safely

    config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    config["dataset"]["batch_size"] = 1
    _, validation_loader = build_dataloader(config["dataset"], workers=0)
    model, _ = build_model(config["model"])
    source_payload = load_checkpoint_safely(source, torch.device("cpu"))
    if int(source_payload.get("epoch", -1)) != 100:
        raise RuntimeError("M56c source payload epoch changed")
    source_state = source_payload.get("model_state")
    if not isinstance(source_state, dict):
        raise RuntimeError("M56c source checkpoint has no model_state")
    model.load_state_dict(source_state, strict=True)

    grouped, m56b_fp32_aliases, inventory = collect_grouped_storage_policy(
        model, torch
    )
    expected_specs = build_candidate_specs(
        grouped,
        m56b_fp32_aliases,
        inventory,
        int(manifest["total_parameter_bytes"]),
    )
    policy_reproduced = all(
        manifest["group_policy"][name]["parameter_state_names"]
        == sorted(grouped[name])
        and manifest["group_policy"][name]["parameter_state_names_sha256"]
        == names_sha256(set(grouped[name]))
        for name in GROUP_ORDER
    )
    candidate_specs_reproduced = expected_specs == manifest["candidate_specs"]

    inputs, calibs, _, info = next(iter(validation_loader))
    sample_value = info["img_id"][0]
    sample_id = int(sample_value.item()) if hasattr(sample_value, "item") else int(sample_value)
    if f"{sample_id:06d}" != manifest["sample_id"]:
        raise RuntimeError(
            f"M56c expected sample {manifest['sample_id']}, got {sample_id:06d}"
        )
    device = torch.device("cuda:0")
    inputs = inputs.to(device)
    calibs = calibs.to(device)
    image_sizes = info["img_size"].to(device)
    model = model.to(device).eval()
    model.load_state_dict(source_state, strict=True)
    with torch.inference_mode():
        baseline_outputs = model(inputs, calibs, None, image_sizes, dn_args=0)
    baseline_flat = {
        name: tensor.detach().cpu()
        for name, tensor in flatten_tensors(baseline_outputs, torch).items()
    }

    all_eligible_names = {
        alias for aliases in grouped.values() for alias in aliases
    }
    results = []
    structural_checks = []
    for index, spec in enumerate(expected_specs, start=1):
        fp16_names = candidate_fp16_names(spec, grouped, m56b_fp32_aliases)
        fp32_names = all_eligible_names - fp16_names
        candidate_state = rounded_state_dict(source_state, fp16_names, torch)
        storage = validate_candidate_storage(
            source_state, candidate_state, fp16_names, fp32_names, torch
        )
        model.load_state_dict(candidate_state, strict=True)
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
        parity = compare_outputs(baseline_flat, candidate_flat, torch)
        integrity = {
            "exact_state_dict_keys": storage["exact_state_dict_keys"],
            "policy_sets_disjoint": storage["policy_sets_disjoint"],
            "policy_names_present": storage["policy_names_present"],
            "all_fp16_names_stored_fp16": storage[
                "all_policy_fp16_names_stored_fp16"
            ],
            "all_fp32_names_exact": storage["all_policy_fp32_names_exact"],
            "all_uncompressed_state_unchanged": storage[
                "all_uncompressed_state_unchanged"
            ],
            "runtime_parameters_fp32": runtime_fp32,
            "output_structure_unchanged": parity["output_structure_unchanged"],
            "finite_outputs": parity["finite_outputs"],
        }
        structural_checks.append(all(integrity.values()))
        result = {"spec": spec, "integrity": integrity, "parity": parity}
        results.append(result)
        depth = parity["parity_summary"]["pred_depth"]["max_abs"]
        status = "PASS" if parity["all_parity_limits_passed"] else "FAIL"
        print(
            f"M56c candidate {index:02d}/{len(expected_specs):02d} "
            f"{spec['name']}: {status} pred_depth_max_abs={depth:.6f}",
            flush=True,
        )
        del candidate_state, candidate_outputs, candidate_flat
        gc.collect()

    by_name = {result["spec"]["name"]: result for result in results}
    m56b_reference = by_name["reference_m56b_geometry_heads_fp32"]["parity"]

    def reference_reproduced(parity: dict, expected: float) -> bool:
        return (
            parity["failed_output_families"] == ["pred_depth"]
            and abs(parity["parity_summary"]["pred_depth"]["max_abs"] - expected)
            <= REFERENCE_DEPTH_TOLERANCE
        )

    singleton_results = [
        result for result in results if result["spec"]["mode"] == "singleton"
    ]
    complement_results = [
        result for result in results if result["spec"]["mode"] == "complement"
    ]
    singleton_depth_ranking = [
        {
            "group": result["spec"]["fp16_groups"][0],
            "pred_depth_max_abs": result["parity"]["parity_summary"]["pred_depth"][
                "max_abs"
            ],
            "all_parity_limits_passed": result["parity"][
                "all_parity_limits_passed"
            ],
            "failed_output_families": result["parity"][
                "failed_output_families"
            ],
        }
        for result in sorted(
            singleton_results,
            key=lambda row: row["parity"]["parity_summary"]["pred_depth"][
                "max_abs"
            ],
            reverse=True,
        )
    ]
    complement_rescues = [
        result["spec"]["fp32_held_group"]
        for result in complement_results
        if result["parity"]["all_parity_limits_passed"]
    ]
    singleton_failures = [
        result["spec"]["fp16_groups"][0]
        for result in singleton_results
        if not result["parity"]["all_parity_limits_passed"]
    ]
    if complement_rescues:
        interpretation = (
            "At least one held-FP32 group rescues the full policy; preserve the listed "
            "groups and test a separately frozen combined candidate next."
        )
    elif singleton_failures:
        interpretation = (
            "One or more groups fail in isolation and must remain FP32 before testing "
            "any combined candidate."
        )
    else:
        interpretation = (
            "All groups pass alone but no single held group rescues the combined policy; "
            "the drift is cumulative and requires a frozen progressive-addition search."
        )

    gates = {
        "manifest_authorized": True,
        "source_checkpoint_sha256": sha256_file(source)
        == PARENT_CHECKPOINT_SHA256,
        "runtime_config_sha256": sha256_file(runtime_config)
        == manifest["runtime_config_sha256"],
        "group_policy_reproduced": policy_reproduced,
        "candidate_specs_reproduced": candidate_specs_reproduced,
        "all_candidates_executed": len(results) == manifest["candidate_count"],
        "all_candidate_integrity_checks_passed": all(structural_checks),
        "m56b_reference_reproduced": reference_reproduced(
            m56b_reference, M56B_FAILED_DEPTH_MAX_ABS
        ),
        "no_training_or_graph_change": (
            manifest["training_performed"] is False
            and manifest["graph_changed"] is False
        ),
        "full_evaluation_remains_locked": manifest[
            "full_evaluation_authorized"
        ]
        is False,
        "coreml_and_product_claims_remain_false": (
            manifest["direct_coreml_conversion_authorized"] is False
            and manifest["product_safety_qualified"] is False
        ),
    }
    report = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M56c grouped FP16-storage sensitivity isolation",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "source_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "sample_id": f"{sample_id:06d}",
        "device": {"name": torch.cuda.get_device_name(device)},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
        },
        "parity_limits": PARITY_LIMITS,
        "candidate_count": len(results),
        "candidate_results": results,
        "analysis": {
            "singleton_depth_ranking": singleton_depth_ranking,
            "singleton_failures": singleton_failures,
            "complement_rescues": complement_rescues,
            "interpretation": interpretation,
        },
        "diagnostic_gate_results": gates,
        "all_diagnostic_gates_passed": all(gates.values()),
        "full_evaluation_authorized": False,
        "offline_compression_candidate_selected": False,
        "training_performed": False,
        "direct_coreml_conversion_authorized": False,
        "product_safety_qualified": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output.with_suffix(".csv")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_csv(csv_path, results)
    print(json.dumps(report["analysis"], indent=2))
    print(f"M56c JSON: {output}")
    print(f"M56c CSV: {csv_path}")
    if not report["all_diagnostic_gates_passed"]:
        raise RuntimeError("M56c diagnostic integrity gate failed")


if __name__ == "__main__":
    main()
