from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.prepare_monodgp_m55_feasibility import (
        CLASS_MAPPING,
        PARENT_CHECKPOINT_SHA256,
        PARENT_METRICS,
        PINNED_COMMIT,
        PRESERVATION_GATES,
    )
    from scripts.prepare_monodgp_m56_fp16_storage import (
        EXPECTED_ELIGIBLE_PARAMETER_BYTES,
        MAX_MODEL_ONLY_SIZE_RATIO,
        M55_AUDIT_SHA256,
        M55_GATE_SHA256,
        M55_PROFILE_SHA256,
        atomic_torch_save,
        checkpoint_payload,
        compress_state_dict,
        same_numbers,
        sha256_file,
        validate_m55_evidence,
    )
except ModuleNotFoundError:
    from prepare_monodgp_m55_feasibility import (
        CLASS_MAPPING,
        PARENT_CHECKPOINT_SHA256,
        PARENT_METRICS,
        PINNED_COMMIT,
        PRESERVATION_GATES,
    )
    from prepare_monodgp_m56_fp16_storage import (
        EXPECTED_ELIGIBLE_PARAMETER_BYTES,
        MAX_MODEL_ONLY_SIZE_RATIO,
        M55_AUDIT_SHA256,
        M55_GATE_SHA256,
        M55_PROFILE_SHA256,
        atomic_torch_save,
        checkpoint_payload,
        compress_state_dict,
        same_numbers,
        sha256_file,
        validate_m55_evidence,
    )


SELECTED_EPOCH = 100
RUN_NAME = "monodgp_m56b_selective_fp16_storage"
FP32_HEAD_ROOTS = ("bbox_embed", "dim_embed_3d", "depth_embed")
M56_MANIFEST_SHA256 = "a9cfbd8e41ed020b4d49ee44594b81966b27601750fe4167454cbbdecffde046"
M56_CANDIDATE_SHA256 = "8da64f181e5c09b17a29e09eb41b1133159313f2d09eeb1ac8218bf1d1ea4404"
M56_FAILED_DEPTH_MAX_ABS = 0.7107391357421875


def validate_m56_result_records(manifest: dict, smoke: dict) -> None:
    failed_gates = sorted(
        name for name, passed in smoke.get("gate_results", {}).items() if not passed
    )
    failed_outputs = sorted(
        name for name, row in smoke.get("parity_summary", {}).items()
        if row.get("passed") is not True
    )
    depth = smoke.get("parity_summary", {}).get("pred_depth", {})
    if (
        manifest.get("complete") is not True
        or manifest.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or manifest.get("candidate_checkpoint_sha256") != M56_CANDIDATE_SHA256
        or smoke.get("complete") is not True
        or smoke.get("manifest_sha256") != M56_MANIFEST_SHA256
        or smoke.get("source_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or smoke.get("candidate_checkpoint_sha256") != M56_CANDIDATE_SHA256
        or smoke.get("all_smoke_gates_passed") is not False
        or smoke.get("full_evaluation_authorized") is not False
        or failed_gates != ["raw_output_parity_within_limits"]
        or failed_outputs != ["pred_depth"]
        or abs(float(depth.get("limit_max_abs", -1.0)) - 0.5) > 1e-12
        or abs(float(depth.get("max_abs", -1.0)) - M56_FAILED_DEPTH_MAX_ABS) > 1e-12
    ):
        raise RuntimeError("M56b is not bound to the exact rejected M56 smoke result")


def validate_m56_evidence(manifest_path: Path, smoke_path: Path) -> tuple[dict, dict]:
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    if not smoke_path.is_file():
        raise FileNotFoundError(smoke_path)
    actual_manifest_sha = sha256_file(manifest_path)
    if actual_manifest_sha != M56_MANIFEST_SHA256:
        raise RuntimeError(
            f"M56 manifest SHA-256 mismatch: {actual_manifest_sha} != {M56_MANIFEST_SHA256}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    validate_m56_result_records(manifest, smoke)
    return manifest, smoke


def _is_preserved_head_alias(name: str) -> bool:
    tokens = name.split(".")
    return any(root in tokens for root in FP32_HEAD_ROOTS)


def collect_alias_aware_storage_policy(model, torch_module) -> tuple[dict, dict, dict]:
    records: dict[int, dict[str, Any]] = {}
    try:
        modules = model.named_modules(remove_duplicate=False)
    except TypeError as exc:
        raise RuntimeError(
            "M56b requires named_modules(remove_duplicate=False) for alias-safe storage"
        ) from exc
    for module_name, module in modules:
        if isinstance(module, torch_module.nn.Conv2d):
            family = "Conv2d"
        elif isinstance(module, torch_module.nn.Linear):
            family = "Linear"
        else:
            continue
        for local_name, parameter in module.named_parameters(recurse=False):
            name = f"{module_name}.{local_name}" if module_name else local_name
            record = records.setdefault(
                id(parameter),
                {"parameter": parameter, "family": family, "aliases": set()},
            )
            if record["family"] != family:
                raise RuntimeError(f"Parameter alias changed operator family: {name}")
            record["aliases"].add(name)

    fp16_aliases: dict[str, str] = {}
    fp32_aliases: dict[str, str] = {}
    unique_bytes_by_family = {"Conv2d": 0, "Linear": 0}
    fp16_unique_bytes = 0
    fp32_unique_bytes = 0
    fp16_unique_parameters = 0
    fp32_unique_parameters = 0
    preserved_by_root = {root: 0 for root in FP32_HEAD_ROOTS}
    for record in records.values():
        aliases = sorted(record["aliases"])
        family = record["family"]
        parameter = record["parameter"]
        size = parameter.numel() * parameter.element_size()
        unique_bytes_by_family[family] += size
        preserve = any(_is_preserved_head_alias(name) for name in aliases)
        destination = fp32_aliases if preserve else fp16_aliases
        for name in aliases:
            destination[name] = family
        if preserve:
            fp32_unique_bytes += size
            fp32_unique_parameters += 1
            roots = {root for root in FP32_HEAD_ROOTS if any(root in name.split(".") for name in aliases)}
            for root in roots:
                preserved_by_root[root] += size
        else:
            fp16_unique_bytes += size
            fp16_unique_parameters += 1

    inventory = {
        "all_eligible_unique_parameters": len(records),
        "all_eligible_unique_parameter_bytes": sum(unique_bytes_by_family.values()),
        "all_eligible_unique_bytes_by_family": unique_bytes_by_family,
        "fp16_unique_parameters": fp16_unique_parameters,
        "fp16_unique_parameter_bytes": fp16_unique_bytes,
        "fp16_state_aliases": len(fp16_aliases),
        "fp32_preserved_unique_parameters": fp32_unique_parameters,
        "fp32_preserved_unique_parameter_bytes": fp32_unique_bytes,
        "fp32_preserved_state_aliases": len(fp32_aliases),
        "fp32_preserved_bytes_by_head_root": preserved_by_root,
    }
    return fp16_aliases, fp32_aliases, inventory


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(
        description="Prepare alias-safe M56b selective FP16 parameter storage."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--m54-selection", type=Path, required=True)
    parser.add_argument("--m55-gate", type=Path, required=True)
    parser.add_argument("--m55-profile", type=Path, required=True)
    parser.add_argument("--m55-audit", type=Path, required=True)
    parser.add_argument("--m56-manifest", type=Path, required=True)
    parser.add_argument("--m56-smoke", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.monodgp_repo.resolve()
    dataset = args.dataset_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for path in (
        repo / "configs/monodgp.yaml",
        dataset / "ImageSets/train.txt",
        dataset / "ImageSets/val.txt",
        dataset / "training/image_2",
        dataset / "training/label_2",
        dataset / "training/calib",
        args.m54_selection.resolve(),
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP commit {PINNED_COMMIT}, found {commit}")

    gate, m55_profile, audit = validate_m55_evidence(
        args.m55_gate.resolve(), args.m55_profile.resolve(), args.m55_audit.resolve()
    )
    m56_manifest, m56_smoke = validate_m56_evidence(
        args.m56_manifest.resolve(), args.m56_smoke.resolve()
    )
    selection = json.loads(args.m54_selection.read_text(encoding="utf-8"))
    if (
        selection.get("complete") is not True
        or selection.get("selected_epoch") != SELECTED_EPOCH
        or selection.get("selected_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or not same_numbers(selection.get("metrics", {}), PARENT_METRICS)
    ):
        raise RuntimeError("M54 selection is not the frozen epoch-100 accuracy parent")
    source_checkpoint = Path(selection["selected_checkpoint"]).expanduser().resolve()
    if (
        not source_checkpoint.is_file()
        or sha256_file(source_checkpoint) != PARENT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("Exact M54 parent checkpoint is missing or changed")

    sys.path.insert(0, str(repo))
    from lib.helpers.model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint_safely

    config = yaml.safe_load((repo / "configs/monodgp.yaml").read_text(encoding="utf-8"))
    model, _ = build_model(config["model"])
    source_payload = load_checkpoint_safely(source_checkpoint, torch.device("cpu"))
    if int(source_payload.get("epoch", -1)) != SELECTED_EPOCH:
        raise RuntimeError("M54 checkpoint payload epoch changed")
    source_state = source_payload.get("model_state")
    if not isinstance(source_state, dict):
        raise RuntimeError("M54 checkpoint has no model_state")
    model.load_state_dict(source_state, strict=True)

    fp16_aliases, fp32_aliases, policy_inventory = collect_alias_aware_storage_policy(
        model, torch
    )
    if (
        policy_inventory["all_eligible_unique_parameter_bytes"]
        != EXPECTED_ELIGIBLE_PARAMETER_BYTES
    ):
        raise RuntimeError("M56b eligible unique bytes do not reproduce M55")
    if policy_inventory["all_eligible_unique_bytes_by_family"] != audit[
        "eligible_parameter_bytes_by_family"
    ]:
        raise RuntimeError("M56b eligible family bytes do not reproduce M55")
    if not fp16_aliases or not fp32_aliases:
        raise RuntimeError("M56b storage policy produced an empty branch")
    missing_aliases = sorted((set(fp16_aliases) | set(fp32_aliases)) - set(source_state))
    if missing_aliases:
        raise RuntimeError(f"M56b parameter aliases missing from state dict: {missing_aliases[:10]}")

    baseline_state, candidate_state, storage_inventory = compress_state_dict(
        source_state, fp16_aliases, torch
    )
    run_dir = output / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline_checkpoint = output / "m56b_parent_model_only_fp32.pth"
    candidate_checkpoint = run_dir / f"checkpoint_epoch_{SELECTED_EPOCH}.pth"
    atomic_torch_save(
        torch, checkpoint_payload(source_payload, baseline_state), baseline_checkpoint
    )
    atomic_torch_save(
        torch, checkpoint_payload(source_payload, candidate_state), candidate_checkpoint
    )

    expected_invariants = {
        "backbone": "resnet50",
        "num_classes": 3,
        "num_feature_levels": 4,
        "num_queries": 50,
        "group_num": 11,
        "hidden_dim": 256,
        "enc_layers": 3,
        "dec_layers": 3,
        "nheads": 8,
    }
    actual_invariants = {key: config["model"].get(key) for key in expected_invariants}
    if actual_invariants != expected_invariants:
        raise RuntimeError(f"Pinned MonoDGP architecture changed: {actual_invariants}")
    config["model_name"] = RUN_NAME
    config["dataset"].update(
        {
            "root_dir": str(dataset),
            "train_split": "train",
            "test_split": "val",
            "batch_size": 1,
            "writelist": ["Car", "Pedestrian"],
            "class_mapping": CLASS_MAPPING,
            "class_merging": False,
            "use_dontcare": False,
            "aug_pd": False,
            "aug_crop": False,
            "random_flip": 0.0,
            "random_crop": 0.0,
            "random_mixup3d": 0.0,
        }
    )
    config["trainer"].update(
        {
            "max_epoch": 0,
            "save_path": os.path.relpath(output, repo),
            "save_all": True,
            "pretrain_model": None,
            "resume_model": False,
        }
    )
    config["tester"].update(
        {"mode": "single", "checkpoint": SELECTED_EPOCH, "threshold": 0.001, "topk": 50}
    )
    runtime_config = repo / "configs/monodgp_m56b_selective_fp16_storage.yaml"
    runtime_config.write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    baseline_size = baseline_checkpoint.stat().st_size
    candidate_size = candidate_checkpoint.stat().st_size
    size_ratio = candidate_size / baseline_size
    preparation_gates = {
        "m55_authorized": True,
        "m56_rejection_verified": True,
        "parent_checkpoint_sha256": sha256_file(source_checkpoint)
        == PARENT_CHECKPOINT_SHA256,
        "eligible_unique_bytes_reproduced": (
            policy_inventory["all_eligible_unique_parameter_bytes"]
            == EXPECTED_ELIGIBLE_PARAMETER_BYTES
        ),
        "eligible_family_bytes_reproduced": (
            policy_inventory["all_eligible_unique_bytes_by_family"]
            == audit["eligible_parameter_bytes_by_family"]
        ),
        "all_eligible_parameters_classified_once": (
            policy_inventory["fp16_unique_parameters"]
            + policy_inventory["fp32_preserved_unique_parameters"]
            == policy_inventory["all_eligible_unique_parameters"]
        ),
        "fp16_storage_halved": (
            storage_inventory["eligible_parameter_bytes_fp16"] * 2
            == storage_inventory["eligible_parameter_bytes_fp32"]
        ),
        "fp32_geometry_heads_preserved": (
            policy_inventory["fp32_preserved_unique_parameter_bytes"] > 0
            and all(policy_inventory["fp32_preserved_bytes_by_head_root"].values())
        ),
        "all_fp16_state_aliases_encoded": (
            storage_inventory["eligible_parameter_tensors"] == len(fp16_aliases)
        ),
        "model_only_size_ratio_le_0_60": size_ratio <= MAX_MODEL_ONLY_SIZE_RATIO,
        "optimizer_state_removed_from_both_artifacts": True,
        "architecture_unchanged": actual_invariants == expected_invariants,
        "runtime_remains_fp32": True,
        "direct_coreml_unauthorized": True,
    }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M56b selective FP16 storage with FP32 geometry heads",
        "training_performed": False,
        "graph_changed": False,
        "runtime_precision": "FP32 after checkpoint load",
        "storage_precision": (
            "FP16 for eligible Conv2d/Linear parameters except bbox_embed, "
            "dim_embed_3d, and depth_embed aliases retained in FP32"
        ),
        "compression_policy": {
            "families": ["Conv2d", "Linear"],
            "parameter_scope": "direct floating-point parameters, including weight and bias",
            "fp16_rule": "all eligible parameter alias groups not owned by a preserved head",
            "fp32_preserved_head_roots": list(FP32_HEAD_ROOTS),
            "alias_consistency": (
                "if any state-dict alias belongs to a preserved head, every alias for that "
                "shared parameter remains FP32"
            ),
            "noneligible_tensors": "unchanged",
            "optimizer_state": "removed from both model-only comparison artifacts",
            "parity_thresholds": "identical to frozen M56 thresholds",
        },
        "upstream_commit": commit,
        "parent_epoch": SELECTED_EPOCH,
        "parent_checkpoint": str(source_checkpoint),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_metrics": PARENT_METRICS,
        "preservation_gates": PRESERVATION_GATES,
        "m55_gate": str(args.m55_gate.resolve()),
        "m55_gate_sha256": M55_GATE_SHA256,
        "m55_profile": str(args.m55_profile.resolve()),
        "m55_profile_sha256": M55_PROFILE_SHA256,
        "m55_audit": str(args.m55_audit.resolve()),
        "m55_audit_sha256": M55_AUDIT_SHA256,
        "m55_baseline_latency": m55_profile["latency"],
        "m55_baseline_memory": m55_profile["memory"],
        "m56_manifest": str(args.m56_manifest.resolve()),
        "m56_manifest_sha256": M56_MANIFEST_SHA256,
        "m56_smoke": str(args.m56_smoke.resolve()),
        "m56_smoke_sha256": sha256_file(args.m56_smoke.resolve()),
        "m56_rejected_candidate_sha256": M56_CANDIDATE_SHA256,
        "m56_failed_depth_max_abs": M56_FAILED_DEPTH_MAX_ABS,
        "baseline_model_only_checkpoint": str(baseline_checkpoint),
        "baseline_model_only_checkpoint_sha256": sha256_file(baseline_checkpoint),
        "baseline_model_only_checkpoint_bytes": baseline_size,
        "candidate_run_dir": str(run_dir),
        "candidate_checkpoint": str(candidate_checkpoint),
        "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
        "candidate_checkpoint_bytes": candidate_size,
        "model_only_checkpoint_size_ratio": size_ratio,
        "model_only_checkpoint_savings_bytes": baseline_size - candidate_size,
        "max_model_only_size_ratio": MAX_MODEL_ONLY_SIZE_RATIO,
        "storage_inventory": storage_inventory,
        "alias_aware_policy_inventory": policy_inventory,
        "fp16_parameter_state_names": sorted(fp16_aliases),
        "fp16_parameter_state_names_sha256": hashlib.sha256(
            "\n".join(sorted(fp16_aliases)).encode("utf-8")
        ).hexdigest(),
        "fp32_preserved_parameter_state_names": sorted(fp32_aliases),
        "fp32_preserved_parameter_state_names_sha256": hashlib.sha256(
            "\n".join(sorted(fp32_aliases)).encode("utf-8")
        ).hexdigest(),
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": sha256_file(runtime_config),
        "dataset_root": str(dataset),
        "split_protocol": "chen_3712_3769",
        "preparation_gate_results": preparation_gates,
        "smoke_authorized": all(preparation_gates.values()),
        "full_evaluation_authorized": False,
        "offline_compression_candidate_selected": False,
        "direct_coreml_conversion_authorized": False,
        "product_safety_qualified": False,
        "expected_artifacts": [
            "m56b_selective_fp16_storage_smoke.json",
            "m56b_selective_fp16_storage_gate.json",
            "m56b_selective_fp16_storage_comparison.csv",
        ],
    }
    manifest_path = output / "m56b_compression_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not manifest["smoke_authorized"]:
        raise RuntimeError("M56b preparation failed; do not run the CUDA smoke test")


if __name__ == "__main__":
    main()
