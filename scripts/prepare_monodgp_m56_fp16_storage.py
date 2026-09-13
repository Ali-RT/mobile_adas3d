from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import OrderedDict
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
except ModuleNotFoundError:
    from prepare_monodgp_m55_feasibility import (
        CLASS_MAPPING,
        PARENT_CHECKPOINT_SHA256,
        PARENT_METRICS,
        PINNED_COMMIT,
        PRESERVATION_GATES,
    )


SELECTED_EPOCH = 100
RUN_NAME = "monodgp_m56_fp16_parameter_storage"
M55_GATE_SHA256 = "684338ae7be11f5394aff76b9e3115c22f583fb06103db02c56091c4208050fc"
M55_PROFILE_SHA256 = "8ef15616614c9852ed6b51f171b69a7a0f6994a5a04b79789f0ebbdb1a3d0751"
M55_AUDIT_SHA256 = "798746036e50d37729adca3e2ef9a966a65386ab000f97feb516962b3e1ee1a3"
EXPECTED_ELIGIBLE_PARAMETER_BYTES = 156331452
MAX_MODEL_ONLY_SIZE_RATIO = 0.60


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_bytes(state: dict[str, Any]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in state.values()
        if hasattr(value, "numel") and hasattr(value, "element_size")
    )


def same_numbers(actual: dict, expected: dict) -> bool:
    if set(actual) != set(expected):
        return False
    return all(abs(float(actual[key]) - float(value)) <= 1e-12 for key, value in expected.items())


def validate_m55_evidence(gate_path: Path, profile_path: Path, audit_path: Path) -> tuple[dict, dict, dict]:
    expected_hashes = {
        gate_path: M55_GATE_SHA256,
        profile_path: M55_PROFILE_SHA256,
        audit_path: M55_AUDIT_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"M55 evidence SHA-256 mismatch for {path}: {actual} != {expected}")

    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        gate.get("complete") is not True
        or gate.get("all_feasibility_gates_passed") is not True
        or not gate.get("gate_results")
        or not all(gate["gate_results"].values())
        or gate.get("offline_weight_compression_authorized") is not True
        or gate.get("direct_coreml_conversion_authorized") is not False
        or gate.get("product_safety_qualified") is not False
    ):
        raise RuntimeError("M55 did not authorize the offline M56 compression experiment")
    if (
        gate.get("parent_epoch") != SELECTED_EPOCH
        or gate.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or profile.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or audit.get("checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
    ):
        raise RuntimeError("M55 evidence is not bound to the exact M54 epoch-100 parent")
    if not same_numbers(gate.get("parent_metrics", {}), PARENT_METRICS):
        raise RuntimeError("M55 parent metrics changed")
    if not same_numbers(gate.get("compression_preservation_gates", {}), PRESERVATION_GATES):
        raise RuntimeError("M55 preservation gates changed")
    if gate.get("artifacts", {}).get("profile_sha256") != M55_PROFILE_SHA256:
        raise RuntimeError("M55 gate/profile binding changed")
    if gate.get("artifacts", {}).get("operator_audit_sha256") != M55_AUDIT_SHA256:
        raise RuntimeError("M55 gate/audit binding changed")
    if (
        int(audit.get("eligible_parameter_bytes", 0)) != EXPECTED_ELIGIBLE_PARAMETER_BYTES
        or audit.get("ordinary_weight_operator_families") != ["Conv2d", "Linear"]
        or audit.get("direct_coreml_export_ready") is not False
    ):
        raise RuntimeError("M55 eligible-parameter or export audit changed")
    return gate, profile, audit


def collect_eligible_parameter_names(model, torch_module) -> tuple[dict[str, str], dict[str, int]]:
    names: dict[str, str] = {}
    bytes_by_family = {"Conv2d": 0, "Linear": 0}
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        family = None
        if isinstance(module, torch_module.nn.Conv2d):
            family = "Conv2d"
        elif isinstance(module, torch_module.nn.Linear):
            family = "Linear"
        if family is None:
            continue
        for local_name, parameter in module.named_parameters(recurse=False):
            identity = id(parameter)
            if identity in seen:
                continue
            seen.add(identity)
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            names[full_name] = family
            bytes_by_family[family] += parameter.numel() * parameter.element_size()
    return names, bytes_by_family


def copy_state_dict_metadata(source, target) -> None:
    metadata = getattr(source, "_metadata", None)
    if metadata is not None:
        target._metadata = metadata


def compress_state_dict(
    source_state: dict[str, Any],
    eligible_names: dict[str, str],
    torch_module,
) -> tuple[OrderedDict, OrderedDict, dict]:
    missing = sorted(set(eligible_names) - set(source_state))
    if missing:
        raise RuntimeError(f"Eligible model parameters missing from checkpoint: {missing[:10]}")

    baseline_state: OrderedDict[str, Any] = OrderedDict()
    candidate_state: OrderedDict[str, Any] = OrderedDict()
    eligible_original_bytes = 0
    eligible_stored_bytes = 0
    eligible_tensors = 0
    eligible_bytes_by_family = {"Conv2d": 0, "Linear": 0}
    noneligible_tensor_bytes = 0

    for name, value in source_state.items():
        if not isinstance(value, torch_module.Tensor):
            baseline_state[name] = value
            candidate_state[name] = value
            continue
        source = value.detach().cpu()
        baseline_state[name] = source.clone()
        if name in eligible_names:
            if source.dtype != torch_module.float32:
                raise RuntimeError(f"Expected FP32 eligible parameter {name}, found {source.dtype}")
            family = eligible_names[name]
            original_bytes = source.numel() * source.element_size()
            compressed = source.to(dtype=torch_module.float16)
            candidate_state[name] = compressed
            eligible_original_bytes += original_bytes
            eligible_stored_bytes += compressed.numel() * compressed.element_size()
            eligible_bytes_by_family[family] += original_bytes
            eligible_tensors += 1
        else:
            candidate_state[name] = source.clone()
            noneligible_tensor_bytes += source.numel() * source.element_size()

    copy_state_dict_metadata(source_state, baseline_state)
    copy_state_dict_metadata(source_state, candidate_state)
    baseline_bytes = tensor_bytes(baseline_state)
    candidate_bytes = tensor_bytes(candidate_state)
    inventory = {
        "eligible_parameter_tensors": eligible_tensors,
        "eligible_parameter_bytes_fp32": eligible_original_bytes,
        "eligible_parameter_bytes_fp16": eligible_stored_bytes,
        "eligible_parameter_bytes_by_family_fp32": eligible_bytes_by_family,
        "noneligible_tensor_bytes_unchanged": noneligible_tensor_bytes,
        "baseline_state_dict_tensor_bytes": baseline_bytes,
        "candidate_state_dict_tensor_bytes": candidate_bytes,
        "state_dict_tensor_size_ratio": candidate_bytes / baseline_bytes,
        "state_dict_tensor_savings_bytes": baseline_bytes - candidate_bytes,
    }
    return baseline_state, candidate_state, inventory


def checkpoint_payload(source_payload: dict, model_state: dict) -> dict:
    best_result = source_payload.get("best_result", 0.0)
    best_epoch = source_payload.get("best_epoch", 0)
    return {
        "epoch": SELECTED_EPOCH,
        "model_state": model_state,
        "optimizer_state": None,
        "best_result": float(best_result if best_result is not None else 0.0),
        "best_epoch": int(best_epoch if best_epoch is not None else 0),
    }


def atomic_torch_save(torch_module, payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch_module.save(payload, temporary)
    temporary.replace(path)


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(
        description="Prepare M56 model-only FP16 Conv2d/Linear parameter storage."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--m54-selection", type=Path, required=True)
    parser.add_argument("--m55-gate", type=Path, required=True)
    parser.add_argument("--m55-profile", type=Path, required=True)
    parser.add_argument("--m55-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.monodgp_repo.resolve()
    dataset = args.dataset_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    required = [
        repo / "configs/monodgp.yaml",
        dataset / "ImageSets/train.txt",
        dataset / "ImageSets/val.txt",
        dataset / "training/image_2",
        dataset / "training/label_2",
        dataset / "training/calib",
        args.m54_selection.resolve(),
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP commit {PINNED_COMMIT}, found {commit}")

    gate, m55_profile, audit = validate_m55_evidence(
        args.m55_gate.resolve(), args.m55_profile.resolve(), args.m55_audit.resolve()
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
    if not source_checkpoint.is_file() or sha256_file(source_checkpoint) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("Exact M54 parent checkpoint is missing or changed")

    sys.path.insert(0, str(repo))
    from lib.helpers.model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint_safely

    base_config = yaml.safe_load((repo / "configs/monodgp.yaml").read_text(encoding="utf-8"))
    model, _ = build_model(base_config["model"])
    source_payload = load_checkpoint_safely(source_checkpoint, torch.device("cpu"))
    if int(source_payload.get("epoch", -1)) != SELECTED_EPOCH:
        raise RuntimeError("M54 checkpoint payload epoch changed")
    source_state = source_payload.get("model_state")
    if not isinstance(source_state, dict):
        raise RuntimeError("M54 checkpoint has no model_state")
    model.load_state_dict(source_state, strict=True)

    eligible_names, model_bytes_by_family = collect_eligible_parameter_names(model, torch)
    baseline_state, candidate_state, inventory = compress_state_dict(
        source_state, eligible_names, torch
    )
    if inventory["eligible_parameter_bytes_fp32"] != EXPECTED_ELIGIBLE_PARAMETER_BYTES:
        raise RuntimeError(
            "M56 eligible bytes do not reproduce M55: "
            f"{inventory['eligible_parameter_bytes_fp32']} != {EXPECTED_ELIGIBLE_PARAMETER_BYTES}"
        )
    if model_bytes_by_family != audit["eligible_parameter_bytes_by_family"]:
        raise RuntimeError("M56 eligible family inventory does not reproduce M55")
    if inventory["eligible_parameter_bytes_fp16"] * 2 != EXPECTED_ELIGIBLE_PARAMETER_BYTES:
        raise RuntimeError("M56 FP16 eligible storage is not exactly half the FP32 bytes")

    run_dir = output / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline_checkpoint = output / "m56_parent_model_only_fp32.pth"
    candidate_checkpoint = run_dir / f"checkpoint_epoch_{SELECTED_EPOCH}.pth"
    atomic_torch_save(torch, checkpoint_payload(source_payload, baseline_state), baseline_checkpoint)
    atomic_torch_save(torch, checkpoint_payload(source_payload, candidate_state), candidate_checkpoint)

    config = base_config
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
    runtime_config = repo / "configs/monodgp_m56_fp16_parameter_storage.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    baseline_size = baseline_checkpoint.stat().st_size
    candidate_size = candidate_checkpoint.stat().st_size
    size_ratio = candidate_size / baseline_size
    preparation_gates = {
        "m55_authorized": True,
        "parent_checkpoint_sha256": sha256_file(source_checkpoint) == PARENT_CHECKPOINT_SHA256,
        "eligible_bytes_reproduced": (
            inventory["eligible_parameter_bytes_fp32"] == EXPECTED_ELIGIBLE_PARAMETER_BYTES
        ),
        "eligible_storage_halved": (
            inventory["eligible_parameter_bytes_fp16"] * 2
            == inventory["eligible_parameter_bytes_fp32"]
        ),
        "noneligible_storage_unchanged": (
            inventory["noneligible_tensor_bytes_unchanged"] > 0
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
        "experiment": "M56 M54 FP16 Conv2d/Linear parameter-storage sensitivity",
        "training_performed": False,
        "graph_changed": False,
        "runtime_precision": "FP32 after checkpoint load",
        "storage_precision": "FP16 for floating Conv2d/Linear-owned parameters",
        "compression_policy": {
            "families": ["Conv2d", "Linear"],
            "parameter_scope": "direct floating-point parameters, including weight and bias",
            "eligible_storage_dtype": "torch.float16",
            "noneligible_tensors": "unchanged",
            "optimizer_state": "removed from both model-only comparison artifacts",
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
        "storage_inventory": inventory,
        "eligible_parameter_name_count": len(eligible_names),
        "eligible_parameter_names_sha256": hashlib.sha256(
            "\n".join(sorted(eligible_names)).encode("utf-8")
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
            "m56_fp16_storage_smoke.json",
            "m56_fp16_storage_gate.json",
            "m56_fp16_storage_comparison.csv",
        ],
    }
    manifest_path = output / "m56_compression_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not manifest["smoke_authorized"]:
        raise RuntimeError("M56 preparation failed; do not run the CUDA smoke test")


if __name__ == "__main__":
    main()
