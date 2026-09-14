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
        M55_AUDIT_SHA256,
        M55_GATE_SHA256,
        M55_PROFILE_SHA256,
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
        M55_AUDIT_SHA256,
        M55_GATE_SHA256,
        M55_PROFILE_SHA256,
        same_numbers,
        sha256_file,
        validate_m55_evidence,
    )


SELECTED_EPOCH = 100
RUN_NAME = "monodgp_m56c_group_sensitivity"
GROUP_ORDER = (
    "backbone",
    "input_projection",
    "region_head",
    "depth_predictor",
    "det2d_transformer",
    "det3d_transformer",
    "prediction_heads",
    "other_eligible",
)
REQUIRED_NONEMPTY_GROUPS = GROUP_ORDER[:-1]
PREDICTION_HEAD_ROOTS = (
    "class_embed",
    "bbox_embed",
    "dim_embed_3d",
    "depth_embed",
    "angle_embed",
)
M56B_FP32_HEAD_ROOTS = ("bbox_embed", "dim_embed_3d", "depth_embed")
M56B_MANIFEST_SHA256 = "6251abd0ca2eb39b4d1376b9efb836221565a38b4359801efb5e2318f1a2f4ef"
M56B_SMOKE_SHA256 = "c9510f67588a4998c1f84e2c309c98c3def3931627874200662fac92690fd8d8"
M56B_CANDIDATE_SHA256 = "ea2d32db0836dc35d226ec17072f445368d79d4d36dbb864b4759e89c8a190f4"
M56B_FAILED_DEPTH_MAX_ABS = 0.7097339630126953


def names_sha256(names: set[str] | list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def validate_m56b_result_records(manifest: dict, smoke: dict) -> None:
    failed_gates = sorted(
        name for name, passed in smoke.get("gate_results", {}).items() if not passed
    )
    failed_outputs = sorted(
        name
        for name, row in smoke.get("parity_summary", {}).items()
        if row.get("passed") is not True
    )
    depth = smoke.get("parity_summary", {}).get("pred_depth", {})
    depth_channel = smoke.get("pred_depth_channel_summary", {}).get("depth_m", {})
    preparation = manifest.get("preparation_gate_results", {})
    if (
        manifest.get("complete") is not True
        or manifest.get("experiment")
        != "M56b selective FP16 storage with FP32 geometry heads"
        or manifest.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or manifest.get("candidate_checkpoint_sha256") != M56B_CANDIDATE_SHA256
        or manifest.get("smoke_authorized") is not True
        or not preparation
        or not all(preparation.values())
        or smoke.get("complete") is not True
        or smoke.get("manifest_sha256") != M56B_MANIFEST_SHA256
        or smoke.get("source_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or smoke.get("candidate_checkpoint_sha256") != M56B_CANDIDATE_SHA256
        or smoke.get("stored_fp16_parameter_state_name_count") != 271
        or smoke.get("preserved_fp32_parameter_state_name_count") != 104
        or smoke.get("all_smoke_gates_passed") is not False
        or smoke.get("full_evaluation_authorized") is not False
        or smoke.get("training_performed") is not False
        or failed_gates != ["raw_output_parity_within_limits"]
        or failed_outputs != ["pred_depth"]
        or abs(float(depth.get("limit_max_abs", -1.0)) - 0.5) > 1e-12
        or abs(float(depth.get("max_abs", -1.0)) - M56B_FAILED_DEPTH_MAX_ABS)
        > 1e-12
        or abs(float(depth_channel.get("max_abs", -1.0)) - M56B_FAILED_DEPTH_MAX_ABS)
        > 1e-12
    ):
        raise RuntimeError("M56c is not bound to the exact rejected M56b result")


def validate_m56b_evidence(
    manifest_path: Path, smoke_path: Path
) -> tuple[dict, dict]:
    expected_hashes = {
        manifest_path: M56B_MANIFEST_SHA256,
        smoke_path: M56B_SMOKE_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(
                f"M56b evidence SHA-256 mismatch for {path}: {actual} != {expected}"
            )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    validate_m56b_result_records(manifest, smoke)
    return manifest, smoke


def _alias_group(name: str) -> str:
    tokens = name.split(".")
    if any(root in tokens for root in PREDICTION_HEAD_ROOTS):
        return "prediction_heads"
    root = tokens[0]
    return {
        "backbone": "backbone",
        "input_proj": "input_projection",
        "region_head": "region_head",
        "depth_predictor": "depth_predictor",
        "det2d_transformer": "det2d_transformer",
        "det3d_transformer": "det3d_transformer",
    }.get(root, "other_eligible")


def collect_grouped_storage_policy(
    model, torch_module
) -> tuple[dict[str, dict[str, str]], set[str], dict]:
    records: dict[int, dict[str, Any]] = {}
    try:
        modules = model.named_modules(remove_duplicate=False)
    except TypeError as exc:
        raise RuntimeError(
            "M56c requires named_modules(remove_duplicate=False) for alias safety"
        ) from exc

    for module_name, module in modules:
        if isinstance(module, torch_module.nn.Conv2d):
            family = "Conv2d"
        elif isinstance(module, torch_module.nn.Linear):
            family = "Linear"
        else:
            continue
        for local_name, parameter in module.named_parameters(recurse=False):
            alias = f"{module_name}.{local_name}" if module_name else local_name
            record = records.setdefault(
                id(parameter),
                {"parameter": parameter, "family": family, "aliases": set()},
            )
            if record["family"] != family:
                raise RuntimeError(f"Parameter alias changed operator family: {alias}")
            record["aliases"].add(alias)

    grouped = {name: {} for name in GROUP_ORDER}
    group_rows = {
        name: {
            "unique_parameters": 0,
            "unique_parameter_bytes": 0,
            "state_aliases": 0,
            "unique_bytes_by_family": {"Conv2d": 0, "Linear": 0},
        }
        for name in GROUP_ORDER
    }
    m56b_fp32_aliases: set[str] = set()
    m56b_fp32_unique_parameters = 0
    m56b_fp32_unique_bytes = 0

    for record in records.values():
        aliases = sorted(record["aliases"])
        groups = {_alias_group(alias) for alias in aliases}
        if len(groups) != 1:
            raise RuntimeError(f"One parameter spans M56c groups: {aliases} -> {groups}")
        group = groups.pop()
        family = record["family"]
        parameter = record["parameter"]
        size = parameter.numel() * parameter.element_size()
        for alias in aliases:
            grouped[group][alias] = family
        row = group_rows[group]
        row["unique_parameters"] += 1
        row["unique_parameter_bytes"] += size
        row["state_aliases"] += len(aliases)
        row["unique_bytes_by_family"][family] += size

        if any(
            root in alias.split(".")
            for alias in aliases
            for root in M56B_FP32_HEAD_ROOTS
        ):
            m56b_fp32_aliases.update(aliases)
            m56b_fp32_unique_parameters += 1
            m56b_fp32_unique_bytes += size

    inventory = {
        "group_order": list(GROUP_ORDER),
        "groups": group_rows,
        "all_eligible_unique_parameters": len(records),
        "all_eligible_unique_parameter_bytes": sum(
            row["unique_parameter_bytes"] for row in group_rows.values()
        ),
        "all_eligible_state_aliases": sum(
            row["state_aliases"] for row in group_rows.values()
        ),
        "m56b_fp32_geometry_unique_parameters": m56b_fp32_unique_parameters,
        "m56b_fp32_geometry_unique_parameter_bytes": m56b_fp32_unique_bytes,
        "m56b_fp32_geometry_state_aliases": len(m56b_fp32_aliases),
    }
    return grouped, m56b_fp32_aliases, inventory


def candidate_fp16_names(
    spec: dict, grouped: dict[str, dict[str, str]], m56b_fp32_aliases: set[str]
) -> set[str]:
    names = {
        alias
        for group in spec["fp16_groups"]
        for alias in grouped[group]
    }
    if spec.get("exclude_m56b_geometry_heads"):
        names -= m56b_fp32_aliases
    return names


def build_candidate_specs(
    grouped: dict[str, dict[str, str]],
    m56b_fp32_aliases: set[str],
    inventory: dict,
    total_parameter_bytes: int,
) -> list[dict]:
    active_groups = [name for name in GROUP_ORDER if grouped[name]]
    raw_specs = [
        {
            "name": "reference_alias_consistent_all_eligible",
            "mode": "reference",
            "fp16_groups": active_groups,
        },
        {
            "name": "reference_m56b_geometry_heads_fp32",
            "mode": "reference",
            "fp16_groups": active_groups,
            "exclude_m56b_geometry_heads": True,
        },
    ]
    raw_specs.extend(
        {
            "name": f"only_{group}",
            "mode": "singleton",
            "fp16_groups": [group],
        }
        for group in active_groups
    )
    raw_specs.extend(
        {
            "name": f"all_except_{group}",
            "mode": "complement",
            "fp16_groups": [name for name in active_groups if name != group],
            "fp32_held_group": group,
        }
        for group in active_groups
    )

    specs = []
    for spec in raw_specs:
        names = candidate_fp16_names(spec, grouped, m56b_fp32_aliases)
        unique_bytes = sum(
            inventory["groups"][group]["unique_parameter_bytes"]
            for group in spec["fp16_groups"]
        )
        unique_parameters = sum(
            inventory["groups"][group]["unique_parameters"]
            for group in spec["fp16_groups"]
        )
        if spec.get("exclude_m56b_geometry_heads"):
            unique_bytes -= inventory["m56b_fp32_geometry_unique_parameter_bytes"]
            unique_parameters -= inventory["m56b_fp32_geometry_unique_parameters"]
        projected_bytes = total_parameter_bytes - unique_bytes // 2
        specs.append(
            {
                **spec,
                "fp16_parameter_state_name_count": len(names),
                "fp16_parameter_state_names_sha256": names_sha256(names),
                "fp16_unique_parameters": unique_parameters,
                "fp16_unique_parameter_bytes": unique_bytes,
                "projected_parameter_bytes": projected_bytes,
                "projected_parameter_size_ratio": projected_bytes
                / total_parameter_bytes,
            }
        )
    return specs


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(
        description="Prepare the M56c MonoDGP grouped FP16-storage sensitivity matrix."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--m54-selection", type=Path, required=True)
    parser.add_argument("--m55-gate", type=Path, required=True)
    parser.add_argument("--m55-profile", type=Path, required=True)
    parser.add_argument("--m55-audit", type=Path, required=True)
    parser.add_argument("--m56b-manifest", type=Path, required=True)
    parser.add_argument("--m56b-smoke", type=Path, required=True)
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

    _, m55_profile, audit = validate_m55_evidence(
        args.m55_gate.resolve(), args.m55_profile.resolve(), args.m55_audit.resolve()
    )
    validate_m56b_evidence(
        args.m56b_manifest.resolve(), args.m56b_smoke.resolve()
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

    grouped, m56b_fp32_aliases, inventory = collect_grouped_storage_policy(
        model, torch
    )
    all_aliases = {alias for names in grouped.values() for alias in names}
    if inventory["all_eligible_unique_parameter_bytes"] != EXPECTED_ELIGIBLE_PARAMETER_BYTES:
        raise RuntimeError("M56c eligible bytes do not reproduce M55")
    family_bytes = {"Conv2d": 0, "Linear": 0}
    for row in inventory["groups"].values():
        for family, value in row["unique_bytes_by_family"].items():
            family_bytes[family] += value
    if family_bytes != audit["eligible_parameter_bytes_by_family"]:
        raise RuntimeError("M56c eligible family bytes do not reproduce M55")
    missing_groups = [name for name in REQUIRED_NONEMPTY_GROUPS if not grouped[name]]
    if missing_groups:
        raise RuntimeError(f"Pinned MonoDGP is missing M56c groups: {missing_groups}")
    if not all_aliases <= set(source_state):
        raise RuntimeError("M56c policy contains aliases absent from the source state")

    total_parameter_bytes = int(m55_profile["parameter_inventory"]["parameter_bytes"])
    candidates = build_candidate_specs(
        grouped, m56b_fp32_aliases, inventory, total_parameter_bytes
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
            "save_all": False,
            "pretrain_model": None,
            "resume_model": False,
        }
    )
    config["tester"].update(
        {"mode": "single", "checkpoint": SELECTED_EPOCH, "threshold": 0.001, "topk": 50}
    )
    runtime_config = repo / "configs/monodgp_m56c_group_sensitivity.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    group_policy = {
        name: {
            **inventory["groups"][name],
            "parameter_state_names": sorted(grouped[name]),
            "parameter_state_names_sha256": names_sha256(set(grouped[name])),
        }
        for name in GROUP_ORDER
    }
    preparation_gates = {
        "m55_authorized": True,
        "m56b_rejection_verified": True,
        "parent_checkpoint_sha256": sha256_file(source_checkpoint)
        == PARENT_CHECKPOINT_SHA256,
        "eligible_unique_bytes_reproduced": inventory[
            "all_eligible_unique_parameter_bytes"
        ]
        == EXPECTED_ELIGIBLE_PARAMETER_BYTES,
        "eligible_family_bytes_reproduced": family_bytes
        == audit["eligible_parameter_bytes_by_family"],
        "required_groups_nonempty": not missing_groups,
        "group_aliases_disjoint": sum(len(names) for names in grouped.values())
        == len(all_aliases),
        "group_aliases_present_in_source": all_aliases <= set(source_state),
        "m56b_geometry_aliases_nonempty": bool(m56b_fp32_aliases),
        "candidate_matrix_complete": len(candidates)
        == 2 + 2 * sum(bool(grouped[name]) for name in GROUP_ORDER),
        "architecture_unchanged": actual_invariants == expected_invariants,
        "runtime_remains_fp32": True,
        "no_durable_candidate_checkpoints": True,
        "full_evaluation_locked": True,
        "direct_coreml_unauthorized": True,
    }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M56c grouped FP16-storage sensitivity isolation",
        "purpose": (
            "Identify which architectural stages create raw-output depth drift; "
            "this diagnostic does not select a compressed model"
        ),
        "training_performed": False,
        "graph_changed": False,
        "runtime_precision": "FP32 after in-memory FP16 rounding",
        "candidate_storage": "transient in-memory policies; no candidate checkpoint retained",
        "upstream_commit": commit,
        "parent_epoch": SELECTED_EPOCH,
        "parent_checkpoint": str(source_checkpoint),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_metrics": PARENT_METRICS,
        "preservation_gates": PRESERVATION_GATES,
        "m55_gate_sha256": M55_GATE_SHA256,
        "m55_profile": str(args.m55_profile.resolve()),
        "m55_profile_sha256": M55_PROFILE_SHA256,
        "m55_audit_sha256": M55_AUDIT_SHA256,
        "m56b_manifest": str(args.m56b_manifest.resolve()),
        "m56b_manifest_sha256": M56B_MANIFEST_SHA256,
        "m56b_smoke": str(args.m56b_smoke.resolve()),
        "m56b_smoke_sha256": M56B_SMOKE_SHA256,
        "m56b_candidate_checkpoint_sha256": M56B_CANDIDATE_SHA256,
        "m56b_failed_depth_max_abs": M56B_FAILED_DEPTH_MAX_ABS,
        "grouping_policy": {
            "operator_families": ["Conv2d", "Linear"],
            "parameter_scope": "direct floating-point weights and biases",
            "alias_consistency": "every alias of one shared parameter belongs to one group",
            "group_order": list(GROUP_ORDER),
            "prediction_head_roots": list(PREDICTION_HEAD_ROOTS),
            "m56b_fp32_head_roots": list(M56B_FP32_HEAD_ROOTS),
            "matrix": "two references, every nonempty singleton, every nonempty complement",
        },
        "group_inventory": inventory,
        "group_policy": group_policy,
        "m56b_fp32_geometry_parameter_state_names": sorted(m56b_fp32_aliases),
        "m56b_fp32_geometry_parameter_state_names_sha256": names_sha256(
            m56b_fp32_aliases
        ),
        "candidate_specs": candidates,
        "candidate_count": len(candidates),
        "total_parameter_bytes": total_parameter_bytes,
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": sha256_file(runtime_config),
        "dataset_root": str(dataset),
        "split_protocol": "chen_3712_3769",
        "sample_id": "000001",
        "parity_thresholds": "identical to M56 and M56b",
        "preparation_gate_results": preparation_gates,
        "diagnostic_authorized": all(preparation_gates.values()),
        "full_evaluation_authorized": False,
        "offline_compression_candidate_selected": False,
        "direct_coreml_conversion_authorized": False,
        "product_safety_qualified": False,
        "expected_artifacts": [
            "m56c_group_sensitivity.json",
            "m56c_group_sensitivity.csv",
            "colab_logs/m56c_prepare.log",
            "colab_logs/m56c_group_sensitivity.log",
        ],
    }
    manifest_path = output / "m56c_group_sensitivity_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not manifest["diagnostic_authorized"]:
        raise RuntimeError("M56c preparation failed; do not run the CUDA diagnostic")


if __name__ == "__main__":
    main()
