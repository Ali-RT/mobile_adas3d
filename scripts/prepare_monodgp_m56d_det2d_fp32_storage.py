from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

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
    from scripts.prepare_monodgp_m56c_group_sensitivity import (
        GROUP_ORDER,
        collect_grouped_storage_policy,
        names_sha256,
    )
    from scripts.smoke_test_monodgp_m56_fp16_storage import PARITY_LIMITS
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
    from prepare_monodgp_m56c_group_sensitivity import (
        GROUP_ORDER,
        collect_grouped_storage_policy,
        names_sha256,
    )
    from smoke_test_monodgp_m56_fp16_storage import PARITY_LIMITS


SELECTED_EPOCH = 100
RUN_NAME = "monodgp_m56d_det2d_fp32_storage"
POLICY_ID = "m56d_det2d_transformer_fp32"
FP32_GROUP = "det2d_transformer"
M56C_MANIFEST_SHA256 = "0c501fa2e0f13e8ad952e492ef44d45a93fa72d3655e15ee4dfca5ae77417796"
M56C_REPORT_SHA256 = "7f1ca25ec2270642c508bb36a8e109bdcf8cdf09d8b05d5749a23a0129321ba1"
M56C_CSV_SHA256 = "8d9eea64c0bf67eb7f4b597e76344d67ceb5e5a392074259050fa51ae32c5abe"
M56C_SINGLETON_DEPTH_MAX_ABS = 0.5688076019287109
M56C_SELECTED_DEPTH_MAX_ABS = 0.03000640869140625
EXPECTED_FP32_UNIQUE_PARAMETERS = 80
EXPECTED_FP32_UNIQUE_BYTES = 9_476_104
EXPECTED_FP32_STATE_ALIASES = 80
EXPECTED_FP16_UNIQUE_PARAMETERS = 247
EXPECTED_FP16_UNIQUE_BYTES = 146_855_348
EXPECTED_FP16_STATE_ALIASES = 295
EXPECTED_FP16_NAMES_SHA256 = "6afced8a5d40e0829f8b5b5cc06c2e55b1161a2bdfe22a7d17a61deb5de10255"
EXPECTED_PROJECTED_SIZE_RATIO = 0.5646268913923198


def _candidate(report: dict, name: str) -> dict:
    matches = [
        row for row in report.get("candidate_results", [])
        if row.get("spec", {}).get("name") == name
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one M56c candidate named {name}, found {len(matches)}")
    return matches[0]


def validate_m56c_result_records(manifest: dict, report: dict, rows: list[dict]) -> None:
    selected = _candidate(report, "all_except_det2d_transformer")
    singleton = _candidate(report, "only_det2d_transformer")
    selected_spec = selected.get("spec", {})
    selected_parity = selected.get("parity", {})
    singleton_parity = singleton.get("parity", {})
    expected_fp16_groups = [
        name
        for name in GROUP_ORDER
        if name != FP32_GROUP
        and manifest.get("group_policy", {}).get(name, {}).get("unique_parameters", 0)
    ]
    selected_csv = [row for row in rows if row.get("candidate") == selected_spec.get("name")]
    singleton_csv = [row for row in rows if row.get("candidate") == "only_det2d_transformer"]
    group = manifest.get("group_policy", {}).get(FP32_GROUP, {})
    analysis = report.get("analysis", {})
    if (
        manifest.get("complete") is not True
        or manifest.get("experiment")
        != "M56c grouped FP16-storage sensitivity isolation"
        or manifest.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or manifest.get("diagnostic_authorized") is not True
        or not manifest.get("preparation_gate_results")
        or not all(manifest["preparation_gate_results"].values())
        or manifest.get("candidate_count") != 16
        or manifest.get("full_evaluation_authorized") is not False
        or manifest.get("offline_compression_candidate_selected") is not False
        or report.get("complete") is not True
        or report.get("manifest_sha256") != M56C_MANIFEST_SHA256
        or report.get("source_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or report.get("candidate_count") != 16
        or report.get("all_diagnostic_gates_passed") is not True
        or not report.get("diagnostic_gate_results")
        or not all(report["diagnostic_gate_results"].values())
        or report.get("full_evaluation_authorized") is not False
        or report.get("offline_compression_candidate_selected") is not False
        or report.get("training_performed") is not False
        or len(rows) != 16
        or group.get("unique_parameters") != EXPECTED_FP32_UNIQUE_PARAMETERS
        or group.get("unique_parameter_bytes") != EXPECTED_FP32_UNIQUE_BYTES
        or group.get("state_aliases") != EXPECTED_FP32_STATE_ALIASES
        or singleton.get("spec", {}).get("mode") != "singleton"
        or singleton.get("spec", {}).get("fp16_groups") != [FP32_GROUP]
        or singleton_parity.get("all_parity_limits_passed") is not False
        or singleton_parity.get("failed_output_families") != ["pred_depth"]
        or abs(
            float(singleton_parity.get("parity_summary", {}).get("pred_depth", {}).get("max_abs", -1.0))
            - M56C_SINGLETON_DEPTH_MAX_ABS
        ) > 1e-12
        or selected_spec.get("mode") != "complement"
        or selected_spec.get("fp32_held_group") != FP32_GROUP
        or selected_spec.get("fp16_groups") != expected_fp16_groups
        or selected_spec.get("fp16_unique_parameters") != EXPECTED_FP16_UNIQUE_PARAMETERS
        or selected_spec.get("fp16_unique_parameter_bytes") != EXPECTED_FP16_UNIQUE_BYTES
        or selected_spec.get("fp16_parameter_state_name_count") != EXPECTED_FP16_STATE_ALIASES
        or selected_spec.get("fp16_parameter_state_names_sha256")
        != EXPECTED_FP16_NAMES_SHA256
        or abs(
            float(selected_spec.get("projected_parameter_size_ratio", -1.0))
            - EXPECTED_PROJECTED_SIZE_RATIO
        ) > 1e-12
        or selected_parity.get("all_parity_limits_passed") is not True
        or selected_parity.get("failed_output_families") != []
        or not selected.get("integrity")
        or not all(selected["integrity"].values())
        or abs(
            float(selected_parity.get("parity_summary", {}).get("pred_depth", {}).get("max_abs", -1.0))
            - M56C_SELECTED_DEPTH_MAX_ABS
        ) > 1e-12
        or any(
            item.get("passed") is not True
            or abs(float(item.get("limit_max_abs", -1.0)) - PARITY_LIMITS[name]) > 1e-12
            for name, item in selected_parity.get("parity_summary", {}).items()
        )
        or set(selected_parity.get("parity_summary", {})) != set(PARITY_LIMITS)
        or analysis.get("singleton_failures") != [FP32_GROUP]
        or FP32_GROUP not in analysis.get("complement_rescues", [])
        or len(selected_csv) != 1
        or len(singleton_csv) != 1
        or selected_csv[0].get("all_parity_limits_passed") != "True"
        or selected_csv[0].get("failed_output_families") != ""
        or abs(float(selected_csv[0].get("pred_depth_max_abs", -1.0)) - M56C_SELECTED_DEPTH_MAX_ABS) > 1e-12
        or singleton_csv[0].get("all_parity_limits_passed") != "False"
        or singleton_csv[0].get("failed_output_families") != "pred_depth"
    ):
        raise RuntimeError("M56d is not bound to the exact reviewed M56c result")


def validate_m56c_evidence(
    manifest_path: Path, report_path: Path, csv_path: Path
) -> tuple[dict, dict, list[dict]]:
    expected = {
        manifest_path: M56C_MANIFEST_SHA256,
        report_path: M56C_REPORT_SHA256,
        csv_path: M56C_CSV_SHA256,
    }
    for path, expected_hash in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected_hash:
            raise RuntimeError(f"M56c SHA-256 mismatch for {path}: {actual} != {expected_hash}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    validate_m56c_result_records(manifest, report, rows)
    return manifest, report, rows


def collect_det2d_fp32_storage_policy(model, torch_module) -> tuple[dict, dict, dict]:
    grouped, _, grouped_inventory = collect_grouped_storage_policy(model, torch_module)
    fp32_aliases = dict(grouped[FP32_GROUP])
    fp16_aliases = {
        alias: family
        for group_name, aliases in grouped.items()
        if group_name != FP32_GROUP
        for alias, family in aliases.items()
    }
    fp32_row = grouped_inventory["groups"][FP32_GROUP]
    fp16_rows = [
        grouped_inventory["groups"][name]
        for name in GROUP_ORDER
        if name != FP32_GROUP
    ]
    inventory = {
        "all_eligible_unique_parameters": grouped_inventory["all_eligible_unique_parameters"],
        "all_eligible_unique_parameter_bytes": grouped_inventory["all_eligible_unique_parameter_bytes"],
        "all_eligible_state_aliases": grouped_inventory["all_eligible_state_aliases"],
        "fp16_unique_parameters": sum(row["unique_parameters"] for row in fp16_rows),
        "fp16_unique_parameter_bytes": sum(row["unique_parameter_bytes"] for row in fp16_rows),
        "fp16_state_aliases": len(fp16_aliases),
        "fp32_preserved_unique_parameters": fp32_row["unique_parameters"],
        "fp32_preserved_unique_parameter_bytes": fp32_row["unique_parameter_bytes"],
        "fp32_preserved_state_aliases": len(fp32_aliases),
        "fp32_preserved_group": FP32_GROUP,
        "group_inventory": grouped_inventory["groups"],
    }
    return fp16_aliases, fp32_aliases, inventory


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(
        description="Prepare M56d with the complete 2D transformer retained in FP32 storage."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--m54-selection", type=Path, required=True)
    parser.add_argument("--m55-gate", type=Path, required=True)
    parser.add_argument("--m55-profile", type=Path, required=True)
    parser.add_argument("--m55-audit", type=Path, required=True)
    parser.add_argument("--m56c-manifest", type=Path, required=True)
    parser.add_argument("--m56c-report", type=Path, required=True)
    parser.add_argument("--m56c-csv", type=Path, required=True)
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
    m56c_manifest, _, _ = validate_m56c_evidence(
        args.m56c_manifest.resolve(),
        args.m56c_report.resolve(),
        args.m56c_csv.resolve(),
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

    config = yaml.safe_load((repo / "configs/monodgp.yaml").read_text(encoding="utf-8"))
    model, _ = build_model(config["model"])
    source_payload = load_checkpoint_safely(source_checkpoint, torch.device("cpu"))
    if int(source_payload.get("epoch", -1)) != SELECTED_EPOCH:
        raise RuntimeError("M54 checkpoint payload epoch changed")
    source_state = source_payload.get("model_state")
    if not isinstance(source_state, dict):
        raise RuntimeError("M54 checkpoint has no model_state")
    model.load_state_dict(source_state, strict=True)

    fp16_aliases, fp32_aliases, policy_inventory = collect_det2d_fp32_storage_policy(
        model, torch
    )
    m56c_fp32_names = set(
        m56c_manifest["group_policy"][FP32_GROUP]["parameter_state_names"]
    )
    selected_spec = next(
        item for item in m56c_manifest["candidate_specs"]
        if item["name"] == "all_except_det2d_transformer"
    )
    all_aliases = set(fp16_aliases) | set(fp32_aliases)
    if policy_inventory["all_eligible_unique_parameter_bytes"] != EXPECTED_ELIGIBLE_PARAMETER_BYTES:
        raise RuntimeError("M56d eligible bytes do not reproduce M55")
    if (
        policy_inventory["fp16_unique_parameters"] != EXPECTED_FP16_UNIQUE_PARAMETERS
        or policy_inventory["fp16_unique_parameter_bytes"] != EXPECTED_FP16_UNIQUE_BYTES
        or policy_inventory["fp16_state_aliases"] != EXPECTED_FP16_STATE_ALIASES
        or names_sha256(set(fp16_aliases)) != EXPECTED_FP16_NAMES_SHA256
        or policy_inventory["fp32_preserved_unique_parameters"] != EXPECTED_FP32_UNIQUE_PARAMETERS
        or policy_inventory["fp32_preserved_unique_parameter_bytes"] != EXPECTED_FP32_UNIQUE_BYTES
        or policy_inventory["fp32_preserved_state_aliases"] != EXPECTED_FP32_STATE_ALIASES
        or set(fp32_aliases) != m56c_fp32_names
        or selected_spec["fp16_parameter_state_names_sha256"] != names_sha256(set(fp16_aliases))
    ):
        raise RuntimeError("M56d policy does not reproduce the reviewed M56c winner")
    if all_aliases - set(source_state):
        raise RuntimeError("M56d parameter aliases are missing from the source state")
    family_bytes = {"Conv2d": 0, "Linear": 0}
    for row in policy_inventory["group_inventory"].values():
        for family, value in row["unique_bytes_by_family"].items():
            family_bytes[family] += value
    if family_bytes != audit["eligible_parameter_bytes_by_family"]:
        raise RuntimeError("M56d eligible family bytes do not reproduce M55")

    baseline_state, candidate_state, storage_inventory = compress_state_dict(
        source_state, fp16_aliases, torch
    )
    run_dir = output / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline_checkpoint = output / "m56d_parent_model_only_fp32.pth"
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
    runtime_config = repo / "configs/monodgp_m56d_det2d_fp32_storage.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    baseline_size = baseline_checkpoint.stat().st_size
    candidate_size = candidate_checkpoint.stat().st_size
    size_ratio = candidate_size / baseline_size
    preparation_gates = {
        "m55_authorized": True,
        "m56c_diagnostic_verified": True,
        "m56c_winner_reproduced": True,
        "parent_checkpoint_sha256": sha256_file(source_checkpoint) == PARENT_CHECKPOINT_SHA256,
        "eligible_unique_bytes_reproduced": policy_inventory["all_eligible_unique_parameter_bytes"] == EXPECTED_ELIGIBLE_PARAMETER_BYTES,
        "eligible_family_bytes_reproduced": family_bytes == audit["eligible_parameter_bytes_by_family"],
        "all_eligible_parameters_classified_once": policy_inventory["fp16_unique_parameters"] + policy_inventory["fp32_preserved_unique_parameters"] == policy_inventory["all_eligible_unique_parameters"],
        "det2d_group_preserved_exactly": set(fp32_aliases) == m56c_fp32_names,
        "fp16_policy_names_match_m56c": names_sha256(set(fp16_aliases)) == EXPECTED_FP16_NAMES_SHA256,
        "fp16_storage_halved": storage_inventory["eligible_parameter_bytes_fp16"] * 2 == storage_inventory["eligible_parameter_bytes_fp32"],
        "all_fp16_state_aliases_encoded": storage_inventory["eligible_parameter_tensors"] == len(fp16_aliases),
        "model_only_size_ratio_le_0_60": size_ratio <= MAX_MODEL_ONLY_SIZE_RATIO,
        "optimizer_state_removed_from_both_artifacts": True,
        "architecture_unchanged": actual_invariants == expected_invariants,
        "runtime_remains_fp32": True,
        "direct_coreml_unauthorized": True,
    }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M56d FP16 storage with complete det2d transformer retained in FP32",
        "training_performed": False,
        "graph_changed": False,
        "runtime_precision": "FP32 after checkpoint load",
        "storage_precision": "FP16 for eligible Conv2d/Linear parameters except the complete det2d_transformer group retained in FP32",
        "compression_policy": {
            "policy_id": POLICY_ID,
            "families": ["Conv2d", "Linear"],
            "parameter_scope": "direct floating-point parameters, including weight and bias",
            "fp16_groups": [
                name
                for name in GROUP_ORDER
                if name != FP32_GROUP
                and policy_inventory["group_inventory"][name]["unique_parameters"]
            ],
            "fp32_preserved_groups": [FP32_GROUP],
            "alias_consistency": "every state-dict alias of one shared parameter uses the same stored dtype",
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
        "m56c_manifest": str(args.m56c_manifest.resolve()),
        "m56c_manifest_sha256": M56C_MANIFEST_SHA256,
        "m56c_report": str(args.m56c_report.resolve()),
        "m56c_report_sha256": M56C_REPORT_SHA256,
        "m56c_csv": str(args.m56c_csv.resolve()),
        "m56c_csv_sha256": M56C_CSV_SHA256,
        "m56c_selected_policy": "all_except_det2d_transformer",
        "m56c_selected_depth_max_abs": M56C_SELECTED_DEPTH_MAX_ABS,
        "m56c_selected_projected_size_ratio": EXPECTED_PROJECTED_SIZE_RATIO,
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
        "fp16_parameter_state_names_sha256": names_sha256(set(fp16_aliases)),
        "fp32_preserved_parameter_state_names": sorted(fp32_aliases),
        "fp32_preserved_parameter_state_names_sha256": names_sha256(set(fp32_aliases)),
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
            "m56d_det2d_fp32_smoke.json",
            "m56d_det2d_fp32_gate.json",
            "m56d_det2d_fp32_comparison.csv",
        ],
    }
    manifest_path = output / "m56d_compression_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not manifest["smoke_authorized"]:
        raise RuntimeError("M56d preparation failed; do not run the CUDA smoke")


if __name__ == "__main__":
    main()
