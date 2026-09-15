from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.evaluate_monodgp_m56_fp16_storage import (
        metric_value,
        prediction_tree_sha256,
        preservation_gate_results,
        run_logged,
    )
    from scripts.prepare_monodgp_m55_feasibility import PRESERVATION_GATES, VAL_SPLIT_SHA256
    from scripts.prepare_monodgp_m56_fp16_storage import sha256_file
    from scripts.prepare_monodgp_m57_deformable_attention import (
        EXPECTED_M56D_METRICS,
        EXPECTED_MODULES,
        M56D_CANDIDATE_SHA256,
        M56D_MANIFEST_SHA256,
        M56D_POLICY_ID,
        M56D_RUNTIME_CONFIG_SHA256,
        M57_PATCHED_SOURCE_SHA256,
        M57_POLICY_ID,
        PINNED_COMMIT,
        PORTABLE_ENV,
        same_numbers,
    )
except ModuleNotFoundError:
    from evaluate_monodgp_m56_fp16_storage import (
        metric_value,
        prediction_tree_sha256,
        preservation_gate_results,
        run_logged,
    )
    from prepare_monodgp_m55_feasibility import PRESERVATION_GATES, VAL_SPLIT_SHA256
    from prepare_monodgp_m56_fp16_storage import sha256_file
    from prepare_monodgp_m57_deformable_attention import (
        EXPECTED_M56D_METRICS,
        EXPECTED_MODULES,
        M56D_CANDIDATE_SHA256,
        M56D_MANIFEST_SHA256,
        M56D_POLICY_ID,
        M56D_RUNTIME_CONFIG_SHA256,
        M57_PATCHED_SOURCE_SHA256,
        M57_POLICY_ID,
        PINNED_COMMIT,
        PORTABLE_ENV,
        same_numbers,
    )


M57_MANIFEST_SHA256 = "7c4757798dfff4d95053912730faff4c5d7a4626ff41a539ad87b2f9193eb313"
M57_SMOKE_SHA256 = "2a96cffd8e1e6b77f2c547c2b94dca4bde2e72bf4185d853ee7bca71ed28b3b0"
MODEL_NAME = "monodgp_m57_portable_eval"
RESULT_PREFIX = "m57_portable_attention"
PREDICTION_FILES = 3769


def changed_config_paths(before: Any, after: Any, prefix: str = "") -> set[str]:
    if isinstance(before, dict) and isinstance(after, dict):
        if set(before) != set(after):
            return {prefix or "<root>"}
        changed: set[str] = set()
        for key in before:
            path = f"{prefix}.{key}" if prefix else str(key)
            changed |= changed_config_paths(before[key], after[key], path)
        return changed
    return set() if before == after else {prefix or "<root>"}


def validate_reviewed_pair(manifest_path: Path, smoke_path: Path) -> tuple[dict, dict]:
    for path, expected in (
        (manifest_path, M57_MANIFEST_SHA256),
        (smoke_path, M57_SMOKE_SHA256),
    ):
        if not path.is_file() or sha256_file(path) != expected:
            raise RuntimeError(f"Reviewed M57 evidence is missing or changed: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    policy = manifest.get("replacement_policy", {})
    if (
        manifest.get("complete") is not True
        or manifest.get("smoke_authorized") is not True
        or not manifest.get("preparation_gate_results")
        or not all(manifest["preparation_gate_results"].values())
        or manifest.get("upstream_commit") != PINNED_COMMIT
        or manifest.get("checkpoint_sha256") != M56D_CANDIDATE_SHA256
        or manifest.get("runtime_config_sha256") != M56D_RUNTIME_CONFIG_SHA256
        or manifest.get("patched_source_sha256") != M57_PATCHED_SOURCE_SHA256
        or manifest.get("m56d_manifest_sha256") != M56D_MANIFEST_SHA256
        or not same_numbers(manifest.get("m56d_metrics", {}), EXPECTED_M56D_METRICS)
        or policy.get("policy_id") != M57_POLICY_ID
        or policy.get("target_modules") != list(EXPECTED_MODULES)
        or manifest.get("portable_environment_variable") != f"{PORTABLE_ENV}=1"
        or manifest.get("native_path_default") is not True
        or manifest.get("portable_path_opt_in") is not True
        or manifest.get("training_performed") is not False
        or manifest.get("weights_changed") is not False
        or manifest.get("direct_coreml_conversion_authorized") is not False
        or manifest.get("product_safety_qualified") is not False
        or smoke.get("complete") is not True
        or smoke.get("manifest_sha256") != M57_MANIFEST_SHA256
        or smoke.get("checkpoint_sha256") != M56D_CANDIDATE_SHA256
        or smoke.get("native_cuda_calls") != 9
        or smoke.get("portable_native_cuda_calls") != 0
        or smoke.get("target_modules") != list(EXPECTED_MODULES)
        or not smoke.get("gate_results")
        or not all(smoke["gate_results"].values())
        or not smoke.get("module_parity")
        or not all(row.get("passed") is True for row in smoke["module_parity"].values())
        or not smoke.get("trace_signatures")
        or not all(row.get("passed") is True for row in smoke["trace_signatures"].values())
        or not smoke.get("raw_output_parity")
        or not all(row.get("passed") is True for row in smoke["raw_output_parity"].values())
        or smoke.get("all_smoke_gates_passed") is not True
        or smoke.get("full_evaluation_authorized") is not True
        or smoke.get("training_performed") is not False
        or smoke.get("direct_coreml_conversion_authorized") is not False
        or smoke.get("product_safety_qualified") is not False
    ):
        raise RuntimeError("Exact reviewed M57 pair did not authorize complete evaluation")
    return manifest, smoke


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Evaluate exact reviewed M57 portable attention on the full Chen validation split."
    )
    parser.add_argument("--mobile-repo", type=Path, required=True)
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--smoke", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--product-config", default="configs/kitti_mobileadas3d_s1.yaml")
    parser.add_argument("--profile", default="colab_drive")
    args = parser.parse_args()

    mobile = args.mobile_repo.resolve()
    monodgp = args.monodgp_repo.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    manifest_path = args.manifest.resolve()
    smoke_path = args.smoke.resolve()
    manifest, smoke = validate_reviewed_pair(manifest_path, smoke_path)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=monodgp,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    attention_source = monodgp / "lib/models/monodgp/ops/modules/ms_deform_attn.py"
    checkpoint = Path(manifest["checkpoint"]).resolve()
    frozen_config = Path(manifest["runtime_config"]).resolve()
    if (
        commit != PINNED_COMMIT
        or not attention_source.is_file()
        or sha256_file(attention_source) != M57_PATCHED_SOURCE_SHA256
    ):
        raise RuntimeError("Pinned MonoDGP source or M57 operator patch changed")
    if not checkpoint.is_file() or sha256_file(checkpoint) != M56D_CANDIDATE_SHA256:
        raise RuntimeError("Selected M56d checkpoint is missing or changed")
    if (
        not frozen_config.is_file()
        or sha256_file(frozen_config) != M56D_RUNTIME_CONFIG_SHA256
    ):
        raise RuntimeError("Frozen M56d runtime YAML is missing or changed")
    m56d_manifest_path = Path(manifest["m56d_manifest"]).resolve()
    if (
        not m56d_manifest_path.is_file()
        or sha256_file(m56d_manifest_path) != M56D_MANIFEST_SHA256
    ):
        raise RuntimeError("Selected M56d manifest is missing or changed")
    m56d_manifest = json.loads(m56d_manifest_path.read_text(encoding="utf-8"))
    if (
        m56d_manifest.get("compression_policy", {}).get("policy_id") != M56D_POLICY_ID
        or m56d_manifest.get("candidate_checkpoint_sha256") != M56D_CANDIDATE_SHA256
    ):
        raise RuntimeError("M57 no longer uses the selected M56d policy")

    val_split = args.split_dir.resolve() / "val.txt"
    dataset = args.dataset_root.resolve()
    if (
        not val_split.is_file()
        or sha256_file(val_split) != VAL_SPLIT_SHA256
        or dataset != Path(manifest["dataset_root"]).resolve()
        or not (dataset / "training/image_2").is_dir()
        or not (dataset / "training/label_2").is_dir()
        or not (dataset / "training/calib").is_dir()
        or not (dataset / "ImageSets/val.txt").is_file()
        or sha256_file(dataset / "ImageSets/val.txt") != VAL_SPLIT_SHA256
    ):
        raise RuntimeError("M57 dataset or Chen validation split changed")

    base = yaml.safe_load(frozen_config.read_text(encoding="utf-8"))
    if (
        base["dataset"]["root_dir"] != str(dataset)
        or base["dataset"]["test_split"] != "val"
        or base["trainer"]["max_epoch"] != 0
        or base["tester"]["checkpoint"] != 100
        or float(base["tester"]["threshold"]) != 0.001
        or base["tester"]["topk"] != 50
    ):
        raise RuntimeError("Frozen M56d runtime config changed the validation protocol")
    runtime = copy.deepcopy(base)
    runtime["model_name"] = MODEL_NAME
    runtime["trainer"]["save_path"] = os.path.relpath(output, monodgp)
    if changed_config_paths(base, runtime) != {"model_name", "trainer.save_path"}:
        raise RuntimeError("M57 isolated runtime config changed more than its output paths")
    runtime_path = output / f"{MODEL_NAME}.yaml"
    runtime_path.write_text(yaml.safe_dump(runtime, sort_keys=False), encoding="utf-8")
    run_dir = output / MODEL_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    isolated_checkpoint = run_dir / "checkpoint_epoch_100.pth"
    if isolated_checkpoint.is_file():
        if sha256_file(isolated_checkpoint) != M56D_CANDIDATE_SHA256:
            raise RuntimeError("Existing isolated checkpoint has different bytes")
    else:
        shutil.copy2(checkpoint, isolated_checkpoint)
        if sha256_file(isolated_checkpoint) != M56D_CANDIDATE_SHA256:
            raise RuntimeError("Isolated checkpoint copy changed bytes")

    prediction_dir = run_dir / "outputs/data"
    prediction_manifest_path = output / f"{RESULT_PREFIX}_prediction_manifest.json"
    reuse = False
    files = sorted(prediction_dir.glob("*.txt")) if prediction_dir.is_dir() else []
    if len(files) == PREDICTION_FILES and prediction_manifest_path.is_file():
        prior = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
        reuse = (
            prior.get("complete") is True
            and prior.get("checkpoint_sha256") == M56D_CANDIDATE_SHA256
            and prior.get("m57_manifest_sha256") == M57_MANIFEST_SHA256
            and prior.get("m57_smoke_sha256") == M57_SMOKE_SHA256
            and prior.get("portable_environment_variable") == f"{PORTABLE_ENV}=1"
            and prior.get("prediction_files") == PREDICTION_FILES
            and prior.get("prediction_tree_sha256") == prediction_tree_sha256(prediction_dir)
        )
    if reuse:
        print("Reusing exact checkpoint- and mode-bound M57 predictions.", flush=True)
    else:
        shutil.rmtree(prediction_dir, ignore_errors=True)
        previous_mode = os.environ.get(PORTABLE_ENV)
        os.environ[PORTABLE_ENV] = "1"
        try:
            run_logged(
                [
                    sys.executable,
                    "-u",
                    "tools/train_val.py",
                    "--config",
                    runtime_path,
                    "--evaluate_only",
                ],
                monodgp,
                logs / "m57_portable_inference.log",
            )
        finally:
            if previous_mode is None:
                os.environ.pop(PORTABLE_ENV, None)
            else:
                os.environ[PORTABLE_ENV] = previous_mode
        files = sorted(prediction_dir.glob("*.txt"))
        if len(files) != PREDICTION_FILES:
            raise RuntimeError(
                f"Portable path produced {len(files)}/{PREDICTION_FILES} predictions"
            )
        prediction_manifest = {
            "schema_version": 1,
            "complete": True,
            "checkpoint_sha256": M56D_CANDIDATE_SHA256,
            "m57_manifest_sha256": M57_MANIFEST_SHA256,
            "m57_smoke_sha256": M57_SMOKE_SHA256,
            "portable_environment_variable": f"{PORTABLE_ENV}=1",
            "prediction_files": len(files),
            "prediction_tree_sha256": prediction_tree_sha256(prediction_dir),
        }
        prediction_manifest_path.write_text(
            json.dumps(prediction_manifest, indent=2) + "\n", encoding="utf-8"
        )

    product_dir = output / "product_ap"
    nearby_dir = output / "nearby_geometry"
    misses_dir = output / "pedestrian_false_negatives"
    run_logged(
        [
            sys.executable,
            "-u",
            "scripts/evaluate_kitti_prediction_dir.py",
            "--config",
            args.product_config,
            "--profile",
            args.profile,
            "--dataset-root",
            dataset,
            "--split-dir",
            args.split_dir,
            "--prediction-dir",
            prediction_dir,
            "--split",
            "val",
            "--classes",
            "Vehicle",
            "Pedestrian",
            "--source-name",
            "MonoDGP_M57_portable_attention",
            "--output-dir",
            product_dir,
        ],
        mobile,
        logs / "m57_product_ap.log",
    )
    run_logged(
        [
            sys.executable,
            "-u",
            "scripts/audit_product_prediction_geometry.py",
            "--dataset-root",
            dataset,
            "--split-file",
            val_split,
            "--prediction-dir",
            prediction_dir,
            "--output-dir",
            nearby_dir,
            "--checkpoint",
            isolated_checkpoint,
            "--expected-checkpoint-sha256",
            M56D_CANDIDATE_SHA256,
            "--expected-images",
            str(PREDICTION_FILES),
            "--score-threshold",
            "0.001",
            "--match-iou-threshold",
            "0.5",
        ],
        mobile,
        logs / "m57_nearby_geometry.log",
    )
    run_logged(
        [
            sys.executable,
            "-u",
            "scripts/diagnose_a2_pedestrian_false_negatives.py",
            "--dataset-root",
            dataset,
            "--split-file",
            val_split,
            "--prediction-dir",
            prediction_dir,
            "--output-dir",
            misses_dir,
            "--checkpoint",
            isolated_checkpoint,
            "--expected-checkpoint-sha256",
            M56D_CANDIDATE_SHA256,
            "--expected-images",
            str(PREDICTION_FILES),
            "--score-threshold",
            "0.001",
            "--iou-threshold",
            "0.5",
            "--weak-iou-threshold",
            "0.1",
        ],
        mobile,
        logs / "m57_pedestrian_false_negatives.log",
    )

    ap = json.loads((product_dir / "kitti_r40_summary.json").read_text(encoding="utf-8"))
    nearby = json.loads(
        (nearby_dir / "nearby_geometry_summary.json").read_text(encoding="utf-8")
    )
    misses = json.loads(
        (misses_dir / "a2_pedestrian_false_negative_summary.json").read_text(
            encoding="utf-8"
        )
    )
    vehicle_3d = metric_value(ap, "3d", "Vehicle")
    pedestrian_3d = metric_value(ap, "3d", "Pedestrian")
    metrics = {
        "vehicle_3d_moderate": vehicle_3d,
        "pedestrian_3d_moderate": pedestrian_3d,
        "mean_3d_moderate": (vehicle_3d + pedestrian_3d) / 2.0,
        "vehicle_bev_moderate": metric_value(ap, "bev", "Vehicle"),
        "pedestrian_bev_moderate": metric_value(ap, "bev", "Pedestrian"),
        "vehicle_near_recall": float(nearby["classes"]["Vehicle"]["near_recall"]),
        "pedestrian_near_recall": float(
            nearby["classes"]["Pedestrian"]["near_recall"]
        ),
        "pedestrian_localization_failure_rate": float(
            misses["near_failure_rates"].get("localization_failure", 0.0)
        ),
    }
    gates = preservation_gate_results(metrics, len(files))
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M57 portable deformable-attention complete validation",
        "training_performed": False,
        "weights_changed": False,
        "checkpoint_sha256": M56D_CANDIDATE_SHA256,
        "m57_manifest_sha256": M57_MANIFEST_SHA256,
        "m57_smoke_sha256": M57_SMOKE_SHA256,
        "portable_environment_variable": f"{PORTABLE_ENV}=1",
        "runtime_config": str(runtime_path),
        "runtime_config_changed_paths": sorted(changed_config_paths(base, runtime)),
        "prediction_files": len(files),
        "prediction_tree_sha256": prediction_tree_sha256(prediction_dir),
        "m56d_metrics": EXPECTED_M56D_METRICS,
        "portable_metrics": metrics,
        "portable_minus_m56d": {
            key: metrics[key] - EXPECTED_M56D_METRICS[key]
            for key in EXPECTED_M56D_METRICS
        },
        "preservation_gates": PRESERVATION_GATES,
        "preservation_gate_results": gates,
        "all_preservation_gates_passed": passed,
        "portable_operator_candidate_selected": passed,
        "next_coreml_conversion_gate_authorized": passed,
        "direct_coreml_conversion_authorized": False,
        "deployment_authorized": False,
        "product_safety_qualified": False,
        "native_latency": smoke["native_latency"],
        "portable_latency": smoke["portable_latency"],
        "native_memory": smoke["native_memory"],
        "portable_memory": smoke["portable_memory"],
        "runtime_comparison": smoke["runtime_comparison"],
        "artifacts": {
            "manifest": str(manifest_path),
            "smoke": str(smoke_path),
            "prediction_manifest": str(prediction_manifest_path),
            "product_ap": str(product_dir / "kitti_r40_summary.json"),
            "nearby_geometry": str(nearby_dir / "nearby_geometry_summary.json"),
            "pedestrian_false_negatives": str(
                misses_dir / "a2_pedestrian_false_negative_summary.json"
            ),
        },
    }
    gate_path = output / f"{RESULT_PREFIX}_gate.json"
    gate_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    comparison_path = output / f"{RESULT_PREFIX}_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "metric",
                "m56d",
                "requirement",
                "portable",
                "delta",
                "passed",
            ],
        )
        writer.writeheader()
        for key in EXPECTED_M56D_METRICS:
            requirement_key = (
                "pedestrian_localization_failure_rate_max"
                if key == "pedestrian_localization_failure_rate"
                else key
            )
            writer.writerow(
                {
                    "metric": key,
                    "m56d": EXPECTED_M56D_METRICS[key],
                    "requirement": PRESERVATION_GATES[requirement_key],
                    "portable": metrics[key],
                    "delta": metrics[key] - EXPECTED_M56D_METRICS[key],
                    "passed": gates[key],
                }
            )
        writer.writerow(
            {
                "metric": "prediction_files",
                "m56d": PREDICTION_FILES,
                "requirement": PREDICTION_FILES,
                "portable": len(files),
                "delta": len(files) - PREDICTION_FILES,
                "passed": gates["prediction_files"],
            }
        )
    print(json.dumps(report, indent=2))
    print("Comparison CSV:", comparison_path)
    if not passed:
        raise RuntimeError("M57 portable path failed the frozen preservation gate")


if __name__ == "__main__":
    main()
