from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from scripts.prepare_monodgp_m55_feasibility import (
        PARENT_CHECKPOINT_SHA256,
        PARENT_METRICS,
        PRESERVATION_GATES,
    )
    from scripts.prepare_monodgp_m56_fp16_storage import sha256_file
except ModuleNotFoundError:
    from prepare_monodgp_m55_feasibility import (
        PARENT_CHECKPOINT_SHA256,
        PARENT_METRICS,
        PRESERVATION_GATES,
    )
    from prepare_monodgp_m56_fp16_storage import sha256_file


def run_logged(command: list[str], cwd: Path, log_path: Path) -> None:
    command = [str(item) for item in command]
    print("+", " ".join(command), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail: list[str] = []
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            tail.append(line.rstrip())
            tail = tail[-100:]
        code = process.wait()
    if code:
        raise RuntimeError(f"Command exited {code}; log={log_path}\n" + "\n".join(tail))


def prediction_tree_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(directory.glob("*.txt")):
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def metric_value(summary: dict[str, Any], metric: str, class_name: str) -> float:
    matches = [
        row for row in summary["metrics"]
        if row["metric"] == metric
        and row["class_name"] == class_name
        and row["difficulty"] == "moderate"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one {metric}/{class_name}/moderate metric, found {len(matches)}"
        )
    return float(matches[0]["ap_r40"])


def preservation_gate_results(metrics: dict[str, float], prediction_files: int) -> dict[str, bool]:
    results = {
        key: metrics[key] >= threshold
        for key, threshold in PRESERVATION_GATES.items()
        if key not in {"pedestrian_localization_failure_rate_max", "prediction_files"}
    }
    results["pedestrian_localization_failure_rate"] = (
        metrics["pedestrian_localization_failure_rate"]
        <= PRESERVATION_GATES["pedestrian_localization_failure_rate_max"]
    )
    results["prediction_files"] = prediction_files == PRESERVATION_GATES["prediction_files"]
    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate M56 FP16 parameter storage against all frozen M54 gates."
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
    log_dir = output / "logs"

    manifest_path = args.manifest.resolve()
    smoke_path = args.smoke.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    if (
        manifest.get("complete") is not True
        or manifest.get("smoke_authorized") is not True
        or manifest.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or smoke.get("complete") is not True
        or smoke.get("all_smoke_gates_passed") is not True
        or smoke.get("full_evaluation_authorized") is not True
        or smoke.get("manifest_sha256") != sha256_file(manifest_path)
    ):
        raise RuntimeError("M56 full evaluation is not authorized by the exact smoke/manifest pair")

    candidate = Path(manifest["candidate_checkpoint"]).resolve()
    run_dir = Path(manifest["candidate_run_dir"]).resolve()
    runtime_config = Path(manifest["runtime_config"]).resolve()
    if not candidate.is_file() or sha256_file(candidate) != manifest["candidate_checkpoint_sha256"]:
        raise RuntimeError("M56 candidate checkpoint is missing or changed")
    if not runtime_config.is_file() or sha256_file(runtime_config) != manifest["runtime_config_sha256"]:
        raise RuntimeError("M56 runtime config is missing or changed")

    prediction_dir = run_dir / "outputs/data"
    prediction_manifest_path = output / "m56_prediction_manifest.json"
    reuse_predictions = False
    prediction_files = list(prediction_dir.glob("*.txt")) if prediction_dir.is_dir() else []
    if len(prediction_files) == 3769 and prediction_manifest_path.is_file():
        prediction_manifest = json.loads(prediction_manifest_path.read_text(encoding="utf-8"))
        current_tree = prediction_tree_sha256(prediction_dir)
        reuse_predictions = (
            prediction_manifest.get("complete") is True
            and prediction_manifest.get("candidate_checkpoint_sha256")
            == manifest["candidate_checkpoint_sha256"]
            and prediction_manifest.get("prediction_files") == 3769
            and prediction_manifest.get("prediction_tree_sha256") == current_tree
        )
    if reuse_predictions:
        print("Reusing the complete checkpoint-bound M56 prediction set.", flush=True)
    else:
        shutil.rmtree(prediction_dir, ignore_errors=True)
        run_logged(
            [sys.executable, "-u", "tools/train_val.py", "--config", runtime_config, "--evaluate_only"],
            monodgp,
            log_dir / "m56_candidate_inference.log",
        )
        prediction_files = list(prediction_dir.glob("*.txt"))
        if len(prediction_files) != 3769:
            raise RuntimeError(f"M56 produced {len(prediction_files)}/3769 prediction files")
        prediction_manifest = {
            "schema_version": 1,
            "complete": True,
            "candidate_checkpoint_sha256": manifest["candidate_checkpoint_sha256"],
            "prediction_files": len(prediction_files),
            "prediction_tree_sha256": prediction_tree_sha256(prediction_dir),
        }
        prediction_manifest_path.write_text(
            json.dumps(prediction_manifest, indent=2) + "\n", encoding="utf-8"
        )

    product_dir = output / "product_ap"
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
            args.dataset_root,
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
            "MonoDGP_M56_FP16_parameter_storage",
            "--output-dir",
            product_dir,
        ],
        mobile,
        log_dir / "m56_product_ap.log",
    )
    nearby_dir = output / "nearby_geometry"
    run_logged(
        [
            sys.executable,
            "-u",
            "scripts/audit_product_prediction_geometry.py",
            "--dataset-root",
            args.dataset_root,
            "--split-file",
            args.split_dir / "val.txt",
            "--prediction-dir",
            prediction_dir,
            "--output-dir",
            nearby_dir,
            "--checkpoint",
            candidate,
            "--expected-checkpoint-sha256",
            manifest["candidate_checkpoint_sha256"],
            "--expected-images",
            "3769",
            "--score-threshold",
            "0.001",
            "--match-iou-threshold",
            "0.5",
        ],
        mobile,
        log_dir / "m56_nearby_geometry.log",
    )
    miss_dir = output / "pedestrian_false_negatives"
    run_logged(
        [
            sys.executable,
            "-u",
            "scripts/diagnose_a2_pedestrian_false_negatives.py",
            "--dataset-root",
            args.dataset_root,
            "--split-file",
            args.split_dir / "val.txt",
            "--prediction-dir",
            prediction_dir,
            "--output-dir",
            miss_dir,
            "--checkpoint",
            candidate,
            "--expected-checkpoint-sha256",
            manifest["candidate_checkpoint_sha256"],
            "--expected-images",
            "3769",
            "--score-threshold",
            "0.001",
            "--iou-threshold",
            "0.5",
            "--weak-iou-threshold",
            "0.1",
        ],
        mobile,
        log_dir / "m56_pedestrian_false_negatives.log",
    )

    ap = json.loads((product_dir / "kitti_r40_summary.json").read_text(encoding="utf-8"))
    nearby = json.loads((nearby_dir / "nearby_geometry_summary.json").read_text(encoding="utf-8"))
    misses = json.loads(
        (miss_dir / "a2_pedestrian_false_negative_summary.json").read_text(encoding="utf-8")
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
        "pedestrian_near_recall": float(nearby["classes"]["Pedestrian"]["near_recall"]),
        "pedestrian_localization_failure_rate": float(
            misses["near_failure_rates"].get("localization_failure", 0.0)
        ),
    }
    prediction_count = len(list(prediction_dir.glob("*.txt")))
    gates = preservation_gate_results(metrics, prediction_count)
    all_passed = all(gates.values())

    report = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M56 M54 FP16 Conv2d/Linear parameter-storage sensitivity",
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "candidate_checkpoint": str(candidate),
        "candidate_checkpoint_sha256": manifest["candidate_checkpoint_sha256"],
        "compression_policy": manifest["compression_policy"],
        "parent_metrics": PARENT_METRICS,
        "candidate_metrics": metrics,
        "candidate_minus_parent": {
            key: metrics[key] - PARENT_METRICS[key] for key in PARENT_METRICS
        },
        "preservation_gates": PRESERVATION_GATES,
        "preservation_gate_results": gates,
        "all_preservation_gates_passed": all_passed,
        "prediction_files": prediction_count,
        "prediction_tree_sha256": prediction_tree_sha256(prediction_dir),
        "baseline_model_only_checkpoint_bytes": manifest[
            "baseline_model_only_checkpoint_bytes"
        ],
        "candidate_checkpoint_bytes": manifest["candidate_checkpoint_bytes"],
        "model_only_checkpoint_size_ratio": manifest["model_only_checkpoint_size_ratio"],
        "model_only_checkpoint_savings_bytes": manifest[
            "model_only_checkpoint_savings_bytes"
        ],
        "m55_baseline_latency": manifest["m55_baseline_latency"],
        "m56_candidate_latency": smoke["latency"],
        "runtime_comparison": smoke["m55_latency_comparison"],
        "offline_compression_candidate_selected": all_passed,
        "next_step_if_passed": (
            "M57 deformable-attention decomposition/replacement with raw-output parity"
            if all_passed
            else None
        ),
        "direct_coreml_conversion_authorized": False,
        "product_safety_qualified": False,
        "product_safety_note": (
            "Compression preservation cannot qualify safety; Pedestrian nearby recall "
            "must still reach 0.80 and external/device gates remain outstanding."
        ),
        "artifacts": {
            "manifest": str(manifest_path),
            "smoke": str(smoke_path),
            "prediction_manifest": str(prediction_manifest_path),
            "product_ap": str(product_dir / "kitti_r40_summary.json"),
            "nearby_geometry": str(nearby_dir / "nearby_geometry_summary.json"),
            "pedestrian_false_negatives": str(
                miss_dir / "a2_pedestrian_false_negative_summary.json"
            ),
        },
    }
    report_path = output / "m56_fp16_storage_gate.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    comparison_path = output / "m56_fp16_storage_comparison.csv"
    with comparison_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["metric", "parent", "requirement", "candidate", "passed"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for key in PARENT_METRICS:
            requirement_key = (
                "pedestrian_localization_failure_rate_max"
                if key == "pedestrian_localization_failure_rate"
                else key
            )
            writer.writerow(
                {
                    "metric": key,
                    "parent": PARENT_METRICS[key],
                    "requirement": PRESERVATION_GATES[requirement_key],
                    "candidate": metrics[key],
                    "passed": gates[key],
                }
            )
        writer.writerow(
            {
                "metric": "prediction_files",
                "parent": 3769,
                "requirement": 3769,
                "candidate": prediction_count,
                "passed": gates["prediction_files"],
            }
        )
    print(json.dumps(report, indent=2))
    print("Comparison CSV:", comparison_path)
    if not all_passed:
        raise RuntimeError("M56 preservation gate failed; compressed candidate is rejected")


if __name__ == "__main__":
    main()
