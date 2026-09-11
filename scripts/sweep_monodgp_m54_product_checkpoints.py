from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_monodgp_m54_adaptation import (
    OFFLINE_PRODUCT_TARGETS,
    R0_COMPARABLE_GATES,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_logged(command: list[str], cwd: Path, log_path: Path) -> None:
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
            tail = tail[-120:]
        code = process.wait()
    if code:
        raise RuntimeError(
            f"Command exited {code}; log={log_path}\n" + "\n".join(tail)
        )


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Select and qualify M54 with complete product AP and nearby gates."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--mobile-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--product-config", default="configs/kitti_mobileadas3d_s1.yaml")
    parser.add_argument("--profile", default="colab_drive")
    parser.add_argument("--score-threshold", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--epochs", type=int, nargs="*")
    args = parser.parse_args()

    monodgp = args.monodgp_repo.resolve()
    mobile = args.mobile_repo.resolve()
    output = args.output_dir.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("training_authorized") is not True:
        raise RuntimeError("M54 manifest is incomplete or training was not authorized")
    training_config = Path(manifest["runtime_config"]).resolve()
    run_dir = Path(manifest["run_dir"]).resolve()
    if not training_config.is_file() or not run_dir.is_dir():
        raise FileNotFoundError("M54 training config or run directory is missing")
    epochs = list(args.epochs or manifest["sweep_epochs"])

    output.mkdir(parents=True, exist_ok=True)
    sweep_command = [
        sys.executable,
        "-u",
        "scripts/sweep_monodetr_r0_product_checkpoints.py",
        "--monodetr-repo",
        str(monodgp),
        "--mobile-repo",
        str(mobile),
        "--training-config",
        str(training_config),
        "--run-dir",
        str(run_dir),
        "--dataset-root",
        str(args.dataset_root),
        "--split-dir",
        str(args.split_dir),
        "--output-dir",
        str(output),
        "--product-config",
        args.product_config,
        "--profile",
        args.profile,
        "--score-threshold",
        str(args.score_threshold),
        "--topk",
        str(args.topk),
        "--source-name-prefix",
        "MonoDGP_M54",
        "--epochs",
        *map(str, epochs),
    ]
    run_logged(sweep_command, mobile, output / "m54_checkpoint_sweep.log")

    base_selection = json.loads(
        (output / "r0_product_selection.json").read_text(encoding="utf-8")
    )
    for row in base_selection["ranked"]:
        row.pop("checkpoint_sha256", None)
    ranked = sorted(
        base_selection["ranked"],
        key=lambda row: (
            float(row["mean_3d_moderate"]),
            float(row["pedestrian_3d_moderate"]),
            float(row["vehicle_3d_moderate"]),
            float(row["mean_bev_moderate"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
    with (output / "m54_product_checkpoint_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)
    selected = ranked[0]
    selected_checkpoint = Path(selected["checkpoint"]).resolve()
    selected_hash = sha256_file(selected_checkpoint)
    selected["checkpoint_sha256"] = selected_hash

    selected_dir = output / "selected_qualification"
    prediction_dir = run_dir / "outputs/data"
    shutil.rmtree(prediction_dir, ignore_errors=True)
    selected_config = yaml.safe_load(training_config.read_text(encoding="utf-8"))
    selected_config["tester"].update(
        {
            "mode": "single",
            "checkpoint": int(selected["epoch"]),
            "threshold": args.score_threshold,
            "topk": args.topk,
        }
    )
    selected_config["trainer"].update(
        {
            "save_all": True,
            "pretrain_model": None,
            "resume_model": False,
            "evaluate_during_training": False,
            "evaluate_after_training": False,
        }
    )
    selected_config_path = monodgp / "configs/monodgp_m54_selected_qualification.yaml"
    selected_config_path.write_text(
        yaml.safe_dump(selected_config, sort_keys=False), encoding="utf-8"
    )
    run_logged(
        [
            sys.executable,
            "-u",
            "tools/train_val.py",
            "--config",
            str(selected_config_path),
            "--evaluate_only",
        ],
        monodgp,
        selected_dir / "selected_inference.log",
    )
    prediction_files = list(prediction_dir.glob("*.txt"))
    if len(prediction_files) != 3769:
        raise RuntimeError(
            f"Selected M54 checkpoint produced {len(prediction_files)}/3769 files"
        )

    product_ap_dir = selected_dir / "product_ap"
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
            str(args.dataset_root),
            "--split-dir",
            str(args.split_dir),
            "--prediction-dir",
            str(prediction_dir),
            "--split",
            "val",
            "--classes",
            "Vehicle",
            "Pedestrian",
            "--source-name",
            f"MonoDGP_M54_epoch_{int(selected['epoch']):03d}",
            "--output-dir",
            str(product_ap_dir),
        ],
        mobile,
        selected_dir / "product_ap.log",
    )
    nearby_dir = selected_dir / "nearby_geometry"
    run_logged(
        [
            sys.executable,
            "-u",
            "scripts/audit_product_prediction_geometry.py",
            "--dataset-root",
            str(args.dataset_root),
            "--split-file",
            str(args.split_dir / "val.txt"),
            "--prediction-dir",
            str(prediction_dir),
            "--output-dir",
            str(nearby_dir),
            "--checkpoint",
            str(selected_checkpoint),
            "--expected-checkpoint-sha256",
            selected_hash,
            "--expected-images",
            "3769",
            "--score-threshold",
            str(args.score_threshold),
            "--match-iou-threshold",
            "0.5",
        ],
        mobile,
        selected_dir / "nearby_geometry.log",
    )
    miss_dir = selected_dir / "pedestrian_false_negative_diagnostic"
    run_logged(
        [
            sys.executable,
            "-u",
            "scripts/diagnose_a2_pedestrian_false_negatives.py",
            "--dataset-root",
            str(args.dataset_root),
            "--split-file",
            str(args.split_dir / "val.txt"),
            "--prediction-dir",
            str(prediction_dir),
            "--output-dir",
            str(miss_dir),
            "--checkpoint",
            str(selected_checkpoint),
            "--expected-checkpoint-sha256",
            selected_hash,
            "--expected-images",
            "3769",
            "--score-threshold",
            str(args.score_threshold),
            "--iou-threshold",
            "0.5",
            "--weak-iou-threshold",
            "0.1",
        ],
        mobile,
        selected_dir / "pedestrian_false_negative.log",
    )

    nearby = json.loads((nearby_dir / "nearby_geometry_summary.json").read_text())
    misses = json.loads(
        (miss_dir / "a2_pedestrian_false_negative_summary.json").read_text()
    )
    metrics = {
        "vehicle_3d_moderate": float(selected["vehicle_3d_moderate"]),
        "pedestrian_3d_moderate": float(selected["pedestrian_3d_moderate"]),
        "mean_3d_moderate": float(selected["mean_3d_moderate"]),
        "vehicle_bev_moderate": float(selected["vehicle_bev_moderate"]),
        "pedestrian_bev_moderate": float(selected["pedestrian_bev_moderate"]),
        "vehicle_near_recall": float(nearby["classes"]["Vehicle"]["near_recall"]),
        "pedestrian_near_recall": float(
            nearby["classes"]["Pedestrian"]["near_recall"]
        ),
        "pedestrian_localization_failure_rate": float(
            misses["near_failure_rates"].get("localization_failure", 0.0)
        ),
    }
    comparable_results = {
        name: metrics[name] >= threshold
        for name, threshold in R0_COMPARABLE_GATES.items()
        if not name.endswith("_max")
    }
    comparable_results["pedestrian_localization_failure_rate"] = (
        metrics["pedestrian_localization_failure_rate"]
        <= R0_COMPARABLE_GATES["pedestrian_localization_failure_rate_max"]
    )
    product_target_results = {
        name: metrics[name] >= threshold
        for name, threshold in OFFLINE_PRODUCT_TARGETS.items()
    }
    complete = len(prediction_files) == 3769
    accuracy_parent_candidate = complete and all(comparable_results.values())
    offline_product_gates_passed = (
        accuracy_parent_candidate and all(product_target_results.values())
    )
    report = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M54 controlled GT-only Vehicle/Pedestrian MonoDGP adaptation",
        "selected_epoch": int(selected["epoch"]),
        "selected_checkpoint": str(selected_checkpoint),
        "selected_checkpoint_sha256": selected_hash,
        "selection_rule": manifest["selection_rule"],
        "evaluated_epochs": epochs,
        "evaluated_images": len(prediction_files),
        "metrics": metrics,
        "r0_comparable_gates": R0_COMPARABLE_GATES,
        "r0_comparable_gate_results": comparable_results,
        "accuracy_parent_candidate": accuracy_parent_candidate,
        "offline_product_targets": OFFLINE_PRODUCT_TARGETS,
        "offline_product_target_results": product_target_results,
        "offline_product_gates_passed": offline_product_gates_passed,
        "product_safety_qualified": False,
        "product_safety_note": (
            "Offline KITTI gates cannot qualify deployment safety; external-domain "
            "and device-runtime qualification remain required."
        ),
    }
    (output / "m54_product_selection.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
