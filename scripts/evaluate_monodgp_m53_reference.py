from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path


PINNED_COMMIT = "aa059a18214aebf644510e7f0793971b403f9d14"
CHECKPOINT_SHA256 = "1d5f30b34b8bef49638079a8b07f05ebf11bb5f85d6a9a11c7b028c69396f05d"
VAL_SPLIT_SHA256 = "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8"
PUBLISHED = {"easy": 30.1314, "moderate": 22.7109, "hard": 19.3978}
MAX_ABS_AP_DIFFERENCE = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(command: list[str], cwd: Path) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def parse_native_car_3d_ap_r40(text: str) -> dict[str, float]:
    pattern = re.compile(
        r"Car AP_R40@0\.70,\s*0\.70,\s*0\.70:\s*"
        r"bbox AP:[^\r\n]*\r?\n"
        r"bev\s+AP:[^\r\n]*\r?\n"
        r"3d\s+AP:\s*([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)"
    )
    matches = pattern.findall(text)
    if not matches:
        raise RuntimeError("Native MonoDGP Car AP_R40 block not found")
    easy, moderate, hard = matches[-1]
    return {
        "easy": float(easy),
        "moderate": float(moderate),
        "hard": float(hard),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete M53 MonoDGP reference gate.")
    parser.add_argument("--mobile-repo", type=Path, required=True)
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, required=True)
    args = parser.parse_args()

    mobile = args.mobile_repo.resolve()
    monodgp = args.monodgp_repo.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    output = args.manifest.resolve().parent
    if not manifest.get("complete") or manifest.get("training_authorized") is not False:
        raise RuntimeError("Invalid M53 manifest")
    if manifest.get("upstream_commit") != PINNED_COMMIT:
        raise RuntimeError("M53 upstream commit mismatch")
    if manifest.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("M53 manifest checkpoint mismatch")
    checkpoint = Path(manifest["evaluation_checkpoint"])
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("M53 checkpoint SHA-256 mismatch")
    val_split = args.split_dir / "val.txt"
    if sha256_file(val_split) != VAL_SPLIT_SHA256:
        raise RuntimeError("M53 requires the exact Chen validation split")

    val_ids = [
        line.strip()
        for line in val_split.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_prediction_ids = set(val_ids)
    run_dir = Path(manifest["run_dir"])
    prediction_dir = run_dir / "outputs/data"
    prediction_ids = {path.stem for path in prediction_dir.glob("*.txt")}
    native_logs = sorted(
        run_dir.glob("train.log.*"), key=lambda path: path.stat().st_mtime
    )
    reused_predictions = (
        len(val_ids) == 3769
        and prediction_ids == expected_prediction_ids
        and bool(native_logs)
    )
    if reused_predictions:
        print("Reusing complete M53 predictions and native evaluator log.", flush=True)
    else:
        if prediction_dir.exists():
            shutil.rmtree(prediction_dir)
        run(
            [
                sys.executable,
                "-u",
                "tools/train_val.py",
                "--config",
                manifest["runtime_config"],
                "--evaluate_only",
            ],
            monodgp,
        )
        native_logs = sorted(
            run_dir.glob("train.log.*"), key=lambda path: path.stat().st_mtime
        )
    if not native_logs:
        raise RuntimeError("MonoDGP native evaluator log is missing")
    native_log = native_logs[-1]
    native_candidate = parse_native_car_3d_ap_r40(
        native_log.read_text(encoding="utf-8", errors="replace")
    )

    prediction_ids = {path.stem for path in prediction_dir.glob("*.txt")}
    complete_predictions = (
        len(val_ids) == 3769 and prediction_ids == expected_prediction_ids
    )
    if not complete_predictions:
        raise RuntimeError(
            f"Incomplete MonoDGP prediction set: {len(prediction_ids)}/{len(val_ids)}"
        )

    metrics_dir = output / "independent_car_metrics"
    run(
        [
            sys.executable,
            "-u",
            "scripts/evaluate_kitti_prediction_dir.py",
            "--config",
            "configs/kitti_monodgp_m53_car_reference.yaml",
            "--profile",
            "colab_drive",
            "--dataset-root",
            str(args.dataset_root),
            "--split-dir",
            str(args.split_dir),
            "--prediction-dir",
            str(prediction_dir),
            "--split",
            "val",
            "--classes",
            "Car",
            "--source-name",
            "MonoDGP_M53_official_checkpoint",
            "--output-dir",
            str(metrics_dir),
        ],
        mobile,
    )
    summary = json.loads(
        (metrics_dir / "kitti_r40_summary.json").read_text(encoding="utf-8")
    )
    independent_candidate = {
        row["difficulty"]: float(row["ap_r40"])
        for row in summary["metrics"]
        if row["metric"] == "3d" and row["class_name"] == "Car"
    }
    rows = []
    for difficulty in ("easy", "moderate", "hard"):
        native_absolute_difference = abs(
            native_candidate[difficulty] - PUBLISHED[difficulty]
        )
        rows.append(
            {
                "difficulty": difficulty,
                "published_car_3d_ap_r40": PUBLISHED[difficulty],
                "native_car_3d_ap_r40": native_candidate[difficulty],
                "independent_car_3d_ap_r40": independent_candidate[difficulty],
                "native_absolute_difference": native_absolute_difference,
                "independent_minus_native": independent_candidate[difficulty]
                - native_candidate[difficulty],
                "max_absolute_difference": MAX_ABS_AP_DIFFERENCE,
                "passed": native_absolute_difference <= MAX_ABS_AP_DIFFERENCE,
            }
        )
    gate_results = {
        "upstream_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=monodgp, check=True, capture_output=True, text=True
        ).stdout.strip()
        == PINNED_COMMIT,
        "checkpoint_sha256": sha256_file(checkpoint) == CHECKPOINT_SHA256,
        "val_split_sha256": sha256_file(val_split) == VAL_SPLIT_SHA256,
        "complete_prediction_set": complete_predictions,
        **{
            f"native_car_3d_{row['difficulty']}_within_tolerance": row["passed"]
            for row in rows
        },
    }
    passed = all(gate_results.values())
    report = {
        "schema_version": 2,
        "complete": True,
        "experiment": "M53 official MonoDGP Car reference reproducibility gate",
        "model_role": "accuracy_challenger_reference",
        "training_performed": False,
        "product_taxonomy_adaptation": False,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "split_protocol": "chen_3712_3769",
        "prediction_files": len(prediction_ids),
        "reused_existing_predictions": reused_predictions,
        "gate_metric_source": "MonoDGP native official KITTI evaluator",
        "native_evaluator_log": str(native_log),
        "published_car_3d_ap_r40": PUBLISHED,
        "reproduced_car_3d_ap_r40": native_candidate,
        "independent_mobileadas3d_car_3d_ap_r40": independent_candidate,
        "independent_metric_role": "diagnostic_not_gate_authority",
        "max_absolute_ap_difference": MAX_ABS_AP_DIFFERENCE,
        "gate_results": gate_results,
        "reference_reproduced": passed,
        "two_class_adaptation_authorized": passed,
        "product_safety_qualified": False,
        "next_step_if_passed": "M54 controlled Vehicle/Pedestrian MonoDGP adaptation",
    }
    (output / "m53_monodgp_reference_gate.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (output / "m53_monodgp_reference_gate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("M53 MonoDGP reference gate failed; do not start M54 adaptation")


if __name__ == "__main__":
    main()
