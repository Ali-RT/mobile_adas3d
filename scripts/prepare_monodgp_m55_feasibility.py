from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path


PINNED_COMMIT = "aa059a18214aebf644510e7f0793971b403f9d14"
PARENT_CHECKPOINT_SHA256 = (
    "8e79f3921d96e1de70cbb4219245e3fcc3fa1fb67ae675468b4ebca90e579847"
)
TRAIN_SPLIT_SHA256 = "e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c"
VAL_SPLIT_SHA256 = "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8"
SELECTED_EPOCH = 100
EXPECTED_SWEEP_EPOCHS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
PARENT_METRICS = {
    "vehicle_3d_moderate": 19.451853874133846,
    "pedestrian_3d_moderate": 6.174878930065441,
    "mean_3d_moderate": 12.813366402099643,
    "vehicle_bev_moderate": 25.77508716658311,
    "pedestrian_bev_moderate": 6.767615777186105,
    "vehicle_near_recall": 0.9099254609650843,
    "pedestrian_near_recall": 0.7248677248677249,
    "pedestrian_localization_failure_rate": 0.23765432098765432,
}
PRESERVATION_GATES = {
    "vehicle_3d_moderate": 18.47926118042715,
    "pedestrian_3d_moderate": 5.866134983562169,
    "mean_3d_moderate": 12.17269808199466,
    "vehicle_bev_moderate": 24.48633280825395,
    "pedestrian_bev_moderate": 6.429234988326799,
    "vehicle_near_recall": 0.8999254609650843,
    "pedestrian_near_recall": 0.7148677248677249,
    "pedestrian_localization_failure_rate_max": 0.24765432098765433,
    "prediction_files": 3769,
}
CLASS_MAPPING = {
    "Car": "Car",
    "Van": "Car",
    "Truck": "Car",
    "Tram": "Car",
    "Pedestrian": "Pedestrian",
    "Person_sitting": "Pedestrian",
}
PROFILE_SETTINGS = {
    "batch_size": 1,
    "warmup_runs": 5,
    "timed_runs": 100,
    "sample_policy": "first Chen-validation image",
    "latency_clock": "CUDA events with synchronization",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_metric_mapping(actual: dict, expected: dict, label: str) -> None:
    for key, expected_value in expected.items():
        if key not in actual or abs(float(actual[key]) - expected_value) > 1e-12:
            raise RuntimeError(f"{label} {key} mismatch: {actual.get(key)} != {expected_value}")


def validate_m54_selection(selection_path: Path, sweep_path: Path) -> tuple[dict, Path]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("schema_version") != 1 or not selection.get("complete"):
        raise RuntimeError("M54 selection is incomplete")
    if selection.get("selected_epoch") != SELECTED_EPOCH:
        raise RuntimeError("M54 selected epoch is not the frozen epoch 100")
    if selection.get("selected_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("M54 selected checkpoint hash mismatch")
    if selection.get("evaluated_epochs") != EXPECTED_SWEEP_EPOCHS:
        raise RuntimeError("M54 evaluated checkpoint set changed")
    if selection.get("evaluated_images") != 3769:
        raise RuntimeError("M54 selection did not evaluate all 3,769 images")
    if not selection.get("accuracy_parent_candidate"):
        raise RuntimeError("M54 selection did not authorize the accuracy parent")
    if not all(selection.get("r0_comparable_gate_results", {}).values()):
        raise RuntimeError("M54 selection contains a failed R0-comparability gate")
    if selection.get("product_safety_qualified") is not False:
        raise RuntimeError("M54 must not claim product-safety qualification")
    require_metric_mapping(selection.get("metrics", {}), PARENT_METRICS, "M54 metric")

    checkpoint = Path(selection["selected_checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if sha256_file(checkpoint) != PARENT_CHECKPOINT_SHA256:
        raise RuntimeError("M54 parent checkpoint bytes do not match the frozen hash")

    with sweep_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    epochs = sorted(int(row["epoch"]) for row in rows)
    if epochs != EXPECTED_SWEEP_EPOCHS:
        raise RuntimeError(f"M54 sweep epochs changed: {epochs}")
    selected_rows = [row for row in rows if int(row["epoch"]) == SELECTED_EPOCH]
    if len(selected_rows) != 1 or int(selected_rows[0]["rank"]) != 1:
        raise RuntimeError("M54 epoch 100 is not the unique rank-one sweep row")
    if selected_rows[0].get("complete_split", "").lower() != "true":
        raise RuntimeError("M54 selected sweep row is incomplete")
    return selection, checkpoint


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="Freeze and prepare M55 MonoDGP feasibility profiling.")
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--m54-manifest", type=Path, required=True)
    parser.add_argument("--m54-selection", type=Path, required=True)
    parser.add_argument("--m54-sweep", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.monodgp_repo.resolve()
    dataset = args.dataset_root.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base_config = repo / "configs/monodgp.yaml"
    train_split = dataset / "ImageSets/train.txt"
    val_split = dataset / "ImageSets/val.txt"
    required = [
        base_config,
        train_split,
        val_split,
        dataset / "training/image_2",
        dataset / "training/label_2",
        dataset / "training/calib",
        args.m54_manifest.resolve(),
        args.m54_selection.resolve(),
        args.m54_sweep.resolve(),
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP commit {PINNED_COMMIT}, found {commit}")
    if len(split_ids(train_split)) != 3712 or sha256_file(train_split) != TRAIN_SPLIT_SHA256:
        raise RuntimeError("Expected the exact 3,712-image Chen training split")
    if len(split_ids(val_split)) != 3769 or sha256_file(val_split) != VAL_SPLIT_SHA256:
        raise RuntimeError("Expected the exact 3,769-image Chen validation split")

    m54_manifest = json.loads(args.m54_manifest.read_text(encoding="utf-8"))
    required_manifest = {
        "complete": True,
        "architecture_changed": False,
        "distillation_enabled": False,
        "upstream_commit": PINNED_COMMIT,
        "split_protocol": "chen_3712_3769",
        "train_split_sha256": TRAIN_SPLIT_SHA256,
        "val_split_sha256": VAL_SPLIT_SHA256,
    }
    for key, expected in required_manifest.items():
        if m54_manifest.get(key) != expected:
            raise RuntimeError(f"M54 manifest {key} mismatch")
    selection, checkpoint = validate_m54_selection(
        args.m54_selection.resolve(), args.m54_sweep.resolve()
    )

    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    invariants = {
        "backbone": config["model"].get("backbone"),
        "num_classes": config["model"].get("num_classes"),
        "num_feature_levels": config["model"].get("num_feature_levels"),
        "num_queries": config["model"].get("num_queries"),
        "group_num": config["model"].get("group_num"),
        "hidden_dim": config["model"].get("hidden_dim"),
        "enc_layers": config["model"].get("enc_layers"),
        "dec_layers": config["model"].get("dec_layers"),
        "nheads": config["model"].get("nheads"),
    }
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
    if invariants != expected_invariants:
        raise RuntimeError(f"Pinned MonoDGP architecture changed: {invariants}")

    config["model_name"] = "monodgp_m55_native_parent"
    config["dataset"].update(
        {
            "root_dir": str(dataset),
            "train_split": "train",
            "test_split": "val",
            "batch_size": PROFILE_SETTINGS["batch_size"],
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
            "pretrain_model": None,
            "resume_model": False,
            "save_all": False,
        }
    )
    config["tester"].update({"mode": "single", "checkpoint": 100, "threshold": 0.001, "topk": 50})
    runtime_config = output / "monodgp_m55_profile.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M55 M54-parent compression baseline and export feasibility",
        "training_performed": False,
        "compression_performed": False,
        "profile_authorized": True,
        "offline_weight_compression_authorized": False,
        "direct_coreml_conversion_authorized": False,
        "product_safety_qualified": False,
        "upstream_repository": "https://github.com/PuFanqi23/MonoDGP",
        "upstream_commit": commit,
        "architecture_invariants": invariants,
        "parent_epoch": SELECTED_EPOCH,
        "parent_checkpoint": str(checkpoint),
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_metrics": PARENT_METRICS,
        "preservation_policy": {
            "ap_retention_fraction": 0.95,
            "max_nearby_recall_drop_absolute": 0.01,
            "max_pedestrian_localization_failure_increase_absolute": 0.01,
        },
        "preservation_gates": PRESERVATION_GATES,
        "dataset_root": str(dataset),
        "split_protocol": "chen_3712_3769",
        "train_split_sha256": TRAIN_SPLIT_SHA256,
        "val_split_sha256": VAL_SPLIT_SHA256,
        "profile_settings": PROFILE_SETTINGS,
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": sha256_file(runtime_config),
        "m54_manifest": str(args.m54_manifest.resolve()),
        "m54_manifest_sha256": sha256_file(args.m54_manifest.resolve()),
        "m54_selection": str(args.m54_selection.resolve()),
        "m54_selection_sha256": sha256_file(args.m54_selection.resolve()),
        "m54_sweep": str(args.m54_sweep.resolve()),
        "m54_sweep_sha256": sha256_file(args.m54_sweep.resolve()),
        "m54_selection_snapshot": {
            "selected_epoch": selection["selected_epoch"],
            "evaluated_epochs": selection["evaluated_epochs"],
            "evaluated_images": selection["evaluated_images"],
        },
        "expected_artifacts": [
            "m55_native_baseline_profile.json",
            "m55_native_baseline_latency.csv",
            "m55_operator_export_audit.json",
            "m55_feasibility_gate.json",
        ],
    }
    manifest_path = output / "m55_feasibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
