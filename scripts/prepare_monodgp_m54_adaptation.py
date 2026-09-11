from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


PINNED_COMMIT = "aa059a18214aebf644510e7f0793971b403f9d14"
CHECKPOINT_SHA256 = "1d5f30b34b8bef49638079a8b07f05ebf11bb5f85d6a9a11c7b028c69396f05d"
TRAIN_SPLIT_SHA256 = "e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c"
VAL_SPLIT_SHA256 = "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8"
RUN_NAME = "monodgp_m54_vehicle_pedestrian_gt"
CLASS_MAPPING = {
    "Car": "Car",
    "Van": "Car",
    "Truck": "Car",
    "Tram": "Car",
    "Pedestrian": "Pedestrian",
    "Person_sitting": "Pedestrian",
}
TRAINING_SCHEDULE = {
    "max_epochs": 100,
    "save_frequency": 5,
    "batch_size": 8,
    "learning_rate": 0.00005,
    "lr_decay_rate": 0.5,
    "lr_decay_epochs": [40, 70, 90],
    "seed": 54054,
}
SWEEP_EPOCHS = [5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
R0_COMPARABLE_GATES = {
    "vehicle_3d_moderate": 17.634769196266316,
    "pedestrian_3d_moderate": 5.721371354710236,
    "mean_3d_moderate": 11.678070275488276,
    "vehicle_bev_moderate": 23.68156332656625,
    "pedestrian_bev_moderate": 6.596148868813419,
    "vehicle_near_recall": 0.8824637112593173,
    "pedestrian_near_recall": 0.6834215167548501,
    "pedestrian_localization_failure_rate_max": 0.24603174603174602,
}
OFFLINE_PRODUCT_TARGETS = {
    "vehicle_near_recall": 0.85,
    "pedestrian_near_recall": 0.80,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def require_m53_authorization(gate_path: Path, manifest_path: Path) -> tuple[dict, dict, Path]:
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    if gate.get("schema_version") != 2 or not gate.get("complete"):
        raise RuntimeError("M53 schema-v2 gate is incomplete")
    if not gate.get("reference_reproduced") or not gate.get("two_class_adaptation_authorized"):
        raise RuntimeError("M53 did not authorize M54")
    if not all(gate.get("gate_results", {}).values()):
        raise RuntimeError("M53 contains a failed frozen gate")
    if gate.get("checkpoint_sha256") != CHECKPOINT_SHA256:
        raise RuntimeError("M53 gate checkpoint SHA-256 mismatch")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("training_authorized") is not False:
        raise RuntimeError("M53 manifest is incomplete or unexpectedly authorizes training")
    required_manifest = {
        "upstream_commit": PINNED_COMMIT,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "train_split_sha256": TRAIN_SPLIT_SHA256,
        "val_split_sha256": VAL_SPLIT_SHA256,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise RuntimeError(f"M53 manifest {key} mismatch")
    checkpoint = Path(manifest["evaluation_checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Pinned M53 checkpoint is missing or has the wrong hash")
    return gate, manifest, checkpoint


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Prepare the frozen M54 GT-only Vehicle/Pedestrian MonoDGP adaptation."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--m53-gate", type=Path, required=True)
    parser.add_argument("--m53-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.monodgp_repo.resolve()
    dataset = args.dataset_root.resolve()
    output = args.output_root.resolve()
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
        args.m53_gate.resolve(),
        args.m53_manifest.resolve(),
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

    gate, m53_manifest, source_checkpoint = require_m53_authorization(
        args.m53_gate.resolve(), args.m53_manifest.resolve()
    )
    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    invariants = {
        "model.backbone": config["model"].get("backbone"),
        "model.num_classes": config["model"].get("num_classes"),
        "model.num_feature_levels": config["model"].get("num_feature_levels"),
        "model.num_queries": config["model"].get("num_queries"),
        "model.group_num": config["model"].get("group_num"),
        "model.hidden_dim": config["model"].get("hidden_dim"),
        "model.enc_layers": config["model"].get("enc_layers"),
        "model.dec_layers": config["model"].get("dec_layers"),
    }
    expected = {
        "model.backbone": "resnet50",
        "model.num_classes": 3,
        "model.num_feature_levels": 4,
        "model.num_queries": 50,
        "model.group_num": 11,
        "model.hidden_dim": 256,
        "model.enc_layers": 3,
        "model.dec_layers": 3,
    }
    if invariants != expected:
        raise RuntimeError(f"Pinned MonoDGP architecture changed: {invariants}")

    run_dir = output / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    initial_checkpoint = run_dir / "checkpoint_m54_init.pth"
    if initial_checkpoint.exists():
        if sha256_file(initial_checkpoint) != CHECKPOINT_SHA256:
            raise RuntimeError(f"Refusing mismatched initialization: {initial_checkpoint}")
    else:
        shutil.copy2(source_checkpoint, initial_checkpoint)

    config["random_seed"] = TRAINING_SCHEDULE["seed"]
    config["model_name"] = RUN_NAME
    config["dataset"].update(
        {
            "root_dir": str(dataset),
            "train_split": "train",
            "test_split": "val",
            "batch_size": TRAINING_SCHEDULE["batch_size"],
            "writelist": ["Car", "Pedestrian"],
            "class_mapping": CLASS_MAPPING,
            "class_merging": False,
            "use_dontcare": False,
        }
    )
    config["optimizer"]["lr"] = TRAINING_SCHEDULE["learning_rate"]
    config["lr_scheduler"].update(
        {
            "type": "step",
            "warmup": False,
            "decay_rate": TRAINING_SCHEDULE["lr_decay_rate"],
            "decay_list": TRAINING_SCHEDULE["lr_decay_epochs"],
        }
    )
    config["trainer"].update(
        {
            "max_epoch": TRAINING_SCHEDULE["max_epochs"],
            "save_frequency": TRAINING_SCHEDULE["save_frequency"],
            "save_all": True,
            "gpu_ids": "0",
            "save_path": os.path.relpath(output, repo),
            "pretrain_model": str(initial_checkpoint),
            "resume_model": False,
            "log_frequency": 20,
            "evaluate_during_training": False,
            "evaluate_after_training": False,
        }
    )
    config["tester"].update(
        {"mode": "single", "checkpoint": 100, "threshold": 0.001, "topk": 50}
    )
    runtime_config = repo / "configs/monodgp_m54_vehicle_pedestrian_gt.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M54 controlled GT-only Vehicle/Pedestrian MonoDGP adaptation",
        "model_role": "accuracy_challenger_adaptation",
        "training_authorized": True,
        "architecture_changed": False,
        "distillation_enabled": False,
        "temperature_tuning": False,
        "upstream_repository": "https://github.com/PuFanqi23/MonoDGP",
        "upstream_commit": PINNED_COMMIT,
        "run_name": RUN_NAME,
        "run_dir": str(run_dir),
        "output_root": str(output),
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": sha256_file(runtime_config),
        "dataset_root": str(dataset),
        "split_protocol": "chen_3712_3769",
        "train_split_sha256": TRAIN_SPLIT_SHA256,
        "val_split_sha256": VAL_SPLIT_SHA256,
        "native_model_classes": ["Pedestrian", "Car", "Cyclist"],
        "native_training_classes": ["Car", "Pedestrian"],
        "unused_native_output_class": "Cyclist",
        "product_classes": ["Vehicle", "Pedestrian"],
        "class_mapping": CLASS_MAPPING,
        "initialization_policy": "exact verified M53 official checkpoint; retain every model tensor",
        "initial_checkpoint": str(initial_checkpoint),
        "initial_checkpoint_sha256": CHECKPOINT_SHA256,
        "m53_gate": str(args.m53_gate.resolve()),
        "m53_gate_sha256": sha256_file(args.m53_gate.resolve()),
        "m53_manifest": str(args.m53_manifest.resolve()),
        "m53_manifest_sha256": sha256_file(args.m53_manifest.resolve()),
        "m53_native_car_3d_ap_r40": gate["reproduced_car_3d_ap_r40"],
        "training_schedule": TRAINING_SCHEDULE,
        "sweep_epochs": SWEEP_EPOCHS,
        "selection_rule": (
            "highest balanced Vehicle/Pedestrian moderate 3D AP_R40; "
            "tie Pedestrian 3D, Vehicle 3D, then balanced BEV"
        ),
        "r0_comparable_gates": R0_COMPARABLE_GATES,
        "offline_product_targets": OFFLINE_PRODUCT_TARGETS,
        "product_safety_qualified": False,
        "m53_provenance_snapshot": {
            "checkpoint_sha256": m53_manifest["checkpoint_sha256"],
            "upstream_commit": m53_manifest["upstream_commit"],
        },
    }
    manifest_path = output / "m54_adaptation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
