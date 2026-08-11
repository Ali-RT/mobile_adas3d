from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
import yaml


PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
CLASS_MAPPING = {
    "Car": "Car",
    "Van": "Car",
    "Truck": "Car",
    "Tram": "Car",
    "Pedestrian": "Pedestrian",
    "Person_sitting": "Pedestrian",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the frozen ResNet50 MonoDETR R0 two-class reference run."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default="monodetr_r0_vehicle_pedestrian")
    parser.add_argument("--max-epochs", type=int, default=195)
    parser.add_argument("--save-frequency", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    repo = args.monodetr_repo.resolve()
    dataset_root = args.dataset_root.resolve()
    checkpoint = args.official_checkpoint.resolve()
    output_root = args.output_root.resolve()
    base_config = repo / "configs/monodetr.yaml"
    required_files = [
        base_config,
        checkpoint,
        dataset_root / "ImageSets/train.txt",
        dataset_root / "ImageSets/val.txt",
    ]
    for path in required_files:
        if not path.is_file():
            raise FileNotFoundError(path)

    official = load_checkpoint(checkpoint)
    if "model_state" not in official:
        raise ValueError("Official checkpoint has no model_state")

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    init_checkpoint = run_dir / "checkpoint_r0_init.pth"
    torch.save(
        {
            "epoch": 0,
            "model_state": official["model_state"],
            "optimizer_state": None,
            "best_result": 0.0,
            "best_epoch": 0,
            "mobileadas3d_initialization": {
                "policy": "published MonoDETR checkpoint; retain native Car/Pedestrian logits",
                "official_checkpoint_sha256": sha256_file(checkpoint),
            },
        },
        init_checkpoint,
    )

    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    config["model_name"] = args.run_name
    config["dataset"].update(
        {
            "root_dir": str(dataset_root),
            "train_split": "train",
            "test_split": "val",
            "batch_size": args.batch_size,
            "writelist": ["Car", "Pedestrian"],
            "class_mapping": CLASS_MAPPING,
            "class_merging": False,
        }
    )
    config["trainer"].update(
        {
            "max_epoch": args.max_epochs,
            "save_frequency": args.save_frequency,
            "save_all": True,
            "gpu_ids": "0",
            "save_path": os.path.relpath(output_root, repo),
            "pretrain_model": str(init_checkpoint),
            "resume_model": False,
        }
    )
    config_path = repo / "configs/monodetr_r0_vehicle_pedestrian.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "experiment": "ResNet50 MonoDETR R0 product-taxonomy reference",
        "upstream_commit": PINNED_COMMIT,
        "run_name": args.run_name,
        "config": str(config_path),
        "output_root": str(output_root),
        "initial_checkpoint": str(init_checkpoint),
        "official_checkpoint_sha256": sha256_file(checkpoint),
        "train_split_sha256": sha256_file(dataset_root / "ImageSets/train.txt"),
        "val_split_sha256": sha256_file(dataset_root / "ImageSets/val.txt"),
        "native_training_classes": ["Car", "Pedestrian"],
        "product_classes": ["Vehicle", "Pedestrian"],
        "class_mapping": CLASS_MAPPING,
        "max_epochs": args.max_epochs,
        "save_frequency": args.save_frequency,
        "batch_size": args.batch_size,
        "selection_metric": "mean Vehicle/Pedestrian moderate 3D AP_R40",
    }
    manifest_path = run_dir / "experiment_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
