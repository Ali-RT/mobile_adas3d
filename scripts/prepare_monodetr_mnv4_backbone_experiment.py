from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml


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
        description=(
            "Create a backbone-only MobileNetV4 MonoDETR config and a strict-load "
            "initialization checkpoint from the official ResNet50 model."
        )
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--run-name", default="monodetr_mnv4_conv_small_backbone_gate")
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--save-frequency", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    args = parser.parse_args()

    repo = args.monodetr_repo.resolve()
    dataset_root = args.dataset_root.resolve()
    official_checkpoint = args.official_checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    base_config_path = repo / "configs/monodetr.yaml"
    for required in (base_config_path, official_checkpoint):
        if not required.is_file():
            raise FileNotFoundError(required)
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    config_output = (
        args.config_output.resolve()
        if args.config_output
        else repo / "configs/monodetr_mnv4_conv_small_backbone_gate.yaml"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    config_output.parent.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(base_config_path.read_text())
    config["model_name"] = args.run_name
    config["dataset"]["root_dir"] = str(dataset_root)
    config["dataset"]["train_split"] = "train"
    config["dataset"]["test_split"] = "val"
    config["dataset"]["batch_size"] = args.batch_size
    config["dataset"]["writelist"] = ["Car"]

    model_config = config["model"]
    model_config["backbone_source"] = "timm"
    model_config["backbone"] = "mobilenetv4_conv_small.e2400_r224_in1k"
    model_config["backbone_out_indices"] = [2, 3, 4]
    model_config["backbone_pretrained"] = not args.no_pretrained_backbone

    trainer = config["trainer"]
    trainer["max_epoch"] = args.max_epochs
    trainer["save_frequency"] = args.save_frequency
    trainer["save_all"] = True
    trainer["gpu_ids"] = "0"
    trainer["save_path"] = os.path.relpath(output_dir.parent, repo)
    config["optimizer"]["lr"] = args.learning_rate

    sys.path.insert(0, str(repo))
    from lib.helpers.model_helper import build_model  # noqa: PLC0415

    model, _ = build_model(model_config)
    mobile_state = model.state_dict()
    official = load_checkpoint(official_checkpoint)
    official_state = official.get("model_state", official)

    copied = []
    shape_mismatch = []
    absent = []
    deliberately_new = []
    for name, tensor in mobile_state.items():
        if name.startswith(("backbone.0.", "input_proj.")):
            deliberately_new.append(name)
            continue
        source = official_state.get(name)
        if source is None:
            absent.append(name)
        elif source.shape != tensor.shape:
            shape_mismatch.append(
                {"name": name, "source": list(source.shape), "target": list(tensor.shape)}
            )
        else:
            mobile_state[name] = source
            copied.append(name)

    if absent or shape_mismatch:
        raise RuntimeError(
            "Unexpected downstream checkpoint incompatibility: "
            f"absent={len(absent)}, shape_mismatch={len(shape_mismatch)}"
        )
    model.load_state_dict(mobile_state, strict=True)

    init_checkpoint = output_dir / "checkpoint_mobile_init.pth"
    torch.save(
        {
            "epoch": 0,
            "model_state": model.state_dict(),
            "optimizer_state": None,
            "best_result": 0.0,
            "best_epoch": 0,
            "mobileadas3d_initialization": {
                "policy": "ImageNet MobileNetV4 backbone plus shape-compatible official MonoDETR downstream weights",
                "official_checkpoint_sha256": sha256_file(official_checkpoint),
                "copied_tensor_count": len(copied),
                "new_backbone_or_projection_tensor_count": len(deliberately_new),
            },
        },
        init_checkpoint,
    )
    trainer["pretrain_model"] = str(init_checkpoint)
    config_output.write_text(yaml.safe_dump(config, sort_keys=False))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    report = {
        "schema_version": 1,
        "experiment": "MonoDETR backbone-only MobileNetV4 Conv Small gate",
        "run_name": args.run_name,
        "config": str(config_output),
        "output_dir": str(output_dir),
        "initial_checkpoint": str(init_checkpoint),
        "official_checkpoint": str(official_checkpoint),
        "official_checkpoint_sha256": sha256_file(official_checkpoint),
        "backbone": model_config["backbone"],
        "feature_strides": [8, 16, 32],
        "feature_channels": [64, 96, 960],
        "copied_downstream_tensors": len(copied),
        "new_backbone_or_projection_tensors": len(deliberately_new),
        "total_parameters": total,
        "trainable_parameters": trainable,
        "gate_epochs": args.max_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
    }
    report_path = output_dir / "experiment_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
