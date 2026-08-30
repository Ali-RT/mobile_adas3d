from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
R0_CHECKPOINT_SHA256 = "fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59"
R0_EPOCH = 185
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
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_r0_selection(path: Path) -> tuple[Path, dict]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    selected = selection.get("selected", {})
    if not selection.get("complete"):
        raise RuntimeError("R0 product selection is not complete")
    if int(selected.get("epoch", -1)) != R0_EPOCH:
        raise RuntimeError(f"Expected R0 epoch {R0_EPOCH}, found {selected.get('epoch')}")
    recorded_hash = selected.get("checkpoint_sha256")
    if recorded_hash != R0_CHECKPOINT_SHA256:
        raise RuntimeError(
            f"Expected frozen R0 SHA-256 {R0_CHECKPOINT_SHA256}, found {recorded_hash}"
        )
    checkpoint = Path(selected["checkpoint"]).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual_hash = sha256_file(checkpoint)
    if actual_hash != R0_CHECKPOINT_SHA256:
        raise RuntimeError(f"R0 checkpoint hash mismatch: {actual_hash}")
    return checkpoint, selection


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(
        description="Prepare the GT-only two-class MobileNetV4 MonoDETR A3 run."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--r0-selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--run-name", default="monodetr_a3_mnv4_vehicle_pedestrian_gt")
    parser.add_argument("--max-epochs", type=int, default=195)
    parser.add_argument("--save-frequency", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260824)
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    args = parser.parse_args()

    repo = args.monodetr_repo.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    selection_path = args.r0_selection.resolve()
    base_config = repo / "configs/monodetr.yaml"
    required = [
        base_config,
        selection_path,
        dataset_root / "ImageSets/train.txt",
        dataset_root / "ImageSets/val.txt",
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    r0_checkpoint, selection = resolve_r0_selection(selection_path)
    r0_payload = load_checkpoint(r0_checkpoint)
    r0_state = r0_payload.get("model_state")
    if r0_state is None:
        raise RuntimeError("Frozen R0 checkpoint has no model_state")

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
    model_config = config["model"]
    model_config.update(
        {
            "backbone_source": "timm",
            "backbone": "mobilenetv4_conv_large.e500_r256_in1k",
            "backbone_out_indices": [2, 3, 4],
            "backbone_pretrained": not args.no_pretrained_backbone,
        }
    )
    config["trainer"].update(
        {
            "max_epoch": args.max_epochs,
            "save_frequency": args.save_frequency,
            "save_all": True,
            "gpu_ids": "0",
            "save_path": os.path.relpath(output_root, repo),
            "resume_model": False,
            "log_frequency": 20,
        }
    )
    config["optimizer"]["lr"] = args.learning_rate

    sys.path.insert(0, str(repo))
    from lib.helpers.model_helper import build_model  # noqa: PLC0415

    torch.manual_seed(args.seed)
    model, _ = build_model(model_config)
    mobile_state = model.state_dict()
    copied: list[str] = []
    deliberately_new: list[str] = []
    incompatible: list[dict] = []
    absent: list[str] = []
    for name, target in mobile_state.items():
        if name.startswith(("backbone.0.", "input_proj.")):
            deliberately_new.append(name)
            continue
        source = r0_state.get(name)
        if source is None:
            absent.append(name)
        elif source.shape != target.shape:
            incompatible.append(
                {"name": name, "source": list(source.shape), "target": list(target.shape)}
            )
        else:
            mobile_state[name] = source
            copied.append(name)
    if absent or incompatible:
        raise RuntimeError(
            "Unexpected downstream R0 incompatibility: "
            f"absent={absent[:5]}, shape_mismatch={incompatible[:5]}"
        )
    model.load_state_dict(mobile_state, strict=True)

    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    init_checkpoint = run_dir / "checkpoint_a3_init.pth"
    torch.save(
        {
            "epoch": 0,
            "model_state": model.state_dict(),
            "optimizer_state": None,
            "best_result": 0.0,
            "best_epoch": 0,
            "mobileadas3d_initialization": {
                "policy": "ImageNet MobileNetV4 backbone plus compatible frozen two-class R0 epoch-185 tensors",
                "r0_checkpoint_sha256": R0_CHECKPOINT_SHA256,
                "copied_tensor_count": len(copied),
                "new_backbone_or_projection_tensor_count": len(deliberately_new),
            },
        },
        init_checkpoint,
    )
    config["trainer"]["pretrain_model"] = str(init_checkpoint)
    config_output = (
        args.config_output.resolve()
        if args.config_output
        else repo / "configs/monodetr_a3_mnv4_vehicle_pedestrian_gt.yaml"
    )
    config_output.parent.mkdir(parents=True, exist_ok=True)
    config_output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "experiment": "MobileMonoDETR-Student-A3 GT-only two-class baseline",
        "upstream_commit": PINNED_COMMIT,
        "run_name": args.run_name,
        "config": str(config_output),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "initial_checkpoint": str(init_checkpoint),
        "r0_selection": str(selection_path),
        "r0_checkpoint": str(r0_checkpoint),
        "r0_checkpoint_sha256": R0_CHECKPOINT_SHA256,
        "r0_selected_epoch": R0_EPOCH,
        "train_split_sha256": sha256_file(dataset_root / "ImageSets/train.txt"),
        "val_split_sha256": sha256_file(dataset_root / "ImageSets/val.txt"),
        "native_training_classes": ["Car", "Pedestrian"],
        "product_classes": ["Vehicle", "Pedestrian"],
        "class_mapping": CLASS_MAPPING,
        "backbone": model_config["backbone"],
        "changed_component": "backbone and required feature projections only",
        "copied_downstream_tensors": len(copied),
        "new_backbone_or_projection_tensors": len(deliberately_new),
        "max_epochs": args.max_epochs,
        "save_frequency": args.save_frequency,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "initialization_seed": args.seed,
        "distillation_enabled": False,
        "selection_metric": "all five 90%-of-R0 gates plus nearby recall review",
        "r0_selection_snapshot": selection["selected"],
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
