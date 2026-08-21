from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from data.collate import mobile_adas3d_collate_fn
from data.class_taxonomy import normalize_class_mapping, validate_taxonomy_manifest
from data.kitti_dataset import KITTIDataset
from data.splits import read_split_file
from data.split_resolver import get_split_file
from models.build import build_model
from scripts.train_mobile_adas3d import build_criterion, move_targets_to_device
from tools.config import apply_runtime_overrides, load_config
from tools.device import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight a MobileADAS3D training run")
    parser.add_argument("--config", required=True)
    parser.add_argument("--profile", default="colab_drive")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--report", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = apply_runtime_overrides(
        load_config(args.config),
        profile=args.profile,
        run_name=args.run_name,
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
        output_dir=args.output_dir,
    )
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    target_cfg = config["targets"]
    loss_cfg = config["loss"]
    root = Path(dataset_cfg["profiles"][dataset_cfg["active_profile"]]["root_dir"])

    directories = {
        "images": root / dataset_cfg["image_dir"],
        "labels": root / dataset_cfg["label_dir"],
        "calib": root / dataset_cfg["calib_dir"],
    }
    counts = {}
    suffixes = {"images": ".png", "labels": ".txt", "calib": ".txt"}
    for name, directory in directories.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Missing KITTI {name} directory: {directory}")
        counts[name] = len(list(directory.glob(f"*{suffixes[name]}")))
        if counts[name] != 7481:
            raise RuntimeError(
                f"Expected 7,481 KITTI {name} files in {directory}; got {counts[name]}"
            )

    train_file = Path(get_split_file(config, "train"))
    val_file = Path(get_split_file(config, "val"))
    train_ids = read_split_file(train_file)
    val_ids = read_split_file(val_file)
    if len(train_ids) != 3712 or len(val_ids) != 3769:
        raise RuntimeError(
            f"Expected Chen split 3712/3769; got {len(train_ids)}/{len(val_ids)}"
        )
    if set(train_ids) & set(val_ids) or len(set(train_ids) | set(val_ids)) != 7481:
        raise RuntimeError("Canonical KITTI split overlap/union validation failed")

    class_mapping = normalize_class_mapping(
        dataset_cfg.get("class_mapping"), dataset_cfg["classes"]
    )
    taxonomy_manifest = None
    if class_mapping and dataset_cfg.get("require_taxonomy_manifest", False):
        active_profile = dataset_cfg["active_profile"]
        taxonomy_manifest = dataset_cfg.get("taxonomy_manifest_paths", {}).get(
            active_profile, dataset_cfg.get("taxonomy_manifest")
        )
        if not taxonomy_manifest:
            raise ValueError(
                f"No taxonomy manifest configured for profile {active_profile!r}"
            )
        validate_taxonomy_manifest(
            taxonomy_manifest,
            dataset_cfg["classes"],
            class_mapping,
            {"train": train_file, "val": val_file},
            root / dataset_cfg["label_dir"],
        )

    device = get_device(config["training"].get("device", "auto"))
    if args.require_cuda and device.type != "cuda":
        raise RuntimeError(
            "CUDA GPU is required. In Colab choose Runtime > Change runtime type > GPU."
        )

    dataset = KITTIDataset(
        root_dir=str(root),
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=str(train_file),
        class_mapping=class_mapping,
    )
    collate = partial(
        mobile_adas3d_collate_fn,
        classes=dataset_cfg["classes"],
        input_height=int(model_cfg["input_height"]),
        input_width=int(model_cfg["input_width"]),
        output_stride=int(model_cfg["output_stride"]),
        class_mean_dims=target_cfg["class_mean_dims"],
        center_sampling_radius=int(target_cfg["center_sampling"]["radius"]),
        class_weights=loss_cfg["class_weights"],
        target_format=("query" if model_cfg["name"] == "MobileADAS3D-H1" else "dense"),
        depth_bins=int(model_cfg.get("depth_bins", 40)),
        min_depth_m=float(target_cfg.get("min_depth_m", 1.0)),
        max_depth_m=float(target_cfg.get("max_depth_m", 80.0)),
    )
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0, collate_fn=collate)))
    model = build_model(config).to(device).train()
    targets = move_targets_to_device(batch["targets"], device)
    outputs = model(batch["images"].to(device))
    losses = build_criterion(config)(outputs, targets)
    if not torch.isfinite(losses["total_loss"]):
        raise RuntimeError(f"Non-finite preflight loss: {losses['total_loss'].item()}")

    if model_cfg["name"] == "MobileADAS3D-H1":
        output_shape = list(outputs["class_logits"].shape)
        expected_shape = [1, int(model_cfg["num_queries"]), len(dataset_cfg["classes"])]
    else:
        output_shape = list(outputs["cls_logits"].shape)
        expected_shape = [
            1,
            len(dataset_cfg["classes"]),
            int(model_cfg["input_height"]) // int(model_cfg["output_stride"]),
            int(model_cfg["input_width"]) // int(model_cfg["output_stride"]),
        ]
    if output_shape != expected_shape:
        raise RuntimeError(f"Expected cls output {expected_shape}; got {output_shape}")

    report = {
        "status": "ready",
        "dataset_root": str(root),
        "file_counts": counts,
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "backbone": model_cfg["backbone"],
        "pretrained": bool(model_cfg["pretrained"]),
        "normalize_imagenet": bool(model_cfg.get("normalize_imagenet", False)),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "cls_output_shape": output_shape,
        "taxonomy_manifest": taxonomy_manifest,
        "total_loss": float(losses["total_loss"].detach().cpu()),
        "output_dir": config["outputs"]["output_dir"],
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
