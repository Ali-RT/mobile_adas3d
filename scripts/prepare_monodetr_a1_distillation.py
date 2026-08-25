from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path


PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
R0_CHECKPOINT_SHA256 = "fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59"
EXPECTED_A1 = {
    "selected_epoch": 140,
    "vehicle_3d_moderate": 12.860398859999606,
    "pedestrian_3d_moderate": 7.266891428067282,
    "mean_3d_moderate": 10.063645144033444,
    "vehicle_bev_moderate": 18.785243556978813,
    "pedestrian_bev_moderate": 8.509545844712921,
}
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


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def validate_a1_selection(path: Path) -> dict:
    report = load_json(path)
    if not report.get("complete"):
        raise RuntimeError("A1 selection report is incomplete")
    if int(report.get("selected_epoch", -1)) != EXPECTED_A1["selected_epoch"]:
        raise RuntimeError("A1 selection is not frozen epoch 140")
    metrics = report.get("metrics", {})
    for name, expected in EXPECTED_A1.items():
        if name == "selected_epoch":
            continue
        actual = float(metrics.get(name, float("nan")))
        if abs(actual - expected) > 1e-9:
            raise RuntimeError(f"A1 metric mismatch for {name}: {actual} != {expected}")
    checkpoint = Path(report["selected_checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return report


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(
        description="Prepare the paired R0-to-A1 query-distillation run."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--r0-selection", type=Path, required=True)
    parser.add_argument("--a1-manifest", type=Path, required=True)
    parser.add_argument("--a1-selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-name", default="monodetr_a1_mnv4_vehicle_pedestrian_distill")
    args = parser.parse_args()

    repo = args.monodetr_repo.resolve()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    base_config_path = repo / "configs/monodetr.yaml"
    for required in (
        base_config_path,
        dataset_root / "ImageSets/train.txt",
        dataset_root / "ImageSets/val.txt",
    ):
        if not required.is_file():
            raise FileNotFoundError(required)

    r0_selection = load_json(args.r0_selection.resolve())
    r0_selected = r0_selection.get("selected", {})
    if not r0_selection.get("complete") or int(r0_selected.get("epoch", -1)) != 185:
        raise RuntimeError("Frozen R0 selection must be complete epoch 185")
    if r0_selected.get("checkpoint_sha256") != R0_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen R0 selection hash mismatch")
    r0_checkpoint = Path(r0_selected["checkpoint"]).resolve()
    if not r0_checkpoint.is_file() or sha256_file(r0_checkpoint) != R0_CHECKPOINT_SHA256:
        raise RuntimeError("Frozen R0 checkpoint file/hash mismatch")

    a1_manifest = load_json(args.a1_manifest.resolve())
    if a1_manifest.get("distillation_enabled") is not False:
        raise RuntimeError("A1 baseline manifest must be GT-only")
    if a1_manifest.get("r0_checkpoint_sha256") != R0_CHECKPOINT_SHA256:
        raise RuntimeError("A1 initialization used a different R0 checkpoint")
    a1_init = Path(a1_manifest["initial_checkpoint"]).resolve()
    if not a1_init.is_file():
        raise FileNotFoundError(a1_init)
    try:
        init_payload = torch.load(a1_init, map_location="cpu", weights_only=False)
    except TypeError:
        init_payload = torch.load(a1_init, map_location="cpu")
    init_meta = init_payload.get("mobileadas3d_initialization", {})
    if int(init_payload.get("epoch", -1)) != 0 or init_payload.get("optimizer_state") is not None:
        raise RuntimeError("A1 initialization checkpoint is not a clean epoch-zero start")
    if init_meta.get("r0_checkpoint_sha256") != R0_CHECKPOINT_SHA256:
        raise RuntimeError("A1 initialization metadata has the wrong R0 hash")
    a1_selection = validate_a1_selection(args.a1_selection.resolve())

    config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    teacher_model = copy.deepcopy(config["model"])
    config["model_name"] = args.run_name
    config["dataset"].update(
        {
            "root_dir": str(dataset_root),
            "train_split": "train",
            "test_split": "val",
            "batch_size": int(a1_manifest["batch_size"]),
            "writelist": ["Car", "Pedestrian"],
            "class_mapping": CLASS_MAPPING,
            "class_merging": False,
        }
    )
    config["model"].update(
        {
            "backbone_source": "timm",
            "backbone": "mobilenetv4_conv_small.e2400_r224_in1k",
            "backbone_out_indices": [2, 3, 4],
            "backbone_pretrained": True,
        }
    )
    config["optimizer"]["lr"] = float(a1_manifest["learning_rate"])
    config["trainer"].update(
        {
            "max_epoch": int(a1_manifest["max_epochs"]),
            "save_frequency": int(a1_manifest["save_frequency"]),
            "save_all": True,
            "gpu_ids": "0",
            "save_path": os.path.relpath(output_root, repo),
            "pretrain_model": str(a1_init),
            "resume_model": False,
            "log_frequency": 20,
        }
    )
    config["distillation"] = {
        "enabled": True,
        "method": "GT-aligned teacher/student query distillation",
        "teacher_checkpoint": str(r0_checkpoint),
        "teacher_model": teacher_model,
        "min_teacher_score": 0.30,
        "min_teacher_iou_2d": 0.50,
        "temperature": 2.0,
        "overall_weight": 0.25,
        "target_class_weights": {0: 1.0, 1: 0.25},
        "loss_weights": {
            "logits": 0.25,
            "boxes": 1.0,
            "depth": 0.20,
            "dims": 0.50,
            "angles": 0.10,
        },
    }

    config_path = repo / "configs/monodetr_a1_mnv4_vehicle_pedestrian_distill.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "experiment": "paired R0-to-A1 query distillation",
        "upstream_commit": PINNED_COMMIT,
        "run_name": args.run_name,
        "config": str(config_path),
        "output_root": str(output_root),
        "run_dir": str(run_dir),
        "distillation_enabled": True,
        "distillation_method": config["distillation"]["method"],
        "teacher_checkpoint": str(r0_checkpoint),
        "teacher_checkpoint_sha256": R0_CHECKPOINT_SHA256,
        "student_initial_checkpoint": str(a1_init),
        "student_initial_checkpoint_sha256": sha256_file(a1_init),
        "paired_gt_baseline_selection": a1_selection,
        "train_split_sha256": sha256_file(dataset_root / "ImageSets/train.txt"),
        "val_split_sha256": sha256_file(dataset_root / "ImageSets/val.txt"),
        "batch_size": config["dataset"]["batch_size"],
        "learning_rate": config["optimizer"]["lr"],
        "max_epochs": config["trainer"]["max_epoch"],
        "save_frequency": config["trainer"]["save_frequency"],
        "distillation": config["distillation"],
        "full_training_started": False,
    }
    (run_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
