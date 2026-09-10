from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

PINNED_COMMIT = "aa059a18214aebf644510e7f0793971b403f9d14"
CHECKPOINT_FILE_ID = "1nfCiFIxCIm0WG--cbllkqzgeuVRPpNI5"
CHECKPOINT_SHA256 = "1d5f30b34b8bef49638079a8b07f05ebf11bb5f85d6a9a11c7b028c69396f05d"
TRAIN_SPLIT_SHA256 = "e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c"
VAL_SPLIT_SHA256 = "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8"
PUBLISHED_CAR_3D_AP_R40 = {"easy": 30.1314, "moderate": 22.7109, "hard": 19.3978}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Prepare the evaluation-only M53 official MonoDGP Car reference."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    repo = args.monodgp_repo.resolve()
    dataset = args.dataset_root.resolve()
    source_checkpoint = args.official_checkpoint.resolve()
    output = args.output_root.resolve()
    base_config = repo / "configs/monodgp.yaml"
    required = [
        base_config,
        source_checkpoint,
        dataset / "ImageSets/train.txt",
        dataset / "ImageSets/val.txt",
        dataset / "training/image_2",
        dataset / "training/label_2",
        dataset / "training/calib",
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP commit {PINNED_COMMIT}, found {commit}")
    if sha256_file(source_checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Official MonoDGP checkpoint SHA-256 mismatch")

    train_split = dataset / "ImageSets/train.txt"
    val_split = dataset / "ImageSets/val.txt"
    if len(split_ids(train_split)) != 3712 or sha256_file(train_split) != TRAIN_SPLIT_SHA256:
        raise RuntimeError("Expected the exact 3,712-image Chen training split")
    if len(split_ids(val_split)) != 3769 or sha256_file(val_split) != VAL_SPLIT_SHA256:
        raise RuntimeError("Expected the exact 3,769-image Chen validation split")

    config = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    invariants = {
        "dataset.writelist": config["dataset"].get("writelist"),
        "model.backbone": config["model"].get("backbone"),
        "model.num_queries": config["model"].get("num_queries"),
        "model.hidden_dim": config["model"].get("hidden_dim"),
        "model.enc_layers": config["model"].get("enc_layers"),
        "model.dec_layers": config["model"].get("dec_layers"),
        "tester.threshold": config["tester"].get("threshold"),
        "tester.topk": config["tester"].get("topk"),
    }
    expected = {
        "dataset.writelist": ["Car"],
        "model.backbone": "resnet50",
        "model.num_queries": 50,
        "model.hidden_dim": 256,
        "model.enc_layers": 3,
        "model.dec_layers": 3,
        "tester.threshold": 0.2,
        "tester.topk": 50,
    }
    if invariants != expected:
        raise RuntimeError(f"Pinned MonoDGP config invariants changed: {invariants}")

    run_name = "monodgp_m53_official_car"
    run_dir = output / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    evaluation_checkpoint = run_dir / "checkpoint_best.pth"
    if evaluation_checkpoint.exists():
        if sha256_file(evaluation_checkpoint) != CHECKPOINT_SHA256:
            raise RuntimeError(f"Refusing to replace mismatched checkpoint: {evaluation_checkpoint}")
    else:
        shutil.copy2(source_checkpoint, evaluation_checkpoint)

    config["model_name"] = run_name
    config["dataset"].update(
        {
            "root_dir": str(dataset),
            "train_split": "train",
            "test_split": "val",
            "batch_size": 8,
            "writelist": ["Car"],
            "class_merging": False,
            "use_dontcare": False,
        }
    )
    config["trainer"].update(
        {
            "gpu_ids": "0",
            "save_path": os.path.relpath(output, repo),
            "save_all": False,
            "pretrain_model": None,
            "resume_model": False,
        }
    )
    config["tester"].update(
        {"mode": "single", "checkpoint": 195, "threshold": 0.2, "topk": 50}
    )
    runtime_config = repo / "configs/monodgp_m53_official_car.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M53 official MonoDGP Car reference reproducibility gate",
        "model_role": "accuracy_challenger_reference",
        "training_authorized": False,
        "architecture_changed": False,
        "product_taxonomy_adaptation": False,
        "native_classes": ["Car"],
        "upstream_repository": "https://github.com/PuFanqi23/MonoDGP",
        "upstream_commit": PINNED_COMMIT,
        "upstream_config": str(base_config),
        "upstream_config_sha256": sha256_file(base_config),
        "checkpoint_google_drive_file_id": CHECKPOINT_FILE_ID,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "source_checkpoint": str(source_checkpoint),
        "evaluation_checkpoint": str(evaluation_checkpoint),
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": sha256_file(runtime_config),
        "run_dir": str(run_dir),
        "dataset_root": str(dataset),
        "split_protocol": "chen_3712_3769",
        "train_split_sha256": TRAIN_SPLIT_SHA256,
        "val_split_sha256": VAL_SPLIT_SHA256,
        "published_car_3d_ap_r40": PUBLISHED_CAR_3D_AP_R40,
        "evaluation_protocol": {
            "split": "val",
            "images": 3769,
            "score_threshold": 0.2,
            "topk": 50,
            "iou_threshold": 0.7,
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "m53_monodgp_reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
