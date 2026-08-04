from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.kitti_prediction_parser import parse_kitti_prediction_file


MANIFEST_NAME = "teacher_cache_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_ids(split_file: Path) -> List[str]:
    sample_ids = [
        Path(line.strip()).stem
        for line in split_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not sample_ids:
        raise ValueError(f"Split file is empty: {split_file}")
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError(f"Split file contains duplicate IDs: {split_file}")
    return sample_ids


def prediction_tree_sha256(prediction_dir: Path, sample_ids: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        path = prediction_dir / f"{sample_id}.txt"
        payload = path.read_bytes()
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def validate_prediction_directory(
    prediction_dir: Path,
    sample_ids: List[str],
    allowed_classes: Iterable[str] | None = None,
) -> Dict[str, Any]:
    expected_names = {f"{sample_id}.txt" for sample_id in sample_ids}
    actual_names = {path.name for path in prediction_dir.glob("*.txt")}
    missing_names = sorted(expected_names - actual_names)
    extra_names = sorted(actual_names - expected_names)
    if missing_names or extra_names:
        raise RuntimeError(
            "Teacher prediction directory does not exactly match the split: "
            f"missing={len(missing_names)}, extra={len(extra_names)}, "
            f"missing_preview={missing_names[:10]}, extra_preview={extra_names[:10]}"
        )

    detection_count = 0
    empty_file_count = 0
    for sample_id in sample_ids:
        predictions = parse_kitti_prediction_file(
            prediction_dir / f"{sample_id}.txt",
            allowed_classes=allowed_classes,
        )
        detection_count += len(predictions)
        empty_file_count += int(not predictions)

    return {
        "prediction_files": len(sample_ids),
        "detection_count": detection_count,
        "empty_prediction_files": empty_file_count,
        "prediction_tree_sha256": prediction_tree_sha256(
            prediction_dir, sample_ids
        ),
    }


def validate_inference_contract(
    runtime_config: Path,
    expected_sample_ids: List[str],
) -> Dict[str, Any]:
    config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    dataset_config = config.get("dataset", {})
    inference_split = dataset_config.get("test_split")
    if inference_split in {"train", "trainval"}:
        raise RuntimeError(
            "Teacher cache inference used a MonoDETR training split, which enables "
            f"random data augmentation: test_split={inference_split!r}"
        )
    if inference_split not in {"val", "test"}:
        raise RuntimeError(
            f"Unsupported teacher-cache inference split: {inference_split!r}"
        )

    inference_root = Path(dataset_config.get("root_dir", ""))
    inference_split_file = inference_root / "ImageSets" / f"{inference_split}.txt"
    if not inference_split_file.is_file():
        raise FileNotFoundError(
            f"Inference-view split file not found: {inference_split_file}"
        )
    inference_ids = load_split_ids(inference_split_file)
    if inference_ids != expected_sample_ids:
        raise RuntimeError(
            "Inference-view IDs do not exactly match the requested cache split"
        )
    return {
        "inference_dataset_root": str(inference_root),
        "inference_dataset_split": inference_split,
        "inference_split_file_sha256": sha256_file(inference_split_file),
        "inference_data_augmentation": False,
    }


def create_cache(
    *,
    prediction_dir: Path,
    split_file: Path,
    output_dir: Path,
    runtime_config: Path,
    teacher_name: str,
    teacher_source_commit: str,
    checkpoint_sha256: str,
    expected_count: int,
    allowed_classes: Iterable[str] | None = None,
) -> Dict[str, Any]:
    sample_ids = load_split_ids(split_file)
    if len(sample_ids) != expected_count:
        raise RuntimeError(
            f"Expected {expected_count} split IDs, found {len(sample_ids)} in {split_file}"
        )

    inference_contract = validate_inference_contract(
        runtime_config,
        sample_ids,
    )

    validation = validate_prediction_directory(
        prediction_dir,
        sample_ids,
        allowed_classes=allowed_classes,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cached_prediction_dir = output_dir / "predictions"
    cached_prediction_dir.mkdir(parents=True, exist_ok=True)

    incomplete_manifest = {
        "schema_version": 1,
        "complete": False,
        "teacher_name": teacher_name,
        "split": split_file.stem,
        "split_images": len(sample_ids),
    }
    manifest_path = output_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(incomplete_manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    for sample_id in sample_ids:
        shutil.copy2(
            prediction_dir / f"{sample_id}.txt",
            cached_prediction_dir / f"{sample_id}.txt",
        )
    shutil.copy2(runtime_config, output_dir / "runtime_config.yaml")

    cached_validation = validate_prediction_directory(
        cached_prediction_dir,
        sample_ids,
        allowed_classes=allowed_classes,
    )
    if cached_validation != validation:
        raise RuntimeError("Cached predictions differ from validated source predictions")

    manifest = {
        "schema_version": 1,
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "teacher_name": teacher_name,
        "teacher_source_commit": teacher_source_commit,
        "checkpoint_sha256": checkpoint_sha256,
        "runtime_config_sha256": sha256_file(runtime_config),
        "split": split_file.stem,
        "split_file": str(split_file),
        "split_file_sha256": sha256_file(split_file),
        "split_images": len(sample_ids),
        "allowed_classes": list(allowed_classes or []),
        **inference_contract,
        **cached_validation,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and persist a provenance-tracked KITTI teacher cache."
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--split-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--teacher-name", required=True)
    parser.add_argument("--teacher-source-commit", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--allowed-classes", nargs="+", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = create_cache(
        prediction_dir=args.prediction_dir,
        split_file=args.split_file,
        output_dir=args.output_dir,
        runtime_config=args.runtime_config,
        teacher_name=args.teacher_name,
        teacher_source_commit=args.teacher_source_commit,
        checkpoint_sha256=args.checkpoint_sha256,
        expected_count=args.expected_count,
        allowed_classes=args.allowed_classes,
    )
    print(json.dumps(manifest, indent=2))
    print(f"Teacher cache complete: {args.output_dir / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
