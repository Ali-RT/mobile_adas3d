from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping


KITTI_PRODUCTION_CLASS_MAPPING = {
    "Car": "Vehicle",
    "Van": "Vehicle",
    "Truck": "Vehicle",
    "Tram": "Vehicle",
    "Pedestrian": "Pedestrian",
    "Person_sitting": "Pedestrian",
}


def normalize_class_mapping(
    mapping: Mapping[str, str] | None,
    classes: Iterable[str],
) -> Dict[str, str]:
    if mapping is None:
        return {}
    normalized = {str(source): str(target) for source, target in mapping.items()}
    targets = set(normalized.values())
    expected = set(classes)
    unknown = targets - expected
    missing = expected - targets
    if unknown or missing:
        raise ValueError(
            "Invalid class_mapping targets: "
            f"unknown={sorted(unknown)}, missing={sorted(missing)}"
        )
    return normalized


def taxonomy_sha256(classes: Iterable[str], mapping: Mapping[str, str]) -> str:
    payload = {
        "classes": list(classes),
        "class_mapping": dict(sorted(mapping.items())),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def map_objects(
    objects: List[Dict[str, Any]],
    mapping: Mapping[str, str],
) -> List[Dict[str, Any]]:
    mapped = []
    for original in objects:
        source_name = str(original["class_name"])
        target_name = mapping.get(source_name)
        if target_name is None:
            continue
        item = dict(original)
        item["source_class_name"] = source_name
        item["class_name"] = target_name
        mapped.append(item)
    return mapped


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_label_tree_sha256(
    label_dir: str | Path, split_file: str | Path
) -> str:
    from data.splits import read_split_file

    root = Path(label_dir)
    digest = hashlib.sha256()
    for sample_id in read_split_file(split_file):
        normalized = f"{int(sample_id):06d}"
        path = root / f"{normalized}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Missing KITTI label for manifest: {path}")
        digest.update(normalized.encode("ascii"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def validate_taxonomy_manifest(
    manifest_path: str | Path,
    classes: Iterable[str],
    mapping: Mapping[str, str],
    split_files: Mapping[str, str | Path],
    label_dir: str | Path,
) -> Dict[str, Any]:
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Required taxonomy manifest missing: {path}. Run "
            "scripts/create_kitti_taxonomy_manifest.py before training."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not manifest.get("complete"):
        raise RuntimeError(f"Taxonomy manifest is incomplete: {path}")
    expected_taxonomy_hash = taxonomy_sha256(classes, mapping)
    if manifest.get("taxonomy_sha256") != expected_taxonomy_hash:
        raise RuntimeError("Taxonomy manifest mapping hash does not match config")
    for split_name, split_file in split_files.items():
        expected_hash = file_sha256(split_file)
        actual_hash = manifest.get("splits", {}).get(split_name, {}).get(
            "split_file_sha256"
        )
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"Taxonomy manifest {split_name} split hash does not match config"
            )
        expected_labels_hash = split_label_tree_sha256(label_dir, split_file)
        actual_labels_hash = manifest.get("splits", {}).get(split_name, {}).get(
            "label_tree_sha256"
        )
        if actual_labels_hash != expected_labels_hash:
            raise RuntimeError(
                f"Taxonomy manifest {split_name} label-tree hash does not match dataset"
            )
    return manifest


def count_mapped_objects(
    objects: Iterable[Mapping[str, Any]], mapping: Mapping[str, str]
) -> Dict[str, Counter]:
    source = Counter()
    mapped = Counter()
    excluded = Counter()
    for obj in objects:
        source_name = str(obj["class_name"])
        source[source_name] += 1
        target = mapping.get(source_name)
        if target is None:
            excluded[source_name] += 1
        else:
            mapped[target] += 1
    return {"source": source, "mapped": mapped, "excluded": excluded}
