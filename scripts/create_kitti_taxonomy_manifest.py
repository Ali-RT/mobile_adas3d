from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.class_taxonomy import (
    file_sha256,
    normalize_class_mapping,
    taxonomy_sha256,
    split_label_tree_sha256,
)
from data.kitti_parser import parse_kitti_label_file
from data.split_resolver import get_split_file
from data.splits import read_split_file
from tools.config import apply_runtime_overrides, load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an audited KITTI source-to-production taxonomy manifest."
    )
    parser.add_argument("--config", default="configs/kitti_mobileadas3d_s1.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def audit_split(
    root: Path,
    label_dir: str,
    split_file: Path,
    mapping: dict[str, str],
) -> dict:
    sample_ids = read_split_file(split_file)
    source_counts = Counter()
    mapped_counts = Counter()
    excluded_counts = Counter()
    mapped_source_counts = defaultdict(Counter)
    dimension_sums = defaultdict(lambda: [0.0, 0.0, 0.0])
    dimension_counts = Counter()
    samples_with_mapped_objects = 0
    for index, sample_id in enumerate(sample_ids, start=1):
        objects = parse_kitti_label_file(
            root / label_dir / f"{int(sample_id):06d}.txt"
        )
        has_mapped = False
        for obj in objects:
            source = obj.class_name
            source_counts[source] += 1
            target = mapping.get(source)
            if target is None:
                excluded_counts[source] += 1
                continue
            has_mapped = True
            mapped_counts[target] += 1
            mapped_source_counts[target][source] += 1
            dimension_counts[target] += 1
            for axis, value in enumerate(obj.dimensions_3d):
                dimension_sums[target][axis] += float(value)
        samples_with_mapped_objects += int(has_mapped)
        if index % 500 == 0 or index == len(sample_ids):
            print(f"{split_file.stem}: audited {index}/{len(sample_ids)} samples")

    mean_dimensions = {
        target: [value / dimension_counts[target] for value in sums]
        for target, sums in sorted(dimension_sums.items())
    }
    return {
        "samples": len(sample_ids),
        "unique_samples": len(set(sample_ids)),
        "samples_with_mapped_objects": samples_with_mapped_objects,
        "split_file": str(split_file),
        "split_file_sha256": file_sha256(split_file),
        "label_tree_sha256": split_label_tree_sha256(root / label_dir, split_file),
        "source_counts": dict(sorted(source_counts.items())),
        "mapped_counts": dict(sorted(mapped_counts.items())),
        "mapped_source_counts": {
            target: dict(sorted(counts.items()))
            for target, counts in sorted(mapped_source_counts.items())
        },
        "excluded_counts": dict(sorted(excluded_counts.items())),
        "mean_dimensions_hwl": mean_dimensions,
    }


def main() -> None:
    args = parse_args()
    config = apply_runtime_overrides(
        load_config(args.config),
        profile=args.profile,
        dataset_root=args.dataset_root,
        split_dir=args.split_dir,
    )
    dataset = config["dataset"]
    profile = dataset["active_profile"]
    root = Path(dataset["profiles"][profile]["root_dir"])
    mapping = normalize_class_mapping(
        dataset.get("class_mapping"), dataset["classes"]
    )
    if not mapping:
        raise ValueError("Config does not define dataset.class_mapping")

    splits = {
        name: audit_split(
            root,
            dataset["label_dir"],
            Path(get_split_file(config, name)),
            mapping,
        )
        for name in ("train", "val")
    }
    complete = (
        splits["train"]["samples"] == 3712
        and splits["val"]["samples"] == 3769
        and all(
            split["mapped_counts"].get(class_name, 0) > 0
            for split in splits.values()
            for class_name in dataset["classes"]
        )
    )
    manifest = {
        "schema_version": 1,
        "complete": complete,
        "protocol": dataset["splits"]["protocol"],
        "classes": dataset["classes"],
        "class_mapping": dict(sorted(mapping.items())),
        "taxonomy_sha256": taxonomy_sha256(dataset["classes"], mapping),
        "dataset_root": str(root),
        "splits": splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    if not complete:
        raise RuntimeError("KITTI taxonomy manifest is incomplete")


if __name__ == "__main__":
    main()
