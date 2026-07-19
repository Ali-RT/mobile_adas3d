from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.splits import write_split_file
from tools.config import apply_runtime_overrides, load_config


SOURCES = {
    "train": {
        "url": (
            "https://raw.githubusercontent.com/xinzhuma/monodle/"
            "e426aa65fdc7ceedcaab0d637acf3d3425d0736c/"
            "data/KITTI/ImageSets/train.txt"
        ),
        "sha256": "b6417a1d9b18c8fdb085128e633d28ff321b7674a6d1b3841b8f43d865b281cb",
        "count": 3712,
    },
    "val": {
        "url": (
            "https://raw.githubusercontent.com/xinzhuma/monodle/"
            "e426aa65fdc7ceedcaab0d637acf3d3425d0736c/"
            "data/KITTI/ImageSets/val.txt"
        ),
        "sha256": "657ac4bcc1e156e5b106a4ca18e1f88e012787ea1d2b5d0adeea97fee903fa86",
        "count": 3769,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the canonical KITTI Chen 3712/3769 split."
    )
    parser.add_argument("--config", default="configs/kitti_mobileadas3d.yaml")
    parser.add_argument("--profile", default=None)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override the split directory resolved from the config/profile.",
    )
    parser.add_argument(
        "--source-dir",
        default=None,
        help="Read train.txt and val.txt locally instead of downloading them.",
    )
    return parser.parse_args()


def _read_source(name: str, source_dir: Path | None) -> bytes:
    if source_dir is not None:
        return (source_dir / f"{name}.txt").read_bytes()

    with urllib.request.urlopen(SOURCES[name]["url"], timeout=30) as response:
        return response.read()


def _validate_ids(name: str, payload: bytes) -> list[str]:
    expected = SOURCES[name]
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected["sha256"]:
        raise RuntimeError(
            f"{name} split checksum mismatch: expected {expected['sha256']}, got {digest}"
        )

    ids = [line.strip() for line in payload.decode("utf-8").splitlines() if line.strip()]
    if len(ids) != expected["count"]:
        raise RuntimeError(
            f"{name} split count mismatch: expected {expected['count']}, got {len(ids)}"
        )
    if len(set(ids)) != len(ids):
        raise RuntimeError(f"{name} split contains duplicate sample IDs")
    if any(len(sample_id) != 6 or not sample_id.isdigit() for sample_id in ids):
        raise RuntimeError(f"{name} split contains malformed sample IDs")
    return ids


def resolve_output_dir(config: dict, override: str | None) -> Path:
    if override is not None:
        return Path(override)

    dataset_cfg = config["dataset"]
    split_cfg = dataset_cfg["splits"]
    profile = dataset_cfg.get("active_profile")
    return Path(
        split_cfg.get("profile_split_dirs", {}).get(
            profile,
            split_cfg["split_dir"],
        )
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    config = apply_runtime_overrides(config, profile=args.profile)
    output_dir = resolve_output_dir(config, args.output_dir)
    source_dir = Path(args.source_dir) if args.source_dir else None

    splits = {
        name: _validate_ids(name, _read_source(name, source_dir))
        for name in ("train", "val")
    }

    overlap = set(splits["train"]) & set(splits["val"])
    union = set(splits["train"]) | set(splits["val"])
    if overlap:
        raise RuntimeError(f"Train/val overlap contains {len(overlap)} IDs")
    if len(union) != 7481:
        raise RuntimeError(f"Expected 7,481 total labeled frames, got {len(union)}")

    split_cfg = config["dataset"]["splits"]
    write_split_file(splits["train"], output_dir / split_cfg["train_file"])
    write_split_file(splits["val"], output_dir / split_cfg["val_file"])

    print("Installed canonical KITTI Chen split")
    print(f"  output: {output_dir}")
    print(f"  train:  {len(splits['train'])}")
    print(f"  val:    {len(splits['val'])}")
    print(f"  total:  {len(union)}")
    print("  overlap: 0")


if __name__ == "__main__":
    main()
