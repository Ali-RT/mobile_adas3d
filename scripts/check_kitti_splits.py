from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.splits import read_split_file
from data.split_resolver import get_split_file
from tools.cli import parse_config_arg
from tools.config import load_runtime_config_from_args


def main() -> None:
    args = parse_config_arg("Validate the canonical KITTI Chen split")
    config = load_runtime_config_from_args(args)
    split_cfg = config["dataset"]["splits"]

    if split_cfg.get("protocol") != "chen_3712_3769":
        raise RuntimeError(
            "Expected dataset.splits.protocol=chen_3712_3769; "
            f"got {split_cfg.get('protocol')!r}"
        )

    train_ids = read_split_file(get_split_file(config, "train"))
    val_ids = read_split_file(get_split_file(config, "val"))
    overlap = set(train_ids) & set(val_ids)
    union = set(train_ids) | set(val_ids)

    errors = []
    if len(train_ids) != 3712 or len(set(train_ids)) != 3712:
        errors.append(f"train must contain 3,712 unique IDs; got {len(train_ids)}")
    if len(val_ids) != 3769 or len(set(val_ids)) != 3769:
        errors.append(f"val must contain 3,769 unique IDs; got {len(val_ids)}")
    if overlap:
        errors.append(f"train and val overlap by {len(overlap)} IDs")
    if len(union) != 7481:
        errors.append(f"train+val union must be 7,481; got {len(union)}")
    if errors:
        raise RuntimeError("Invalid KITTI split:\n- " + "\n- ".join(errors))

    print("KITTI canonical split validation passed")
    print(f"  protocol: {split_cfg['protocol']}")
    print(f"  train:    {len(train_ids)}")
    print(f"  val:      {len(val_ids)}")
    print(f"  overlap:  {len(overlap)}")
    print(f"  union:    {len(union)}")


if __name__ == "__main__":
    main()
