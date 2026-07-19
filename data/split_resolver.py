from pathlib import Path
from typing import Any, Dict


def get_split_file(config: Dict[str, Any], split_name: str) -> str:
    dataset_cfg = config["dataset"]

    if "splits" not in dataset_cfg:
        raise KeyError("Missing dataset.splits section in config.")

    split_cfg = dataset_cfg["splits"]
    active_profile = dataset_cfg.get("active_profile")
    profile_split_dirs = split_cfg.get("profile_split_dirs", {})
    split_dir = Path(
        profile_split_dirs.get(active_profile, split_cfg["split_dir"])
    )

    if split_name == "train":
        filename = split_cfg["train_file"]
    elif split_name == "val":
        filename = split_cfg["val_file"]
    elif split_name == "test":
        filename = split_cfg["test_file"]
    else:
        raise ValueError(f"Unsupported split_name: {split_name}")

    return str(split_dir / filename)
