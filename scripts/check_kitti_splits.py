from pathlib import Path

from data.splits import create_train_val_test_split, write_split_file
from tools.cli import parse_config_arg
from tools.config import load_runtime_config_from_args


def discover_valid_sample_ids(root_dir: Path, image_dir: str, label_dir: str, calib_dir: str) -> list[str]:
    image_path = root_dir / image_dir
    label_path = root_dir / label_dir
    calib_path = root_dir / calib_dir

    image_ids = {p.stem for p in image_path.glob("*.png")}
    label_ids = {p.stem for p in label_path.glob("*.txt")}
    calib_ids = {p.stem for p in calib_path.glob("*.txt")}

    valid_ids = sorted(image_ids & label_ids & calib_ids)

    print(f"Images found: {len(image_ids)}")
    print(f"Labels found: {len(label_ids)}")
    print(f"Calib files found: {len(calib_ids)}")
    print(f"Valid complete samples: {len(valid_ids)}")

    if not valid_ids:
        raise RuntimeError("No valid KITTI samples found.")

    return valid_ids


def main() -> None:
    args = parse_config_arg("Create KITTI train/val/test splits")
    config = load_runtime_config_from_args(args)

    dataset_cfg = config["dataset"]
    split_cfg = dataset_cfg["splits"]

    active_profile = dataset_cfg["active_profile"]
    root_dir = Path(dataset_cfg["profiles"][active_profile]["root_dir"])

    print("Creating KITTI splits.")
    print(f"Using config: {args.config}")
    print(f"Active profile: {active_profile}")
    print(f"KITTI root: {root_dir}")

    sample_ids = discover_valid_sample_ids(
        root_dir=root_dir,
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
    )

    splits = create_train_val_test_split(
        sample_ids=sample_ids,
        train_ratio=float(split_cfg["train_ratio"]),
        val_ratio=float(split_cfg["val_ratio"]),
        test_ratio=float(split_cfg["test_ratio"]),
        seed=int(split_cfg["seed"]),
    )

    split_dir = Path(split_cfg["split_dir"])
    train_path = split_dir / split_cfg["train_file"]
    val_path = split_dir / split_cfg["val_file"]
    test_path = split_dir / split_cfg["test_file"]

    write_split_file(splits["train"], train_path)
    write_split_file(splits["val"], val_path)
    write_split_file(splits["test"], test_path)

    print("\nSplit files saved:")
    print(f"Train: {train_path} ({len(splits['train'])})")
    print(f"Val:   {val_path} ({len(splits['val'])})")
    print(f"Test:  {test_path} ({len(splits['test'])})")

    print("\nKITTI split creation complete.")


if __name__ == "__main__":
    main()