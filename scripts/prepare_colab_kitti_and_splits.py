from pathlib import Path
import shutil
import subprocess

from data.splits import create_train_val_test_split, write_split_file
from tools.cli import parse_config_arg
from tools.config import load_config


def run_cmd(cmd: list[str]) -> None:
    print("\nRunning:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def copy_with_rsync(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)

    run_cmd([
        "rsync",
        "-ah",
        "--info=progress2",
        f"{source}/",
        f"{destination}/",
    ])


def copy_with_shutil(source: Path, destination: Path) -> None:
    if destination.exists():
        print(f"Destination already exists, removing: {destination}")
        shutil.rmtree(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Copying {source} -> {destination}")
    shutil.copytree(source, destination)


def copy_directory(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")

    if shutil.which("rsync") is not None:
        copy_with_rsync(source, destination)
    else:
        copy_with_shutil(source, destination)


def count_files(path: Path, suffix: str) -> int:
    if not path.exists():
        return 0
    return len(list(path.glob(f"*{suffix}")))


def discover_valid_sample_ids(root_dir: Path) -> list[str]:
    image_dir = root_dir / "training/image_2"
    label_dir = root_dir / "training/label_2"
    calib_dir = root_dir / "training/calib"

    image_ids = {p.stem for p in image_dir.glob("*.png")}
    label_ids = {p.stem for p in label_dir.glob("*.txt")}
    calib_ids = {p.stem for p in calib_dir.glob("*.txt")}

    valid_ids = sorted(image_ids & label_ids & calib_ids)

    missing_labels = sorted(image_ids - label_ids)
    missing_calibs = sorted(image_ids - calib_ids)

    print(f"Images found: {len(image_ids)}")
    print(f"Labels found: {len(label_ids)}")
    print(f"Calib files found: {len(calib_ids)}")
    print(f"Valid complete samples: {len(valid_ids)}")

    if missing_labels:
        print(f"Warning: {len(missing_labels)} images missing labels.")

    if missing_calibs:
        print(f"Warning: {len(missing_calibs)} images missing calibration.")

    if len(valid_ids) == 0:
        raise RuntimeError("No valid KITTI samples found.")

    return valid_ids


def main() -> None:
    args = parse_config_arg("Prepare KITTI local copy and train/val/test splits")
    config = load_config(args.config)

    if "colab_data" not in config:
        raise KeyError("Missing colab_data section in config.")

    dataset_cfg = config["dataset"]
    split_cfg = dataset_cfg["splits"]
    colab_data_cfg = config["colab_data"]

    drive_root = Path(colab_data_cfg["drive_root_dir"])
    local_root = Path(colab_data_cfg["local_root_dir"])
    required_subdirs = colab_data_cfg["required_subdirs"]

    print("Preparing KITTI for Colab.")
    print(f"Using config: {args.config}")
    print(f"Drive KITTI root: {drive_root}")
    print(f"Local KITTI root: {local_root}")

    if not drive_root.exists():
        raise FileNotFoundError(
            f"Drive KITTI root does not exist: {drive_root}\n"
            "Mount Google Drive and download/extract KITTI first."
        )

    for subdir in required_subdirs:
        source = drive_root / subdir
        destination = local_root / subdir

        print("\n----------------------------------------")
        print(f"Copying subdir: {subdir}")
        print(f"Source: {source}")
        print(f"Destination: {destination}")

        copy_directory(source, destination)

    print("\nLocal KITTI copy summary:")
    print("Images:", count_files(local_root / "training/image_2", ".png"))
    print("Labels:", count_files(local_root / "training/label_2", ".txt"))
    print("Calib:", count_files(local_root / "training/calib", ".txt"))

    sample_ids = discover_valid_sample_ids(local_root)

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

    total = len(splits["train"]) + len(splits["val"]) + len(splits["test"])
    print(f"Total split samples: {total}")

    if total != len(sample_ids):
        raise RuntimeError("Split sample count mismatch.")

    print("\nKITTI local copy and split creation complete.")
    print("No GPU was required for this step.")


if __name__ == "__main__":
    main()