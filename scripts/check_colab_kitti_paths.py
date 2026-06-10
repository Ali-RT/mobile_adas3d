from pathlib import Path

from tools.cli import parse_config_arg
from tools.config import load_runtime_config_from_args

def main() -> None:
    args = parse_config_arg("Check KITTI dataset paths")
    config = load_runtime_config_from_args(args)

    dataset_cfg = config["dataset"]
    active_profile = dataset_cfg["active_profile"]
    root_dir = Path(dataset_cfg["profiles"][active_profile]["root_dir"])

    image_dir = root_dir / dataset_cfg["image_dir"]
    label_dir = root_dir / dataset_cfg["label_dir"]
    calib_dir = root_dir / dataset_cfg["calib_dir"]

    print("Checking KITTI paths...")
    print(f"Using config: {args.config}")
    print(f"Active profile: {active_profile}")
    print(f"Root dir: {root_dir}")

    expected_dirs = {
        "image_dir": image_dir,
        "label_dir": label_dir,
        "calib_dir": calib_dir,
    }

    for name, path in expected_dirs.items():
        print(f"{name}: {path}")
        print(f"  exists: {path.exists()}")

    if image_dir.exists():
        print(f"Number of images: {len(list(image_dir.glob('*.png')))}")

    if label_dir.exists():
        print(f"Number of labels: {len(list(label_dir.glob('*.txt')))}")

    if calib_dir.exists():
        print(f"Number of calib files: {len(list(calib_dir.glob('*.txt')))}")

    required_sample = {
        "image": image_dir / "000000.png",
        "label": label_dir / "000000.txt",
        "calib": calib_dir / "000000.txt",
    }

    print("\nChecking sample 000000:")
    for name, path in required_sample.items():
        print(f"{name}: {path.exists()} -> {path}")

    missing = [
        str(path)
        for path in expected_dirs.values()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Some KITTI directories are missing:\n" + "\n".join(missing)
        )

    print("\nKITTI path check complete.")


if __name__ == "__main__":
    main()