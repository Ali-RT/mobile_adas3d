from pathlib import Path

from data.splits import read_split_file
from data.split_resolver import get_split_file
from tools.cli import parse_config_arg
from tools.config import load_config


def main() -> None:
    args = parse_config_arg("Check KITTI train/val/test splits")
    config = load_config(args.config)

    dataset_cfg = config["dataset"]
    active_profile = dataset_cfg["active_profile"]
    root_dir = Path(dataset_cfg["profiles"][active_profile]["root_dir"])

    print("Checking KITTI splits.")
    print(f"Using config: {args.config}")
    print(f"Active profile: {active_profile}")
    print(f"Root dir: {root_dir}")

    all_ids = {}

    for split_name in ["train", "val", "test"]:
        split_file = Path(get_split_file(config, split_name))
        sample_ids = read_split_file(split_file)
        all_ids[split_name] = set(sample_ids)

        print(f"\n{split_name.upper()}")
        print(f"Split file: {split_file}")
        print(f"Samples: {len(sample_ids)}")

        print("First 5 sample IDs:", sample_ids[:5])

        for sample_id in sample_ids[:5]:
            image_path = root_dir / dataset_cfg["image_dir"] / f"{sample_id}.png"
            label_path = root_dir / dataset_cfg["label_dir"] / f"{sample_id}.txt"
            calib_path = root_dir / dataset_cfg["calib_dir"] / f"{sample_id}.txt"

            print(
                f"  {sample_id}: "
                f"image={image_path.exists()} "
                f"label={label_path.exists()} "
                f"calib={calib_path.exists()}"
            )

    train_val_overlap = all_ids["train"] & all_ids["val"]
    train_test_overlap = all_ids["train"] & all_ids["test"]
    val_test_overlap = all_ids["val"] & all_ids["test"]

    print("\nOverlap checks:")
    print(f"train ∩ val: {len(train_val_overlap)}")
    print(f"train ∩ test: {len(train_test_overlap)}")
    print(f"val ∩ test: {len(val_test_overlap)}")

    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise RuntimeError("Split overlap detected.")

    total_unique = len(all_ids["train"] | all_ids["val"] | all_ids["test"])
    total_sum = sum(len(v) for v in all_ids.values())

    print(f"\nTotal unique split samples: {total_unique}")
    print(f"Total split rows: {total_sum}")

    print("\nKITTI split check complete.")


if __name__ == "__main__":
    main()