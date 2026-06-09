from pprint import pprint

from data.kitti_parser import load_kitti_sample
from tools.config import load_config


def main() -> None:
    config = load_config("configs/kitti_mobileadas3d.yaml")

    dataset_cfg = config["dataset"]
    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]
    classes = config["dataset"]["classes"]
    
    # Change this if you test another sample.
    sample_id = "000000"

    sample = load_kitti_sample(
        root_dir=root_dir,
        sample_id=sample_id,
        allowed_classes=classes,
    )

    print("Loaded KITTI sample successfully.")
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Image path: {sample['image_path']}")
    print(f"Number of selected objects: {len(sample['objects'])}")

    print("\nCamera intrinsics K:")
    pprint(sample["K"])

    print("\nFirst object:")
    if sample["objects"]:
        pprint(sample["objects"][0])
    else:
        print("No objects found for selected classes.")


if __name__ == "__main__":
    main()