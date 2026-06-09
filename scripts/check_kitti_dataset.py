from pathlib import Path

from data.kitti_dataset import KITTIDataset
from data.visualization import draw_2d_boxes
from tools.config import load_config


def main() -> None:
    config = load_config("configs/kitti_mobileadas3d.yaml")

    dataset_cfg = config["dataset"]
    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
    )

    print("KITTI Dataset loaded successfully.")
    print(f"Active profile: {active_profile}")
    print(f"Root dir: {root_dir}")
    print(f"Number of samples: {len(dataset)}")
    print(f"Classes: {dataset.classes}")
    print(f"Class mapping: {dataset.class_to_id}")

    sample = dataset[0]

    print("\nFirst sample:")
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Image tensor shape: {tuple(sample['image'].shape)}")
    print(f"Original size: {sample['original_size']}")
    print(f"Number of objects: {len(sample['objects'])}")

    for obj in sample["objects"]:
        print(
            f"  class={obj['class_name']} "
            f"class_id={obj['class_id']} "
            f"bbox={obj['bbox_2d']} "
            f"depth={obj['location_3d'][2]:.2f}m"
        )

    output_path = Path(config["outputs"]["visualization_dir"]) / f"{sample['sample_id']}_2d_boxes.png"

    draw_2d_boxes(
        image_rgb=sample["image_rgb"],
        objects=sample["objects"],
        output_path=output_path,
    )

    print(f"\nSaved visualization to: {output_path}")


if __name__ == "__main__":
    main()