from pathlib import Path

import numpy as np

from data.kitti_dataset import KITTIDataset
from data.visualization import draw_2d_boxes, draw_projected_3d_boxes
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

    sample = dataset[0]

    print("Loaded sample for 3D projection.")
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Number of objects: {len(sample['objects'])}")
    print(f"P2 shape: {tuple(sample['P2'].shape)}")

    output_dir = Path(config["outputs"]["visualization_dir"])

    output_2d_path = output_dir / f"{sample['sample_id']}_2d_boxes.png"
    output_3d_path = output_dir / f"{sample['sample_id']}_projected_3d_boxes.png"

    draw_2d_boxes(
        image_rgb=sample["image_rgb"],
        objects=sample["objects"],
        output_path=output_2d_path,
    )

    draw_projected_3d_boxes(
        image_rgb=sample["image_rgb"],
        objects=sample["objects"],
        P2=sample["P2"].numpy().astype(np.float32),
        output_path=output_3d_path,
    )

    print(f"Saved 2D visualization to: {output_2d_path}")
    print(f"Saved 3D projection visualization to: {output_3d_path}")

    for obj in sample["objects"]:
        print(
            f"class={obj['class_name']} "
            f"location_3d={obj['location_3d']} "
            f"dimensions_3d={obj['dimensions_3d']} "
            f"rotation_y={obj['rotation_y']}"
        )


if __name__ == "__main__":
    main()