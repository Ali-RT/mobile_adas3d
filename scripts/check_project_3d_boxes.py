import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path

import numpy as np

from data.kitti_dataset import KITTIDataset
from data.visualization import draw_2d_boxes, draw_projected_3d_boxes
from tools.cli import parse_config_arg
from tools.config import load_runtime_config_from_args


def main() -> None:
    args = parse_config_arg("Check projected KITTI 3D boxes")
    config = load_runtime_config_from_args(args)

    dataset_cfg = config["dataset"]
    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        class_mapping=dataset_cfg.get("class_mapping"),
    )

    sample = dataset[0]

    print("Loaded sample for 3D projection.")
    print(f"Using config: {args.config}")
    print(f"Active profile: {active_profile}")
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
