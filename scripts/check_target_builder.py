from data.kitti_dataset import KITTIDataset
from data.target_builder import build_targets_for_sample
from tools.config import load_config


def main() -> None:
    config = load_config("configs/kitti_mobileadas3d.yaml")

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    target_cfg = config["targets"]

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

    targets = build_targets_for_sample(
        sample=sample,
        classes=dataset_cfg["classes"],
        input_height=model_cfg["input_height"],
        input_width=model_cfg["input_width"],
        output_stride=model_cfg["output_stride"],
        class_mean_dims=target_cfg["class_mean_dims"],
    )

    print("Target builder ran successfully.")
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Number of objects in sample: {len(sample['objects'])}")

    print("\nTarget shapes:")
    for name, tensor in targets.items():
        print(f"  {name}: {tuple(tensor.shape)}")

    valid_mask = targets["valid_mask"]
    num_positive_cells = int(valid_mask.sum().item())

    print(f"\nNumber of positive cells: {num_positive_cells}")

    cls_target = targets["cls_target"]
    box2d_target = targets["box2d_target"]
    log_depth_target = targets["log_depth_target"]
    dim_target = targets["dim_target"]
    yaw_target = targets["yaw_target"]
    offset_target = targets["offset_target"]

    positive_indices = valid_mask[0].nonzero(as_tuple=False)

    print("\nPositive cell details:")
    for idx in positive_indices:
        gy = int(idx[0].item())
        gx = int(idx[1].item())

        class_id = int(cls_target[:, gy, gx].argmax().item())
        class_name = dataset_cfg["classes"][class_id]

        box = box2d_target[:, gy, gx].tolist()
        log_depth = float(log_depth_target[0, gy, gx].item())
        dim_residual = dim_target[:, gy, gx].tolist()
        yaw_sincos = yaw_target[:, gy, gx].tolist()
        offset = offset_target[:, gy, gx].tolist()

        print(
            f"  cell=(y={gy}, x={gx}) "
            f"class={class_name} "
            f"box={[round(v, 2) for v in box]} "
            f"log_depth={log_depth:.3f} "
            f"dim_residual={[round(v, 3) for v in dim_residual]} "
            f"yaw_sincos={[round(v, 3) for v in yaw_sincos]} "
            f"offset={[round(v, 3) for v in offset]}"
        )


if __name__ == "__main__":
    main()