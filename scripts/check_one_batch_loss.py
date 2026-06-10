import torch
import torch.nn.functional as F

from data.kitti_dataset import KITTIDataset
from data.target_builder import build_targets_for_sample
from losses.mobile_adas3d_loss import MobileADAS3DLoss
from models.build import build_model
from tools.cli import parse_config_arg
from tools.config import load_config


def add_batch_dim_to_targets(targets):
    return {
        name: tensor.unsqueeze(0)
        for name, tensor in targets.items()
    }


def main() -> None:
    args = parse_config_arg("Check one-batch MobileADAS3D loss")
    config = load_config(args.config)

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

    image = sample["image"].unsqueeze(0)

    input_height = model_cfg["input_height"]
    input_width = model_cfg["input_width"]

    image = F.interpolate(
        image,
        size=(input_height, input_width),
        mode="bilinear",
        align_corners=False,
    )

    targets = build_targets_for_sample(
        sample=sample,
        classes=dataset_cfg["classes"],
        input_height=input_height,
        input_width=input_width,
        output_stride=model_cfg["output_stride"],
        class_mean_dims=target_cfg["class_mean_dims"],
    )

    targets = add_batch_dim_to_targets(targets)

    model = build_model(config)
    model.train()

    criterion = MobileADAS3DLoss(
        input_height=input_height,
        input_width=input_width,
    )

    outputs = model(image)
    losses = criterion(outputs, targets)

    print("One-batch loss check successful.")
    print(f"Using config: {args.config}")
    print(f"Active profile: {active_profile}")
    print(f"Sample ID: {sample['sample_id']}")
    print(f"Image shape: {tuple(image.shape)}")

    print("\nOutput shapes:")
    for name, tensor in outputs.items():
        print(f"  {name}: {tuple(tensor.shape)}")

    print("\nTarget shapes:")
    for name, tensor in targets.items():
        print(f"  {name}: {tuple(tensor.shape)}")

    print("\nLoss values:")
    for name, value in losses.items():
        print(f"  {name}: {float(value.item()):.6f}")

    total_loss = losses["total_loss"]

    if not torch.isfinite(total_loss):
        raise RuntimeError(f"total_loss is not finite: {total_loss}")

    total_loss.backward()

    print("\nBackward pass successful.")
    print("Gradients were computed.")


if __name__ == "__main__":
    main()