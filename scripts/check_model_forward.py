import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F

from data.kitti_dataset import KITTIDataset
from models.build import build_model
from tools.cli import parse_config_arg
from tools.config import load_runtime_config_from_args

def main() -> None:
    args = parse_config_arg("Check MobileADAS3D model forward pass")
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
    )

    sample = dataset[0]
    image = sample["image"].unsqueeze(0)

    input_height = config["model"]["input_height"]
    input_width = config["model"]["input_width"]

    image = F.interpolate(
        image,
        size=(input_height, input_width),
        mode="bilinear",
        align_corners=False,
    )

    model = build_model(config)
    model.eval()

    with torch.no_grad():
        outputs = model(image)

    print("Model forward pass successful.")
    print(f"Using config: {args.config}")
    print(f"Active profile: {active_profile}")
    print(f"Input image shape: {tuple(image.shape)}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    print("\nOutput shapes:")
    for name, tensor in outputs.items():
        print(f"  {name}: {tuple(tensor.shape)}")

    cls_logits = outputs["cls_logits"]
    feature_height = cls_logits.shape[-2]
    feature_width = cls_logits.shape[-1]

    stride_h = input_height / feature_height
    stride_w = input_width / feature_width

    print("\nFeature map info:")
    print(f"  feature height: {feature_height}")
    print(f"  feature width: {feature_width}")
    print(f"  approximate stride_h: {stride_h:.2f}")
    print(f"  approximate stride_w: {stride_w:.2f}")


if __name__ == "__main__":
    main()