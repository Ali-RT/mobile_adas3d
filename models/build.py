from __future__ import annotations

from typing import Any, Dict

from models.mobile_adas3d import MobileADAS3D


def build_model(config: Dict[str, Any]) -> MobileADAS3D:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    model = MobileADAS3D(
        num_classes=len(dataset_cfg["classes"]),
        backbone_name=model_cfg.get("backbone", "mobilenet_v3_small"),
        pretrained=bool(model_cfg.get("pretrained", True)),
        normalize_imagenet=bool(model_cfg.get("normalize_imagenet", False)),
        input_height=int(model_cfg["input_height"]),
        input_width=int(model_cfg["input_width"]),
        fpn_channels=int(model_cfg.get("fpn_channels", 128)),
        head_channels=int(model_cfg.get("head_channels", 256)),
        use_projected_center=bool(
            model_cfg.get("use_projected_center", False)
            or model_cfg.get("heads", {}).get("projected_center_offset", False)
        ),
        use_quality=bool(
            model_cfg.get("use_quality", False)
            or model_cfg.get("heads", {}).get("quality", False)
        ),
        use_yaw_axis_direction=bool(
            model_cfg.get("use_yaw_axis_direction", False)
            or model_cfg.get("heads", {}).get("yaw_direction", False)
        ),
    )

    return model
