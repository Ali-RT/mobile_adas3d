from __future__ import annotations

from typing import Any, Dict

from models.mobile_adas3d import MobileADAS3D
from models.mobile_adas3d_s1 import MobileADAS3DS1
from models.mobile_adas3d_h1 import MobileADAS3DH1
from models.mobile_adas3d_h2 import MobileADAS3DH2


def build_model(config: Dict[str, Any]) -> MobileADAS3D | MobileADAS3DS1 | MobileADAS3DH1:
    dataset_cfg = config["dataset"]
    model_cfg = config["model"]

    if model_cfg.get("name") in {"MobileADAS3D-H1", "MobileADAS3D-H2"}:
        model_class = (
            MobileADAS3DH2
            if model_cfg.get("name") == "MobileADAS3D-H2"
            else MobileADAS3DH1
        )
        return model_class(
            num_classes=len(dataset_cfg["classes"]),
            backbone_name=model_cfg.get(
                "backbone", "mobilenetv4_conv_small.e2400_r224_in1k"
            ),
            pretrained=bool(model_cfg.get("pretrained", True)),
            input_height=int(model_cfg["input_height"]),
            input_width=int(model_cfg["input_width"]),
            fpn_channels=int(model_cfg.get("fpn_channels", 128)),
            transformer_width=int(model_cfg.get("transformer_width", 192)),
            attention_heads=int(model_cfg.get("attention_heads", 6)),
            encoder_layers=int(model_cfg.get("encoder_layers", 2)),
            decoder_layers=int(model_cfg.get("decoder_layers", 2)),
            feedforward_width=int(model_cfg.get("feedforward_width", 768)),
            num_queries=int(model_cfg.get("num_queries", 50)),
            depth_bins=int(model_cfg.get("depth_bins", 40)),
            **(
                {"center_offset_scale": float(model_cfg.get("center_offset_scale", 0.10))}
                if model_cfg.get("name") == "MobileADAS3D-H2"
                else {}
            ),
        )

    if model_cfg.get("name") == "MobileADAS3D-S1":
        return MobileADAS3DS1(
            num_classes=len(dataset_cfg["classes"]),
            backbone_name=model_cfg.get(
                "backbone", "mobilenetv4_conv_small.e2400_r224_in1k"
            ),
            pretrained=bool(model_cfg.get("pretrained", True)),
            fpn_channels=int(model_cfg.get("fpn_channels", 96)),
            yaw_encoding=str(model_cfg.get("yaw_encoding", "axis_direction")),
        )

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
