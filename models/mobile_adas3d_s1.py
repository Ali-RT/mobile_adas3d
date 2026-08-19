from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mobile_adas3d import decode_yaw_axis_direction


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        groups: int = 1,
    ) -> None:
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseSeparableBlock(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            ConvBNReLU(channels, channels, kernel_size=3, groups=channels),
            ConvBNReLU(channels, channels, kernel_size=1),
        )


class MobileNetV4S1Pyramid(nn.Module):
    """MobileNetV4 Conv Small features at strides 8, 16, and 32."""

    def __init__(self, model_name: str, pretrained: bool) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError(
                "MobileADAS3D-S1 requires timm. Install requirements.txt."
            ) from error

        self.features = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )
        reductions = list(self.features.feature_info.reduction())
        channels = list(self.features.feature_info.channels())
        if reductions != [8, 16, 32]:
            raise RuntimeError(
                f"{model_name} returned reductions {reductions}; expected [8, 16, 32]"
            )
        self.output_channels = tuple(int(value) for value in channels)

    def forward(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.features(images)
        return features[0], features[1], features[2]


class MobileADAS3DS1(nn.Module):
    """Core-ML-native stride-8 MobileADAS3D student."""

    architecture_name = "MobileADAS3D-S1"

    def __init__(
        self,
        num_classes: int = 2,
        backbone_name: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        pretrained: bool = True,
        fpn_channels: int = 96,
        yaw_encoding: str = "axis_direction",
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError(
                f"MobileADAS3D-S1 requires exactly 2 classes, got {num_classes}"
            )
        if not backbone_name.startswith("mobilenetv4_conv_small"):
            raise ValueError(
                "MobileADAS3D-S1 is locked to a MobileNetV4 Conv Small backbone"
            )
        if fpn_channels not in (64, 96):
            raise ValueError("MobileADAS3D-S1 fpn_channels must be 96 or fallback 64")
        if yaw_encoding not in ("axis_direction", "continuous_sincos"):
            raise ValueError(
                "MobileADAS3D-S1 yaw_encoding must be 'axis_direction' or "
                f"'continuous_sincos', got {yaw_encoding!r}"
            )

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.fpn_channels = fpn_channels
        self.yaw_encoding = yaw_encoding
        self.architecture_name = (
            "MobileADAS3D-S1-V2"
            if yaw_encoding == "continuous_sincos"
            else "MobileADAS3D-S1"
        )
        self.normalize_imagenet = True
        self.output_stride = 8
        self.backbone = MobileNetV4S1Pyramid(backbone_name, pretrained)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.register_buffer("input_mean", mean.view(1, 3, 1, 1))
        self.register_buffer("input_std", std.view(1, 3, 1, 1))

        c8, c16, c32 = self.backbone.output_channels
        self.lateral8 = ConvBNReLU(c8, fpn_channels, kernel_size=1)
        self.lateral16 = ConvBNReLU(c16, fpn_channels, kernel_size=1)
        self.lateral32 = ConvBNReLU(c32, fpn_channels, kernel_size=1)
        self.refine16 = DepthwiseSeparableBlock(fpn_channels)
        self.refine8 = DepthwiseSeparableBlock(fpn_channels)
        self.prediction_tower = DepthwiseSeparableBlock(fpn_channels)

        self.cls_head = nn.Conv2d(fpn_channels, num_classes, kernel_size=1)
        self.quality_head = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        self.box2d_head = nn.Conv2d(fpn_channels, 4, kernel_size=1)
        self.center_offset_head = nn.Conv2d(fpn_channels, 2, kernel_size=1)
        self.projected_center_offset_head = nn.Conv2d(
            fpn_channels, 2, kernel_size=1
        )
        self.depth_head = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        self.depth_uncertainty_head = nn.Conv2d(fpn_channels, 1, kernel_size=1)
        self.dim_head = nn.Conv2d(fpn_channels, 3, kernel_size=1)
        if yaw_encoding == "axis_direction":
            self.yaw_axis_head = nn.Conv2d(fpn_channels, 2, kernel_size=1)
            self.yaw_direction_head = nn.Conv2d(fpn_channels, 1, kernel_size=1)
            self.export_output_names = S1_OUTPUT_NAMES
        else:
            self.yaw_head = nn.Conv2d(fpn_channels, 2, kernel_size=1)
            self.export_output_names = S1_V2_OUTPUT_NAMES
        self.loc_xy_head = nn.Conv2d(fpn_channels, 2, kernel_size=1)

    @staticmethod
    def _resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            source,
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        images = (images - self.input_mean) / self.input_std
        feat8, feat16, feat32 = self.backbone(images)
        lateral8 = self.lateral8(feat8)
        lateral16 = self.lateral16(feat16)
        lateral32 = self.lateral32(feat32)
        p16 = self.refine16(lateral16 + self._resize_like(lateral32, lateral16))
        p8 = self.refine8(lateral8 + self._resize_like(p16, lateral8))
        feature = self.prediction_tower(p8)

        outputs = {
            "cls_logits": self.cls_head(feature),
            "quality": self.quality_head(feature),
            "box2d": F.softplus(self.box2d_head(feature)),
            "center_offset": self.center_offset_head(feature),
            "projected_center_offset": self.projected_center_offset_head(feature),
            "log_depth": self.depth_head(feature),
            "depth_uncertainty": self.depth_uncertainty_head(feature),
            "dim": self.dim_head(feature),
            "loc_xy": self.loc_xy_head(feature),
        }
        if self.yaw_encoding == "axis_direction":
            yaw_axis = self.yaw_axis_head(feature)
            yaw_direction = self.yaw_direction_head(feature)
            outputs.update(
                {
                    "yaw": decode_yaw_axis_direction(yaw_axis, yaw_direction),
                    "yaw_axis": yaw_axis,
                    "yaw_direction": yaw_direction,
                }
            )
        else:
            outputs["yaw"] = self.yaw_head(feature)
        return outputs


S1_OUTPUT_NAMES = (
    "cls_logits",
    "quality",
    "box2d",
    "center_offset",
    "projected_center_offset",
    "log_depth",
    "depth_uncertainty",
    "dim",
    "yaw_axis",
    "yaw_direction",
    "loc_xy",
)

S1_V2_OUTPUT_NAMES = (
    "cls_logits",
    "quality",
    "box2d",
    "center_offset",
    "projected_center_offset",
    "log_depth",
    "depth_uncertainty",
    "dim",
    "yaw",
    "loc_xy",
)


class MobileADAS3DS1TupleWrapper(nn.Module):
    """Export the learned heads for the selected S1 yaw representation."""

    def __init__(self, model: MobileADAS3DS1) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = self.model(images)
        return tuple(outputs[name] for name in self.model.export_output_names)
