from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ConvHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileNetV3SmallPyramid(nn.Module):
    def __init__(self, pretrained: bool, input_height: int, input_width: int) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.features = mobilenet_v3_small(weights=weights).features
        (
            self.stride16_index,
            self.stride32_index,
            self.stride16_channels,
            self.stride32_channels,
        ) = self._infer_feature_metadata(input_height, input_width)

    def _infer_feature_metadata(
        self, input_height: int, input_width: int
    ) -> Tuple[int, int, int, int]:
        was_training = self.features.training
        self.features.eval()
        output = torch.zeros(1, 3, input_height, input_width)
        found = {}
        with torch.no_grad():
            for index, layer in enumerate(self.features):
                output = layer(output)
                stride_h = int(round(input_height / float(output.shape[-2])))
                stride_w = int(round(input_width / float(output.shape[-1])))
                if stride_h != stride_w:
                    raise RuntimeError(
                        f"Non-square backbone stride at layer {index}: "
                        f"{stride_h} x {stride_w}"
                    )
                if stride_h in (16, 32):
                    found[stride_h] = (index, int(output.shape[1]))
        self.features.train(was_training)
        if 16 not in found or 32 not in found:
            raise RuntimeError("MobileNetV3 must expose stride-16 and stride-32 features")
        return found[16][0], found[32][0], found[16][1], found[32][1]

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = images
        feat16 = images
        feat32 = images
        for index, layer in enumerate(self.features):
            output = layer(output)
            if index == self.stride16_index:
                feat16 = output
            if index == self.stride32_index:
                feat32 = output
        return feat16, feat32


class MobileNetV4Pyramid(nn.Module):
    """timm MobileNetV4 feature extractor returning strides 16 and 32."""

    def __init__(self, model_name: str, pretrained: bool) -> None:
        super().__init__()
        try:
            import timm
        except ImportError as error:
            raise ImportError(
                "MobileNetV4 requires timm. Install requirements.txt or "
                "`pip install timm==1.0.27`."
            ) from error

        self.features = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(3, 4),
        )
        reductions = list(self.features.feature_info.reduction())
        channels = list(self.features.feature_info.channels())
        if reductions != [16, 32]:
            raise RuntimeError(
                f"{model_name} returned reductions {reductions}; expected [16, 32]"
            )
        self.stride16_channels = int(channels[0])
        self.stride32_channels = int(channels[1])

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(images)
        return features[0], features[1]


class MobileADAS3D(nn.Module):
    """Anchor-free mobile monocular-3D detector with a stride-16 FPN."""

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        normalize_imagenet: bool = False,
        input_height: int = 384,
        input_width: int = 1280,
        fpn_channels: int = 128,
        head_channels: int = 256,
        use_projected_center: bool = False,
        use_quality: bool = False,
    ) -> None:
        super().__init__()

        if backbone_name == "mobilenet_v3_small":
            self.backbone = MobileNetV3SmallPyramid(
                pretrained=pretrained,
                input_height=input_height,
                input_width=input_width,
            )
        elif backbone_name.startswith("mobilenetv4_conv_"):
            self.backbone = MobileNetV4Pyramid(
                model_name=backbone_name,
                pretrained=pretrained,
            )
        else:
            raise ValueError(
                f"Unsupported backbone {backbone_name!r}; expected "
                "mobilenet_v3_small or a timm mobilenetv4_conv_* model"
            )

        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.backbone_name = backbone_name
        self.normalize_imagenet = normalize_imagenet
        self.use_projected_center = use_projected_center
        self.use_quality = use_quality

        if normalize_imagenet:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        else:
            mean = [0.0, 0.0, 0.0]
            std = [1.0, 1.0, 1.0]
        self.register_buffer(
            "input_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "input_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

        c16 = self.backbone.stride16_channels
        c32 = self.backbone.stride32_channels
        print(
            f"MobileADAS3D backbone={backbone_name} pretrained={pretrained} "
            f"normalize_imagenet={normalize_imagenet} c16={c16} c32={c32}"
        )

        self.proj16 = nn.Sequential(
            nn.Conv2d(c16, fpn_channels, kernel_size=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.proj32 = nn.Sequential(
            nn.Conv2d(c32, fpn_channels, kernel_size=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(fpn_channels * 2, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )

        self.cls_head = ConvHead(head_channels, head_channels, num_classes)
        self.box2d_head = ConvHead(head_channels, head_channels, 4)
        self.depth_head = ConvHead(head_channels, head_channels, 1)
        self.dim_head = ConvHead(head_channels, head_channels, 3)
        self.yaw_head = ConvHead(head_channels, head_channels, 2)
        self.center_offset_head = ConvHead(head_channels, head_channels, 2)
        self.depth_uncertainty_head = ConvHead(head_channels, head_channels, 1)
        self.loc_xy_head = ConvHead(head_channels, head_channels, 2)
        if self.use_projected_center:
            self.projected_center_offset_head = ConvHead(head_channels, head_channels, 2)
        if self.use_quality:
            self.quality_head = ConvHead(head_channels, head_channels, 1)

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        images = (images - self.input_mean) / self.input_std
        feat16, feat32 = self.backbone(images)
        p16 = self.proj16(feat16)
        p32 = self.proj32(feat32)
        p32 = F.interpolate(p32, size=p16.shape[-2:], mode="bilinear", align_corners=False)
        fused = self.fusion(torch.cat([p16, p32], dim=1))

        outputs = {
            "cls_logits": self.cls_head(fused),
            "box2d": F.softplus(self.box2d_head(fused)),
            "log_depth": self.depth_head(fused),
            "dim": self.dim_head(fused),
            "yaw": self.yaw_head(fused),
            "center_offset": self.center_offset_head(fused),
            "depth_uncertainty": self.depth_uncertainty_head(fused),
            "loc_xy": self.loc_xy_head(fused),
        }

        if self.use_projected_center:
            outputs["projected_center_offset"] = self.projected_center_offset_head(fused)
        if self.use_quality:
            outputs["quality"] = self.quality_head(fused)

        return outputs
