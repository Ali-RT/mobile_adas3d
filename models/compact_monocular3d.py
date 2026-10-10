"""Compact CNN architecture candidate; not a trained or qualified detector.

The backbone is ResNet-18 or MobileNetV4 Conv Small. A narrow top-down pyramid replaces RTM3D's three
256-channel transposed convolutions. All nine raw outputs share one 64-channel
feature block. Output meanings follow the RTM3D/KM3D geometry interface, but
the class heatmap is Vehicle/Pedestrian rather than Car/Pedestrian/Cyclist.
New neck/head weights need training; this is not a checkpoint conversion.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet18_Weights, resnet18


CLASSES = ("Vehicle", "Pedestrian")
OUTPUT_CHANNELS = {
    "hm": 2, "wh": 2, "hps": 18, "rot": 8, "dim": 3,
    "prob": 1, "reg": 2, "hm_hp": 9, "hp_offset": 2,
}
OUTPUT_NAMES = tuple(OUTPUT_CHANNELS)
ARCHITECTURE_REVISION = "COMPACT-MONOCULAR3D-RES18-2026-10-09-r1"
MOBILENETV4_ARCHITECTURE_REVISION = "COMPACT-MONOCULAR3D-MNV4-SMALL-2026-10-09-r1"
MOBILENETV4_MODEL = "mobilenetv4_conv_small.e2400_r224_in1k"
MOBILENETV4_TIMM_VERSION = "1.0.27"
BACKBONES = ("resnet18", "mobilenetv4_conv_small")


class ResNet18Pyramid(nn.Module):
    """Return stride-4/8/16/32 features without the ImageNet classifier."""

    feature_channels = (64, 128, 256, 512)

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        source = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        for name in ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3", "layer4"):
            setattr(self, name, getattr(source, name))

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        x = self.maxpool(self.relu(self.bn1(self.conv1(image))))
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c2, c3, c4, c5


class MobileNetV4SmallPyramid(nn.Module):
    """Pinned convolution-only timm model; four validated pyramid features.

    The speed screen uses pretrained=False: no weight download or learned
    detector is implied by the ImageNet model identifier.
    """

    feature_channels = (32, 64, 96, 960)

    def __init__(self, pretrained: bool = False) -> None:
        super().__init__()
        import timm

        if timm.__version__ != MOBILENETV4_TIMM_VERSION:
            raise RuntimeError(f"This candidate requires timm=={MOBILENETV4_TIMM_VERSION}; got {timm.__version__}")
        self.body = timm.create_model(MOBILENETV4_MODEL, pretrained=pretrained,
                                      features_only=True, out_indices=(1, 2, 3, 4))
        if (tuple(self.body.feature_info.channels()) != self.feature_channels
                or self.body.feature_info.reduction() != [4, 8, 16, 32]):
            raise RuntimeError("MobileNetV4 feature channels/strides changed")

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return tuple(self.body(image))


def conv_bn_relu(in_channels: int, out_channels: int, kernel: int, groups: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=kernel // 2, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=False),
    )


def separable_refine(channels: int) -> nn.Sequential:
    return nn.Sequential(
        conv_bn_relu(channels, channels, 3, groups=channels),
        conv_bn_relu(channels, channels, 1),
    )


class CompactPyramid(nn.Module):
    """128 -> 64 -> 64 channels; nearest resize, skip fusion, separable refine."""

    def __init__(self, feature_channels: tuple[int, ...] = (64, 128, 256, 512)) -> None:
        super().__init__()
        if len(feature_channels) != 4 or any(v <= 0 for v in feature_channels):
            raise ValueError("Expected four positive stride-4/8/16/32 feature-channel counts")
        c2, c3, c4, c5 = feature_channels
        self.lateral32 = nn.Conv2d(c5, 128, 1)
        self.lateral16 = nn.Conv2d(c4, 128, 1)
        self.refine16 = separable_refine(128)
        self.reduce8 = nn.Conv2d(128, 64, 1)
        self.lateral8 = nn.Conv2d(c3, 64, 1)
        self.refine8 = separable_refine(64)
        self.lateral4 = nn.Conv2d(c2, 64, 1)
        self.refine4 = separable_refine(64)

    def forward(self, features: tuple[torch.Tensor, ...]) -> torch.Tensor:
        c2, c3, c4, c5 = features
        p16 = self.refine16(
            F.interpolate(self.lateral32(c5), size=c4.shape[-2:], mode="nearest") + self.lateral16(c4)
        )
        p8 = self.refine8(
            F.interpolate(self.reduce8(p16), size=c3.shape[-2:], mode="nearest") + self.lateral8(c3)
        )
        return self.refine4(
            F.interpolate(p8, size=c2.shape[-2:], mode="nearest") + self.lateral4(c2)
        )


class CompactMonocular3D(nn.Module):
    """Nine raw stride-4 heads, sharing one 64-channel 3x3 feature block.

    Inputs must already have the documented normalization and channel order;
    this model does not convert the upstream RTM3D BGR preprocessing to RGB. Calibration
    and the geometry decoder stay outside this neural graph. Raw regression
    outputs from this untrained candidate must not be interpreted as boxes.
    """

    def __init__(self, pretrained_backbone: bool = False, backbone: str = "resnet18") -> None:
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"Unsupported compact backbone: {backbone}")
        self.backbone_name = backbone
        self.architecture_revision = (ARCHITECTURE_REVISION if backbone == "resnet18"
                                      else MOBILENETV4_ARCHITECTURE_REVISION)
        self.backbone = (ResNet18Pyramid(pretrained_backbone) if backbone == "resnet18"
                         else MobileNetV4SmallPyramid(pretrained_backbone))
        self.neck = CompactPyramid(self.backbone.feature_channels)
        self.shared_head = conv_bn_relu(64, 64, 3)
        self.outputs = nn.ModuleDict({name: nn.Conv2d(64, channels, 1)
                                      for name, channels in OUTPUT_CHANNELS.items()})
        nn.init.constant_(self.outputs["hm"].bias, -2.19)
        nn.init.constant_(self.outputs["hm_hp"].bias, -2.19)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        if not torch.jit.is_tracing():
            if image.ndim != 4 or image.shape[1] != 3:
                raise ValueError("Expected an NCHW three-channel image tensor")
            if image.shape[-2] % 32 or image.shape[-1] % 32:
                raise ValueError("Input height and width must be multiples of 32")
        shared = self.shared_head(self.neck(self.backbone(image)))
        return {name: head(shared) for name, head in self.outputs.items()}


class CompactRawHeadWrapper(nn.Module):
    """Stable export order; no activation, thresholding or geometry decode."""

    def __init__(self, model: CompactMonocular3D) -> None:
        super().__init__()
        self.model = model

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, ...]:
        values = self.model(image)
        return tuple(values[name] for name in OUTPUT_NAMES)
