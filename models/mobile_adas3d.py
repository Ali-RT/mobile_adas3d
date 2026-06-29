from __future__ import annotations

from typing import Dict, Optional, Tuple

from PIL.features import features
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small


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


class MobileADAS3D(nn.Module):
    """
    MobileADAS3D v6.

    Structure:
      RGB image
        -> MobileNetV3-Small backbone
        -> stride-16 feature + stride-32 feature
        -> lightweight FPN fusion at stride 16
        -> dense prediction heads on 24 x 80 feature map for 384 x 1280 input

    Heads:
      cls_logits:        [B, num_classes, H/16, W/16]
      box2d:             [B, 4, H/16, W/16] local l/t/r/b normalized distances
      log_depth:         [B, 1, H/16, W/16]
      dim:               [B, 3, H/16, W/16]
      yaw:               [B, 2, H/16, W/16]
      center_offset:     [B, 2, H/16, W/16]
      depth_uncertainty: [B, 1, H/16, W/16]
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        input_height: int = 384,
        input_width: int = 1280,
        fpn_channels: int = 128,
        head_channels: int = 256,
    ) -> None:
        super().__init__()

        if backbone_name != "mobilenet_v3_small":
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. "
                "This implementation currently supports mobilenet_v3_small."
            )

        # Keep old torchvision compatibility simple.
        backbone = mobilenet_v3_small(pretrained=pretrained)
        self.backbone_features = backbone.features

        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width

        (
            self.stride16_feature_index,
            self.stride32_feature_index,
            c16,
            c32,
        ) = self._infer_stride_feature_indices_and_channels(
            input_height=input_height,
            input_width=input_width,
        )

        print(
            "MobileADAS3D FPN channels: "
            f"stride16={c16}, stride32={c32}"
        )
        print(
            "MobileADAS3D FPN feature indices: "
            f"stride16_index={self.stride16_feature_index}, "
            f"stride32_index={self.stride32_feature_index}"
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

        self.cls_head = ConvHead(
            in_channels=head_channels,
            hidden_channels=head_channels,
            out_channels=num_classes,
        )

        self.box2d_head = ConvHead(
            in_channels=head_channels,
            hidden_channels=head_channels,
            out_channels=4,
        )

        self.depth_head = ConvHead(
            in_channels=head_channels,
            hidden_channels=head_channels,
            out_channels=1,
        )

        self.dim_head = ConvHead(
            in_channels=head_channels,
            hidden_channels=head_channels,
            out_channels=3,
        )

        self.yaw_head = ConvHead(
            in_channels=head_channels,
            hidden_channels=head_channels,
            out_channels=2,
        )

        self.center_offset_head = ConvHead(
            in_channels=head_channels,
            hidden_channels=head_channels,
            out_channels=2,
        )

        self.depth_uncertainty_head = ConvHead(
            in_channels=head_channels,
            hidden_channels=head_channels,
            out_channels=1,
        )

        self.loc_xy_head = ConvHead(
            in_channels=head_channels,
            hidden_channels=head_channels,
            out_channels=2,
        )

    def _infer_stride_feature_indices_and_channels(
        self,
        input_height: int,
        input_width: int,
    ) -> Tuple[int, int, int, int]:
        """
        Find which MobileNetV3 feature layers correspond to stride 16 and stride 32.

        This runs only during __init__, not during forward.
        That makes the actual forward path TorchScript-trace friendly.
        """
        was_training = self.backbone_features.training
        self.backbone_features.eval()

        dummy = torch.zeros(1, 3, input_height, input_width)

        stride16_index: Optional[int] = None
        stride32_index: Optional[int] = None
        stride16_channels: Optional[int] = None
        stride32_channels: Optional[int] = None

        out = dummy

        with torch.no_grad():
            for layer_idx, layer in enumerate(self.backbone_features):
                out = layer(out)

                out_h = int(out.shape[-2])
                out_w = int(out.shape[-1])

                stride_h = int(round(float(input_height) / float(out_h)))
                stride_w = int(round(float(input_width) / float(out_w)))

                if stride_h != stride_w:
                    raise RuntimeError(
                        f"Unexpected non-square stride at layer {layer_idx}: "
                        f"stride_h={stride_h}, stride_w={stride_w}"
                    )

                stride = stride_h

                # Keep updating stride16 so we use the deepest stride-16 feature.
                if stride == 16:
                    stride16_index = layer_idx
                    stride16_channels = int(out.shape[1])

                # Keep updating stride32 so we use the deepest stride-32 feature.
                if stride == 32:
                    stride32_index = layer_idx
                    stride32_channels = int(out.shape[1])

        if was_training:
            self.backbone_features.train()

        if stride16_index is None or stride16_channels is None:
            raise RuntimeError("Could not find stride-16 feature in backbone.")

        if stride32_index is None or stride32_channels is None:
            raise RuntimeError("Could not find stride-32 feature in backbone.")

        return (
            stride16_index,
            stride32_index,
            stride16_channels,
            stride32_channels,
        )

    def _extract_stride_features(
        self,
        images: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        TorchScript-trace friendly feature extraction.

        Important:
          Do not compute dynamic stride here.
          Just collect features from fixed layer indices found during __init__.
        """
        out = images

        feat16: Optional[torch.Tensor] = None
        feat32: Optional[torch.Tensor] = None

        for layer_idx, layer in enumerate(self.backbone_features):
            out = layer(out)

            if layer_idx == self.stride16_feature_index:
                feat16 = out

            if layer_idx == self.stride32_feature_index:
                feat32 = out

        if feat16 is None:
            raise RuntimeError("stride-16 feature was not extracted.")

        if feat32 is None:
            raise RuntimeError("stride-32 feature was not extracted.")

        return feat16, feat32

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        feat16, feat32 = self._extract_stride_features(images)

        p16 = self.proj16(feat16)
        p32 = self.proj32(feat32)

        p32_up = F.interpolate(
            p32,
            size=p16.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )

        fused = torch.cat([p16, p32_up], dim=1)
        fused = self.fusion(fused)

        cls_logits = self.cls_head(fused)

        # Local l/t/r/b distances should be non-negative.
        # Targets are normalized by image width/height.
        box2d = F.softplus(self.box2d_head(fused))

        log_depth = self.depth_head(fused)
        dim = self.dim_head(fused)
        yaw = self.yaw_head(fused)
        center_offset = self.center_offset_head(fused)
        depth_uncertainty = self.depth_uncertainty_head(fused)
        loc_xy = self.loc_xy_head(fused)

        return {
            "cls_logits": cls_logits,
            "box2d": box2d,
            "log_depth": log_depth,
            "dim": dim,
            "yaw": yaw,
            "center_offset": center_offset,
            "depth_uncertainty": depth_uncertainty,
            "loc_xy": loc_xy,
        }