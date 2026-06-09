from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


class ConvHead(nn.Module):
    """
    Small convolutional prediction head.

    Input:
      feature map [B, C, H, W]

    Output:
      prediction map [B, out_channels, H, W]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),

            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MobileADAS3D(nn.Module):
    """
    First version of MobileADAS3D.

    This model predicts dense feature-map outputs.

    Heads:
      cls_logits:     [B, num_classes, Hf, Wf]
      box2d:          [B, 4, Hf, Wf]
      log_depth:      [B, 1, Hf, Wf]
      dim_residual:   [B, 3, Hf, Wf]
      yaw_sincos:     [B, 2, Hf, Wf]
      center_offset:  [B, 2, Hf, Wf]
      depth_logvar:   [B, 1, Hf, Wf]
    """

    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        use_sparse_depth: bool = False,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.use_sparse_depth = use_sparse_depth

        if pretrained:
            weights = MobileNet_V3_Small_Weights.DEFAULT
        else:
            weights = None

        backbone = mobilenet_v3_small(weights=weights)

        # MobileNetV3 features output [B, 576, H/32, W/32]
        self.rgb_backbone = backbone.features
        rgb_channels = 576

        if use_sparse_depth:
            self.depth_stem = nn.Sequential(
                nn.Conv2d(2, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.SiLU(inplace=True),

                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.SiLU(inplace=True),

                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.SiLU(inplace=True),
            )
            fusion_in_channels = rgb_channels + 64
        else:
            self.depth_stem = None
            fusion_in_channels = rgb_channels

        self.fusion = nn.Sequential(
            nn.Conv2d(fusion_in_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True),
        )

        self.cls_head = ConvHead(256, num_classes)
        self.box2d_head = ConvHead(256, 4)
        self.depth_head = ConvHead(256, 1)
        self.dim_head = ConvHead(256, 3)
        self.yaw_head = ConvHead(256, 2)
        self.center_offset_head = ConvHead(256, 2)
        self.depth_uncertainty_head = ConvHead(256, 1)

        self._init_head_biases()

    def _init_head_biases(self) -> None:
        """
        Initialize classification head bias for sparse object detection.

        This prevents the model from starting with high confidence everywhere.
        """
        final_cls_layer = self.cls_head.net[-1]

        if isinstance(final_cls_layer, nn.Conv2d):
            nn.init.constant_(final_cls_layer.bias, -4.6)

    def forward(
        self,
        image: torch.Tensor,
        sparse_depth: torch.Tensor | None = None,
        sparse_depth_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
          image:
            [B, 3, H, W]

          sparse_depth:
            optional [B, 1, H, W]

          sparse_depth_mask:
            optional [B, 1, H, W]

        Returns:
          dictionary of prediction maps
        """
        if image.ndim != 4:
            raise ValueError(f"Expected image shape [B, 3, H, W], got {tuple(image.shape)}")

        if image.shape[1] != 3:
            raise ValueError(f"Expected image with 3 channels, got {image.shape[1]}")

        rgb_feat = self.rgb_backbone(image)

        if self.use_sparse_depth:
            if sparse_depth is None or sparse_depth_mask is None:
                raise ValueError(
                    "sparse_depth and sparse_depth_mask are required when use_sparse_depth=True"
                )

            depth_input = torch.cat([sparse_depth, sparse_depth_mask], dim=1)
            depth_feat = self.depth_stem(depth_input)

            depth_feat = F.interpolate(
                depth_feat,
                size=rgb_feat.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

            feat = torch.cat([rgb_feat, depth_feat], dim=1)
        else:
            feat = rgb_feat

        feat = self.fusion(feat)

        outputs = {
            "cls_logits": self.cls_head(feat),
            "box2d": self.box2d_head(feat),
            "log_depth": self.depth_head(feat),
            "dim_residual": self.dim_head(feat),
            "yaw_sincos": self.yaw_head(feat),
            "center_offset": self.center_offset_head(feat),
            "depth_logvar": self.depth_uncertainty_head(feat),
        }

        return outputs