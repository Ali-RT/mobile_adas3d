from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mobile_adas3d_s1 import (
    ConvBNReLU,
    DepthwiseSeparableBlock,
    MobileNetV4S1Pyramid,
)


H1_OUTPUT_NAMES = (
    "class_logits",
    "box2d_cxcywh",
    "projected_center",
    "depth_logits",
    "depth_residual",
    "dimensions",
    "yaw",
    "location_xy",
    "quality",
)


def fixed_2d_sincos_position(
    height: int,
    width: int,
    channels: int,
) -> torch.Tensor:
    """Return fixed row-major 2D sine/cosine positions as ``[1, HW, C]``."""
    if channels % 4 != 0:
        raise ValueError("2D sine/cosine position channels must be divisible by 4")
    quarter = channels // 4
    omega = torch.arange(quarter, dtype=torch.float32)
    omega = 1.0 / (10_000.0 ** (omega / max(quarter - 1, 1)))
    y = torch.arange(height, dtype=torch.float32).unsqueeze(1) * omega.unsqueeze(0)
    x = torch.arange(width, dtype=torch.float32).unsqueeze(1) * omega.unsqueeze(0)
    y_embedding = torch.cat((y.sin(), y.cos()), dim=1)
    x_embedding = torch.cat((x.sin(), x.cos()), dim=1)
    position = torch.cat(
        (
            y_embedding[:, None, :].expand(height, width, -1),
            x_embedding[None, :, :].expand(height, width, -1),
        ),
        dim=2,
    )
    return position.reshape(1, height * width, channels)


class FixedMultiheadAttention(nn.Module):
    """Standard attention expressed with Core-ML-friendly fixed-shape ops."""

    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        if width % heads != 0:
            raise ValueError("attention width must be divisible by heads")
        self.heads = heads
        self.head_width = width // heads
        self.scale = self.head_width**-0.5
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        batch = query.shape[0]
        query_tokens = query.shape[1]
        key_tokens = key.shape[1]
        q = self.query(query).reshape(
            batch, query_tokens, self.heads, self.head_width
        ).transpose(1, 2)
        k = self.key(key).reshape(
            batch, key_tokens, self.heads, self.head_width
        ).transpose(1, 2)
        v = self.value(value).reshape(
            batch, key_tokens, self.heads, self.head_width
        ).transpose(1, 2)
        weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        weights = weights.softmax(dim=-1)
        context = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch, query_tokens, self.heads * self.head_width
        )
        return self.output(context)


class FixedFeedForward(nn.Module):
    def __init__(self, width: int, feedforward_width: int) -> None:
        super().__init__()
        self.input = nn.Linear(width, feedforward_width)
        self.output = nn.Linear(feedforward_width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.output(F.gelu(self.input(value)))


class FixedEncoderLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward_width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = FixedMultiheadAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.feedforward = FixedFeedForward(width, feedforward_width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(value)
        value = value + self.attention(normalized, normalized, normalized)
        return value + self.feedforward(self.norm2(value))


class FixedDecoderLayer(nn.Module):
    def __init__(self, width: int, heads: int, feedforward_width: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.self_attention = FixedMultiheadAttention(width, heads)
        self.norm2 = nn.LayerNorm(width)
        self.cross_attention = FixedMultiheadAttention(width, heads)
        self.norm3 = nn.LayerNorm(width)
        self.feedforward = FixedFeedForward(width, feedforward_width)

    def forward(self, queries: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(queries)
        queries = queries + self.self_attention(normalized, normalized, normalized)
        queries = queries + self.cross_attention(self.norm2(queries), memory, memory)
        return queries + self.feedforward(self.norm3(queries))


class MobileADAS3DH1(nn.Module):
    """Fixed-shape MobileNetV4 depth-aware query student for Core ML."""

    architecture_name = "MobileADAS3D-H1"
    export_output_names = H1_OUTPUT_NAMES

    def __init__(
        self,
        num_classes: int = 2,
        backbone_name: str = "mobilenetv4_conv_small.e2400_r224_in1k",
        pretrained: bool = True,
        input_height: int = 384,
        input_width: int = 1280,
        fpn_channels: int = 128,
        transformer_width: int = 192,
        attention_heads: int = 6,
        encoder_layers: int = 2,
        decoder_layers: int = 2,
        feedforward_width: int = 768,
        num_queries: int = 50,
        depth_bins: int = 40,
    ) -> None:
        super().__init__()
        if num_classes != 2:
            raise ValueError(f"MobileADAS3D-H1 requires 2 classes, got {num_classes}")
        if (input_height, input_width) != (384, 1280):
            raise ValueError("MobileADAS3D-H1 input is locked to 1280x384")
        if fpn_channels != 128 or transformer_width != 192:
            raise ValueError("MobileADAS3D-H1 is locked to FPN 128 / transformer 192")
        if transformer_width % attention_heads != 0:
            raise ValueError("transformer width must be divisible by attention heads")
        if (encoder_layers, decoder_layers, num_queries, depth_bins) != (2, 2, 50, 40):
            raise ValueError("MobileADAS3D-H1 layer/query/depth dimensions are locked")

        self.num_classes = num_classes
        self.input_height = input_height
        self.input_width = input_width
        self.fpn_channels = fpn_channels
        self.transformer_width = transformer_width
        self.num_queries = num_queries
        self.depth_bins = depth_bins
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

        self.spatial_depth_head = nn.Conv2d(fpn_channels, depth_bins, kernel_size=1)
        self.depth_context_projection = nn.Conv2d(1, fpn_channels, kernel_size=1)
        depth_centers = torch.linspace(1.0 / depth_bins, 1.0, depth_bins)
        self.register_buffer("depth_bin_centers", depth_centers.view(1, depth_bins, 1, 1))

        self.memory8 = nn.Conv2d(fpn_channels, transformer_width, kernel_size=1)
        self.memory16 = nn.Conv2d(fpn_channels, transformer_width, kernel_size=1)
        self.memory32 = nn.Conv2d(fpn_channels, transformer_width, kernel_size=1)
        self.level_embedding = nn.Parameter(torch.zeros(3, transformer_width))
        nn.init.normal_(self.level_embedding, std=0.02)

        self.encoder = nn.ModuleList(
            FixedEncoderLayer(
                transformer_width,
                attention_heads,
                feedforward_width,
            )
            for _ in range(encoder_layers)
        )
        self.decoder = nn.ModuleList(
            FixedDecoderLayer(
                transformer_width,
                attention_heads,
                feedforward_width,
            )
            for _ in range(decoder_layers)
        )
        self.query_embedding = nn.Parameter(torch.empty(1, num_queries, transformer_width))
        nn.init.normal_(self.query_embedding, std=0.02)

        self.register_buffer("position8", fixed_2d_sincos_position(48, 160, transformer_width))
        self.register_buffer("position16", fixed_2d_sincos_position(24, 80, transformer_width))
        self.register_buffer("position32", fixed_2d_sincos_position(12, 40, transformer_width))

        self.class_head = nn.Linear(transformer_width, num_classes)
        self.box2d_head = nn.Linear(transformer_width, 4)
        self.projected_center_head = nn.Linear(transformer_width, 2)
        self.query_depth_head = nn.Linear(transformer_width, depth_bins)
        self.depth_residual_head = nn.Linear(transformer_width, 1)
        self.dimensions_head = nn.Linear(transformer_width, 3)
        self.yaw_head = nn.Linear(transformer_width, 2)
        self.location_xy_head = nn.Linear(transformer_width, 2)
        self.quality_head = nn.Linear(transformer_width, 1)
        self._initialize_query_heads()

    def _initialize_query_heads(self) -> None:
        # Start the untrained detector near neutral logits/residuals. Besides
        # improving early optimization, this avoids magnifying insignificant
        # FP16 feature-rounding noise before any head has learned a signal.
        heads = (
            self.class_head,
            self.box2d_head,
            self.projected_center_head,
            self.query_depth_head,
            self.depth_residual_head,
            self.dimensions_head,
            self.yaw_head,
            self.location_xy_head,
            self.quality_head,
        )
        for head in heads:
            nn.init.normal_(head.weight, mean=0.0, std=1e-3)
            nn.init.zeros_(head.bias)

    @staticmethod
    def _resize_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)

    @staticmethod
    def _tokens(feature: torch.Tensor) -> torch.Tensor:
        return feature.flatten(2).transpose(1, 2)

    def _decode_queries(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.input_mean) / self.input_std
        feature8, feature16, feature32 = self.backbone(images)
        lateral8 = self.lateral8(feature8)
        lateral16 = self.lateral16(feature16)
        lateral32 = self.lateral32(feature32)
        pyramid16 = self.refine16(
            lateral16 + self._resize_like(lateral32, lateral16)
        )
        pyramid8 = self.refine8(
            lateral8 + self._resize_like(pyramid16, lateral8)
        )
        pyramid32 = lateral32

        spatial_depth_logits = self.spatial_depth_head(pyramid16)
        depth_probability = spatial_depth_logits.softmax(dim=1)
        expected_depth = (depth_probability * self.depth_bin_centers).sum(
            dim=1, keepdim=True
        )
        depth_context16 = self.depth_context_projection(expected_depth)
        pyramid16 = pyramid16 + depth_context16
        pyramid8 = pyramid8 + self._resize_like(depth_context16, pyramid8)
        pyramid32 = pyramid32 + self._resize_like(depth_context16, pyramid32)

        memory8 = self._tokens(self.memory8(pyramid8))
        memory16 = self._tokens(self.memory16(pyramid16))
        memory32 = self._tokens(self.memory32(pyramid32))
        encoded32 = memory32 + self.position32 + self.level_embedding[2].view(1, 1, -1)
        for encoder_layer in self.encoder:
            encoded32 = encoder_layer(encoded32)
        memory = torch.cat(
            (
                memory8 + self.position8 + self.level_embedding[0].view(1, 1, -1),
                memory16 + self.position16 + self.level_embedding[1].view(1, 1, -1),
                encoded32,
            ),
            dim=1,
        )
        queries = self._initial_queries(images.shape[0])
        decoded = queries
        for decoder_layer in self.decoder:
            decoded = decoder_layer(decoded, memory)
        return decoded

    def _initial_queries(self, batch_size: int) -> torch.Tensor:
        return self.query_embedding.expand(batch_size, -1, -1)

    def _format_outputs(self, decoded: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "class_logits": self.class_head(decoded),
            "box2d_cxcywh": self.box2d_head(decoded).sigmoid(),
            "projected_center": self.projected_center_head(decoded).sigmoid(),
            "depth_logits": self.query_depth_head(decoded),
            "depth_residual": self.depth_residual_head(decoded),
            "dimensions": self.dimensions_head(decoded),
            "yaw": self.yaw_head(decoded),
            "location_xy": self.location_xy_head(decoded),
            "quality": self.quality_head(decoded),
        }

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._format_outputs(self._decode_queries(images))


class MobileADAS3DH1TupleWrapper(nn.Module):
    def __init__(self, model: MobileADAS3DH1) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = self.model(images)
        return tuple(outputs[name] for name in self.model.export_output_names)
