from __future__ import annotations

from typing import Dict

import torch

from models.mobile_adas3d_h1 import H1_OUTPUT_NAMES, MobileADAS3DH1


def fixed_query_reference_grid(rows: int = 5, columns: int = 10) -> torch.Tensor:
    """Return row-major normalized cell centers as ``[1, rows*columns, 2]``."""
    y = (torch.arange(rows, dtype=torch.float32) + 0.5) / rows
    x = (torch.arange(columns, dtype=torch.float32) + 0.5) / columns
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack((grid_x, grid_y), dim=-1).reshape(1, rows * columns, 2)


class MobileADAS3DH2(MobileADAS3DH1):
    """H1 graph with spatially anchored queries and bounded center offsets."""

    architecture_name = "MobileADAS3D-H2"
    export_output_names = H1_OUTPUT_NAMES

    def __init__(self, *args, center_offset_scale: float = 0.10, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.num_queries != 50:
            raise ValueError("MobileADAS3D-H2 requires the locked 50-query grid")
        if center_offset_scale <= 0.0 or center_offset_scale > 0.25:
            raise ValueError("H2 center_offset_scale must be in (0, 0.25]")
        self.center_offset_scale = float(center_offset_scale)
        references = fixed_query_reference_grid()
        self.register_buffer("query_reference_points", references)
        # Reuse the same deterministic 2D positional basis as image memory.
        from models.mobile_adas3d_h1 import fixed_2d_sincos_position

        self.register_buffer(
            "query_reference_encoding",
            fixed_2d_sincos_position(5, 10, self.transformer_width),
        )

    def _initial_queries(self, batch_size: int) -> torch.Tensor:
        return (
            self.query_embedding + self.query_reference_encoding
        ).expand(batch_size, -1, -1)

    def _bounded_center(self, raw: torch.Tensor) -> torch.Tensor:
        reference = self.query_reference_points.expand(raw.shape[0], -1, -1)
        return (reference + self.center_offset_scale * raw.tanh()).clamp(0.0, 1.0)

    def _format_outputs(self, decoded: torch.Tensor) -> Dict[str, torch.Tensor]:
        raw_box = self.box2d_head(decoded)
        return {
            "class_logits": self.class_head(decoded),
            "box2d_cxcywh": torch.cat(
                (self._bounded_center(raw_box[..., :2]), raw_box[..., 2:].sigmoid()),
                dim=-1,
            ),
            "projected_center": self._bounded_center(
                self.projected_center_head(decoded)
            ),
            "depth_logits": self.query_depth_head(decoded),
            "depth_residual": self.depth_residual_head(decoded),
            "dimensions": self.dimensions_head(decoded),
            "yaw": self.yaw_head(decoded),
            "location_xy": self.location_xy_head(decoded),
            "quality": self.quality_head(decoded),
        }


class MobileADAS3DH2TupleWrapper(torch.nn.Module):
    def __init__(self, model: MobileADAS3DH2) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor):
        outputs = self.model(images)
        return tuple(outputs[name] for name in self.model.export_output_names)
