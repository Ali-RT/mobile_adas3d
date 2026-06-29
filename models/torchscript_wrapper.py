from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


TORCHSCRIPT_OUTPUT_NAMES = [
    "cls_logits",
    "box2d",
    "log_depth",
    "dim",
    "yaw",
    "center_offset",
    "depth_uncertainty",
    "loc_xy",
]


class MobileADAS3DTupleWrapper(nn.Module):
    """
    TorchScript/CoreML-friendly wrapper.

    The normal model returns a Python dict:
      {
        "cls_logits": Tensor,
        "box2d": Tensor,
        ...
      }

    Export tools are usually happier with a fixed tuple:
      (
        cls_logits,
        box2d,
        log_depth,
        dim,
        yaw,
        center_offset,
        depth_uncertainty,
        loc_xy,
      )
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs = self.model(x)

        return tuple(
            outputs[name]
            for name in TORCHSCRIPT_OUTPUT_NAMES
        )


def tuple_outputs_to_dict(
    outputs: Tuple[torch.Tensor, ...],
) -> Dict[str, torch.Tensor]:
    if len(outputs) != len(TORCHSCRIPT_OUTPUT_NAMES):
        raise ValueError(
            f"Expected {len(TORCHSCRIPT_OUTPUT_NAMES)} outputs, "
            f"got {len(outputs)}"
        )

    return {
        name: tensor
        for name, tensor in zip(TORCHSCRIPT_OUTPUT_NAMES, outputs)
    }