from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from data.target_builder import build_targets_for_sample
from data.h1_query_targets import build_h1_query_targets_for_sample, pad_h1_query_targets
from data.teacher_target_adapter import TeacherTargetAdapter


def resize_image_tensor(
    image: torch.Tensor,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    """
    image: [3, H, W], float in [0, 1]
    """
    resized = F.interpolate(
        image.unsqueeze(0),
        size=(input_height, input_width),
        mode="bilinear",
        align_corners=False,
    )

    return resized.squeeze(0)


def stack_target_dicts(
    target_dicts: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    keys = target_dicts[0].keys()

    return {
        key: torch.stack([target[key] for target in target_dicts], dim=0)
        for key in keys
    }


def mobile_adas3d_collate_fn(
    batch: List[Dict[str, Any]],
    classes: List[str],
    input_height: int,
    input_width: int,
    output_stride: int,
    class_mean_dims: Dict[str, List[float]],
    center_sampling_radius: int = 1,
    class_weights: Optional[Dict[str, float]] = None,
    quality_center_sigma: float = 1.0,
    teacher_adapter: Optional[TeacherTargetAdapter] = None,
    target_format: str = "dense",
    depth_bins: int = 40,
    min_depth_m: float = 1.0,
    max_depth_m: float = 80.0,
) -> Dict[str, Any]:
    images = []
    targets = []
    metadata = []

    for sample in batch:
        image = resize_image_tensor(
            image=sample["image"],
            input_height=input_height,
            input_width=input_width,
        )

        teacher_targets = None
        if teacher_adapter is not None:
            teacher_targets = teacher_adapter.build_for_sample(
                sample["sample_id"], sample["objects"]
            )

        if target_format == "query":
            if teacher_targets is not None:
                raise ValueError("H1 query targets do not support distillation")
            target = build_h1_query_targets_for_sample(
                objects=sample["objects"],
                original_width=int(sample["original_size"]["width"]),
                original_height=int(sample["original_size"]["height"]),
                classes=classes,
                class_mean_dims=class_mean_dims,
                P2=sample["P2"],
                depth_bins=depth_bins,
                min_depth_m=min_depth_m,
                max_depth_m=max_depth_m,
            )
        elif target_format == "dense":
            target = build_targets_for_sample(
                objects=sample["objects"],
                original_width=int(sample["original_size"]["width"]),
                original_height=int(sample["original_size"]["height"]),
                input_width=input_width,
                input_height=input_height,
                output_stride=output_stride,
                classes=classes,
                class_mean_dims=class_mean_dims,
                center_sampling_radius=center_sampling_radius,
                class_weights=class_weights,
                P2=sample.get("P2"),
                quality_center_sigma=quality_center_sigma,
                teacher_targets=teacher_targets,
            )
        else:
            raise ValueError(f"Unsupported target_format: {target_format}")

        images.append(image)
        targets.append(target)

        metadata.append(
            {
                "sample_id": sample["sample_id"],
                "image_path": sample["image_path"],
                "original_size": sample["original_size"],
                "objects": sample["objects"],
                "K": sample["K"],
                "P2": sample["P2"],
            }
        )

    return {
        "images": torch.stack(images, dim=0),
        "targets": (
            pad_h1_query_targets(targets)
            if target_format == "query"
            else stack_target_dicts(targets)
        ),
        "metadata": metadata,
    }
