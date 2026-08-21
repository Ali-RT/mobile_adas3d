from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

import torch

from data.target_builder import project_kitti_location_to_image


def build_h1_query_targets_for_sample(
    objects: List[Dict[str, Any]],
    original_width: int,
    original_height: int,
    classes: List[str],
    class_mean_dims: Dict[str, List[float]],
    P2: Sequence[Sequence[float]],
    depth_bins: int,
    min_depth_m: float,
    max_depth_m: float,
) -> Dict[str, torch.Tensor]:
    if not 0.0 < min_depth_m < max_depth_m:
        raise ValueError("Require 0 < min_depth_m < max_depth_m")
    log_centers = torch.linspace(
        math.log(min_depth_m), math.log(max_depth_m), depth_bins
    )
    rows = []
    for obj in objects:
        class_name = obj["class_name"]
        if class_name not in classes:
            continue
        x1, y1, x2, y2 = (float(value) for value in obj["bbox_2d"])
        x1 = max(0.0, min(x1, float(original_width)))
        x2 = max(0.0, min(x2, float(original_width)))
        y1 = max(0.0, min(y1, float(original_height)))
        y2 = max(0.0, min(y2, float(original_height)))
        depth = float(obj["location_3d"][2])
        if x2 <= x1 or y2 <= y1 or depth <= 0.0:
            continue

        cx = (x1 + x2) * 0.5 / float(original_width)
        cy = (y1 + y2) * 0.5 / float(original_height)
        width = (x2 - x1) / float(original_width)
        height = (y2 - y1) / float(original_height)
        projected = project_kitti_location_to_image(obj["location_3d"], P2)
        projected_valid = (
            projected is not None
            and 0.0 <= projected[0] < float(original_width)
            and 0.0 <= projected[1] < float(original_height)
        )
        projected_normalized = (
            [projected[0] / original_width, projected[1] / original_height]
            if projected_valid
            else [0.0, 0.0]
        )
        log_depth = math.log(max(depth, min_depth_m))
        depth_bin = int(torch.argmin((log_centers - log_depth).abs()).item())
        mean_dims = class_mean_dims[class_name]
        dimensions = [
            math.log(max(float(value), 1e-4) / float(mean))
            for value, mean in zip(obj["dimensions_3d"], mean_dims)
        ]
        yaw = float(obj["rotation_y"])
        location_x, location_y, location_z = (
            float(value) for value in obj["location_3d"]
        )
        rows.append(
            {
                "class_id": classes.index(class_name),
                "box2d": [cx, cy, width, height],
                "projected_center": projected_normalized,
                "projected_center_valid": projected_valid,
                "depth_bin": depth_bin,
                "depth_residual": log_depth - float(log_centers[depth_bin]),
                "dimensions": dimensions,
                "yaw": [math.sin(yaw), math.cos(yaw)],
                "location_xy": [location_x / location_z, location_y / location_z],
            }
        )

    count = len(rows)
    def tensor(key: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        if count:
            return torch.tensor([row[key] for row in rows], dtype=dtype)
        widths = {
            "box2d": 4,
            "projected_center": 2,
            "dimensions": 3,
            "yaw": 2,
            "location_xy": 2,
        }
        if key in widths:
            return torch.empty((0, widths[key]), dtype=dtype)
        return torch.empty((0,), dtype=dtype)

    return {
        "class_ids": tensor("class_id", torch.long),
        "box2d": tensor("box2d"),
        "projected_center": tensor("projected_center"),
        "projected_center_valid": tensor("projected_center_valid", torch.bool),
        "depth_bin": tensor("depth_bin", torch.long),
        "depth_residual": tensor("depth_residual"),
        "dimensions": tensor("dimensions"),
        "yaw": tensor("yaw"),
        "location_xy": tensor("location_xy"),
    }


def pad_h1_query_targets(
    targets: List[Dict[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    batch_size = len(targets)
    max_objects = max((target["class_ids"].numel() for target in targets), default=0)
    max_objects = max(max_objects, 1)
    shapes = {
        "class_ids": (),
        "box2d": (4,),
        "projected_center": (2,),
        "projected_center_valid": (),
        "depth_bin": (),
        "depth_residual": (),
        "dimensions": (3,),
        "yaw": (2,),
        "location_xy": (2,),
    }
    padded: Dict[str, torch.Tensor] = {
        "object_mask": torch.zeros(batch_size, max_objects, dtype=torch.bool)
    }
    for key, trailing_shape in shapes.items():
        dtype = targets[0][key].dtype
        padded[key] = torch.zeros(
            (batch_size, max_objects, *trailing_shape), dtype=dtype
        )
    for batch_index, target in enumerate(targets):
        count = target["class_ids"].numel()
        if not count:
            continue
        padded["object_mask"][batch_index, :count] = True
        for key in shapes:
            padded[key][batch_index, :count] = target[key]
    return padded
