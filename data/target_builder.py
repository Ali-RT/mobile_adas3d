from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def scale_bbox_2d(
    bbox: List[float],
    original_width: int,
    original_height: int,
    input_width: int,
    input_height: int,
) -> List[float]:
    x_scale = input_width / float(original_width)
    y_scale = input_height / float(original_height)

    x1, y1, x2, y2 = bbox

    return [
        float(x1 * x_scale),
        float(y1 * y_scale),
        float(x2 * x_scale),
        float(y2 * y_scale),
    ]


def compute_feature_shape(
    input_height: int,
    input_width: int,
    output_stride: int,
) -> Tuple[int, int]:
    if input_height % output_stride != 0:
        raise ValueError(
            f"input_height={input_height} must be divisible by output_stride={output_stride}"
        )

    if input_width % output_stride != 0:
        raise ValueError(
            f"input_width={input_width} must be divisible by output_stride={output_stride}"
        )

    return input_height // output_stride, input_width // output_stride


def get_class_id(class_name: str, classes: List[str]) -> int:
    if class_name not in classes:
        raise ValueError(f"Unknown class_name={class_name}; classes={classes}")

    return classes.index(class_name)


def get_positive_cells(
    center_x: float,
    center_y: float,
    input_width: int,
    input_height: int,
    output_stride: int,
    radius: int,
) -> List[Tuple[int, int]]:
    feature_h, feature_w = compute_feature_shape(
        input_height=input_height,
        input_width=input_width,
        output_stride=output_stride,
    )

    center_cell_x = int(center_x // output_stride)
    center_cell_y = int(center_y // output_stride)

    center_cell_x = max(0, min(center_cell_x, feature_w - 1))
    center_cell_y = max(0, min(center_cell_y, feature_h - 1))

    cells: List[Tuple[int, int]] = []

    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            cell_x = center_cell_x + dx
            cell_y = center_cell_y + dy

            if cell_x < 0 or cell_x >= feature_w:
                continue

            if cell_y < 0 or cell_y >= feature_h:
                continue

            cells.append((cell_x, cell_y))

    return cells


def build_ltrb_box_target(
    bbox: List[float],
    center_x: float,
    center_y: float,
    input_width: int,
    input_height: int,
) -> List[float]:
    """
    Local l/t/r/b box encoding relative to object center.

    Normalized by image width/height:
      l = (cx - x1) / input_width
      t = (cy - y1) / input_height
      r = (x2 - cx) / input_width
      b = (y2 - cy) / input_height
    """
    x1, y1, x2, y2 = bbox

    l = max(0.0, center_x - x1) / float(input_width)
    t = max(0.0, center_y - y1) / float(input_height)
    r = max(0.0, x2 - center_x) / float(input_width)
    b = max(0.0, y2 - center_y) / float(input_height)

    return [l, t, r, b]


def build_center_offset_target(
    center_x: float,
    center_y: float,
    cell_x: int,
    cell_y: int,
    output_stride: int,
) -> List[float]:
    """
    Offset from feature-cell center to object center, normalized by stride.

    Feature point:
      px = (cell_x + 0.5) * stride
      py = (cell_y + 0.5) * stride

    Target:
      dx = (center_x - px) / stride
      dy = (center_y - py) / stride

    With center_sampling radius=1, this can be roughly in [-1.5, 1.5].
    """
    point_x = (cell_x + 0.5) * output_stride
    point_y = (cell_y + 0.5) * output_stride

    dx = (center_x - point_x) / float(output_stride)
    dy = (center_y - point_y) / float(output_stride)

    return [float(dx), float(dy)]


def build_targets_for_sample(
    objects: List[Dict[str, Any]],
    original_width: int,
    original_height: int,
    input_width: int,
    input_height: int,
    output_stride: int,
    classes: List[str],
    class_mean_dims: Dict[str, List[float]],
    center_sampling_radius: int = 1,
    class_weights: Optional[Dict[str, float]] = None,
) -> Dict[str, torch.Tensor]:
    feature_h, feature_w = compute_feature_shape(
        input_height=input_height,
        input_width=input_width,
        output_stride=output_stride,
    )

    num_classes = len(classes)

    cls_target = torch.zeros(num_classes, feature_h, feature_w, dtype=torch.float32)
    box2d_target = torch.zeros(4, feature_h, feature_w, dtype=torch.float32)
    log_depth_target = torch.zeros(1, feature_h, feature_w, dtype=torch.float32)
    loc_xy_target = torch.zeros(2, feature_h, feature_w, dtype=torch.float32)
    location_xyz_target = torch.zeros(3, feature_h, feature_w, dtype=torch.float32)
    dim_target = torch.zeros(3, feature_h, feature_w, dtype=torch.float32)
    yaw_target = torch.zeros(2, feature_h, feature_w, dtype=torch.float32)
    offset_target = torch.zeros(2, feature_h, feature_w, dtype=torch.float32)
    valid_mask = torch.zeros(1, feature_h, feature_w, dtype=torch.float32)

    # Used to apply class-balanced regression/object losses.
    loss_weight_target = torch.ones(1, feature_h, feature_w, dtype=torch.float32)

    # For collision handling: closer object wins.
    priority_depth = torch.full(
        (feature_h, feature_w),
        fill_value=float("inf"),
        dtype=torch.float32,
    )

    class_weights = class_weights or {}

    for obj in objects:
        class_name = obj["class_name"]

        if class_name not in classes:
            continue

        class_id = get_class_id(class_name, classes)

        bbox = scale_bbox_2d(
            bbox=obj["bbox_2d"],
            original_width=original_width,
            original_height=original_height,
            input_width=input_width,
            input_height=input_height,
        )

        x1, y1, x2, y2 = bbox

        # Filter invalid / tiny boxes.
        if x2 <= x1 or y2 <= y1:
            continue

        center_x = 0.5 * (x1 + x2)
        center_y = 0.5 * (y1 + y2)

        if center_x < 0 or center_x >= input_width:
            continue

        if center_y < 0 or center_y >= input_height:
            continue

        depth = float(obj["location_3d"][2])

        if depth <= 0.0:
            continue

        location_x = float(obj["location_3d"][0])
        location_y = float(obj["location_3d"][1])
        location_z = depth

        # KITTI camera-frame location encoded as [x / z, y / z].
        # Decode is unambiguous: x = loc_xy[0] * z, y = loc_xy[1] * z.
        loc_xy = [
            location_x / max(location_z, 1e-3),
            location_y / max(location_z, 1e-3),
        ]

        positive_cells = get_positive_cells(
            center_x=center_x,
            center_y=center_y,
            input_width=input_width,
            input_height=input_height,
            output_stride=output_stride,
            radius=center_sampling_radius,
        )

        ltrb_target = build_ltrb_box_target(
            bbox=bbox,
            center_x=center_x,
            center_y=center_y,
            input_width=input_width,
            input_height=input_height,
        )

        dims = obj["dimensions_3d"]  # KITTI [h, w, l]
        mean_dims = class_mean_dims[class_name]

        dim_residual = [
            torch.log(torch.tensor(max(dims[i], 1e-6) / max(mean_dims[i], 1e-6))).item()
            for i in range(3)
        ]

        yaw = float(obj["rotation_y"])
        yaw_sin = torch.sin(torch.tensor(yaw)).item()
        yaw_cos = torch.cos(torch.tensor(yaw)).item()

        sample_class_weight = float(class_weights.get(class_name, 1.0))

        for cell_x, cell_y in positive_cells:
            # Collision policy: closer object owns the cell.
            if depth > float(priority_depth[cell_y, cell_x]):
                continue

            # If replacing an old assignment, clear all class channels for this cell.
            cls_target[:, cell_y, cell_x] = 0.0

            cls_target[class_id, cell_y, cell_x] = 1.0

            box2d_target[:, cell_y, cell_x] = torch.tensor(
                ltrb_target,
                dtype=torch.float32,
            )

            log_depth_target[0, cell_y, cell_x] = torch.log(
                torch.tensor(depth, dtype=torch.float32)
            )

            loc_xy_target[:, cell_y, cell_x] = torch.tensor(
                loc_xy,
                dtype=torch.float32,
            )

            location_xyz_target[:, cell_y, cell_x] = torch.tensor(
                [location_x, location_y, location_z],
                dtype=torch.float32,
            )

            dim_target[:, cell_y, cell_x] = torch.tensor(
                dim_residual,
                dtype=torch.float32,
            )

            yaw_target[:, cell_y, cell_x] = torch.tensor(
                [yaw_sin, yaw_cos],
                dtype=torch.float32,
            )

            offset_target[:, cell_y, cell_x] = torch.tensor(
                build_center_offset_target(
                    center_x=center_x,
                    center_y=center_y,
                    cell_x=cell_x,
                    cell_y=cell_y,
                    output_stride=output_stride,
                ),
                dtype=torch.float32,
            )

            valid_mask[0, cell_y, cell_x] = 1.0
            loss_weight_target[0, cell_y, cell_x] = sample_class_weight
            priority_depth[cell_y, cell_x] = depth

    return {
        "cls_target": cls_target,
        "box2d_target": box2d_target,
        "log_depth_target": log_depth_target,
        "loc_xy_target": loc_xy_target,
        "location_xyz_target": location_xyz_target,
        "dim_target": dim_target,
        "yaw_target": yaw_target,
        "offset_target": offset_target,
        "valid_mask": valid_mask,
        "loss_weight_target": loss_weight_target,
    }


# Backward-compatible alias if older code imports build_targets.
def build_targets(*args, **kwargs):
    return build_targets_for_sample(*args, **kwargs)