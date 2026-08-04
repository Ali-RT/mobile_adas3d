from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from data.geometry import scale_p2_for_resize


def encode_yaw_axis_direction(yaw: float) -> Tuple[List[float], float]:
    """Encode yaw as a 180-degree-invariant double-angle axis plus direction."""
    axis = [math.sin(2.0 * yaw), math.cos(2.0 * yaw)]
    canonical_axis_yaw = 0.5 * math.atan2(axis[0], axis[1])
    direction = 1.0 if math.cos(yaw - canonical_axis_yaw) < 0.0 else 0.0
    return axis, direction


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


def build_quality_target_from_center_offset(
    center_offset: Sequence[float],
    sigma: float = 1.0,
) -> float:
    """
    Soft center-quality target for ranking/objectness.

    Positive cells close to the true object center should rank above neighboring
    positive cells assigned by center sampling. Background remains 0.
    """
    if sigma <= 0.0:
        return 1.0

    dx = float(center_offset[0])
    dy = float(center_offset[1])
    distance_sq = dx * dx + dy * dy
    return float(torch.exp(torch.tensor(-0.5 * distance_sq / (sigma * sigma))).item())


def project_kitti_location_to_image(
    location_3d: Sequence[float],
    P2: Sequence[Sequence[float]],
    min_z: float = 0.1,
) -> Optional[Tuple[float, float]]:
    """
    Project a KITTI camera-frame location into image pixels with P2.

    KITTI object locations are bottom-center 3D points. For v2 geometry training
    we use this projected bottom-center as an explicit image-space anchor and
    back-project it at decode time with the predicted depth.
    """
    if len(location_3d) != 3:
        raise ValueError(f"Expected location_3d length 3, got {location_3d}")

    p2_tensor = torch.as_tensor(P2, dtype=torch.float32)
    if tuple(p2_tensor.shape) != (3, 4):
        raise ValueError(f"Expected P2 shape [3, 4], got {tuple(p2_tensor.shape)}")

    point = torch.tensor(
        [float(location_3d[0]), float(location_3d[1]), float(location_3d[2]), 1.0],
        dtype=torch.float32,
    )
    uvw = p2_tensor @ point
    z = float(uvw[2].item())

    if z <= float(min_z):
        return None

    return float((uvw[0] / uvw[2]).item()), float((uvw[1] / uvw[2]).item())


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
    P2: Optional[Sequence[Sequence[float]]] = None,
    quality_center_sigma: float = 1.0,
    teacher_targets: Optional[Dict[str, torch.Tensor]] = None,
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
    yaw_axis_target = torch.zeros(2, feature_h, feature_w, dtype=torch.float32)
    yaw_direction_target = torch.zeros(1, feature_h, feature_w, dtype=torch.float32)
    offset_target = torch.zeros(2, feature_h, feature_w, dtype=torch.float32)
    projected_center_offset_target = torch.zeros(2, feature_h, feature_w, dtype=torch.float32)
    projected_center_valid_mask = torch.zeros(1, feature_h, feature_w, dtype=torch.float32)
    quality_target = torch.zeros(1, feature_h, feature_w, dtype=torch.float32)
    valid_mask = torch.zeros(1, feature_h, feature_w, dtype=torch.float32)

    teacher_valid_mask = torch.zeros(1, feature_h, feature_w, dtype=torch.float32)
    teacher_score_target = torch.zeros(1, feature_h, feature_w, dtype=torch.float32)
    teacher_log_depth_target = torch.zeros(1, feature_h, feature_w, dtype=torch.float32)
    teacher_loc_xy_target = torch.zeros(2, feature_h, feature_w, dtype=torch.float32)
    teacher_dim_target = torch.zeros(3, feature_h, feature_w, dtype=torch.float32)
    teacher_yaw_target = torch.zeros(2, feature_h, feature_w, dtype=torch.float32)

    # Used to apply class-balanced regression/object losses.
    loss_weight_target = torch.ones(1, feature_h, feature_w, dtype=torch.float32)

    # For collision handling: closer object wins.
    priority_depth = torch.full(
        (feature_h, feature_w),
        fill_value=float("inf"),
        dtype=torch.float32,
    )

    class_weights = class_weights or {}
    scaled_p2: Optional[Sequence[Sequence[float]]] = None

    if P2 is not None:
        scaled_p2 = scale_p2_for_resize(
            P2=torch.as_tensor(P2, dtype=torch.float32).numpy(),
            orig_w=original_width,
            orig_h=original_height,
            input_w=input_width,
            input_h=input_height,
        ).tolist()

    if teacher_targets is not None:
        expected_objects = len(objects)
        for key, value in teacher_targets.items():
            if int(value.shape[0]) != expected_objects:
                raise ValueError(
                    f"Teacher target {key!r} has {value.shape[0]} objects; "
                    f"expected {expected_objects}"
                )

    for object_index, obj in enumerate(objects):
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
        projected_center: Optional[Tuple[float, float]] = None

        if scaled_p2 is not None:
            projected_center = project_kitti_location_to_image(
                location_3d=obj["location_3d"],
                P2=scaled_p2,
            )
            if projected_center is not None:
                projected_x, projected_y = projected_center
                if not (
                    0.0 <= projected_x < float(input_width)
                    and 0.0 <= projected_y < float(input_height)
                ):
                    projected_center = None

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
        yaw_axis, yaw_direction = encode_yaw_axis_direction(yaw)

        sample_class_weight = float(class_weights.get(class_name, 1.0))

        for cell_x, cell_y in positive_cells:
            # Collision policy: closer object owns the cell.
            if depth > float(priority_depth[cell_y, cell_x]):
                continue

            # If replacing an old assignment, clear all class channels for this cell.
            cls_target[:, cell_y, cell_x] = 0.0
            projected_center_offset_target[:, cell_y, cell_x] = 0.0
            projected_center_valid_mask[0, cell_y, cell_x] = 0.0
            quality_target[0, cell_y, cell_x] = 0.0
            teacher_valid_mask[0, cell_y, cell_x] = 0.0
            teacher_score_target[0, cell_y, cell_x] = 0.0
            teacher_log_depth_target[0, cell_y, cell_x] = 0.0
            teacher_loc_xy_target[:, cell_y, cell_x] = 0.0
            teacher_dim_target[:, cell_y, cell_x] = 0.0
            teacher_yaw_target[:, cell_y, cell_x] = 0.0

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
            yaw_axis_target[:, cell_y, cell_x] = torch.tensor(
                yaw_axis,
                dtype=torch.float32,
            )
            yaw_direction_target[0, cell_y, cell_x] = yaw_direction

            center_offset_values = build_center_offset_target(
                center_x=center_x,
                center_y=center_y,
                cell_x=cell_x,
                cell_y=cell_y,
                output_stride=output_stride,
            )

            offset_target[:, cell_y, cell_x] = torch.tensor(
                center_offset_values,
                dtype=torch.float32,
            )
            quality_target[0, cell_y, cell_x] = build_quality_target_from_center_offset(
                center_offset=center_offset_values,
                sigma=quality_center_sigma,
            )

            if (
                teacher_targets is not None
                and bool(teacher_targets["teacher_valid_mask"][object_index])
            ):
                teacher_location = teacher_targets["teacher_location_3d"][object_index]
                teacher_depth = teacher_location[2].clamp(min=1e-3)
                teacher_dims = teacher_targets["teacher_dimensions_3d"][object_index]
                teacher_yaw = teacher_targets["teacher_yaw"][object_index]
                teacher_valid_mask[0, cell_y, cell_x] = 1.0
                teacher_score_target[0, cell_y, cell_x] = teacher_targets[
                    "teacher_score"
                ][object_index]
                teacher_log_depth_target[0, cell_y, cell_x] = torch.log(teacher_depth)
                teacher_loc_xy_target[:, cell_y, cell_x] = torch.stack(
                    [
                        teacher_location[0] / teacher_depth,
                        teacher_location[1] / teacher_depth,
                    ]
                )
                teacher_dim_target[:, cell_y, cell_x] = torch.log(
                    teacher_dims.clamp(min=1e-6)
                    / torch.tensor(mean_dims, dtype=torch.float32).clamp(min=1e-6)
                )
                teacher_yaw_target[:, cell_y, cell_x] = torch.stack(
                    [torch.sin(teacher_yaw), torch.cos(teacher_yaw)]
                )

            if projected_center is not None:
                projected_center_offset_target[:, cell_y, cell_x] = torch.tensor(
                    build_center_offset_target(
                        center_x=projected_center[0],
                        center_y=projected_center[1],
                        cell_x=cell_x,
                        cell_y=cell_y,
                        output_stride=output_stride,
                    ),
                    dtype=torch.float32,
                )
                projected_center_valid_mask[0, cell_y, cell_x] = 1.0

            valid_mask[0, cell_y, cell_x] = 1.0
            loss_weight_target[0, cell_y, cell_x] = sample_class_weight
            priority_depth[cell_y, cell_x] = depth

    targets = {
        "cls_target": cls_target,
        "box2d_target": box2d_target,
        "log_depth_target": log_depth_target,
        "loc_xy_target": loc_xy_target,
        "location_xyz_target": location_xyz_target,
        "dim_target": dim_target,
        "yaw_target": yaw_target,
        "yaw_axis_target": yaw_axis_target,
        "yaw_direction_target": yaw_direction_target,
        "offset_target": offset_target,
        "projected_center_offset_target": projected_center_offset_target,
        "projected_center_valid_mask": projected_center_valid_mask,
        "quality_target": quality_target,
        "valid_mask": valid_mask,
        "loss_weight_target": loss_weight_target,
    }
    if teacher_targets is not None:
        targets.update(
            {
                "teacher_valid_mask": teacher_valid_mask,
                "teacher_score_target": teacher_score_target,
                "teacher_log_depth_target": teacher_log_depth_target,
                "teacher_loc_xy_target": teacher_loc_xy_target,
                "teacher_dim_target": teacher_dim_target,
                "teacher_yaw_target": teacher_yaw_target,
            }
        )
    return targets


# Backward-compatible alias if older code imports build_targets.
def build_targets(*args, **kwargs):
    return build_targets_for_sample(*args, **kwargs)
