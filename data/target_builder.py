from typing import Any, Dict, List, Tuple

import torch


def scale_bbox_2d(
    bbox: List[float],
    original_width: int,
    original_height: int,
    input_width: int,
    input_height: int,
) -> List[float]:
    """
    Scale a 2D bounding box from original image size to model input size.
    """
    x1, y1, x2, y2 = bbox

    scale_x = input_width / original_width
    scale_y = input_height / original_height

    return [
        x1 * scale_x,
        y1 * scale_y,
        x2 * scale_x,
        y2 * scale_y,
    ]


def get_box_center(bbox: List[float]) -> Tuple[float, float]:
    """
    Return center x/y of a 2D box.
    """
    x1, y1, x2, y2 = bbox
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    return cx, cy


def build_targets_for_sample(
    sample: Dict[str, Any],
    classes: List[str],
    input_height: int,
    input_width: int,
    output_stride: int,
    class_mean_dims: Dict[str, List[float]],
) -> Dict[str, torch.Tensor]:
    """
    Build dense target maps for one KITTI sample.

    Assignment rule:
      - Each object is assigned to one feature-map cell.
      - The cell is determined by the resized 2D box center.
      - If multiple objects land in the same cell, keep the closer object.

    Returns:
      Dictionary of target tensors.
    """
    num_classes = len(classes)
    class_to_id = {name: idx for idx, name in enumerate(classes)}

    feature_height = input_height // output_stride
    feature_width = input_width // output_stride

    cls_target = torch.zeros(
        num_classes,
        feature_height,
        feature_width,
        dtype=torch.float32,
    )

    box2d_target = torch.zeros(
        4,
        feature_height,
        feature_width,
        dtype=torch.float32,
    )

    log_depth_target = torch.zeros(
        1,
        feature_height,
        feature_width,
        dtype=torch.float32,
    )

    dim_target = torch.zeros(
        3,
        feature_height,
        feature_width,
        dtype=torch.float32,
    )

    yaw_target = torch.zeros(
        2,
        feature_height,
        feature_width,
        dtype=torch.float32,
    )

    offset_target = torch.zeros(
        2,
        feature_height,
        feature_width,
        dtype=torch.float32,
    )

    valid_mask = torch.zeros(
        1,
        feature_height,
        feature_width,
        dtype=torch.float32,
    )

    # Used to resolve collisions. Smaller z means closer object.
    assigned_depth = torch.full(
        (feature_height, feature_width),
        fill_value=float("inf"),
        dtype=torch.float32,
    )

    original_height = int(sample["original_size"]["height"])
    original_width = int(sample["original_size"]["width"])

    objects = sample["objects"]

    for obj in objects:
        class_name = obj["class_name"]

        if class_name not in class_to_id:
            continue

        bbox_resized = scale_bbox_2d(
            bbox=obj["bbox_2d"],
            original_width=original_width,
            original_height=original_height,
            input_width=input_width,
            input_height=input_height,
        )

        x1, y1, x2, y2 = bbox_resized

        # Clamp box to resized image bounds.
        x1 = max(0.0, min(float(input_width - 1), x1))
        y1 = max(0.0, min(float(input_height - 1), y1))
        x2 = max(0.0, min(float(input_width - 1), x2))
        y2 = max(0.0, min(float(input_height - 1), y2))

        if x2 <= x1 or y2 <= y1:
            continue

        center_x, center_y = get_box_center([x1, y1, x2, y2])

        grid_x_float = center_x / output_stride
        grid_y_float = center_y / output_stride

        grid_x = int(grid_x_float)
        grid_y = int(grid_y_float)

        if grid_x < 0 or grid_x >= feature_width:
            continue

        if grid_y < 0 or grid_y >= feature_height:
            continue

        depth_z = float(obj["location_3d"][2])

        if depth_z <= 0:
            continue

        # If cell is occupied, keep the closer object.
        if depth_z >= float(assigned_depth[grid_y, grid_x]):
            continue

        assigned_depth[grid_y, grid_x] = depth_z

        class_id = class_to_id[class_name]

        # Clear previous class assignment at this cell, then assign current object.
        cls_target[:, grid_y, grid_x] = 0.0
        cls_target[class_id, grid_y, grid_x] = 1.0

        # Store resized absolute box coordinates in input-image pixel space.
        box2d_target[:, grid_y, grid_x] = torch.tensor(
            [x1, y1, x2, y2],
            dtype=torch.float32,
        )

        # Predict log depth for stability.
        log_depth_target[0, grid_y, grid_x] = torch.log(
            torch.tensor(depth_z, dtype=torch.float32)
        )

        # Dimension residual relative to class mean dimensions.
        # KITTI dimensions are [h, w, l].
        dims = torch.tensor(obj["dimensions_3d"], dtype=torch.float32)
        mean_dims = torch.tensor(class_mean_dims[class_name], dtype=torch.float32)

        dim_target[:, grid_y, grid_x] = torch.log(dims / mean_dims)

        # Predict yaw as sin/cos to avoid angle wrap discontinuity.
        yaw = torch.tensor(float(obj["rotation_y"]), dtype=torch.float32)
        yaw_target[:, grid_y, grid_x] = torch.tensor(
            [torch.sin(yaw), torch.cos(yaw)],
            dtype=torch.float32,
        )

        # Fractional offset inside feature cell.
        offset_x = grid_x_float - grid_x
        offset_y = grid_y_float - grid_y

        offset_target[:, grid_y, grid_x] = torch.tensor(
            [offset_x, offset_y],
            dtype=torch.float32,
        )

        valid_mask[0, grid_y, grid_x] = 1.0

    targets = {
        "cls_target": cls_target,
        "box2d_target": box2d_target,
        "log_depth_target": log_depth_target,
        "dim_target": dim_target,
        "yaw_target": yaw_target,
        "offset_target": offset_target,
        "valid_mask": valid_mask,
    }

    return targets