from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


try:
    from torchvision.ops import batched_nms as torchvision_batched_nms

    HAS_TORCHVISION_NMS = True
except Exception:
    torchvision_batched_nms = None
    HAS_TORCHVISION_NMS = False


def _box_iou_torch(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    boxes1: [N, 4] in xyxy
    boxes2: [M, 4] in xyxy
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)

    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])

    wh = (rb - lt).clamp(min=0)
    intersection = wh[:, :, 0] * wh[:, :, 1]

    union = area1[:, None] + area2[None, :] - intersection
    return intersection / union.clamp(min=1e-6)

def box_iou(box_a: Any, box_b: Any) -> torch.Tensor:
    """
    Public compatibility helper used by evaluation scripts.

    Supports:
      box_a: [4] or [N, 4]
      box_b: [4] or [M, 4]

    Returns:
      if box_a is [4] and box_b is [M, 4]:
        Tensor [M]

      if box_a is [N, 4] and box_b is [M, 4]:
        Tensor [N, M]

      if both are [4]:
        Tensor scalar-like [1]
    """
    if not torch.is_tensor(box_a):
        box_a_tensor = torch.tensor(box_a, dtype=torch.float32)
    else:
        box_a_tensor = box_a.detach().to(dtype=torch.float32)

    if not torch.is_tensor(box_b):
        box_b_tensor = torch.tensor(box_b, dtype=torch.float32)
    else:
        box_b_tensor = box_b.detach().to(dtype=torch.float32)

    box_a_was_1d = box_a_tensor.ndim == 1
    box_b_was_1d = box_b_tensor.ndim == 1

    if box_a_was_1d:
        box_a_tensor = box_a_tensor.unsqueeze(0)

    if box_b_was_1d:
        box_b_tensor = box_b_tensor.unsqueeze(0)

    ious = _box_iou_torch(box_a_tensor, box_b_tensor)

    # Common evaluation case:
    #   one prediction box vs many GT boxes
    # Return [M], so caller can do ious.tolist().
    if box_a_was_1d and not box_b_was_1d:
        return ious.squeeze(0)

    # Less common:
    #   many prediction boxes vs one GT box
    # Return [N].
    if not box_a_was_1d and box_b_was_1d:
        return ious.squeeze(1)

    # Single box vs single box.
    # Return [1], not float, so .tolist() still works.
    if box_a_was_1d and box_b_was_1d:
        return ious.reshape(1)

    # Full matrix [N, M].
    return ious

def _nms_torch_fallback(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Pure PyTorch NMS fallback.

    This is slower than torchvision.ops.batched_nms but avoids breaking if
    torchvision C++/CUDA NMS is unavailable.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: List[torch.Tensor] = []

    while order.numel() > 0:
        current = order[0]
        keep.append(current)

        if order.numel() == 1:
            break

        current_box = boxes[current].unsqueeze(0)
        remaining = order[1:]
        remaining_boxes = boxes[remaining]

        ious = _box_iou_torch(current_box, remaining_boxes).squeeze(0)
        order = remaining[ious <= iou_threshold]

    return torch.stack(keep) if keep else torch.empty((0,), dtype=torch.long, device=boxes.device)


def _class_aware_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    class_ids: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    Class-aware NMS.

    Prefer torchvision.ops.batched_nms when available.
    Fallback to per-class PyTorch NMS otherwise.
    """
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    if HAS_TORCHVISION_NMS and torchvision_batched_nms is not None:
        return torchvision_batched_nms(
            boxes=boxes,
            scores=scores,
            idxs=class_ids,
            iou_threshold=iou_threshold,
        )

    keep_indices: List[torch.Tensor] = []

    for cls_id in class_ids.unique():
        cls_mask = class_ids == cls_id
        cls_indices = torch.nonzero(cls_mask, as_tuple=False).flatten()

        cls_keep_local = _nms_torch_fallback(
            boxes=boxes[cls_indices],
            scores=scores[cls_indices],
            iou_threshold=iou_threshold,
        )

        if cls_keep_local.numel() > 0:
            keep_indices.append(cls_indices[cls_keep_local])

    if not keep_indices:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    keep = torch.cat(keep_indices, dim=0)

    # Sort final detections by score descending.
    keep = keep[scores[keep].argsort(descending=True)]

    return keep


def _select_p2_for_batch(
    P2: Optional[Any],
    batch_index: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if P2 is None:
        return None

    p2_tensor = torch.as_tensor(P2, device=device, dtype=dtype)

    if p2_tensor.ndim == 2:
        if tuple(p2_tensor.shape) != (3, 4):
            raise ValueError(f"Expected P2 shape [3, 4], got {tuple(p2_tensor.shape)}")
        return p2_tensor

    if p2_tensor.ndim == 3:
        if tuple(p2_tensor.shape[1:]) != (3, 4):
            raise ValueError(f"Expected P2 shape [B, 3, 4], got {tuple(p2_tensor.shape)}")
        return p2_tensor[batch_index]

    raise ValueError(f"Expected P2 shape [3, 4] or [B, 3, 4], got {tuple(p2_tensor.shape)}")


def _backproject_uv_depth_with_p2(
    u: torch.Tensor,
    v: torch.Tensor,
    depth: torch.Tensor,
    P2: torch.Tensor,
) -> torch.Tensor:
    """
    Back-project image coordinates and KITTI z-depth into camera coordinates.

    KITTI P2 is rectified, so:
      u = (fx * x + cx * z + tx) / z
      v = (fy * y + cy * z + ty) / z
    """
    fx = P2[0, 0].clamp(min=1e-6)
    fy = P2[1, 1].clamp(min=1e-6)
    cx = P2[0, 2]
    cy = P2[1, 2]
    tx = P2[0, 3]
    ty = P2[1, 3]

    x = ((u - cx) * depth - tx) / fx
    y = ((v - cy) * depth - ty) / fy

    return torch.stack([x, y, depth], dim=1)


def _make_class_mean_dims_tensor(
    classes: List[str],
    class_mean_dims: Dict[str, List[float]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Returns [num_classes, 3] tensor in h/w/l order.

    Expected class_mean_dims:
      {
        "Car": [h, w, l],
        "Pedestrian": [h, w, l],
        "Cyclist": [h, w, l],
      }
    """
    values = []

    for class_name in classes:
        if class_name not in class_mean_dims:
            raise KeyError(f"Missing mean dimensions for class: {class_name}")

        dims = class_mean_dims[class_name]

        if len(dims) != 3:
            raise ValueError(
                f"class_mean_dims[{class_name}] must have 3 values [h, w, l]. "
                f"Got: {dims}"
            )

        values.append(dims)

    return torch.tensor(values, device=device, dtype=dtype)


def _decode_single_image_vectorized(
    outputs: Dict[str, torch.Tensor],
    batch_index: int,
    classes: List[str],
    class_mean_dims_tensor: torch.Tensor,
    input_height: int,
    input_width: int,
    score_threshold: float,
    topk: int,
    nms_iou_threshold: float,
    max_depth: float = 200.0,
    P2: Optional[torch.Tensor] = None,
    location_source: str = "loc_xy",
    score_mode: str = "class",
    quality_score_power: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Vectorized decode for one image.

    Main optimization:
    - topK is computed over flattened class/cell scores
    - candidate tensors are gathered in batch
    - NMS runs on tensors
    - CPU conversion happens once after final NMS
    """

    cls_logits = outputs["cls_logits"][batch_index]          # [C, H, W]
    box2d = outputs["box2d"][batch_index]                    # [4, H, W]
    log_depth = outputs["log_depth"][batch_index]            # [1, H, W]
    dim = outputs["dim"][batch_index]                        # [3, H, W]
    yaw = outputs["yaw"][batch_index]                        # [2, H, W]
    center_offset = outputs["center_offset"][batch_index]    # [2, H, W]

    quality_logits: Optional[torch.Tensor]
    if "quality" in outputs:
        quality_logits = outputs["quality"][batch_index]     # [1, H, W]
    else:
        quality_logits = None

    loc_xy: Optional[torch.Tensor]
    if "loc_xy" in outputs:
        loc_xy = outputs["loc_xy"][batch_index]              # [2, H, W]
    else:
        loc_xy = None

    projected_center_offset: Optional[torch.Tensor]
    if "projected_center_offset" in outputs:
        projected_center_offset = outputs["projected_center_offset"][batch_index]  # [2, H, W]
    else:
        projected_center_offset = None

    depth_uncertainty: Optional[torch.Tensor]
    if "depth_uncertainty" in outputs:
        depth_uncertainty = outputs["depth_uncertainty"][batch_index]  # [1, H, W]
    else:
        depth_uncertainty = None

    num_classes, feature_h, feature_w = cls_logits.shape
    num_cells = feature_h * feature_w

    stride_x = float(input_width) / float(feature_w)
    stride_y = float(input_height) / float(feature_h)

    # [C, H, W] -> [C * H * W]
    class_scores = torch.sigmoid(cls_logits)
    quality_scores: Optional[torch.Tensor] = None

    if score_mode == "class_quality" and quality_logits is not None:
        quality_scores = torch.sigmoid(quality_logits).clamp(min=1e-6, max=1.0)
        final_scores = class_scores * quality_scores.pow(float(quality_score_power))
        effective_score_mode = "class_quality"
    else:
        final_scores = class_scores
        effective_score_mode = "class"

    scores_flat = final_scores.reshape(-1)
    class_scores_flat_all = class_scores.reshape(-1)

    k = min(int(topk), int(scores_flat.numel()))

    if k <= 0:
        return []

    top_scores, top_indices = torch.topk(scores_flat, k=k, largest=True, sorted=True)

    keep_score_mask = top_scores >= score_threshold

    if keep_score_mask.sum().item() == 0:
        return []

    top_scores = top_scores[keep_score_mask]
    top_indices = top_indices[keep_score_mask]
    top_class_scores = class_scores_flat_all[top_indices]

    class_ids = torch.div(top_indices, num_cells, rounding_mode="floor").long()
    spatial_indices = top_indices % num_cells

    cell_y = torch.div(spatial_indices, feature_w, rounding_mode="floor").long()
    cell_x = (spatial_indices % feature_w).long()

    # Flatten spatial tensors once.
    # [channels, H, W] -> [H * W, channels]
    box_flat = box2d.permute(1, 2, 0).reshape(num_cells, 4)
    dim_flat = dim.permute(1, 2, 0).reshape(num_cells, 3)
    yaw_flat = yaw.permute(1, 2, 0).reshape(num_cells, 2)
    offset_flat = center_offset.permute(1, 2, 0).reshape(num_cells, 2)
    log_depth_flat = log_depth.reshape(-1)

    candidate_ltrb = box_flat[spatial_indices]       # [N, 4]
    candidate_dim_residual = dim_flat[spatial_indices]
    candidate_yaw_vec = yaw_flat[spatial_indices]
    candidate_offset = offset_flat[spatial_indices]
    candidate_log_depth = log_depth_flat[spatial_indices]
    candidate_class_scores = top_class_scores

    if quality_scores is not None:
        quality_scores_flat = quality_scores.reshape(-1)
        candidate_quality_scores = quality_scores_flat[spatial_indices]
    else:
        candidate_quality_scores = torch.ones_like(candidate_class_scores)

    if loc_xy is not None:
        loc_xy_flat = loc_xy.permute(1, 2, 0).reshape(num_cells, 2)
        candidate_loc_xy = loc_xy_flat[spatial_indices]
    else:
        candidate_loc_xy = torch.zeros(
            (spatial_indices.shape[0], 2),
            device=candidate_log_depth.device,
            dtype=candidate_log_depth.dtype,
        )

    if projected_center_offset is not None:
        projected_center_offset_flat = projected_center_offset.permute(1, 2, 0).reshape(
            num_cells,
            2,
        )
        candidate_projected_center_offset = projected_center_offset_flat[spatial_indices]
    else:
        candidate_projected_center_offset = torch.zeros(
            (spatial_indices.shape[0], 2),
            device=candidate_log_depth.device,
            dtype=candidate_log_depth.dtype,
        )

    if depth_uncertainty is not None:
        depth_uncertainty_flat = depth_uncertainty.reshape(-1)
        candidate_depth_uncertainty = depth_uncertainty_flat[spatial_indices]
    else:
        candidate_depth_uncertainty = torch.zeros_like(candidate_log_depth)

    # Decode center.
    # center_offset target was normalized by stride, so:
    # center = feature_cell_center + offset * stride
    center_x = (cell_x.to(candidate_offset.dtype) + 0.5 + candidate_offset[:, 0]) * stride_x
    center_y = (cell_y.to(candidate_offset.dtype) + 0.5 + candidate_offset[:, 1]) * stride_y

    # Decode local l/t/r/b box distances.
    # These were normalized by input width/height.
    left = candidate_ltrb[:, 0] * float(input_width)
    top = candidate_ltrb[:, 1] * float(input_height)
    right = candidate_ltrb[:, 2] * float(input_width)
    bottom = candidate_ltrb[:, 3] * float(input_height)

    x1 = (center_x - left).clamp(min=0.0, max=float(input_width - 1))
    y1 = (center_y - top).clamp(min=0.0, max=float(input_height - 1))
    x2 = (center_x + right).clamp(min=0.0, max=float(input_width - 1))
    y2 = (center_y + bottom).clamp(min=0.0, max=float(input_height - 1))

    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # Remove invalid boxes.
    box_w = boxes[:, 2] - boxes[:, 0]
    box_h = boxes[:, 3] - boxes[:, 1]
    valid_box_mask = (box_w > 1.0) & (box_h > 1.0)

    if valid_box_mask.sum().item() == 0:
        return []

    boxes = boxes[valid_box_mask]
    top_scores = top_scores[valid_box_mask]
    candidate_class_scores = candidate_class_scores[valid_box_mask]
    candidate_quality_scores = candidate_quality_scores[valid_box_mask]
    class_ids = class_ids[valid_box_mask]
    spatial_indices = spatial_indices[valid_box_mask]
    cell_x = cell_x[valid_box_mask]
    cell_y = cell_y[valid_box_mask]
    center_x = center_x[valid_box_mask]
    center_y = center_y[valid_box_mask]
    candidate_log_depth = candidate_log_depth[valid_box_mask]
    candidate_dim_residual = candidate_dim_residual[valid_box_mask]
    candidate_yaw_vec = candidate_yaw_vec[valid_box_mask]
    candidate_depth_uncertainty = candidate_depth_uncertainty[valid_box_mask]
    candidate_loc_xy = candidate_loc_xy[valid_box_mask]
    candidate_projected_center_offset = candidate_projected_center_offset[valid_box_mask]

    # Class-aware NMS.
    keep_nms = _class_aware_nms(
        boxes=boxes,
        scores=top_scores,
        class_ids=class_ids,
        iou_threshold=nms_iou_threshold,
    )

    if keep_nms.numel() == 0:
        return []

    boxes = boxes[keep_nms]
    scores = top_scores[keep_nms]
    class_score_values = candidate_class_scores[keep_nms]
    quality_score_values = candidate_quality_scores[keep_nms]
    class_ids = class_ids[keep_nms]
    cell_x = cell_x[keep_nms]
    cell_y = cell_y[keep_nms]
    center_x = center_x[keep_nms]
    center_y = center_y[keep_nms]
    log_depth_values = candidate_log_depth[keep_nms]
    dim_residual = candidate_dim_residual[keep_nms]
    yaw_vec = candidate_yaw_vec[keep_nms]
    depth_uncertainty_values = candidate_depth_uncertainty[keep_nms]
    loc_xy_values = candidate_loc_xy[keep_nms]
    projected_center_offset_values = candidate_projected_center_offset[keep_nms]

    # Decode depth.
    depth = torch.exp(log_depth_values).clamp(min=0.1, max=max_depth)

    effective_location_source = location_source

    if (
        location_source == "projected_center"
        and projected_center_offset is not None
        and P2 is not None
    ):
        projected_u = (
            cell_x.to(projected_center_offset_values.dtype)
            + 0.5
            + projected_center_offset_values[:, 0]
        ) * stride_x
        projected_v = (
            cell_y.to(projected_center_offset_values.dtype)
            + 0.5
            + projected_center_offset_values[:, 1]
        ) * stride_y
        location_xyz = _backproject_uv_depth_with_p2(
            u=projected_u,
            v=projected_v,
            depth=depth,
            P2=P2,
        )
    else:
        # Decode KITTI camera-frame location from loc_xy + depth.
        #   z = exp(log_depth)
        #   x = loc_xy[0] * z
        #   y = loc_xy[1] * z
        effective_location_source = "loc_xy"
        location_x = loc_xy_values[:, 0] * depth
        location_y = loc_xy_values[:, 1] * depth
        location_xyz = torch.stack([location_x, location_y, depth], dim=1)

    # Decode dimensions.
    # Target builder uses log-class-ratio encoding:
    #   dim_target = log(dims_hwl / class_mean_dims)
    # so decode is:
    #   dims = class_mean_dims * exp(dim_residual)
    dims = class_mean_dims_tensor[class_ids] * torch.exp(dim_residual)
    dims = dims.clamp(min=0.01)

    # Decode yaw.
    # Assumption:
    #   yaw head predicts [sin(yaw), cos(yaw)]
    yaw_angle = torch.atan2(yaw_vec[:, 0], yaw_vec[:, 1])

    # Move final compact tensors to CPU once.
    boxes_cpu = boxes.detach().cpu()
    scores_cpu = scores.detach().cpu()
    class_scores_cpu = class_score_values.detach().cpu()
    quality_scores_cpu = quality_score_values.detach().cpu()
    class_ids_cpu = class_ids.detach().cpu()
    center_x_cpu = center_x.detach().cpu()
    center_y_cpu = center_y.detach().cpu()
    depth_cpu = depth.detach().cpu()
    location_xyz_cpu = location_xyz.detach().cpu()
    dims_cpu = dims.detach().cpu()
    yaw_cpu = yaw_angle.detach().cpu()
    depth_uncertainty_cpu = depth_uncertainty_values.detach().cpu()
    cell_x_cpu = cell_x.detach().cpu()
    cell_y_cpu = cell_y.detach().cpu()

    predictions: List[Dict[str, Any]] = []

    for i in range(boxes_cpu.shape[0]):
        class_id = int(class_ids_cpu[i].item())
        class_name = classes[class_id]

        bbox = boxes_cpu[i].tolist()
        dims_hwl = dims_cpu[i].tolist()

        predictions.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "score": float(scores_cpu[i].item()),
                "class_score": float(class_scores_cpu[i].item()),
                "quality_score": float(quality_scores_cpu[i].item()),
                "score_mode": effective_score_mode,
                "bbox_2d": [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ],
                "center_2d": [
                    float(center_x_cpu[i].item()),
                    float(center_y_cpu[i].item()),
                ],
                "depth": float(depth_cpu[i].item()),
                "location_3d": [
                    float(location_xyz_cpu[i][0].item()),
                    float(location_xyz_cpu[i][1].item()),
                    float(location_xyz_cpu[i][2].item()),
                ],
                "dimensions_3d_hwl": [
                    float(dims_hwl[0]),
                    float(dims_hwl[1]),
                    float(dims_hwl[2]),
                ],
                "yaw": float(yaw_cpu[i].item()),
                "depth_uncertainty": float(depth_uncertainty_cpu[i].item()),
                "cell_x": int(cell_x_cpu[i].item()),
                "cell_y": int(cell_y_cpu[i].item()),
                "location_decode_source": effective_location_source,
            }
        )

    return predictions


def decode_mobile_adas3d_outputs(
    outputs: Dict[str, torch.Tensor],
    classes: List[str],
    class_mean_dims: Dict[str, List[float]],
    input_height: int,
    input_width: int,
    score_threshold: float = 0.55,
    topk: int = 50,
    nms_iou_threshold: float = 0.5,
    P2: Optional[Any] = None,
    location_source: str = "loc_xy",
    score_mode: str = "class",
    quality_score_power: float = 1.0,
) -> List[List[Dict[str, Any]]]:
    """
    Decode MobileADAS3D raw output tensors into per-image prediction dictionaries.

    Returns:
      List over batch:
        [
          [
            {
              "class_id": int,
              "class_name": str,
              "score": float,
              "bbox_2d": [x1, y1, x2, y2],
              "center_2d": [cx, cy],
              "depth": float,
              "dimensions_3d_hwl": [h, w, l],
              "yaw": float,
              "depth_uncertainty": float,
              "cell_x": int,
              "cell_y": int,
            },
            ...
          ],
          ...
        ]
    """

    required_keys = [
        "cls_logits",
        "box2d",
        "log_depth",
        "dim",
        "yaw",
        "center_offset",
    ]

    for key in required_keys:
        if key not in outputs:
            raise KeyError(f"Missing output tensor: {key}")

    if location_source not in ("loc_xy", "projected_center"):
        raise ValueError(
            "location_source must be 'loc_xy' or 'projected_center', "
            f"got {location_source!r}"
        )
    if score_mode not in ("class", "class_quality"):
        raise ValueError(
            "score_mode must be 'class' or 'class_quality', "
            f"got {score_mode!r}"
        )

    cls_logits = outputs["cls_logits"]
    batch_size = cls_logits.shape[0]

    device = cls_logits.device
    dtype = cls_logits.dtype

    class_mean_dims_tensor = _make_class_mean_dims_tensor(
        classes=classes,
        class_mean_dims=class_mean_dims,
        device=device,
        dtype=dtype,
    )

    batch_predictions: List[List[Dict[str, Any]]] = []

    for batch_index in range(batch_size):
        p2_for_image = _select_p2_for_batch(
            P2=P2,
            batch_index=batch_index,
            device=device,
            dtype=dtype,
        )
        preds = _decode_single_image_vectorized(
            outputs=outputs,
            batch_index=batch_index,
            classes=classes,
            class_mean_dims_tensor=class_mean_dims_tensor,
            input_height=input_height,
            input_width=input_width,
            score_threshold=score_threshold,
            topk=topk,
            nms_iou_threshold=nms_iou_threshold,
            P2=p2_for_image,
            location_source=location_source,
            score_mode=score_mode,
            quality_score_power=quality_score_power,
        )

        batch_predictions.append(preds)

    return batch_predictions
