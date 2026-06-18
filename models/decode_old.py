from __future__ import annotations

import math
from typing import Dict, List

import torch


def box_iou(
    box: torch.Tensor,
    boxes: torch.Tensor,
) -> torch.Tensor:
    """
    box: [4]
    boxes: [N, 4]
    """
    if boxes.numel() == 0:
        return torch.zeros(0, dtype=box.dtype, device=box.device)

    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = torch.clamp(x2 - x1, min=0)
    inter_h = torch.clamp(y2 - y1, min=0)
    inter = inter_w * inter_h

    box_area = torch.clamp(box[2] - box[0], min=0) * torch.clamp(box[3] - box[1], min=0)
    boxes_area = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0) * torch.clamp(
        boxes[:, 3] - boxes[:, 1],
        min=0,
    )

    union = box_area + boxes_area - inter
    return inter / torch.clamp(union, min=1e-6)


def nms_2d(
    predictions: List[Dict],
    iou_threshold: float = 0.5,
) -> List[Dict]:
    if len(predictions) == 0:
        return []

    predictions = sorted(predictions, key=lambda p: p["score"], reverse=True)

    kept: List[Dict] = []

    while predictions:
        current = predictions.pop(0)
        kept.append(current)

        remaining = []

        current_box = torch.tensor(current["bbox_2d"], dtype=torch.float32)

        for pred in predictions:
            # Class-aware NMS.
            if pred["class_name"] != current["class_name"]:
                remaining.append(pred)
                continue

            pred_box = torch.tensor(pred["bbox_2d"], dtype=torch.float32)
            iou = box_iou(current_box, pred_box.unsqueeze(0))[0].item()

            if iou <= iou_threshold:
                remaining.append(pred)

        predictions = remaining

    return kept


def decode_mobile_adas3d_outputs(
    outputs: Dict[str, torch.Tensor],
    classes: List[str],
    class_mean_dims: Dict[str, List[float]],
    input_height: int,
    input_width: int,
    score_threshold: float = 0.20,
    topk: int = 200,
    nms_iou_threshold: float = 0.5,
) -> List[List[Dict]]:
    """
    Decode v6 outputs.

    box2d is l/t/r/b normalized relative to predicted object center.
    center_offset predicts offset from feature-cell center to object center.
    """
    cls_logits = outputs["cls_logits"]
    box2d = outputs["box2d"]
    log_depth = outputs["log_depth"]
    dim = outputs["dim"]
    yaw = outputs["yaw"]
    center_offset = outputs["center_offset"]
    depth_uncertainty = outputs["depth_uncertainty"]

    batch_size, num_classes, feature_h, feature_w = cls_logits.shape

    stride_y = input_height / float(feature_h)
    stride_x = input_width / float(feature_w)

    scores = torch.sigmoid(cls_logits)

    batch_predictions: List[List[Dict]] = []

    for b in range(batch_size):
        scores_b = scores[b]

        flat_scores = scores_b.reshape(-1)
        k = min(topk, flat_scores.numel())

        top_scores, top_indices = torch.topk(flat_scores, k=k)

        predictions: List[Dict] = []

        for score, flat_index in zip(top_scores, top_indices):
            score_value = float(score.item())

            if score_value < score_threshold:
                continue

            flat_index_value = int(flat_index.item())

            class_id = flat_index_value // (feature_h * feature_w)
            rem = flat_index_value % (feature_h * feature_w)
            cell_y = rem // feature_w
            cell_x = rem % feature_w

            class_name = classes[class_id]

            point_x = (cell_x + 0.5) * stride_x
            point_y = (cell_y + 0.5) * stride_y

            offset_x = float(center_offset[b, 0, cell_y, cell_x].item())
            offset_y = float(center_offset[b, 1, cell_y, cell_x].item())

            center_x = point_x + offset_x * stride_x
            center_y = point_y + offset_y * stride_y

            l_norm = float(box2d[b, 0, cell_y, cell_x].item())
            t_norm = float(box2d[b, 1, cell_y, cell_x].item())
            r_norm = float(box2d[b, 2, cell_y, cell_x].item())
            b_norm = float(box2d[b, 3, cell_y, cell_x].item())

            l = l_norm * input_width
            r = r_norm * input_width
            t = t_norm * input_height
            bb = b_norm * input_height

            x1 = center_x - l
            y1 = center_y - t
            x2 = center_x + r
            y2 = center_y + bb

            x1 = max(0.0, min(float(x1), float(input_width - 1)))
            y1 = max(0.0, min(float(y1), float(input_height - 1)))
            x2 = max(0.0, min(float(x2), float(input_width - 1)))
            y2 = max(0.0, min(float(y2), float(input_height - 1)))

            if x2 <= x1 or y2 <= y1:
                continue

            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue

            depth = float(torch.exp(log_depth[b, 0, cell_y, cell_x]).item())
            depth = max(0.1, min(depth, 200.0))

            dim_residual = dim[b, :, cell_y, cell_x]
            mean_dims = torch.tensor(
                class_mean_dims[class_name],
                device=dim_residual.device,
                dtype=dim_residual.dtype,
            )
            dims = torch.exp(dim_residual) * mean_dims

            yaw_vec = yaw[b, :, cell_y, cell_x]
            yaw_sin = float(yaw_vec[0].item())
            yaw_cos = float(yaw_vec[1].item())
            yaw_angle = math.atan2(yaw_sin, yaw_cos)

            unc = float(depth_uncertainty[b, 0, cell_y, cell_x].item())

            predictions.append(
                {
                    "class_id": int(class_id),
                    "class_name": class_name,
                    "score": score_value,
                    "bbox_2d": [x1, y1, x2, y2],
                    "center_2d": [float(center_x), float(center_y)],
                    "depth": depth,
                    "dimensions_3d_hwl": [
                        float(dims[0].item()),
                        float(dims[1].item()),
                        float(dims[2].item()),
                    ],
                    "yaw": yaw_angle,
                    "depth_uncertainty": unc,
                    "cell_x": int(cell_x),
                    "cell_y": int(cell_y),
                }
            )

        predictions = nms_2d(
            predictions=predictions,
            iou_threshold=nms_iou_threshold,
        )

        batch_predictions.append(predictions)

    return batch_predictions