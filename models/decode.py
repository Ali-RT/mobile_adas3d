from typing import Any, Dict, List

import torch


def box_iou(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """
    box:   [4]
    boxes: [N, 4]
    format: x1, y1, x2, y2
    """
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter = inter_w * inter_h

    area1 = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    area2 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)

    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


def nms_2d(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    if boxes.numel() == 0:
        return torch.empty((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        idx = order[0]
        keep.append(idx)

        if order.numel() == 1:
            break

        ious = box_iou(boxes[idx], boxes[order[1:]])
        order = order[1:][ious <= iou_threshold]

    return torch.stack(keep)


def decode_mobile_adas3d_outputs(
    outputs: Dict[str, torch.Tensor],
    classes: List[str],
    class_mean_dims: Dict[str, List[float]],
    input_height: int,
    input_width: int,
    score_threshold: float = 0.25,
    topk: int = 100,
    nms_iou_threshold: float = 0.5,
) -> List[List[Dict[str, Any]]]:
    """
    Decode dense model outputs into prediction dictionaries.

    Returns:
      batch_predictions: list of length B.
      Each element is a list of predictions for one image.
    """
    cls_scores = torch.sigmoid(outputs["cls_logits"])
    box2d = outputs["box2d"]
    log_depth = outputs["log_depth"]
    dim_residual = outputs["dim_residual"]
    yaw_sincos = outputs["yaw_sincos"]
    center_offset = outputs["center_offset"]
    depth_logvar = outputs["depth_logvar"]

    B, C, Hf, Wf = cls_scores.shape

    batch_predictions = []

    for b in range(B):
        scores_flat = cls_scores[b].reshape(-1)

        top_scores, top_indices = torch.topk(
            scores_flat,
            k=min(topk, scores_flat.numel()),
        )

        predictions = []

        for score, flat_idx in zip(top_scores, top_indices):
            score_value = float(score.item())

            if score_value < score_threshold:
                continue

            class_id = int(flat_idx // (Hf * Wf))
            rem = int(flat_idx % (Hf * Wf))
            gy = rem // Wf
            gx = rem % Wf

            class_name = classes[class_id]

            raw_box = box2d[b, :, gy, gx]

            x1 = raw_box[0].clamp(0, input_width - 1)
            y1 = raw_box[1].clamp(0, input_height - 1)
            x2 = raw_box[2].clamp(0, input_width - 1)
            y2 = raw_box[3].clamp(0, input_height - 1)

            # Ensure valid ordering.
            x_min = torch.minimum(x1, x2)
            y_min = torch.minimum(y1, y2)
            x_max = torch.maximum(x1, x2)
            y_max = torch.maximum(y1, y2)

            if (x_max - x_min) < 2 or (y_max - y_min) < 2:
                continue

            depth = torch.exp(log_depth[b, 0, gy, gx]).clamp(min=0.1, max=200.0)

            mean_dims = torch.tensor(
                class_mean_dims[class_name],
                dtype=dim_residual.dtype,
                device=dim_residual.device,
            )

            dims = torch.exp(dim_residual[b, :, gy, gx]) * mean_dims

            yaw_vec = yaw_sincos[b, :, gy, gx]
            yaw = torch.atan2(yaw_vec[0], yaw_vec[1])

            uncertainty = torch.exp(depth_logvar[b, 0, gy, gx]).clamp(min=1e-6, max=1e6)

            stride_x = input_width / Wf
            stride_y = input_height / Hf

            offset = center_offset[b, :, gy, gx]
            center_x = (gx + offset[0]) * stride_x
            center_y = (gy + offset[1]) * stride_y

            predictions.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "score": score_value,
                    "bbox_2d": [
                        float(x_min.item()),
                        float(y_min.item()),
                        float(x_max.item()),
                        float(y_max.item()),
                    ],
                    "depth": float(depth.item()),
                    "dimensions_3d_hwl": [
                        float(dims[0].item()),
                        float(dims[1].item()),
                        float(dims[2].item()),
                    ],
                    "yaw": float(yaw.item()),
                    "center_2d": [
                        float(center_x.item()),
                        float(center_y.item()),
                    ],
                    "depth_uncertainty": float(uncertainty.item()),
                }
            )

        if predictions:
            boxes = torch.tensor(
                [p["bbox_2d"] for p in predictions],
                dtype=torch.float32,
                device=cls_scores.device,
            )
            scores = torch.tensor(
                [p["score"] for p in predictions],
                dtype=torch.float32,
                device=cls_scores.device,
            )

            keep = nms_2d(boxes, scores, nms_iou_threshold)
            predictions = [predictions[int(i.item())] for i in keep]

        batch_predictions.append(predictions)

    return batch_predictions