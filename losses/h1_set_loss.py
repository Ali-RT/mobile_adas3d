from __future__ import annotations

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width * 0.5,
            center_y - height * 0.5,
            center_x + width * 0.5,
            center_y + height * 0.5,
        ),
        dim=-1,
    )


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    intersection = (right_bottom - left_top).clamp(min=0).prod(-1)
    area1 = (boxes1[:, 2:] - boxes1[:, :2]).clamp(min=0).prod(-1)
    area2 = (boxes2[:, 2:] - boxes2[:, :2]).clamp(min=0).prod(-1)
    union = area1[:, None] + area2[None, :] - intersection
    iou = intersection / union.clamp(min=1e-6)
    enclosing_left_top = torch.minimum(boxes1[:, None, :2], boxes2[None, :, :2])
    enclosing_right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    enclosing_area = (enclosing_right_bottom - enclosing_left_top).clamp(min=0).prod(-1)
    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-6)


class H1SetCriterion(nn.Module):
    distillation_enabled = False

    def __init__(
        self,
        num_classes: int,
        class_weights: List[float],
        match_class_cost: float = 2.0,
        match_box_cost: float = 5.0,
        match_giou_cost: float = 2.0,
        match_center_cost: float = 1.0,
        cls_weight: float = 2.0,
        box2d_weight: float = 5.0,
        giou_weight: float = 2.0,
        projected_center_weight: float = 1.0,
        depth_weight: float = 2.0,
        dim_weight: float = 1.0,
        yaw_weight: float = 1.0,
        loc_xy_weight: float = 1.0,
        quality_weight: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        self.register_buffer("class_weights", torch.tensor(class_weights))
        self.match_class_cost = float(match_class_cost)
        self.match_box_cost = float(match_box_cost)
        self.match_giou_cost = float(match_giou_cost)
        self.match_center_cost = float(match_center_cost)
        self.cls_weight = float(cls_weight)
        self.box2d_weight = float(box2d_weight)
        self.giou_weight = float(giou_weight)
        self.projected_center_weight = float(projected_center_weight)
        self.depth_weight = float(depth_weight)
        self.dim_weight = float(dim_weight)
        self.yaw_weight = float(yaw_weight)
        self.loc_xy_weight = float(loc_xy_weight)
        self.quality_weight = float(quality_weight)
        self.focal_alpha = float(focal_alpha)
        self.focal_gamma = float(focal_gamma)

    @torch.no_grad()
    def _match(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        matches = []
        for batch_index in range(outputs["class_logits"].shape[0]):
            mask = targets["object_mask"][batch_index]
            count = int(mask.sum().item())
            if count == 0:
                empty = torch.empty(0, dtype=torch.long, device=mask.device)
                matches.append((empty, empty))
                continue
            target_classes = targets["class_ids"][batch_index, mask]
            target_boxes = targets["box2d"][batch_index, mask]
            target_centers = targets["projected_center"][batch_index, mask]
            target_center_valid = targets["projected_center_valid"][batch_index, mask]
            class_probability = outputs["class_logits"][batch_index].sigmoid()
            class_cost = -class_probability[:, target_classes]
            box_cost = torch.cdist(outputs["box2d_cxcywh"][batch_index], target_boxes, p=1)
            giou_cost = -generalized_box_iou(
                box_cxcywh_to_xyxy(outputs["box2d_cxcywh"][batch_index]),
                box_cxcywh_to_xyxy(target_boxes),
            )
            center_cost = torch.cdist(
                outputs["projected_center"][batch_index], target_centers, p=1
            )
            center_cost = center_cost * target_center_valid.to(center_cost.dtype).unsqueeze(0)
            cost = (
                self.match_class_cost * class_cost
                + self.match_box_cost * box_cost
                + self.match_giou_cost * giou_cost
                + self.match_center_cost * center_cost
            )
            query_indices, target_indices = linear_sum_assignment(
                cost.detach().float().cpu().numpy()
            )
            matches.append(
                (
                    torch.as_tensor(query_indices, dtype=torch.long, device=mask.device),
                    torch.as_tensor(target_indices, dtype=torch.long, device=mask.device),
                )
            )
        return matches

    def forward(
        self, outputs: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        matches = self._match(outputs, targets)
        class_target = torch.zeros_like(outputs["class_logits"])
        quality_target = torch.zeros_like(outputs["quality"])
        matched_predictions: Dict[str, List[torch.Tensor]] = {
            key: [] for key in outputs if key not in {"class_logits", "quality"}
        }
        matched_targets: Dict[str, List[torch.Tensor]] = {
            key: []
            for key in (
                "box2d", "projected_center", "projected_center_valid", "depth_bin",
                "depth_residual", "dimensions", "yaw", "location_xy"
            )
        }
        for batch_index, (query_indices, target_indices) in enumerate(matches):
            if not query_indices.numel():
                continue
            valid_targets = targets["object_mask"][batch_index]
            for local_index, query_index in enumerate(query_indices):
                target_index = target_indices[local_index]
                padded_index = torch.nonzero(valid_targets, as_tuple=False).flatten()[target_index]
                class_id = targets["class_ids"][batch_index, padded_index]
                class_target[batch_index, query_index, class_id] = 1.0
            target_boxes = targets["box2d"][batch_index, valid_targets][target_indices]
            predicted_boxes = outputs["box2d_cxcywh"][batch_index, query_indices]
            pair_iou = generalized_box_iou(
                box_cxcywh_to_xyxy(predicted_boxes), box_cxcywh_to_xyxy(target_boxes)
            ).diag().clamp(0.0, 1.0)
            quality_target[batch_index, query_indices, 0] = pair_iou.detach()
            for key in matched_predictions:
                matched_predictions[key].append(outputs[key][batch_index, query_indices])
            for key in matched_targets:
                matched_targets[key].append(
                    targets[key][batch_index, valid_targets][target_indices]
                )

        probability = outputs["class_logits"].sigmoid()
        bce = F.binary_cross_entropy_with_logits(
            outputs["class_logits"], class_target, reduction="none"
        )
        pt = probability * class_target + (1.0 - probability) * (1.0 - class_target)
        alpha = self.focal_alpha * class_target + (1.0 - self.focal_alpha) * (1.0 - class_target)
        class_weights = self.class_weights.to(
            device=class_target.device, dtype=class_target.dtype
        )
        class_balance = torch.where(
            class_target > 0,
            class_weights.view(1, 1, -1),
            torch.ones_like(class_target),
        )
        cls_loss = (alpha * (1.0 - pt).pow(self.focal_gamma) * bce * class_balance).mean()
        quality_loss = F.binary_cross_entropy_with_logits(outputs["quality"], quality_target)

        zero = outputs["class_logits"].sum() * 0.0
        if not matched_targets["box2d"]:
            box_l1 = giou_loss = projected_loss = depth_loss = dim_loss = yaw_loss = loc_loss = zero
        else:
            prediction = {key: torch.cat(value) for key, value in matched_predictions.items()}
            target = {key: torch.cat(value) for key, value in matched_targets.items()}
            box_l1 = F.l1_loss(prediction["box2d_cxcywh"], target["box2d"])
            giou_loss = 1.0 - generalized_box_iou(
                box_cxcywh_to_xyxy(prediction["box2d_cxcywh"]),
                box_cxcywh_to_xyxy(target["box2d"]),
            ).diag().mean()
            projected_mask = target["projected_center_valid"]
            projected_loss = (
                F.l1_loss(
                    prediction["projected_center"][projected_mask],
                    target["projected_center"][projected_mask],
                )
                if projected_mask.any()
                else zero
            )
            depth_loss = F.cross_entropy(prediction["depth_logits"], target["depth_bin"])
            depth_loss = depth_loss + F.smooth_l1_loss(
                prediction["depth_residual"].squeeze(-1), target["depth_residual"]
            )
            dim_loss = F.smooth_l1_loss(prediction["dimensions"], target["dimensions"])
            predicted_yaw = F.normalize(prediction["yaw"], dim=-1, eps=1e-4)
            yaw_loss = (1.0 - (predicted_yaw * target["yaw"]).sum(-1)).mean()
            loc_loss = F.smooth_l1_loss(prediction["location_xy"], target["location_xy"])

        box2d_loss = self.box2d_weight * box_l1 + self.giou_weight * giou_loss
        total = (
            self.cls_weight * cls_loss
            + box2d_loss
            + self.projected_center_weight * projected_loss
            + self.depth_weight * depth_loss
            + self.dim_weight * dim_loss
            + self.yaw_weight * yaw_loss
            + self.loc_xy_weight * loc_loss
            + self.quality_weight * quality_loss
        )
        return {
            "total_loss": total,
            "cls_loss": cls_loss,
            "box2d_loss": box2d_loss,
            "depth_loss": depth_loss,
            "depth_uncertainty_loss": zero,
            "dim_loss": dim_loss,
            "yaw_loss": yaw_loss,
            "yaw_cosine_loss": yaw_loss,
            "yaw_direction_loss": zero,
            "offset_loss": zero,
            "loc_xy_loss": loc_loss,
            "projected_center_loss": projected_loss,
            "quality_loss": quality_loss,
            "corner3d_loss": zero,
        }
