from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    alpha: float = 0.25,
    gamma: float = 2.0,
    normalizer: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)

    ce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)
    focal_factor = (1.0 - p_t).pow(gamma)

    loss = ce_loss * focal_factor

    if alpha >= 0:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * loss

    if weights is not None:
        loss = loss * weights

    if normalizer is None:
        normalizer = torch.clamp(targets.sum(), min=1.0)

    return loss.sum() / torch.clamp(normalizer, min=1.0)


def masked_weighted_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_weight: Optional[torch.Tensor] = None,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    pred/target: [B, C, H, W]
    valid_mask: [B, 1, H, W]
    loss_weight: [B, 1, H, W]
    """
    loss = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=beta,
    )

    mask = valid_mask.expand_as(loss)

    if loss_weight is not None:
        mask = mask * loss_weight.expand_as(loss)

    numerator = (loss * mask).sum()
    denominator = torch.clamp(mask.sum(), min=1.0)

    return numerator / denominator


def masked_weighted_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    loss = torch.abs(pred - target)
    mask = valid_mask.expand_as(loss)

    if loss_weight is not None:
        mask = mask * loss_weight.expand_as(loss)

    numerator = (loss * mask).sum()
    denominator = torch.clamp(mask.sum(), min=1.0)

    return numerator / denominator


class MobileADAS3DLoss(nn.Module):
    def __init__(
        self,
        input_height: int,
        input_width: int,
        classes: Optional[List[str]] = None,
        class_weights: Optional[Dict[str, float]] = None,
        cls_weight: float = 1.0,
        box2d_weight: float = 2.0,
        depth_weight: float = 1.0,
        depth_uncertainty_weight: float = 0.0,
        dim_weight: float = 1.0,
        yaw_weight: float = 1.0,
        offset_weight: float = 0.5,
    ) -> None:
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width

        self.cls_weight = cls_weight
        self.box2d_weight = box2d_weight
        self.depth_weight = depth_weight
        self.depth_uncertainty_weight = depth_uncertainty_weight
        self.dim_weight = dim_weight
        self.yaw_weight = yaw_weight
        self.offset_weight = offset_weight

        classes = classes or []
        class_weights = class_weights or {}

        weights = [
            float(class_weights.get(class_name, 1.0))
            for class_name in classes
        ]

        if len(weights) == 0:
            weights = [1.0]

        self.register_buffer(
            "class_weights_tensor",
            torch.tensor(weights, dtype=torch.float32),
        )

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cls_logits = outputs["cls_logits"]
        box2d_pred = outputs["box2d"]
        log_depth_pred = outputs["log_depth"]
        dim_pred = outputs["dim"]
        yaw_pred = outputs["yaw"]
        offset_pred = outputs["center_offset"]
        depth_uncertainty_pred = outputs["depth_uncertainty"]

        cls_target = targets["cls_target"]
        box2d_target = targets["box2d_target"]
        log_depth_target = targets["log_depth_target"]
        dim_target = targets["dim_target"]
        yaw_target = targets["yaw_target"]
        offset_target = targets["offset_target"]
        valid_mask = targets["valid_mask"]

        loss_weight_target = targets.get(
            "loss_weight_target",
            torch.ones_like(valid_mask),
        )

        if cls_logits.shape != cls_target.shape:
            raise RuntimeError(
                f"cls shape mismatch: pred={cls_logits.shape}, target={cls_target.shape}"
            )

        # Positive class balancing for focal classification.
        class_weights = self.class_weights_tensor.to(cls_logits.device)

        if class_weights.numel() != cls_logits.shape[1]:
            class_weights = torch.ones(
                cls_logits.shape[1],
                device=cls_logits.device,
                dtype=cls_logits.dtype,
            )

        class_weights_view = class_weights.view(1, -1, 1, 1)

        cls_loss_weights = torch.where(
            cls_target > 0,
            class_weights_view.expand_as(cls_target),
            torch.ones_like(cls_target),
        )

        cls_normalizer = torch.clamp(
            (cls_target * class_weights_view).sum(),
            min=1.0,
        )

        cls_loss = sigmoid_focal_loss(
            logits=cls_logits,
            targets=cls_target,
            weights=cls_loss_weights,
            alpha=0.25,
            gamma=2.0,
            normalizer=cls_normalizer,
        )

        box2d_loss = masked_weighted_smooth_l1_loss(
            pred=box2d_pred,
            target=box2d_target,
            valid_mask=valid_mask,
            loss_weight=loss_weight_target,
            beta=1.0,
        )

        depth_loss = masked_weighted_smooth_l1_loss(
            pred=log_depth_pred,
            target=log_depth_target,
            valid_mask=valid_mask,
            loss_weight=loss_weight_target,
            beta=1.0,
        )

        # Optional uncertainty NLL. Kept safe and clamped.
        log_scale = torch.clamp(depth_uncertainty_pred, min=-3.0, max=3.0)
        depth_abs_error = torch.abs(log_depth_pred - log_depth_target)

        depth_uncertainty_raw = torch.exp(-log_scale) * depth_abs_error + log_scale

        depth_uncertainty_loss = masked_weighted_l1_loss(
            pred=depth_uncertainty_raw,
            target=torch.zeros_like(depth_uncertainty_raw),
            valid_mask=valid_mask,
            loss_weight=loss_weight_target,
        )

        dim_loss = masked_weighted_smooth_l1_loss(
            pred=dim_pred,
            target=dim_target,
            valid_mask=valid_mask,
            loss_weight=loss_weight_target,
            beta=1.0,
        )

        yaw_pred_norm = F.normalize(yaw_pred, dim=1, eps=1e-6)

        yaw_loss = masked_weighted_smooth_l1_loss(
            pred=yaw_pred_norm,
            target=yaw_target,
            valid_mask=valid_mask,
            loss_weight=loss_weight_target,
            beta=1.0,
        )

        offset_loss = masked_weighted_smooth_l1_loss(
            pred=offset_pred,
            target=offset_target,
            valid_mask=valid_mask,
            loss_weight=loss_weight_target,
            beta=1.0,
        )

        total_loss = (
            self.cls_weight * cls_loss
            + self.box2d_weight * box2d_loss
            + self.depth_weight * depth_loss
            + self.depth_uncertainty_weight * depth_uncertainty_loss
            + self.dim_weight * dim_loss
            + self.yaw_weight * yaw_loss
            + self.offset_weight * offset_loss
        )

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "box2d_loss": box2d_loss,
            "depth_loss": depth_loss,
            "depth_uncertainty_loss": depth_uncertainty_loss,
            "dim_loss": dim_loss,
            "yaw_loss": yaw_loss,
            "offset_loss": offset_loss,
        }