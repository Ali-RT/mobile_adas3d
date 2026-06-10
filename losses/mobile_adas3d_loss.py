from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def sigmoid_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    Binary focal loss for dense multi-label class maps.

    Args:
      logits:  [B, C, H, W]
      targets: [B, C, H, W], values 0 or 1
    """
    prob = torch.sigmoid(logits)

    ce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
    )

    p_t = prob * targets + (1.0 - prob) * (1.0 - targets)

    focal_weight = (1.0 - p_t).pow(gamma)

    alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)

    loss = alpha_t * focal_weight * ce_loss

    if reduction == "mean":
        return loss.mean()

    if reduction == "sum":
        return loss.sum()

    if reduction == "none":
        return loss

    raise ValueError(f"Unsupported reduction: {reduction}")


def masked_smooth_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    SmoothL1 loss only on positive object cells.

    Args:
      pred:       [B, C, H, W]
      target:     [B, C, H, W]
      valid_mask: [B, 1, H, W]
    """
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target shape mismatch: pred={pred.shape}, target={target.shape}"
        )

    if valid_mask.ndim != 4 or valid_mask.shape[1] != 1:
        raise ValueError(f"Expected valid_mask shape [B, 1, H, W], got {valid_mask.shape}")

    mask = valid_mask.expand_as(pred)

    loss = F.smooth_l1_loss(
        pred,
        target,
        reduction="none",
        beta=beta,
    )

    loss = loss * mask

    denom = mask.sum().clamp(min=1.0)

    return loss.sum() / denom

def masked_depth_uncertainty_loss(
    pred_log_depth: torch.Tensor,
    target_log_depth: torch.Tensor,
    pred_logvar: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Uncertainty-weighted depth loss.

    pred_log_depth:    [B, 1, H, W]
    target_log_depth:  [B, 1, H, W]
    pred_logvar:       [B, 1, H, W]
    valid_mask:        [B, 1, H, W]

    This trains both:
      - the depth prediction
      - the uncertainty prediction

    Loss form:
      0.5 * exp(-logvar) * error^2 + 0.5 * logvar
    """
    if pred_log_depth.shape != target_log_depth.shape:
        raise ValueError(
            f"Depth shape mismatch: pred={pred_log_depth.shape}, target={target_log_depth.shape}"
        )

    if pred_logvar.shape != pred_log_depth.shape:
        raise ValueError(
            f"logvar shape mismatch: logvar={pred_logvar.shape}, depth={pred_log_depth.shape}"
        )

    # Clamp for numerical stability.
    pred_logvar = torch.clamp(pred_logvar, min=-5.0, max=5.0)

    error = pred_log_depth - target_log_depth

    loss = 0.5 * torch.exp(-pred_logvar) * error.pow(2) + 0.5 * pred_logvar

    loss = loss * valid_mask

    denom = valid_mask.sum().clamp(min=1.0)

    return loss.sum() / denom

def normalize_box2d(
    box: torch.Tensor,
    input_height: int,
    input_width: int,
) -> torch.Tensor:
    """
    Normalize absolute box coordinates to [0, 1]-scale.

    box format:
      [x1, y1, x2, y2]
    """
    scale = torch.tensor(
        [input_width, input_height, input_width, input_height],
        dtype=box.dtype,
        device=box.device,
    )

    return box / scale.view(1, 4, 1, 1)


class MobileADAS3DLoss(nn.Module):
    """
    Combined loss for MobileADAS3D.

    Loss components:
      - classification focal loss over all cells
      - 2D box loss on positive cells
      - log-depth loss on positive cells
      - dimension residual loss on positive cells
      - yaw sin/cos loss on positive cells
      - center offset loss on positive cells
    """

    def __init__(
        self,
        input_height: int,
        input_width: int,
        cls_weight: float = 1.0,
        box2d_weight: float = 2.0,
        depth_weight: float = 1.0,
        dim_weight: float = 1.0,
        yaw_weight: float = 1.0,
        offset_weight: float = 0.5,
        depth_uncertainty_weight: float = 0.2,
    ) -> None:
        super().__init__()

        self.input_height = input_height
        self.input_width = input_width

        self.cls_weight = cls_weight
        self.box2d_weight = box2d_weight
        self.depth_weight = depth_weight
        self.dim_weight = dim_weight
        self.yaw_weight = yaw_weight
        self.offset_weight = offset_weight
        self.depth_uncertainty_weight = depth_uncertainty_weight

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        cls_target = targets["cls_target"]
        box2d_target = targets["box2d_target"]
        log_depth_target = targets["log_depth_target"]
        dim_target = targets["dim_target"]
        yaw_target = targets["yaw_target"]
        offset_target = targets["offset_target"]
        valid_mask = targets["valid_mask"]

        cls_loss = sigmoid_focal_loss(
            logits=outputs["cls_logits"],
            targets=cls_target,
        )

        pred_box_norm = normalize_box2d(
            outputs["box2d"],
            input_height=self.input_height,
            input_width=self.input_width,
        )

        target_box_norm = normalize_box2d(
            box2d_target,
            input_height=self.input_height,
            input_width=self.input_width,
        )

        box2d_loss = masked_smooth_l1_loss(
            pred=pred_box_norm,
            target=target_box_norm,
            valid_mask=valid_mask,
        )

        depth_loss = masked_smooth_l1_loss(
            pred=outputs["log_depth"],
            target=log_depth_target,
            valid_mask=valid_mask,
        )

        depth_uncertainty_loss = masked_depth_uncertainty_loss(
            pred_log_depth=outputs["log_depth"],
            target_log_depth=log_depth_target,
            pred_logvar=outputs["depth_logvar"],
            valid_mask=valid_mask,
        )
        dim_loss = masked_smooth_l1_loss(
            pred=outputs["dim_residual"],
            target=dim_target,
            valid_mask=valid_mask,
        )

        # Normalize predicted yaw vector before comparing to target sin/cos.
        pred_yaw = outputs["yaw_sincos"]
        pred_yaw = F.normalize(pred_yaw, dim=1, eps=1e-6)

        yaw_loss = masked_smooth_l1_loss(
            pred=pred_yaw,
            target=yaw_target,
            valid_mask=valid_mask,
        )

        offset_loss = masked_smooth_l1_loss(
            pred=outputs["center_offset"],
            target=offset_target,
            valid_mask=valid_mask,
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