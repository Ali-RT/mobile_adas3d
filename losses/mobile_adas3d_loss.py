from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.geometry import compute_kitti_corners_3d_torch


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


def masked_weighted_yaw_cosine_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_weight: Optional[torch.Tensor] = None,
    pred_is_normalized: bool = False,
) -> torch.Tensor:
    """Angular loss for normalized ``[sin(yaw), cos(yaw)]`` vectors."""
    pred_norm = pred if pred_is_normalized else F.normalize(pred, dim=1, eps=1e-6)
    target_norm = F.normalize(target, dim=1, eps=1e-6)
    loss = 1.0 - (pred_norm * target_norm).sum(dim=1, keepdim=True)
    mask = valid_mask

    if loss_weight is not None:
        mask = mask * loss_weight

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
        yaw_cosine_weight: float = 0.0,
        yaw_direction_weight: float = 0.0,
        offset_weight: float = 0.5,
        loc_xy_weight: float = 1.0,
        projected_center_weight: float = 0.0,
        quality_weight: float = 0.0,
        corner3d_weight: float = 0.0,
        detach_yaw_in_corner3d: bool = False,
        yaw_pred_is_direct_sincos: bool = False,
        class_mean_dims: Optional[Dict[str, List[float]]] = None,
        distillation_enabled: bool = False,
        teacher_depth_weight: float = 0.0,
        teacher_dim_weight: float = 0.0,
        teacher_loc_xy_weight: float = 0.0,
        teacher_yaw_weight: float = 0.0,
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
        self.yaw_cosine_weight = yaw_cosine_weight
        self.yaw_direction_weight = yaw_direction_weight
        self.offset_weight = offset_weight
        self.loc_xy_weight = loc_xy_weight
        self.projected_center_weight = projected_center_weight
        self.quality_weight = quality_weight
        self.corner3d_weight = corner3d_weight
        self.detach_yaw_in_corner3d = bool(detach_yaw_in_corner3d)
        self.yaw_pred_is_direct_sincos = bool(yaw_pred_is_direct_sincos)
        self.distillation_enabled = bool(distillation_enabled)
        self.teacher_depth_weight = float(teacher_depth_weight)
        self.teacher_dim_weight = float(teacher_dim_weight)
        self.teacher_loc_xy_weight = float(teacher_loc_xy_weight)
        self.teacher_yaw_weight = float(teacher_yaw_weight)

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

        # [num_classes, 3] in h/w/l order. Only used by the optional corner3d loss.
        class_mean_dims = class_mean_dims or {}

        mean_dims_values = [
            [float(v) for v in class_mean_dims.get(class_name, [1.0, 1.0, 1.0])]
            for class_name in classes
        ]

        if len(mean_dims_values) == 0:
            mean_dims_values = [[1.0, 1.0, 1.0]]

        self.register_buffer(
            "class_mean_dims_tensor",
            torch.tensor(mean_dims_values, dtype=torch.float32),
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
        yaw_axis_pred = outputs.get("yaw_axis")
        yaw_direction_pred = outputs.get("yaw_direction")
        offset_pred = outputs["center_offset"]
        depth_uncertainty_pred = outputs["depth_uncertainty"]
        loc_xy_pred = outputs["loc_xy"]
        projected_center_pred = outputs.get("projected_center_offset")
        quality_pred = outputs.get("quality")

        cls_target = targets["cls_target"]
        box2d_target = targets["box2d_target"]
        log_depth_target = targets["log_depth_target"]
        dim_target = targets["dim_target"]
        yaw_target = targets["yaw_target"]
        yaw_axis_target = targets.get("yaw_axis_target")
        yaw_direction_target = targets.get("yaw_direction_target")
        offset_target = targets["offset_target"]
        loc_xy_target = targets["loc_xy_target"]
        projected_center_target = targets.get("projected_center_offset_target")
        quality_target = targets.get("quality_target")
        valid_mask = targets["valid_mask"]
        projected_center_valid_mask = targets.get(
            "projected_center_valid_mask",
            valid_mask,
        )

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

        yaw_reg_pred = yaw_axis_pred if yaw_axis_pred is not None else yaw_pred
        yaw_reg_target = yaw_axis_target if yaw_axis_target is not None and yaw_axis_pred is not None else yaw_target
        yaw_pred_norm = (
            yaw_reg_pred
            if self.yaw_pred_is_direct_sincos and yaw_axis_pred is None
            else F.normalize(yaw_reg_pred, dim=1, eps=1e-6)
        )

        yaw_loss = masked_weighted_smooth_l1_loss(
            pred=yaw_pred_norm,
            target=yaw_reg_target,
            valid_mask=valid_mask,
            loss_weight=loss_weight_target,
            beta=1.0,
        )

        yaw_cosine_loss = masked_weighted_yaw_cosine_loss(
            pred=yaw_reg_pred,
            target=yaw_reg_target,
            valid_mask=valid_mask,
            loss_weight=loss_weight_target,
            pred_is_normalized=(
                self.yaw_pred_is_direct_sincos and yaw_axis_pred is None
            ),
        )

        yaw_direction_loss = torch.zeros(
            (),
            device=cls_logits.device,
            dtype=cls_logits.dtype,
        )
        if yaw_direction_pred is not None and yaw_direction_target is not None:
            direction_raw = F.binary_cross_entropy_with_logits(
                yaw_direction_pred,
                yaw_direction_target,
                reduction="none",
            )
            direction_mask = valid_mask * loss_weight_target
            yaw_direction_loss = (
                direction_raw * direction_mask
            ).sum() / direction_mask.sum().clamp(min=1.0)

        offset_loss = masked_weighted_smooth_l1_loss(
            pred=offset_pred,
            target=offset_target,
            valid_mask=valid_mask,
            loss_weight=loss_weight_target,
            beta=1.0,
        )

        loc_xy_loss = masked_weighted_smooth_l1_loss(
            pred=loc_xy_pred,
            target=loc_xy_target,
            valid_mask=valid_mask,
            loss_weight=loss_weight_target,
            beta=1.0,
        )

        projected_center_loss = torch.zeros(
            (),
            device=cls_logits.device,
            dtype=cls_logits.dtype,
        )

        if (
            self.projected_center_weight > 0.0
            and projected_center_pred is not None
            and projected_center_target is not None
        ):
            projected_center_loss = masked_weighted_smooth_l1_loss(
                pred=projected_center_pred,
                target=projected_center_target,
                valid_mask=valid_mask * projected_center_valid_mask,
                loss_weight=loss_weight_target,
                beta=1.0,
            )

        quality_loss = torch.zeros(
            (),
            device=cls_logits.device,
            dtype=cls_logits.dtype,
        )

        if (
            self.quality_weight > 0.0
            and quality_pred is not None
            and quality_target is not None
        ):
            quality_loss = sigmoid_focal_loss(
                logits=quality_pred,
                targets=quality_target,
                weights=None,
                alpha=0.25,
                gamma=2.0,
                normalizer=valid_mask.sum().clamp(min=1.0),
            )

        # Optional cuboid corner consistency loss.
        # Reconstructs the KITTI cuboid from predicted physical pose
        # (location_xyz + dimensions + yaw) and compares against GT corners.
        corner3d_loss = torch.zeros((), device=cls_logits.device, dtype=cls_logits.dtype)

        if self.corner3d_weight > 0.0:
            corner3d_loss = self._compute_corner3d_loss(
                outputs=outputs,
                targets=targets,
                valid_mask=valid_mask,
            )

        total_loss = (
            self.cls_weight * cls_loss
            + self.box2d_weight * box2d_loss
            + self.depth_weight * depth_loss
            + self.depth_uncertainty_weight * depth_uncertainty_loss
            + self.dim_weight * dim_loss
            + self.yaw_weight * yaw_loss
            + self.yaw_cosine_weight * yaw_cosine_loss
            + self.yaw_direction_weight * yaw_direction_loss
            + self.offset_weight * offset_loss
            + self.loc_xy_weight * loc_xy_loss
            + self.projected_center_weight * projected_center_loss
            + self.quality_weight * quality_loss
            + self.corner3d_weight * corner3d_loss
        )

        teacher_losses: Dict[str, torch.Tensor] = {}
        if self.distillation_enabled:
            teacher_mask = targets.get("teacher_valid_mask")
            if teacher_mask is None:
                zero = cls_logits.sum() * 0.0
                teacher_depth_loss = zero
                teacher_dim_loss = zero
                teacher_loc_xy_loss = zero
                teacher_yaw_loss = zero
            else:
                teacher_score = targets.get(
                    "teacher_score_target", torch.ones_like(teacher_mask)
                )
                teacher_depth_loss = masked_weighted_smooth_l1_loss(
                    log_depth_pred,
                    targets["teacher_log_depth_target"],
                    teacher_mask,
                    teacher_score,
                    pred_is_normalized=self.yaw_pred_is_direct_sincos,
                )
                teacher_dim_loss = masked_weighted_smooth_l1_loss(
                    dim_pred,
                    targets["teacher_dim_target"],
                    teacher_mask,
                    teacher_score,
                )
                teacher_loc_xy_loss = masked_weighted_smooth_l1_loss(
                    loc_xy_pred,
                    targets["teacher_loc_xy_target"],
                    teacher_mask,
                    teacher_score,
                )
                teacher_yaw_loss = masked_weighted_yaw_cosine_loss(
                    yaw_pred,
                    targets["teacher_yaw_target"],
                    teacher_mask,
                    teacher_score,
                )
            total_loss = (
                total_loss
                + self.teacher_depth_weight * teacher_depth_loss
                + self.teacher_dim_weight * teacher_dim_loss
                + self.teacher_loc_xy_weight * teacher_loc_xy_loss
                + self.teacher_yaw_weight * teacher_yaw_loss
            )
            teacher_losses = {
                "teacher_depth_loss": teacher_depth_loss,
                "teacher_dim_loss": teacher_dim_loss,
                "teacher_loc_xy_loss": teacher_loc_xy_loss,
                "teacher_yaw_loss": teacher_yaw_loss,
            }

        losses = {
            "total_loss": total_loss,
            "cls_loss": cls_loss,
            "box2d_loss": box2d_loss,
            "depth_loss": depth_loss,
            "depth_uncertainty_loss": depth_uncertainty_loss,
            "dim_loss": dim_loss,
            "yaw_loss": yaw_loss,
            "yaw_cosine_loss": yaw_cosine_loss,
            "yaw_direction_loss": yaw_direction_loss,
            "offset_loss": offset_loss,
            "loc_xy_loss": loc_xy_loss,
            "projected_center_loss": projected_center_loss,
            "quality_loss": quality_loss,
            "corner3d_loss": corner3d_loss,
        }
        losses.update(teacher_losses)
        return losses

    def _compute_corner3d_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Corner consistency loss over positive cells only.

        Decodes predicted location_xyz, dimensions, and yaw, builds the KITTI
        cuboid corners, and compares against GT corners reconstructed from the
        target location/dim/yaw.
        """
        pos = valid_mask[:, 0].bool()  # [B, H, W]

        if pos.sum() == 0:
            return outputs["loc_xy"].sum() * 0.0

        device = outputs["loc_xy"].device
        dtype = outputs["loc_xy"].dtype

        class_mean_dims = self.class_mean_dims_tensor.to(device=device, dtype=dtype)

        # Per-cell class id from one-hot cls target.
        cls_target = targets["cls_target"]  # [B, C, H, W]
        class_ids = cls_target.argmax(dim=1)  # [B, H, W]

        # Gather positive-cell channel-last tensors.
        def gather(tensor: torch.Tensor) -> torch.Tensor:
            # tensor: [B, C, H, W] -> [N, C]
            return tensor.permute(0, 2, 3, 1)[pos]

        loc_xy = gather(outputs["loc_xy"])            # [N, 2]
        log_depth = gather(outputs["log_depth"])      # [N, 1]
        dim_residual = gather(outputs["dim"])         # [N, 3]
        # V5 reconstructs final yaw with atan2 plus a hard direction decision.
        # Its dedicated axis and direction losses supervise the yaw heads.
        # Do not backpropagate the cuboid loss through that discontinuous
        # reconstruction: atan2(0, 0) can otherwise produce NaN gradients
        # early in training when the raw axis head is close to zero.
        yaw_for_corners = outputs["yaw"]
        if "yaw_axis" in outputs or self.detach_yaw_in_corner3d:
            yaw_for_corners = yaw_for_corners.detach()
        yaw_vec = F.normalize(gather(yaw_for_corners), dim=-1, eps=1e-6)  # [N, 2]

        pos_class_ids = class_ids[pos]                # [N]
        mean_dims = class_mean_dims[pos_class_ids]    # [N, 3]

        z = torch.exp(log_depth[:, 0]).clamp(min=0.1, max=200.0)  # [N]
        x = loc_xy[:, 0] * z
        y = loc_xy[:, 1] * z
        pred_location = torch.stack([x, y, z], dim=-1)  # [N, 3]

        pred_dims = (mean_dims * torch.exp(dim_residual)).clamp(min=0.01)  # [N, 3]
        pred_yaw = torch.atan2(yaw_vec[:, 0], yaw_vec[:, 1])               # [N]

        pred_corners = compute_kitti_corners_3d_torch(
            dims_hwl=pred_dims,
            location_xyz=pred_location,
            rotation_y=pred_yaw,
        )  # [N, 8, 3]

        # GT corners from target location/dim/yaw.
        gt_location = gather(targets["location_xyz_target"])  # [N, 3]
        gt_dim_residual = gather(targets["dim_target"])       # [N, 3]
        gt_yaw_vec = gather(targets["yaw_target"])            # [N, 2]

        gt_dims = (mean_dims * torch.exp(gt_dim_residual)).clamp(min=0.01)
        gt_yaw = torch.atan2(gt_yaw_vec[:, 0], gt_yaw_vec[:, 1])

        gt_corners = compute_kitti_corners_3d_torch(
            dims_hwl=gt_dims,
            location_xyz=gt_location,
            rotation_y=gt_yaw,
        )  # [N, 8, 3]

        return F.smooth_l1_loss(
            pred_corners.contiguous(),
            gt_corners.contiguous(),
            reduction="mean",
        )
