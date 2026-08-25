from __future__ import annotations

import torch
import torch.nn.functional as F

from utils import box_ops


def _weighted_mean(values, weights):
    if values.ndim > 1:
        values = values.flatten(1).mean(1)
    return (values * weights).sum() / weights.sum().clamp(min=1e-6)


def compute_a1_distillation_losses(
    student_outputs,
    teacher_outputs,
    targets,
    matcher,
    config,
    student_group_num=11,
):
    """Align independently matched teacher/student queries through shared GT IDs."""
    zero = student_outputs["pred_logits"].sum() * 0.0
    student_final = {k: v for k, v in student_outputs.items() if k != "aux_outputs"}
    teacher_final = {k: v for k, v in teacher_outputs.items() if k != "aux_outputs"}
    student_indices = matcher(student_final, targets, group_num=student_group_num)
    teacher_indices = matcher(teacher_final, targets, group_num=1)
    min_score = float(config.get("min_teacher_score", 0.30))
    min_iou = float(config.get("min_teacher_iou_2d", 0.50))
    temperature = float(config.get("temperature", 2.0))
    class_weights = {
        int(key): float(value)
        for key, value in config.get("target_class_weights", {0: 1.0, 1: 0.25}).items()
    }

    aligned = []
    for batch_index, ((student_query, student_gt), (teacher_query, teacher_gt)) in enumerate(
        zip(student_indices, teacher_indices)
    ):
        teacher_by_gt = {
            int(gt): int(query) for query, gt in zip(teacher_query.tolist(), teacher_gt.tolist())
        }
        for query, gt in zip(student_query.tolist(), student_gt.tolist()):
            teacher_query_index = teacher_by_gt.get(int(gt))
            if teacher_query_index is None:
                continue
            label = int(targets[batch_index]["labels"][gt].item())
            teacher_score = teacher_final["pred_logits"][
                batch_index, teacher_query_index, label
            ].sigmoid()
            teacher_box = box_ops.box_cxcylrtb_to_xyxy(
                teacher_final["pred_boxes"][batch_index, teacher_query_index].unsqueeze(0)
            )
            target_box = box_ops.box_cxcylrtb_to_xyxy(
                targets[batch_index]["boxes_3d"][gt].unsqueeze(0)
            )
            teacher_iou = box_ops.box_iou(teacher_box, target_box)[0][0, 0]
            if float(teacher_score) < min_score or float(teacher_iou) < min_iou:
                continue
            aligned.append(
                (batch_index, int(query), teacher_query_index, label, teacher_score, teacher_iou)
            )

    if not aligned:
        return {
            "distill_logits": zero,
            "distill_boxes": zero,
            "distill_depth": zero,
            "distill_dims": zero,
            "distill_angles": zero,
            "distill_total": zero,
            "distill_pairs": zero.detach(),
        }

    device = student_final["pred_logits"].device
    batch = torch.tensor([row[0] for row in aligned], device=device, dtype=torch.long)
    student_query = torch.tensor([row[1] for row in aligned], device=device, dtype=torch.long)
    teacher_query = torch.tensor([row[2] for row in aligned], device=device, dtype=torch.long)
    labels = torch.tensor([row[3] for row in aligned], device=device, dtype=torch.long)
    weights = torch.tensor(
        [class_weights.get(int(label), 1.0) for label in labels.tolist()],
        device=device,
        dtype=student_final["pred_logits"].dtype,
    )

    student_logits = student_final["pred_logits"][batch, student_query]
    teacher_logits = teacher_final["pred_logits"][batch, teacher_query].detach()
    soft_logits = torch.sigmoid(teacher_logits / temperature)
    logits_loss = F.binary_cross_entropy_with_logits(
        student_logits / temperature, soft_logits, reduction="none"
    ) * (temperature ** 2)

    student_boxes = student_final["pred_boxes"][batch, student_query]
    teacher_boxes = teacher_final["pred_boxes"][batch, teacher_query].detach()
    box_loss = F.smooth_l1_loss(student_boxes, teacher_boxes, reduction="none")

    student_depth = student_final["pred_depth"][batch, student_query, 0]
    teacher_depth = teacher_final["pred_depth"][batch, teacher_query, 0].detach()
    depth_loss = (student_depth - teacher_depth).abs() / teacher_depth.abs().clamp(min=1.0)

    student_dims = student_final["pred_3d_dim"][batch, student_query]
    teacher_dims = teacher_final["pred_3d_dim"][batch, teacher_query].detach()
    dim_loss = (student_dims - teacher_dims).abs() / teacher_dims.abs().clamp(min=0.25)

    student_angles = student_final["pred_angle"][batch, student_query]
    teacher_angles = teacher_final["pred_angle"][batch, teacher_query].detach()
    teacher_angle_prob = F.softmax(teacher_angles[:, :12] / temperature, dim=1)
    angle_cls = F.kl_div(
        F.log_softmax(student_angles[:, :12] / temperature, dim=1),
        teacher_angle_prob,
        reduction="none",
    ).sum(1) * (temperature ** 2)
    teacher_bin = teacher_angles[:, :12].argmax(1, keepdim=True)
    student_residual = student_angles[:, 12:].gather(1, teacher_bin).squeeze(1)
    teacher_residual = teacher_angles[:, 12:].gather(1, teacher_bin).squeeze(1)
    angle_loss = angle_cls + F.smooth_l1_loss(
        student_residual, teacher_residual, reduction="none"
    )

    losses = {
        "distill_logits": _weighted_mean(logits_loss, weights),
        "distill_boxes": _weighted_mean(box_loss, weights),
        "distill_depth": _weighted_mean(depth_loss, weights),
        "distill_dims": _weighted_mean(dim_loss, weights),
        "distill_angles": _weighted_mean(angle_loss, weights),
    }
    configured_weights = config.get("loss_weights", {})
    overall_weight = float(config.get("overall_weight", 0.25))
    total = zero
    for name, value in losses.items():
        total = total + value * float(configured_weights.get(name.removeprefix("distill_"), 1.0))
    losses["distill_total"] = total * overall_weight
    losses["distill_pairs"] = torch.as_tensor(float(len(aligned)), device=device)
    return losses
