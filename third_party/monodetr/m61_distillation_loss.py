"""Vehicle-only geometry KD through student Hungarian -> shared GT identity.

Teacher outputs are frozen, audited train-only targets, not same-index queries.
No teacher class-logit, 2D box, Pedestrian, or temperature distillation is used.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

COMPONENTS = ("depth", "dimensions", "center", "angle")


def camera_centers(boxes, depth, calibration, image_size):
    uv = boxes[:, :2] * image_size
    x = (uv[:, 0] * depth - calibration[:, 0, 2] * depth - calibration[:, 0, 3]) / calibration[:, 0, 0]
    y = (uv[:, 1] * depth - calibration[:, 1, 2] * depth - calibration[:, 1, 3]) / calibration[:, 1, 1]
    return torch.stack((x, y, depth), dim=-1)


def geometry_distillation(outputs, targets, assignments, approved, enabled, weight=0.25):
    zero = outputs["pred_depth"].sum() * 0.0
    terms = {c: [] for c in COMPONENTS}
    counts = {c: 0 for c in COMPONENTS}
    for batch, (queries, gt_ids) in enumerate(assignments):
        a = approved[batch]
        device = outputs["pred_depth"].device
        # Upstream Hungarian matching returns CPU indices even for CUDA outputs.
        queries = queries.to(device=device, dtype=torch.long)
        gt_ids = gt_ids.to(device=device, dtype=torch.long)
        masks = torch.as_tensor(a["masks"], device=device, dtype=torch.bool)
        labels = targets[batch]["labels"].long()
        if masks.shape != (len(labels), 4) or torch.any(masks[labels != 1]):
            raise ValueError("Only native class 1 (Vehicle/Car) may receive M61 teacher loss")
        for component_index, component in enumerate(COMPONENTS):
            if component not in enabled:
                continue
            selected = masks[gt_ids, component_index]
            q, g = queries[selected], gt_ids[selected]
            if not len(q):
                continue
            counts[component] += len(q)
            if component == "depth":
                teacher = torch.as_tensor(a["pred_depth"], device=device).float()[g, 0].detach()
                student = outputs["pred_depth"][batch, q, 0].float()
                value = (student - teacher).abs() / targets[batch]["depth"][g, 0].float().clamp(min=1)
            elif component == "dimensions":
                teacher = torch.as_tensor(a["pred_3d_dim"], device=device).float()[g].detach()
                student = outputs["pred_3d_dim"][batch, q].float()
                value = ((student - teacher).abs() / targets[batch]["size_3d"][g].float().clamp(min=0.25)).mean(-1)
            elif component == "center":
                calibration = targets[batch]["calibs"][g].float()
                image_size = torch.as_tensor(a["image_size"], device=device).float()
                teacher = camera_centers(
                    torch.as_tensor(a["pred_boxes"], device=device).float()[g].detach(),
                    torch.as_tensor(a["pred_depth"], device=device).float()[g, 0].detach(),
                    calibration, image_size)
                student = camera_centers(outputs["pred_boxes"][batch, q].float(),
                                         outputs["pred_depth"][batch, q, 0].float(),
                                         calibration, image_size)
                value = (student - teacher).abs().sum(-1) / targets[batch]["depth"][g, 0].float().clamp(min=1)
            else:
                teacher = torch.as_tensor(a["pred_angle"], device=device).float()[g].detach()
                student = outputs["pred_angle"][batch, q].float()
                bins = teacher[:, :12].argmax(-1)
                teacher_res = teacher[:, 12:].gather(1, bins[:, None]).squeeze(1)
                student_res = student[:, 12:].gather(1, bins[:, None]).squeeze(1)
                value = F.cross_entropy(student[:, :12], bins, reduction="none") + F.smooth_l1_loss(student_res, teacher_res, reduction="none")
            terms[component].append(value)
    losses = {c: torch.cat(v).mean() if v else zero for c, v in terms.items()}
    losses["total"] = weight * sum(losses.values())
    return losses, counts
