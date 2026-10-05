"""Object-matched original-A2 preservation; not external-teacher KD."""
from __future__ import annotations

import torch
from torch.nn import functional as F

RULES = dict(score=0.30, iou_2d=0.50, depth_relative_error=0.15,
             dimension_relative_error=0.25)
WEIGHTS = dict(classification=2.0, box=5.0, depth=1.0,
               dimensions=1.0, angle=1.0)


def xyxy(boxes):
    """Native boxes are projected center plus left/right/top/bottom distances."""
    return torch.stack((boxes[..., 0] - boxes[..., 2],
                        boxes[..., 1] - boxes[..., 4],
                        boxes[..., 0] + boxes[..., 3],
                        boxes[..., 1] + boxes[..., 5]), dim=-1)


def paired_iou(a, b):
    a, b = xyxy(a), xyxy(b)
    intersection = (torch.minimum(a[:, 2:], b[:, 2:])
                    - torch.maximum(a[:, :2], b[:, :2])).clamp(min=0).prod(-1)
    area_a = (a[:, 2:] - a[:, :2]).clamp(min=0).prod(-1)
    area_b = (b[:, 2:] - b[:, :2]).clamp(min=0).prod(-1)
    return intersection / (area_a + area_b - intersection).clamp(min=1e-8)


def bernoulli_kl(logits, reference_logits):
    reference = reference_logits.detach().sigmoid().clamp(1e-6, 1 - 1e-6)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, reference, reduction="none")
    entropy = -(reference * reference.log() + (1 - reference) * (1 - reference).log())
    return (cross_entropy - entropy).clamp(min=0).mean(-1)


def preservation(outputs, reference, targets, matcher):
    """Match each model independently to the same transformed GT object.

    Both forwards use the single inference group. Query index is not object
    identity. Normalize each component within each class, then average classes
    with eligible objects. Native GT losses for both classes remain unchanged.
    """
    for values in (outputs, reference):
        for key in ("pred_logits", "pred_boxes", "pred_depth", "pred_3d_dim", "pred_angle"):
            if not torch.isfinite(values[key]).all():
                raise RuntimeError(f"Non-finite preservation output: {key}")
    zero = outputs["pred_logits"].sum() * 0.0
    rows = {key: {0: [], 1: []} for key in WEIGHTS}
    counts = {key: {0: 0, 1: 0} for key in WEIGHTS}
    student_matches = matcher(outputs, targets, group_num=1)
    anchor_matches = matcher(reference, targets, group_num=1)
    for batch, target in enumerate(targets):
        si, sg = student_matches[batch]
        ai, ag = anchor_matches[batch]
        by_gt = {int(gt): int(query) for query, gt in zip(si, sg)}
        pairs = [(by_gt[int(gt)], int(query), int(gt)) for query, gt in zip(ai, ag)
                 if int(gt) in by_gt]
        if not pairs:
            continue
        device = outputs["pred_logits"].device
        s, a, g = (torch.tensor(v, dtype=torch.long, device=device) for v in zip(*pairs))
        labels = target["labels"][g].long()
        if not bool(((labels == 0) | (labels == 1)).all()):
            raise RuntimeError("M65 preserves only native Pedestrian/Car product IDs")
        anchor_logits = reference["pred_logits"][batch, a].detach()
        probabilities = anchor_logits.sigmoid()
        correct_class = probabilities.argmax(-1) == labels
        score = probabilities.gather(1, labels[:, None]).squeeze(1)
        iou = paired_iou(reference["pred_boxes"][batch, a], target["boxes_3d"][g])
        eligible = correct_class & (score >= RULES["score"]) & (iou >= RULES["iou_2d"])
        anchor_depth = reference["pred_depth"][batch, a, 0].detach()
        gt_depth = target["depth"][g].reshape(-1)
        depth_good = (anchor_depth > 0) & (gt_depth > 0)
        depth_good &= (anchor_depth - gt_depth).abs() / gt_depth.clamp(min=1e-3) <= RULES["depth_relative_error"]
        anchor_dim = reference["pred_3d_dim"][batch, a].detach()
        gt_dim = target["size_3d"][g]
        dim_good = (anchor_dim > 0).all(-1) & (gt_dim > 0).all(-1)
        dim_good &= ((anchor_dim - gt_dim).abs() / gt_dim.clamp(min=0.05)).mean(-1) <= RULES["dimension_relative_error"]
        values = dict(
            classification=bernoulli_kl(outputs["pred_logits"][batch, s], anchor_logits),
            box=F.smooth_l1_loss(outputs["pred_boxes"][batch, s],
                                 reference["pred_boxes"][batch, a].detach(), reduction="none").mean(-1),
            depth=F.smooth_l1_loss(outputs["pred_depth"][batch, s, 0] / anchor_depth.clamp(min=1),
                                   anchor_depth / anchor_depth.clamp(min=1), reduction="none"),
            dimensions=F.smooth_l1_loss(outputs["pred_3d_dim"][batch, s] / anchor_dim.clamp(min=0.05),
                                        anchor_dim / anchor_dim.clamp(min=0.05), reduction="none").mean(-1))
        # Preserve angle bins only when the anchor chooses the correct GT bin.
        aa = reference["pred_angle"][batch, a].detach()
        sa = outputs["pred_angle"][batch, s]
        bins = aa[:, :12].argmax(-1)
        angle_good = bins == target["heading_bin"][g].reshape(-1)
        angle_good &= (aa[:, 12:].gather(1, bins[:, None]).squeeze(1)
                       - target["heading_res"][g].reshape(-1)).abs() <= 0.15
        angle_prob = aa[:, :12].softmax(-1)
        values["angle"] = F.kl_div(sa[:, :12].log_softmax(-1), angle_prob,
                                    reduction="none").sum(-1).clamp(min=0)
        values["angle"] += F.smooth_l1_loss(sa[:, 12:].gather(1, bins[:, None]).squeeze(1),
                                             aa[:, 12:].gather(1, bins[:, None]).squeeze(1), reduction="none")
        masks = dict(classification=eligible, box=eligible, depth=eligible & depth_good,
                     dimensions=eligible & dim_good, angle=eligible & angle_good)
        for key in WEIGHTS:
            for class_id in (0, 1):
                mask = masks[key] & (labels == class_id)
                if bool(mask.any()):
                    rows[key][class_id].append(values[key][mask])
                    counts[key][class_id] += int(mask.sum())
    losses = {}
    for key in WEIGHTS:
        class_means = [torch.cat(values).mean() for values in rows[key].values() if values]
        losses[key] = torch.stack(class_means).mean() if class_means else zero
    losses["total"] = sum(losses[key] * weight for key, weight in WEIGHTS.items())
    return losses, counts
