"""Scale-normalized original-A2 preservation, including depth log uncertainty."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .m65_preservation import RULES as OLD_RULES, bernoulli_kl, paired_iou

RULES = dict(OLD_RULES)
WEIGHTS = dict(classification=2.0, box=5.0, depth=1.0,
               dimensions=1.0, angle=1.0, uncertainty=1.0)
# Fixed normalization scales, not accuracy tolerances or acceptance gates.
SCALES = dict(probability_kl=0.01, normalized_box=0.01,
              relative_depth=0.02, relative_dimensions=0.02,
              angle_kl=0.01, angle_residual_rad=0.05, log_uncertainty=0.10)


def preservation(outputs, reference, targets, matcher):
    """Associate queries through transformed GT, not their query indices.

    Reliable anchor masks and per-class averaging match M65. All anchor
    targets are detached. Uncertainty is preserved only for reliable depths.
    """
    for values in (outputs, reference):
        for key in ("pred_logits", "pred_boxes", "pred_depth", "pred_3d_dim", "pred_angle"):
            if not bool(torch.isfinite(values[key]).all()):
                raise RuntimeError(f"Non-finite preservation output: {key}")
        if values["pred_depth"].shape[-1] != 2:
            raise RuntimeError("Expected point depth and native depth log uncertainty")
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
            raise RuntimeError("Only native Pedestrian/Car product IDs may be preserved")
        anchor_logits = reference["pred_logits"][batch, a].detach()
        probabilities = anchor_logits.sigmoid()
        score = probabilities.gather(1, labels[:, None]).squeeze(1)
        anchor_boxes = reference["pred_boxes"][batch, a].detach()
        eligible = ((probabilities.argmax(-1) == labels) & (score >= RULES["score"])
                    & (paired_iou(anchor_boxes, target["boxes_3d"][g]) >= RULES["iou_2d"]))
        anchor_depth = reference["pred_depth"][batch, a, 0].detach()
        gt_depth = target["depth"][g].reshape(-1)
        depth_good = (anchor_depth > 0) & (gt_depth > 0)
        depth_good &= (anchor_depth - gt_depth).abs() / gt_depth.clamp(min=1e-3) <= RULES["depth_relative_error"]
        anchor_dim = reference["pred_3d_dim"][batch, a].detach()
        gt_dim = target["size_3d"][g]
        dim_good = (anchor_dim > 0).all(-1) & (gt_dim > 0).all(-1)
        dim_good &= ((anchor_dim - gt_dim).abs() / gt_dim.clamp(min=0.05)).mean(-1) <= RULES["dimension_relative_error"]
        values = dict(
            classification=bernoulli_kl(outputs["pred_logits"][batch, s], anchor_logits) / SCALES["probability_kl"],
            box=F.smooth_l1_loss((outputs["pred_boxes"][batch, s] - anchor_boxes) / SCALES["normalized_box"],
                                 torch.zeros_like(anchor_boxes), reduction="none").mean(-1),
            depth=F.smooth_l1_loss((outputs["pred_depth"][batch, s, 0] - anchor_depth)
                                   / (anchor_depth.clamp(min=1) * SCALES["relative_depth"]),
                                   torch.zeros_like(anchor_depth), reduction="none"),
            dimensions=F.smooth_l1_loss((outputs["pred_3d_dim"][batch, s] - anchor_dim)
                                        / (anchor_dim.clamp(min=0.05) * SCALES["relative_dimensions"]),
                                        torch.zeros_like(anchor_dim), reduction="none").mean(-1),
            uncertainty=F.smooth_l1_loss((outputs["pred_depth"][batch, s, 1]
                                          - reference["pred_depth"][batch, a, 1].detach())
                                         / SCALES["log_uncertainty"],
                                         torch.zeros_like(anchor_depth), reduction="none"))
        aa = reference["pred_angle"][batch, a].detach()
        sa = outputs["pred_angle"][batch, s]
        bins = aa[:, :12].argmax(-1)
        anchor_residual = aa[:, 12:].gather(1, bins[:, None]).squeeze(1)
        angle_good = bins == target["heading_bin"][g].reshape(-1)
        angle_good &= (anchor_residual - target["heading_res"][g].reshape(-1)).abs() <= 0.15
        values["angle"] = F.kl_div(sa[:, :12].log_softmax(-1), aa[:, :12].softmax(-1),
                                    reduction="none").sum(-1).clamp(min=0) / SCALES["angle_kl"]
        values["angle"] += F.smooth_l1_loss((sa[:, 12:].gather(1, bins[:, None]).squeeze(1)
                                              - anchor_residual) / SCALES["angle_residual_rad"],
                                             torch.zeros_like(anchor_residual), reduction="none")
        masks = dict(classification=eligible, box=eligible, depth=eligible & depth_good,
                     dimensions=eligible & dim_good, angle=eligible & angle_good,
                     uncertainty=eligible & depth_good)
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
