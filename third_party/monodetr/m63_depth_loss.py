"""Vehicle-only, metre-L1 distillation through GT identity, not query index."""
import torch
COMPONENTS = ("depth", "dimensions", "center", "angle")

def geometry_distillation(outputs, targets, assignments, approved, enabled, weight=0.25):
    if list(enabled) != ["depth"] or weight != 0.25:
        raise ValueError("M63 freezes depth-only KD at weight 0.25")
    depth = outputs["pred_depth"]
    zero = depth[..., 0].sum() * 0.0
    terms = []
    for batch, (queries, gt_ids) in enumerate(assignments):
        a = approved[batch]
        masks = torch.as_tensor(a["masks"], device=depth.device, dtype=torch.bool)
        labels = targets[batch]["labels"].to(depth.device)
        if (masks.shape != (len(labels), 4) or masks[:, 1:].any()
                or masks[labels != 1].any()):
            raise ValueError("Only Vehicle depth may receive teacher supervision")
        q = queries.to(device=depth.device, dtype=torch.long)
        g = gt_ids.to(device=depth.device, dtype=torch.long)
        selected = masks[g, 0]
        q, g = q[selected], g[selected]
        if not len(q):
            continue
        teacher = torch.as_tensor(a["pred_depth"], device=depth.device).float()[g, 0].detach()
        if not torch.isfinite(teacher).all() or (teacher <= 0).any():
            raise ValueError("Invalid frozen teacher depth")
        terms.append((depth[batch, q, 0].float() - teacher).abs())
    loss = torch.cat(terms).mean() if terms else zero
    counts = dict.fromkeys(COMPONENTS, 0)
    counts["depth"] = sum(t.numel() for t in terms)
    losses = {c: loss if c == "depth" else zero for c in COMPONENTS}
    losses["total"] = weight * loss
    return losses, counts
