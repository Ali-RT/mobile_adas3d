"""GT-only losses and calibration decode for the compact nine-head CNN.

No custom operators, teacher cache, CUDA compiler, or namespace patches.
Depth is solved from projected box keypoints + dimensions + orientation;
there is deliberately no new depth head in the phone-screened architecture.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from data.kitti_r40 import iou_3d

LOSS_WEIGHTS = {"hm": 1., "wh": 0.1, "reg": 1., "hps": 1., "dim": 1.,
                "rot": 1., "hm_hp": 1., "hp_offset": 1., "position": 0.2, "prob": 1.}


def gather(head, indices):
    return head.flatten(2).transpose(1, 2).gather(
        1, indices[..., None].expand(-1, -1, head.shape[1]))


def masked_l1(pred, target, mask):
    mask = mask[..., None].expand_as(pred)
    return ((pred - target).abs() * mask).sum() / mask.sum().clamp_min(1)


def focal(logits, target, valid):
    p = logits.sigmoid().clamp(1e-5, 1 - 1e-5)
    positive = target.eq(1)
    negative = target.lt(1) * valid
    # A positive always wins over an overlapping ignore region.
    loss = -(p.log() * (1-p).square() * positive
             + (1-p).log() * p.square() * (1-target).pow(4) * negative)
    return loss.sum() / positive.sum().clamp_min(1)


def rotation_loss(pred, bins, residuals, mask):
    count = mask.sum().clamp_min(1)
    loss = pred.sum() * 0
    for bin_id, offset in enumerate((0, 4)):
        classification = F.cross_entropy(pred[..., offset:offset+2].transpose(1, 2),
                                         bins[..., bin_id], reduction="none")
        loss = loss + (classification * mask).sum() / count
        active = mask * bins[..., bin_id]
        expected = torch.stack([residuals[..., bin_id].sin(), residuals[..., bin_id].cos()], -1)
        error = F.smooth_l1_loss(pred[..., offset+2:offset+4], expected, reduction="none").sum(-1)
        loss = loss + (error * active).sum() / active.sum().clamp_min(1)
    return loss


def decode_alpha(rotation):
    # atan2 is quadrant-safe; raw sine/cosine heads are not probabilities.
    alpha1 = torch.atan2(rotation[..., 2], rotation[..., 3]) - math.pi / 2
    alpha2 = torch.atan2(rotation[..., 6], rotation[..., 7]) + math.pi / 2
    # Compare in-bin probabilities rather than unnormalized logits.
    first = rotation[..., :2].softmax(-1)[..., 1] > rotation[..., 4:6].softmax(-1)[..., 1]
    return torch.where(first, alpha1, alpha2)


def offsets(dim, yaw):
    signs = dim.new_tensor([[1, 1, 1], [1, 1, -1], [-1, 1, -1], [-1, 1, 1],
                            [1, -1, 1], [1, -1, -1], [-1, -1, -1], [-1, -1, 1], [0, 0, 0]])
    xyz = signs * dim[..., None, [2, 0, 1]] / 2
    x, y, z = xyz.unbind(-1)
    c, s = yaw[..., None].cos(), yaw[..., None].sin()
    return torch.stack([c*x + s*z, y, -s*x + c*z], -1)


def solve_center(keypoints, dim, yaw, projection):
    """Full P2 least-squares solve; manual 3x3 inverse avoids a CUDA solver.

    Tiny Tikhonov regularization keeps collapsed predicted keypoints finite.
    The output is geometric center, not KITTI bottom center. Negative depth
    is not silently made valid: training and evaluation track invalid boxes.
    """
    p = projection[:, None, None, :, :]
    u, v = keypoints.unbind(-1)
    ax = (p[..., 0, :3] - u[..., None] * p[..., 2, :3]) / p[..., 0, 0:1]
    ay = (p[..., 1, :3] - v[..., None] * p[..., 2, :3]) / p[..., 1, 1:2]
    local = offsets(dim, yaw)
    bx = -(ax * local).sum(-1) - (p[..., 0, 3] - u*p[..., 2, 3]) / p[..., 0, 0]
    by = -(ay * local).sum(-1) - (p[..., 1, 3] - v*p[..., 2, 3]) / p[..., 1, 1]
    a = torch.stack([ax, ay], -2).flatten(-3, -2)
    b = torch.stack([bx, by], -1).flatten(-2)
    normal = a.transpose(-1, -2) @ a
    normal = normal + torch.eye(3, device=a.device, dtype=a.dtype) * 1e-8
    rhs = (a.transpose(-1, -2) @ b[..., None]).squeeze(-1)
    r0, r1, r2 = normal.unbind(-2)
    c0, c1, c2 = torch.cross(r1, r2, dim=-1), torch.cross(r2, r0, dim=-1), torch.cross(r0, r1, dim=-1)
    determinant = (r0*c0).sum(-1).clamp_min(1e-10)
    return (rhs[..., 0:1]*c0 + rhs[..., 1:2]*c1 + rhs[..., 2:3]*c2) / determinant[..., None]


def image_keypoints(hps, indices, inverse, output_width):
    centers = torch.stack([indices % output_width, indices // output_width], -1)
    grid = hps.reshape(*hps.shape[:-1], 9, 2) + centers[..., None, :]
    return torch.einsum("bmkc,bac->bmka", grid, inverse[..., :2]) + inverse[:, None, None, :, 2]


def geometry(hps, indices, dim, rotation, projection, inverse, output_width):
    keypoints = image_keypoints(hps, indices, inverse, output_width)
    alpha = decode_alpha(rotation)
    yaw = alpha + torch.atan2(keypoints[..., 8, 0] - projection[:, None, 0, 2],
                              projection[:, None, 0, 0])
    # Refinement accounts for camera-2 translation in P2 and KITTI's alpha
    # (bearing about the reference rectified camera, not image center).
    for _ in range(3):
        center = solve_center(keypoints, dim, yaw, projection)
        yaw = alpha + torch.atan2(center[..., 0], center[..., 2])
    center = solve_center(keypoints, dim, yaw, projection)
    yaw = torch.atan2(yaw.sin(), yaw.cos())
    bottom = center + torch.stack([torch.zeros_like(dim[..., 0]), dim[..., 0]/2,
                                   torch.zeros_like(dim[..., 0])], -1)
    return bottom, yaw


def quality_targets(location, dim, yaw, target, mask):
    import numpy as np

    arrays = [x.detach().cpu().numpy() for x in (location, dim, yaw, target["location"],
                                                target["dim"], target["yaw"], mask)]
    loc, dims, angle, gt_loc, gt_dim, gt_angle, valid = arrays
    scores = np.zeros(valid.shape, np.float32)
    for batch, slot in zip(*valid.nonzero()):
        if (dims[batch, slot] <= 0).any() or loc[batch, slot, 2] <= 0:
            continue
        box = dict(location_3d=loc[batch, slot], dimensions_3d=dims[batch, slot], rotation_y=angle[batch, slot])
        gt = dict(location_3d=gt_loc[batch, slot], dimensions_3d=gt_dim[batch, slot], rotation_y=gt_angle[batch, slot])
        scores[batch, slot] = max(0., min(1., iou_3d(box, gt)))
    return torch.from_numpy(scores).to(mask.device, dtype=mask.dtype)


class CompactKittiLoss(torch.nn.Module):
    def forward(self, output, batch, geometry_scale=1.0):
        t = batch["targets"]
        mask, gm = t["mask"], t["geom_mask"]
        values = {name: gather(output[name], t["indices"])
                  for name in ("wh", "reg", "hps", "dim", "rot", "prob")}
        losses = {"hm": focal(output["hm"], t["hm"], t["hm_valid"]),
                  "hm_hp": focal(output["hm_hp"], t["hm_hp"], t["kp_valid"]),
                  "wh": masked_l1(values["wh"], t["wh"], mask),
                  "reg": masked_l1(values["reg"], t["reg"], mask),
                  "hps": masked_l1(values["hps"], t["hps"], gm),
                  "dim": masked_l1(values["dim"], t["dim"], gm),
                  "rot": rotation_loss(values["rot"], t["rotbin"], t["rotres"], gm),
                  "hp_offset": masked_l1(gather(output["hp_offset"], t["hp_indices"]),
                                          t["hp_offset"], t["hp_mask"])}
        # Use GT yaw for the differentiable position term: noisy bin decisions
        # must not destabilize the keypoint/dimension gradients at startup.
        keypoints = image_keypoints(values["hps"], t["indices"], batch["inverse_affine"], output["hm"].shape[-1])
        center = solve_center(keypoints, values["dim"], t["yaw"], batch["projection"])
        gt_center = t["location"].clone()
        gt_center[..., 1] -= t["dim"][..., 0] / 2
        error = F.smooth_l1_loss(center / t["location"][..., 2:3].clamp_min(1),
                                 gt_center / t["location"][..., 2:3].clamp_min(1), reduction="none").mean(-1)
        losses["position"] = (error * gm).sum() / gm.sum().clamp_min(1)
        with torch.no_grad():
            location, yaw = geometry(values["hps"], t["indices"], values["dim"], values["rot"],
                                     batch["projection"], batch["inverse_affine"], output["hm"].shape[-1])
            quality = quality_targets(location, values["dim"], yaw, t, gm)
        loss_prob = F.binary_cross_entropy_with_logits(values["prob"].squeeze(-1), quality, reduction="none")
        losses["prob"] = (loss_prob * gm).sum() / gm.sum().clamp_min(1)
        total = sum(value * LOSS_WEIGHTS[name] * (geometry_scale if name in ("position", "prob") else 1)
                    for name, value in losses.items())
        return total, losses


@torch.no_grad()
def decode(output, projection, inverse, topk=50, threshold=0.001):
    """Deterministic center NMS + geometry decode, shared by validation/export.

    This first baseline uses regressed keypoints for the solve; hm_hp and
    hp_offset remain supervised auxiliaries, not a hidden decoder heuristic.
    Scores are center confidence * predicted 3D-IoU quality. Finite but invalid
    geometry remains in the prediction set, and is counted as invalid.
    """
    from data.compact_kitti import CLASSES
    hm = output["hm"].sigmoid()
    hm = hm * (hm == F.max_pool2d(hm, 3, stride=1, padding=1))
    batch, _, height, width = hm.shape
    scores, flat = hm.flatten(1).topk(min(topk, hm[0].numel()), dim=-1)
    classes, indices = flat // (height*width), flat % (height*width)
    values = {name: gather(output[name], indices) for name in ("reg", "wh", "hps", "dim", "rot", "prob")}
    location, yaw = geometry(values["hps"], indices, values["dim"], values["rot"], projection, inverse, width)
    centers = torch.stack([indices % width, indices // width], -1) + values["reg"]
    lo, hi = centers - values["wh"]/2, centers + values["wh"]/2
    points = torch.stack([lo, hi], -2)
    boxes = torch.einsum("bmkc,bac->bmka", points, inverse[..., :2]) + inverse[:, None, None, :, 2]
    scores = scores * values["prob"].squeeze(-1).sigmoid()
    result = []
    for b in range(batch):
        rows = []
        for j in range(scores.shape[1]):
            if scores[b, j] < threshold:
                continue
            box, dims, loc = boxes[b, j], values["dim"][b, j], location[b, j]
            if not all(torch.isfinite(v).all() for v in (box, dims, loc, yaw[b, j], scores[b, j])):
                raise RuntimeError("Non-finite decoded prediction; do not report partial accuracy")
            # Do not remove finite bad 3D boxes and artificially improve AP.
            invalid = bool((dims <= 0).any() or loc[2] <= 0 or (box[1] <= box[0]).any())
            rows.append(dict(class_name=CLASSES[int(classes[b, j])], score=float(scores[b, j]),
                             bbox_2d=box.flatten().cpu().tolist(), dimensions_3d=dims.cpu().tolist(),
                             location_3d=loc.cpu().tolist(), rotation_y=float(yaw[b, j]),
                             invalid_geometry=invalid))
        result.append(rows)
    return result
