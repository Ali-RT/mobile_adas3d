"""Training-only A2 guidance; the compact inference graph is unchanged.

Teachers are matched to labelled objects, not to dense grid/query indices.
Each class/component must improve on the frozen student in the train-only
audit. Individual targets are also rejected when worse than that student.
Ground-truth supervision is always retained at its original weight.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F

from models.compact_kitti_loss import (decode_alpha, gather, geometry,
                                     masked_l1, rotation_loss)

COMPONENTS = ("hps", "dim", "rot", "depth")
KD_WEIGHTS = {"hps": 0.25, "dim": 0.25, "rot": 0.1, "depth": 0.2}


def wrapped_error(a, b):
    return abs(math.atan2(math.sin(a-b), math.cos(a-b)))


def box_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    area = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = np.maximum(0, a[2:]-a[:2]).prod() + np.maximum(0, b[2:]-b[:2]).prod()-area
    return float(area / union) if union > 0 else 0.


def match_teacher(objects, predictions, mapping, threshold=0.5):
    """Deterministic, class-aware, one-to-one IoU matching to TRAIN GT."""
    candidates = []
    for gi, obj in enumerate(objects):
        name = mapping.get(obj.class_name)
        if name is None:
            continue
        for pi, pred in enumerate(predictions):
            if pred["class_name"] != name or pred["invalid_geometry"]:
                continue
            overlap = box_iou(obj.bbox_2d, pred["bbox_2d"])
            if overlap >= threshold:
                candidates.append((-overlap, -pred["score"], gi, pi))
    used_gt, used_pred, result = set(), set(), {}
    for _, _, gi, pi in sorted(candidates):
        if gi not in used_gt and pi not in used_pred:
            used_gt.add(gi); used_pred.add(pi); result[gi] = predictions[pi]
    return result


def qualification(rows, min_count=16, retention=0.9):
    """Mean AND median train error improvement; no val-driven target selection."""
    report = {}
    for name in ("Vehicle", "Pedestrian"):
        report[name] = {}
        for component in COMPONENTS:
            pairs = [(r["student_errors"][component], r["teacher_errors"][component])
                     for r in rows if r["class_name"] == name]
            a = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
            count = len(a)
            finite = bool(np.isfinite(a).all())
            report[name][component] = dict(
                count=count, student_mean=float(a[:, 0].mean()) if count else None,
                teacher_mean=float(a[:, 1].mean()) if count else None,
                student_median=float(np.median(a[:, 0])) if count else None,
                teacher_median=float(np.median(a[:, 1])) if count else None,
                approved=bool(count >= min_count and finite
                              and a[:, 1].mean() < retention * a[:, 0].mean()
                              and np.median(a[:, 1]) < np.median(a[:, 0])))
    return report


def usable_components(row, approved):
    return {name: bool(approved[row["class_name"]][name]["approved"]
                       and np.isfinite(row["teacher_errors"][name])
                       and row["teacher_errors"][name] < 0.9 * row["student_errors"][name])
            for name in COMPONENTS}


def distillation_loss(output, batch):
    t, kd = batch["targets"], batch["kd"]
    values = {k: gather(output[k], t["indices"]) for k in ("hps", "dim", "rot")}
    losses = {"hps": masked_l1(values["hps"], kd["hps"], kd["hps_mask"]),
              "dim": masked_l1(values["dim"], kd["dim"], kd["dim_mask"]),
              "rot": rotation_loss(values["rot"], kd["rotbin"], kd["rotres"], kd["rot_mask"])}
    location, _ = geometry(values["hps"], t["indices"], values["dim"], values["rot"],
                           batch["projection"], batch["inverse_affine"], output["hm"].shape[-1])
    # Normalize depth errors by GT depth, as in the GT position objective.
    denominator = t["location"][..., 2].clamp_min(1)
    error = F.smooth_l1_loss(location[..., 2]/denominator,
                           kd["depth"]/denominator, reduction="none")
    losses["depth"] = (error * kd["depth_mask"]).sum() / kd["depth_mask"].sum().clamp_min(1)
    return sum(KD_WEIGHTS[k] * v for k, v in losses.items()), losses
