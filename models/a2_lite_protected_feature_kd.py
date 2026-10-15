"""Detached, per-step safeguards for the unchanged feature-KD objective.

The controller chooses one nonnegative scalar coefficient. It never edits
GT gradients or projects parameter updates. Bounds apply before global
clipping/AdamW and do not guarantee class AP or deployment performance.
"""
from __future__ import annotations

import math
import torch

TERMS = ("gt_all", "gt/Pedestrian", "gt/Vehicle", "kd")
CAP = .05
COSINE_TOLERANCE = 1e-6


def coordinates(model):
    selected = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        group = ("visual_encoder" if name.startswith("depthaware_transformer.encoder.") else
                 "late_backbone" if name.startswith("backbone.") else "other_trainable")
        selected.append((name, parameter, group))
    if not selected: raise RuntimeError("No trainable protected-KD coordinates")
    return tuple(zip(*selected))


def gradient_grams(groups, gradients):
    if set(gradients) != set(TERMS) or any(len(v) != len(groups) for v in gradients.values()):
        raise ValueError("All four gradient terms must use identical trainable coordinates")
    matrices = {}
    for index, group in enumerate(groups):
        rows = [gradients[k][index] for k in TERMS]; active = [v for v in rows if v is not None]
        if not active: continue
        if any(v.shape != active[0].shape or not torch.isfinite(v).all() for v in active):
            raise RuntimeError("Nonfinite or inconsistent protected-KD gradient coordinates")
        stacked = torch.stack([(v.detach() if v is not None else torch.zeros_like(active[0])).reshape(-1).double() for v in rows])
        product = stacked @ stacked.T
        for tag in ("all_trainable", group):
            if tag not in matrices: matrices[tag] = torch.zeros_like(product)
            matrices[tag] += product
    return {k: v.cpu().tolist() for k, v in matrices.items()}


def choose_coefficient(matrices, base_coefficient):
    """Reject any class/total conflict, then cap every measured group."""
    if not math.isfinite(base_coefficient) or base_coefficient < 0:
        raise ValueError("Invalid scheduled feature coefficient")
    if "all_trainable" not in matrices: raise ValueError("Missing complete trainable-coordinate guard")
    coefficient = base_coefficient; conflicts = []; checks = {}; active = False
    for group, matrix in matrices.items():
        if (len(matrix) != 4 or any(len(row) != 4 for row in matrix)
                or any(not math.isfinite(v) for row in matrix for v in row)
                or any(matrix[i][i] < 0 for i in range(4))):
            raise ValueError("Invalid protected-KD Gram matrix")
        norms = [math.sqrt(matrix[i][i]) for i in range(4)]
        kd_norm = norms[3]; active = active or kd_norm > 0
        cosines = {}
        for index, name in enumerate(TERMS[:3]):
            denominator = norms[index] * kd_norm
            cosine = max(-1., min(1., matrix[3][index] / denominator)) if denominator > 0 else None
            cosines[name] = cosine
            if cosine is not None and cosine < -COSINE_TOLERANCE:
                conflicts.append(group + ":" + name)
        if kd_norm > 0:
            coefficient = min(coefficient, CAP * norms[0] / kd_norm * (1 - 1e-7))
        checks[group] = dict(gt_norm=norms[0], teacher_raw_norm=kd_norm, cosines=cosines)
    if conflicts or not active: coefficient = 0.
    for value in checks.values():
        value["applied_teacher_to_gt_norm_ratio"] = (coefficient * value["teacher_raw_norm"] / value["gt_norm"]
                                                        if value["gt_norm"] > 0 else 0.)
        if value["applied_teacher_to_gt_norm_ratio"] > CAP + 1e-8:
            raise RuntimeError("Protected feature gradient cap violated")
    return dict(coefficient=coefficient, base_coefficient=base_coefficient,
                accepted=bool(coefficient > 0), rejected_for_conflict=bool(conflicts), conflicts=conflicts,
                scaled_for_cap=bool(not conflicts and active and coefficient < base_coefficient),
                no_teacher_gradient=not active, cap=CAP, checks=checks)
