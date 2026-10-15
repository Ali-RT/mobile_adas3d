"""Training-only, box-focused encoder feature KD for A2 -> A2-Lite.

This is a small FGD-inspired foreground experiment, NOT a reproduction of
FGD/DETRDistill. It has no global/background/attention/logit KD or adapter.
Teacher and student retain the inherited 256-channel coordinate system.
Final encoder outputs are compared by semantic stage, not layer index.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from models.a2_lite_recovery_loss import corners

CHANNELS = 256
ROI_SIZE = 3
STUDENT_SHAPES = ((24, 80), (12, 40), (6, 20), (3, 10))
TEACHER_SHAPES = ((48, 160), (24, 80), (12, 40), (6, 20))


class EncoderFeatureTap:
    """Keep the autograd-connected final visual memory; never change outputs."""

    def __init__(self, model):
        self.value = None
        self.handle = model.depthaware_transformer.encoder.register_forward_hook(self._capture)

    def _capture(self, module, args, output):
        if self.value is not None:
            raise RuntimeError("Encoder feature tap must be consumed before the next forward")
        if not torch.is_tensor(output) or output.ndim != 3:
            raise RuntimeError("Expected batch/token/channel visual encoder memory")
        self.value = output

    def take(self):
        if self.value is None:
            raise RuntimeError("Visual encoder feature tap did not fire")
        value, self.value = self.value, None
        return value

    def close(self):
        self.handle.remove()
        self.value = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def object_features(memory, target, shapes):
    """Sample equal 3x3 normalized-box grids at all four feature levels.

    Original image FOV and normalized GT coordinates are shared. Sampling
    with align_corners=False handles different feature resolutions without
    resizing/cropping the labels or privileging large boxes. Box interiors
    are not segmentation masks and can contain occluders/background.
    """
    if (tuple(shapes) not in (STUDENT_SHAPES, TEACHER_SHAPES)
            or memory.shape != (1, sum(h * w for h, w in shapes), CHANNELS)
            or memory.dtype != torch.float32 or not torch.isfinite(memory).all()):
        raise ValueError("Invalid four-level FP32 visual memory")
    boxes = target["boxes_3d"]
    labels = target["labels"]
    if (boxes.shape != (len(labels), 6) or not torch.isfinite(boxes).all()
            or not ((labels == 0) | (labels == 1)).all()):
        raise ValueError("Invalid foreground GT boxes/classes")
    if len(boxes) == 0:
        return memory[:, :0].reshape(0, 1, CHANNELS, 1, 1).expand(0, 4, CHANNELS, ROI_SIZE, ROI_SIZE)
    bounds = corners(boxes.detach()).clamp(0, 1)
    if (bounds[:, 2:] <= bounds[:, :2]).any():
        raise ValueError("Foreground box has no visible area")
    fraction = (torch.arange(ROI_SIZE, device=memory.device, dtype=memory.dtype) + .5) / ROI_SIZE
    xs = bounds[:, 0, None] + (bounds[:, 2] - bounds[:, 0])[:, None] * fraction
    ys = bounds[:, 1, None] + (bounds[:, 3] - bounds[:, 1])[:, None] * fraction
    grid = torch.stack((xs[:, None, :].expand(-1, ROI_SIZE, -1),
                        ys[:, :, None].expand(-1, -1, ROI_SIZE)), dim=-1) * 2 - 1
    grid = grid.reshape(1, len(boxes) * ROI_SIZE, ROI_SIZE, 2)
    rows, offset = [], 0
    for height, width in shapes:
        count = height * width
        level = memory[:, offset:offset + count].transpose(1, 2).reshape(1, CHANNELS, height, width)
        sampled = F.grid_sample(level, grid, mode="bilinear", padding_mode="border", align_corners=False)
        rows.append(sampled.reshape(CHANNELS, len(boxes), ROI_SIZE, ROI_SIZE).permute(1, 0, 2, 3))
        offset += count
    return torch.stack(rows, dim=1)


def feature_kd_loss(student, teacher, labels, eligible):
    """Mean cosine distance: points -> levels -> objects -> present classes.

    Teacher qualification comes from correct class/confidence/2D GT overlap,
    not a claim that a teacher feature is ground truth. Both classes receive
    equal class shares when present; no direct teacher head loss is used.
    """
    expected = (len(labels), 4, CHANNELS, ROI_SIZE, ROI_SIZE)
    if (student.shape != expected or teacher.shape != expected or eligible.shape != labels.shape
            or eligible.dtype != torch.bool or labels.ndim != 1
            or not ((labels == 0) | (labels == 1)).all()
            or student.dtype != torch.float32 or teacher.dtype != torch.float32
            or not torch.isfinite(student).all() or not torch.isfinite(teacher).all()):
        raise ValueError("Invalid object-aligned feature KD inputs")
    zero = student.sum() * 0
    counts = {name: int((eligible & (labels == cls)).sum())
              for cls, name in ((0, "Pedestrian"), (1, "Vehicle"))}
    if not eligible.any():
        return zero, counts
    active_s, active_t = student[eligible], teacher.detach()[eligible]
    if (active_s.detach().norm(dim=2) <= 1e-8).any() or (active_t.norm(dim=2) <= 1e-8).any():
        raise ValueError("Zero-norm active foreground feature")
    values = (1 - F.cosine_similarity(active_s, active_t, dim=2, eps=1e-8)).clamp(0, 2).mean((1, 2, 3))
    classes = labels[eligible]
    means = [values[classes == cls].mean() for cls in (0, 1) if (classes == cls).any()]
    return torch.stack(means).mean(), counts
