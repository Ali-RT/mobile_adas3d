"""Shared Urban image features plus a lightweight, initially untrained depth head.

Prototype phase: frozen pretrained Urban path (including BatchNorm statistics),
new trainable depth decoder. No metric-depth accuracy is implied by exp(log_z).
The deployed graph returns raw product candidates and quarter-resolution depth;
neither output is an ultrasonic range or a calibrated metric 3D cuboid.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

REVISION = "PARKING-SHARED-NANO-2026-10-27-r1"
COCO_CLASSES = (0, 2, 5, 7)  # person, car, bus, truck; product mapping outside graph
OUTPUT_NAMES = ("candidates", "depth_untrained")


def block(cin, cout, kernel=3, groups=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, padding=kernel//2, groups=groups, bias=False),
        nn.BatchNorm2d(cout), nn.SiLU(inplace=False))


def refine(channels):
    return nn.Sequential(block(channels, channels, groups=channels), block(channels, channels, kernel=1))


class ParkingDepthDecoder(nn.Module):
    """P3/P4/P5 -> stride-4 log-depth, separable fusion, no custom operators."""

    def __init__(self, channels=(64, 128, 256), width=64):
        super().__init__()
        if len(channels) != 3 or any(c <= 0 for c in channels) or width < 8 or width % 2:
            raise ValueError("Expected three feature channels and an even decoder width >=8")
        self.projections = nn.ModuleList(block(c, width, kernel=1) for c in channels)
        self.refinements = nn.ModuleList(refine(width) for _ in range(2))
        self.output = nn.Sequential(block(width, width//2), nn.Conv2d(width//2, 1, 1))
        # Small random weights retain an input-dependent runtime graph. The
        # nominal ~4m initialization is NOT a learned or calibrated distance.
        nn.init.normal_(self.output[-1].weight, std=.001)
        nn.init.constant_(self.output[-1].bias, 1.3862943611198906)

    def forward(self, features):
        p3, p4, p5 = [projection(x) for projection, x in zip(self.projections, features)]
        x = self.refinements[0](F.interpolate(p5, size=p4.shape[-2:], mode="nearest") + p4)
        x = self.refinements[1](F.interpolate(x, size=p3.shape[-2:], mode="nearest") + p3)
        x = F.interpolate(x, scale_factor=2., mode="nearest")
        return torch.exp(self.output(x).clamp(-4., 5.))


class SharedParkingNet(nn.Module):
    """Traverse Urban's backbone/neck once; feed both heads from the same tensors.

    The adapter deliberately depends on the pinned Ultralytics 8.4.61 Pose26
    interface. A mismatch fails closed instead of silently changing branches.
    Source Urban weights and BN statistics stay frozen in this first stage.
    """

    def __init__(self, urban, channels=(64, 128, 256), width=64):
        super().__init__()
        head = urban.model[-1]
        if (list(head.f) != [16, 19, 22] or head.nc != 80 or list(head.kpt_shape) != [8,3]
                or not head.end2end or not hasattr(head, "one2one")):
            raise ValueError("Expected pinned Urban Nano COCO-80/eight-corner one-to-one head")
        self.body = urban.model
        self.save = frozenset(urban.save)
        self.depth_decoder = ParkingDepthDecoder(channels, width)
        self.register_buffer("candidate_channels", torch.tensor([0,1,2,3] + [4+c for c in COCO_CLASSES]
            + list(range(84,108)), dtype=torch.long), persistent=False)
        for parameter in self.body.parameters():
            parameter.requires_grad_(False)
        head.export = True
        head.dynamic = False
        head.format = "onnx"
        self.train(False)

    def train(self, mode=True):
        super().train(mode)
        # Frozen weights alone do not protect BN running statistics.
        self.body.eval()
        return self

    def image_features(self, image):
        saved = []
        x = image
        for module in self.body[:-1]:
            if module.f != -1:
                x = saved[module.f] if isinstance(module.f, int) else [x if j == -1 else saved[j] for j in module.f]
            x = module(x)
            saved.append(x if module.i in self.save else None)
        return [saved[index] for index in self.body[-1].f]

    def full_urban_candidates(self, features):
        head = self.body[-1]
        # Keep the pretrained one-to-one branch. Avoid TopK ties in the graph
        # and preserve decoded boxes, all 80 scores and all eight corners.
        raw = head.forward_head(list(features), **head.one2one)
        return head._inference(raw)

    def forward(self, image):
        features = self.image_features(image)
        urban = self.full_urban_candidates(features)
        candidates = torch.index_select(urban, 1, self.candidate_channels)
        return candidates, self.depth_decoder(features)


def metric_depth_loss(prediction, truth, valid, weights=None, delta=.2):
    """Huber error in raw metres at explicitly valid pixels; no scale fitting.

    This loss is ready for a later GT depth-head warmup. The export screen only
    checks gradients on synthetic references and never takes an optimizer step.
    """
    if prediction.shape != truth.shape or valid.shape != truth.shape or valid.dtype != torch.bool:
        raise ValueError("Matched depth/reference arrays and a boolean valid mask are required")
    if delta <= 0 or not torch.isfinite(prediction).all() or not (prediction > 0).all():
        raise ValueError("Depth predictions must be positive and finite")
    if not torch.isfinite(truth[valid]).all() or not (truth[valid] > 0).all():
        raise ValueError("Valid reference pixels must contain positive finite metre depths")
    if not valid.any():
        return prediction.sum()*0.
    if weights is not None and weights.shape != truth.shape:
        raise ValueError("Explicit depth weights must have the reference shape")
    selected_weights = torch.ones_like(truth[valid]) if weights is None else weights[valid]
    if not torch.isfinite(selected_weights).all() or not (selected_weights > 0).all():
        raise ValueError("Explicit depth weights must be positive, finite and matched")
    errors = F.huber_loss(prediction[valid], truth[valid], reduction="none", delta=delta)
    return (errors*selected_weights).sum()/selected_weights.sum()
