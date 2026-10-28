"""Small camera-conditioned residual for the unchanged shared depth decoder.

The final convolution starts at zero: both experimental arms initially return
the reviewed epoch-7 prediction exactly. This is a hypothesis test, not a
qualified distance model or an ultrasonic measurement.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class CameraAwareDepthDecoder(nn.Module):
    """P3 appearance + pinhole rays/focal length -> bounded depth correction.

    camera is Bx4 [fx, fy, cx, cy] in letterboxed input-pixel coordinates.
    Both arms have exactly this architecture; only the camera input differs.
    No BatchNorm is introduced by the residual adapter.
    """

    def __init__(self, base, input_hw=(384, 640), channels=64, width=16):
        super().__init__()
        self.base = base
        self.input_hw = tuple(input_hw)
        self.appearance = nn.Conv2d(channels, width, 1)
        self.fusion = nn.Conv2d(width + 4, width, 1)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.residual = nn.Conv2d(width, 1, 1)
        nn.init.zeros_(self.residual.weight)
        nn.init.zeros_(self.residual.bias)

    def camera_fields(self, camera, spatial_hw):
        h, w = spatial_hw
        ih, iw = self.input_hw
        y = (torch.arange(h, dtype=camera.dtype, device=camera.device) + .5) * (ih / h) - .5
        x = (torch.arange(w, dtype=camera.dtype, device=camera.device) + .5) * (iw / w) - .5
        fx, fy, cx, cy = [v[:, None, None] for v in camera.unbind(1)]
        rx = ((x[None, None, :] - cx) / fx).expand(-1, h, -1)
        ry = ((y[None, :, None] - cy) / fy).expand(-1, -1, w)
        lx = torch.log(fx / iw).expand(-1, h, w)
        ly = torch.log(fy / ih).expand(-1, h, w)
        return torch.stack([rx, ry, lx, ly], dim=1)

    def forward(self, features, camera):
        base = self.base(features)
        appearance = F.silu(self.appearance(features[0]))
        camera = camera.to(dtype=appearance.dtype, device=appearance.device)
        fields = self.camera_fields(camera, appearance.shape[-2:])
        x = F.silu(self.fusion(torch.cat([appearance, fields], dim=1)))
        raw = self.residual(F.silu(self.spatial(x)))
        delta = F.interpolate(math.log(2.) * torch.tanh(raw), size=base.shape[-2:], mode="nearest")
        # Multiplication by exp(0)==1 preserves the base bit-for-bit at step 0.
        # No log/exp round-trip or new clamp is applied to the base prediction.
        return base * torch.exp(delta)
