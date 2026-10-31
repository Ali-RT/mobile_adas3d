"""Two-input inference wrapper for the unchanged trained shared models."""
from torch import nn


class TrainedParkingExport(nn.Module):
    """One shared feature pass; frozen Urban candidates + calibrated metric Z.

    No additional NMS, scaling, clamping, learned weights or geometry decoder.
    Camera is Bx4 [fx,fy,cx,cy] in the letterboxed image's pixel coordinates.
    """

    def __init__(self, native, decoder):
        super().__init__()
        self.native = native
        self.decoder = decoder
        self.eval().requires_grad_(False)

    def forward(self, image, camera):
        features = self.native.image_features(image)
        candidates = self.native.full_urban_candidates(features)
        candidates = candidates[:, self.native.candidate_channels]
        return candidates, self.decoder(features, camera)
