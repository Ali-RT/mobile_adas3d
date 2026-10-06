"""Vehicle-only R0 depth-head feature transfer without an inference-graph change."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from third_party.monodetr.m65_preservation import paired_iou

RULES = dict(native_vehicle_id=1, score=0.30, iou_2d=0.50,
             minimum_depth=2.0, maximum_depth=60.0,
             teacher_relative_error=0.15, anchor_minimum_relative_error=0.02,
             teacher_error_ratio=0.90, feature_width=256)


class DepthFeatureTap:
    """Capture the activation immediately before the last depth-head linear.

    This feature is downstream of trainable depth-head parameters. Capturing
    the frozen decoder output instead would give the KD term no student gradient.
    Hooks live only in the training process and are removed for inference.
    """

    def __init__(self, model):
        self.value = None
        self.calls = 0
        head = model.depth_embed[-1]
        if len(head.layers) != 2 or head.layers[-1].in_features != RULES["feature_width"]:
            raise RuntimeError("Unexpected native depth-head layout")
        self.handle = head.layers[-1].register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        if self.value is not None:
            raise RuntimeError("Depth feature tap called twice without a reset")
        self.value = args[0]
        self.calls += 1

    def reset(self):
        self.value = None
        self.calls = 0

    def take(self):
        if self.calls != 1 or self.value is None:
            raise RuntimeError("Missing or ambiguous depth-head feature")
        value = self.value
        self.reset()
        return value

    def close(self):
        self.handle.remove()
        self.reset()


def configure_trainable(model):
    """Only native depth MLPs learn; shared features and other heads stay fixed."""
    model.requires_grad_(False)
    model.depth_embed.requires_grad_(True)
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if not names or any(not name.startswith("depth_embed.") for name in names):
        raise RuntimeError("Unexpected trainable parameter scope")
    return names


def feature_kd(outputs, teacher, anchor, features, teacher_features, targets, matcher):
    """Independently Hungarian-match each model to transformed GT identity.

    Teach only Vehicle objects where R0 depth is demonstrably more accurate
    than frozen original A2 on this training view. No Pedestrian/background KD,
    no validation labels, and no direct external depth/logit copying.
    Direct cosine alignment assumes the shared 256-channel head coordinates are
    useful for transfer; the held-out pilot tests that assumption.
    """
    for values in (outputs, teacher, anchor):
        for key in ("pred_logits", "pred_boxes", "pred_depth"):
            if not bool(torch.isfinite(values[key]).all()):
                raise RuntimeError(f"Non-finite KD output: {key}")
    for values, feature in ((outputs, features), (teacher, teacher_features)):
        if (feature.shape[:2] != values["pred_logits"].shape[:2]
                or feature.ndim != 3 or feature.shape[-1] != RULES["feature_width"]
                or not bool(torch.isfinite(feature).all())):
            raise RuntimeError("Invalid depth-head feature shape/value")
    zero = features.sum() * 0.0
    matches = [matcher(values, targets, group_num=1) for values in (outputs, teacher, anchor)]
    rows = []
    counts = dict(Vehicle=0, Pedestrian=0, vehicle_near=0, vehicle_far=0)
    for batch, target in enumerate(targets):
        maps = [{int(gt): int(query) for query, gt in zip(*items[batch])} for items in matches]
        for gt in sorted(set(maps[0]) & set(maps[1]) & set(maps[2])):
            if int(target["labels"][gt]) != RULES["native_vehicle_id"]:
                continue
            s, t, a = (mapping[gt] for mapping in maps)
            prob = teacher["pred_logits"][batch, t].detach().sigmoid()
            overlap = paired_iou(teacher["pred_boxes"][batch, t:t + 1], target["boxes_3d"][gt:gt + 1])[0]
            z = target["depth"][gt].reshape(())
            tz, az = teacher["pred_depth"][batch, t, 0].detach(), anchor["pred_depth"][batch, a, 0].detach()
            if not bool((z >= RULES["minimum_depth"]) & (z < RULES["maximum_depth"]) & (tz > 0) & (az > 0)):
                continue
            te, ae = (tz - z).abs() / z, (az - z).abs() / z
            good = ((prob.argmax() == RULES["native_vehicle_id"]) & (prob[1] >= RULES["score"])
                    & (overlap >= RULES["iou_2d"]) & (te <= RULES["teacher_relative_error"])
                    & (ae >= RULES["anchor_minimum_relative_error"])
                    & (te <= RULES["teacher_error_ratio"] * ae))
            if not bool(good):
                continue
            sf, tf = features[batch, s], teacher_features[batch, t].detach()
            if float(sf.detach().norm()) <= 1e-8 or float(tf.norm()) <= 1e-8:
                raise RuntimeError("Zero-norm eligible head feature")
            rows.append((1.0 - F.cosine_similarity(sf[None], tf[None], dim=-1))[0].clamp(min=0))
            counts["Vehicle"] += 1
            counts["vehicle_near" if float(z) < 40 else "vehicle_far"] += 1
    return torch.stack(rows).mean() if rows else zero, counts
