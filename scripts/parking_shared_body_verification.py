"""Verify the Mini warm-start body without requiring cross-CPU fusion bytes.

The source checkpoint and unfused FP32 tensors remain exact identities. Only
Conv/BN fusion arithmetic may differ by tightly bounded FP32 rounding. The
runtime's actual fused digest still binds the feature cache and frozen-state
checks; no reference tensor is copied into the model and no weights are updated.
"""
from __future__ import annotations

import json

from scripts import parking_edge_common as common
from scripts.screen_parking_shared_model import state_digest

CHECKPOINT_SHA256 = "6fc044215d4804b5a9df4616dd19d56fe43343ac67300fb8abd67fb99b9619dd"
UNFUSED_BODY_SHA256 = "7187475b94de186ecc5f7e15817e96d81f696228a9ac5fa28f5aaf46d8ff66fc"
PILOT_FUSED_BODY_SHA256 = "76025834b6d2f6ec18efc91128e9fcb2035cfddcc2665de19a2910be669e7ab8"
FUSION_ATOL = 1e-6
FUSION_RTOL = 1e-6


def compare_fused_tensors(body, reference, original):
    """Compare all tensors to FP64 fusion; untouched values must match exactly."""
    import torch
    actual, expected = body.state_dict(), reference.state_dict()
    if actual.keys() != expected.keys():
        raise ValueError("Fused backbone tensor inventory differs from the pinned source")
    values = exact = numerical = 0
    maximum = fraction = 0.
    for name, value in actual.items():
        value = value.detach().cpu(); target = expected[name].detach().cpu()
        if value.shape != target.shape:
            raise ValueError(f"Fused backbone shape differs: {name}")
        values += value.numel()
        if value.is_floating_point():
            if value.dtype != torch.float32 or target.dtype != torch.float64:
                raise ValueError(f"Expected FP32 body and FP64 fusion reference: {name}")
            if not torch.isfinite(value).all() or not torch.isfinite(target).all():
                raise ValueError(f"Non-finite frozen backbone tensor: {name}")
            before = original.get(name)
            untouched = before is not None and torch.equal(before.double(), target)
            if untouched:
                if not torch.equal(value, before):
                    raise ValueError(f"Unchanged source tensor differs: {name}")
                exact += 1
            else:
                delta = (value.double() - target).abs()
                bound = FUSION_ATOL + FUSION_RTOL * target.abs()
                if (delta > bound).any():
                    raise ValueError(f"Fused backbone exceeds FP32 rounding bounds: {name}; "
                                     f"max_abs_delta={float(delta.max()):.9g}")
                maximum = max(maximum, float(delta.max()))
                fraction = max(fraction, float((delta / bound).max()))
                numerical += 1
        else:
            if value.dtype != target.dtype or not torch.equal(value, target):
                raise ValueError(f"Non-floating source tensor differs: {name}")
            exact += 1
    if not values or not numerical:
        raise ValueError("Expected a non-empty Conv/BN-fused body")
    return {"passed": True, "tensor_values_compared": values,
            "exact_source_tensors": exact, "numerically_checked_fused_tensors": numerical,
            "fusion_atol": FUSION_ATOL, "fusion_rtol": FUSION_RTOL,
            "maximum_fusion_abs_delta": maximum, "maximum_tolerance_fraction": fraction}


def fusion_reference(checkpoint):
    from ultralytics import YOLO
    native = YOLO(str(checkpoint), task="pose").model.float().cpu().eval()
    if state_digest(native.model) != UNFUSED_BODY_SHA256:
        raise ValueError("Unfused Urban tensors differ from the reviewed Mini source")
    original = {k: v.detach().cpu().clone() for k, v in native.model.state_dict().items()}
    # Recalculate fusion in higher precision; do not mutate the training model.
    native.double().fuse(verbose=False)
    return native.model, original


def verify_warmstart_body(model, model_id, config, weights):
    architecture = json.loads((common.ROOT / config["architecture_config"]).read_text())
    spec = architecture["urban_checkpoint"]
    if (config["initial_body_sha256"] != PILOT_FUSED_BODY_SHA256
            or spec["sha256"] != CHECKPOINT_SHA256
            or model_id["source_checkpoint_sha256"] != CHECKPOINT_SHA256
            or model_id["architecture"] != architecture):
        raise ValueError("Warm-start source identity differs from the reviewed Mini pilot")
    checkpoint = weights / spec["file"]
    if checkpoint.is_symlink() or common.file_sha(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Pinned Urban checkpoint bytes changed before fusion verification")
    before = state_digest(model.body)
    reference, original = fusion_reference(checkpoint)
    check = compare_fused_tensors(model.body, reference, original)
    if state_digest(model.body) != before:
        raise ValueError("Backbone verification changed model weights")
    return {"policy": "exact_source_and_unfused_tensors_plus_fp64_fusion_check_v1",
            "source_checkpoint_sha256": CHECKPOINT_SHA256,
            "unfused_body_sha256": UNFUSED_BODY_SHA256,
            "reviewed_pilot_fused_body_sha256": PILOT_FUSED_BODY_SHA256,
            "runtime_fused_body_sha256": before,
            "bitwise_equal_to_pilot_fusion": before == PILOT_FUSED_BODY_SHA256,
            "weights_changed": False, **check}
