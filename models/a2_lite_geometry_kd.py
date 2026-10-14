"""Geometry-only adapter; frozen all-head/GT losses remain unchanged.

Matching still uses class and box information. Only their *direct teacher
losses* are disabled. Geometry can still affect shared features/box geometry;
this does not freeze detection heads or promise recall preservation.
"""
from models.a2_lite_recovery_loss import kd_loss as all_head_kd_loss

ENABLED_COMPONENTS = ("dimensions", "depth", "angle")
DISABLED_COMPONENTS = ("classification", "boxes")


def geometry_kd_loss(student, teacher, target, mapping):
    _, original_parts, coverage = all_head_kd_loss(student, teacher, target, mapping)
    zero = student["pred_logits"].sum() * 0
    parts = {name: value if name in ENABLED_COMPONENTS else zero
             for name, value in original_parts.items()}
    counts = {name: values if name in ENABLED_COMPONENTS else {label: 0 for label in values}
              for name, values in coverage.items()}
    return sum(parts[name] for name in ENABLED_COMPONENTS), parts, counts
