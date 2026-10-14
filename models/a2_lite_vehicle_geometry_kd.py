"""Vehicle-only geometry teaching without reallocating Pedestrian weight.

Compute the original per-component eligibility on all targets, then take
only the Vehicle mean and retain the original eligible-class denominator.
Ground-truth targets and student Hungarian matching still include both
classes. The frozen .01 teacher-angle multiplier is retained.
"""
from models.a2_lite_balanced_geometry_kd import balanced_geometry_kd_loss, COMPONENT_WEIGHTS

CLASS_WEIGHTS = dict(Pedestrian=0., Vehicle=1.)


def vehicle_geometry_kd_loss(student, teacher, target, mapping):
    _, _, original_counts = balanced_geometry_kd_loss(student, teacher, target, mapping)
    vehicle_mapping = mapping.clone()
    vehicle_mapping[target["labels"] != 1] = -1
    _, vehicle_parts, vehicle_counts = balanced_geometry_kd_loss(student, teacher, target, vehicle_mapping)
    parts = {}
    for name, value in vehicle_parts.items():
        denominator = sum(n > 0 for n in original_counts[name].values())
        parts[name] = value / max(denominator, 1) if name in COMPONENT_WEIGHTS else value
    return sum(parts[name] for name in COMPONENT_WEIGHTS), parts, vehicle_counts
