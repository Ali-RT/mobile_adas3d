"""One coefficient change to the frozen selective geometry teacher loss.

Teacher angle guidance is multiplied by .01, not removed. Ground-truth
losses, class-balanced means, matching, eligibility and temperature stay
in the frozen helpers. There is no renormalization of the teacher sum.
"""
from models.a2_lite_geometry_kd import geometry_kd_loss

COMPONENT_WEIGHTS = dict(dimensions=1., depth=1., angle=.01)


def balanced_geometry_kd_loss(student, teacher, target, mapping):
    _, original, counts = geometry_kd_loss(student, teacher, target, mapping)
    parts = {name: value * COMPONENT_WEIGHTS.get(name, 1.)
             for name, value in original.items()}
    return sum(parts[name] for name in COMPONENT_WEIGHTS), parts, counts
