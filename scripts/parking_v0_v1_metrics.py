"""Frozen sparse-depth scoring reference. No models, Torch or inference.

Conditional errors never hide invalid predictions: keep the complete frozen
GT denominator, publish coverage, and compare V0/V1 on a common valid mask.
"""
from collections import defaultdict
import math

from scripts import parking_near_field_geometry as geometry


def cell_names(protocol):
    return [f"{cls}:{geometry.band_name(lo, hi)}" for cls in protocol["classes"] for lo, hi in protocol["range_bands_m"]]


def make_masks(points, objects, protocol):
    import numpy as np
    masks = []
    for cls in protocol["classes"]:
        for lo, hi in protocol["range_bands_m"]:
            mask = np.zeros(len(points["z"]), bool)
            band = geometry.band_mask(points["z"], lo, hi)
            for index, obj in enumerate(objects):
                selected = band & (points["object_index"] == index)
                if obj["class_name"] == cls and int(selected.sum()) >= protocol["minimum_points_per_object_camera_cell"]:
                    mask |= selected
            masks.append(mask)
    return np.asarray(masks, dtype=bool)


def validate_gt(points, sample, protocol):
    import numpy as np
    n = len(points["z"])
    if (set(points) != {"xy", "z", "semantic", "object_index", "cell_masks"}
            or points["xy"].shape != (n, 2) or points["xy"].dtype != np.int32
            or points["z"].shape != (n,) or points["z"].dtype != np.float32 or points["semantic"].shape != (n,)
            or points["semantic"].dtype != np.uint8 or points["object_index"].shape != (n,)
            or points["object_index"].dtype != np.int32
            or points["cell_masks"].shape != (len(cell_names(protocol)), n) or points["cell_masks"].dtype != bool
            or not n or not np.isfinite(points["z"]).all() or not ((points["z"] > 0) & (points["z"] <= 6)).all()
            or not ((points["xy"][:, 0] >= 0) & (points["xy"][:, 0] < sample["width"])).all()
            or not ((points["xy"][:, 1] >= 0) & (points["xy"][:, 1] < sample["height"])).all()
            or not ((points["object_index"] >= 0) & (points["object_index"] < len(sample["objects"]))).all()):
        raise ValueError("Malformed frozen sparse GT arrays")
    if len(np.unique(points["xy"], axis=0)) != n:
        raise ValueError("GT contains duplicate raster pixels")
    for index, obj in enumerate(sample["objects"]):
        if obj["class_name"] not in protocol["classes"] or not (points["semantic"][points["object_index"] == index] == obj["semantic_index"]).all():
            raise ValueError("GT semantic class/owner differs")
    if not np.array_equal(points["cell_masks"], make_masks(points, sample["objects"], protocol)):
        raise ValueError("Frozen object/cell mask differs")


class Metrics:
    def __init__(self, protocol):
        self.protocol = protocol; self.cameras = set(); self.images_with_invalid = set()
        self.errors = {cell: [] for cell in cell_names(protocol)}
        self.counts = {cell: [0, 0] for cell in self.errors}
        self.groups = {cell: defaultdict(lambda: defaultdict(lambda: defaultdict(dict))) for cell in self.errors}
        self.references = {cell: {"annotations": set(), "tracks": set(), "logs": set()} for cell in self.errors}

    def add(self, sample, points, prediction):
        import numpy as np
        validate_gt(points, sample, self.protocol)
        prediction = np.asarray(prediction)
        if prediction.shape != points["z"].shape or prediction.dtype.kind != "f":
            raise ValueError("Prediction must be a floating-point value for every frozen GT point")
        camera = sample["camera_token"]
        if camera in self.cameras:
            raise ValueError("Repeated prediction image")
        self.cameras.add(camera)
        valid = np.isfinite(prediction) & (prediction > 0)
        if not valid.all():
            self.images_with_invalid.add(camera)
        absolute = np.abs(prediction.astype(np.float64) - points["z"].astype(np.float64))
        for cell, mask in zip(cell_names(self.protocol), points["cell_masks"]):
            chosen = mask & valid
            self.counts[cell][0] += int(mask.sum()); self.counts[cell][1] += int(chosen.sum())
            self.errors[cell].append(absolute[chosen])
            for index in np.unique(points["object_index"][mask]):
                obj = sample["objects"][int(index)]; log = sample["recording_log"]
                ref = self.references[cell]; ref["annotations"].add(obj["annotation_token"])
                ref["tracks"].add(obj["instance_token"]); ref["logs"].add(log)
                selected = chosen & (points["object_index"] == index)
                if selected.any():
                    self.groups[cell][log][obj["instance_token"]][obj["annotation_token"]][camera] = float(absolute[selected].mean())

    def result(self):
        import numpy as np
        cells = {}
        for cell in cell_names(self.protocol):
            errors = np.concatenate(self.errors[cell]) if self.errors[cell] else np.empty(0)
            log_means = []
            for tracks in self.groups[cell].values():
                track_means = []
                for annotations in tracks.values():
                    track_means.append(float(np.mean([np.mean(list(cameras.values())) for cameras in annotations.values()])))
                log_means.append(float(np.mean(track_means)))
            total, valid = self.counts[cell]
            predicted_tracks = {track for tracks in self.groups[cell].values() for track in tracks}
            predicted_annotations = {ann for tracks in self.groups[cell].values() for annotations in tracks.values() for ann in annotations}
            cells[cell] = {"reference_points": total, "valid_predicted_points": valid,
                "prediction_coverage": valid / total if total else None,
                "conditional_point_mae_m": float(errors.mean()) if len(errors) else None,
                "conditional_point_median_abs_error_m": float(np.median(errors)) if len(errors) else None,
                "conditional_point_p90_abs_error_m": float(np.quantile(errors, .9, method="linear")) if len(errors) else None,
                "conditional_log_track_observation_macro_mae_m": float(np.mean(log_means)) if log_means else None,
                "reference_annotation_observations": len(self.references[cell]["annotations"]),
                "reference_instance_tracks": len(self.references[cell]["tracks"]),
                "reference_recording_logs": len(self.references[cell]["logs"]),
                "predicted_annotation_observations": len(predicted_annotations),
                "predicted_instance_tracks": len(predicted_tracks),
                "predicted_recording_logs": len(log_means)}
        primary = [cells[k]["conditional_log_track_observation_macro_mae_m"] for k in self.protocol["primary_cells"]]
        score = float(np.mean(primary)) if all(v is not None and math.isfinite(v) for v in primary) else None
        return {"cells": cells, "conditional_equal_four_cell_mae_m": score,
                "images": len(self.cameras), "images_with_invalid_predictions": len(self.images_with_invalid),
                "accuracy_pass_fail_enabled": False, "promotion_authorized": False,
                "warning": "Conditional errors require coverage alongside them; missing cells are null, not zero."}


def paired_metrics(samples, gt_arrays, v0, v1, protocol):
    """No missing images; emit own-valid and exactly paired common-mask errors."""
    import numpy as np
    expected = {s["camera_token"] for s in samples}
    if len(expected) != len(samples) or any(set(values) != expected for values in (gt_arrays, v0, v1)):
        raise ValueError("V0/V1 must provide exactly the frozen image inventory")
    accumulators = [Metrics(protocol) for _ in range(4)]
    for sample in samples:
        token = sample["camera_token"]; points = gt_arrays[token]
        p0 = np.asarray(v0[token]); p1 = np.asarray(v1[token])
        if p0.shape != points["z"].shape or p1.shape != points["z"].shape:
            raise ValueError("V0/V1 point order/shape differs")
        both = np.isfinite(p0) & (p0 > 0) & np.isfinite(p1) & (p1 > 0)
        for accumulator, pred in zip(accumulators, (p0, p1, np.where(both, p0, np.nan), np.where(both, p1, np.nan))):
            accumulator.add(sample, points, pred)
    return dict(zip(("v0_own_valid", "v1_own_valid", "v0_common_valid", "v1_common_valid"), (m.result() for m in accumulators)))
