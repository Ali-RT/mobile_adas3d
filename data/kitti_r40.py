from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Sequence

import numpy as np


DIFFICULTIES = ("easy", "moderate", "hard")
MIN_HEIGHT = {"easy": 40.0, "moderate": 25.0, "hard": 25.0}
MAX_OCCLUSION = {"easy": 0, "moderate": 1, "hard": 2}
MAX_TRUNCATION = {"easy": 0.15, "moderate": 0.30, "hard": 0.50}
IOU_THRESHOLDS = {"Car": 0.70, "Pedestrian": 0.50, "Cyclist": 0.50}
NEIGHBOR_CLASSES = {"Car": {"Van"}, "Pedestrian": {"Person_sitting"}, "Cyclist": set()}
PRODUCT_IOU_THRESHOLDS = {"Vehicle": 0.70, "Pedestrian": 0.50}
PRODUCT_NEIGHBOR_CLASSES = {"Vehicle": set(), "Pedestrian": set()}


@dataclass(frozen=True)
class EvalResult:
    class_name: str
    difficulty: str
    metric: str
    iou_threshold: float
    ap_r40: float
    num_valid_gt: int
    num_scored_predictions: int
    num_true_positives: int
    num_false_positives: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "class_name": self.class_name,
            "difficulty": self.difficulty,
            "metric": self.metric,
            "iou_threshold": self.iou_threshold,
            "ap_r40": self.ap_r40,
            "num_valid_gt": self.num_valid_gt,
            "num_scored_predictions": self.num_scored_predictions,
            "num_true_positives": self.num_true_positives,
            "num_false_positives": self.num_false_positives,
        }


def _signed_polygon_area(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0
    x = polygon[:, 0]
    y = polygon[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def polygon_area(polygon: np.ndarray) -> float:
    return abs(_signed_polygon_area(np.asarray(polygon, dtype=np.float64)))


def _as_ccw(polygon: np.ndarray) -> np.ndarray:
    polygon = np.asarray(polygon, dtype=np.float64)
    if _signed_polygon_area(polygon) < 0.0:
        return polygon[::-1].copy()
    return polygon


def _inside(point: np.ndarray, edge_start: np.ndarray, edge_end: np.ndarray) -> bool:
    edge = edge_end - edge_start
    relative = point - edge_start
    return float(edge[0] * relative[1] - edge[1] * relative[0]) >= -1e-9


def _line_intersection(
    segment_start: np.ndarray,
    segment_end: np.ndarray,
    edge_start: np.ndarray,
    edge_end: np.ndarray,
) -> np.ndarray:
    segment = segment_end - segment_start
    edge = edge_end - edge_start
    denominator = segment[0] * edge[1] - segment[1] * edge[0]
    if abs(float(denominator)) < 1e-12:
        return segment_end.copy()
    offset = edge_start - segment_start
    t = (offset[0] * edge[1] - offset[1] * edge[0]) / denominator
    return segment_start + t * segment


def convex_polygon_intersection(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    output = [point.copy() for point in _as_ccw(subject)]
    clip = _as_ccw(clip)

    for edge_index in range(len(clip)):
        if not output:
            break
        edge_start = clip[edge_index]
        edge_end = clip[(edge_index + 1) % len(clip)]
        input_points = output
        output = []
        previous = input_points[-1]

        for current in input_points:
            current_inside = _inside(current, edge_start, edge_end)
            previous_inside = _inside(previous, edge_start, edge_end)
            if current_inside:
                if not previous_inside:
                    output.append(
                        _line_intersection(previous, current, edge_start, edge_end)
                    )
                output.append(current)
            elif previous_inside:
                output.append(
                    _line_intersection(previous, current, edge_start, edge_end)
                )
            previous = current

    if not output:
        return np.empty((0, 2), dtype=np.float64)
    return np.asarray(output, dtype=np.float64)


def box_bev_corners(box: Mapping[str, Any]) -> np.ndarray:
    h, w, length = [float(v) for v in _dimensions(box)]
    del h
    x, _, z = [float(v) for v in box["location_3d"]]
    yaw = float(_yaw(box))

    local_x = np.asarray([length / 2, -length / 2, -length / 2, length / 2])
    local_z = np.asarray([w / 2, w / 2, -w / 2, -w / 2])
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotated_x = cos_yaw * local_x + sin_yaw * local_z + x
    rotated_z = -sin_yaw * local_x + cos_yaw * local_z + z
    return _as_ccw(np.stack([rotated_x, rotated_z], axis=1))


def _dimensions(box: Mapping[str, Any]) -> Sequence[float]:
    if "dimensions_3d_hwl" in box:
        return box["dimensions_3d_hwl"]
    return box["dimensions_3d"]


def _yaw(box: Mapping[str, Any]) -> float:
    if "yaw" in box:
        return float(box["yaw"])
    return float(box["rotation_y"])


def bev_iou(box_a: Mapping[str, Any], box_b: Mapping[str, Any]) -> float:
    corners_a = box_bev_corners(box_a)
    corners_b = box_bev_corners(box_b)
    area_a = polygon_area(corners_a)
    area_b = polygon_area(corners_b)
    intersection = polygon_area(convex_polygon_intersection(corners_a, corners_b))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def iou_3d(box_a: Mapping[str, Any], box_b: Mapping[str, Any]) -> float:
    dims_a = [float(v) for v in _dimensions(box_a)]
    dims_b = [float(v) for v in _dimensions(box_b)]
    area_intersection = polygon_area(
        convex_polygon_intersection(box_bev_corners(box_a), box_bev_corners(box_b))
    )

    y_a = float(box_a["location_3d"][1])
    y_b = float(box_b["location_3d"][1])
    top_a, bottom_a = y_a - dims_a[0], y_a
    top_b, bottom_b = y_b - dims_b[0], y_b
    height_intersection = max(0.0, min(bottom_a, bottom_b) - max(top_a, top_b))
    intersection = area_intersection * height_intersection
    volume_a = dims_a[0] * dims_a[1] * dims_a[2]
    volume_b = dims_b[0] * dims_b[1] * dims_b[2]
    union = volume_a + volume_b - intersection
    return intersection / union if union > 0.0 else 0.0


def _bbox_height(box: Mapping[str, Any]) -> float:
    bbox = box["bbox_2d"]
    return max(0.0, float(bbox[3]) - float(bbox[1]))


def _gt_status(
    gt: Mapping[str, Any],
    class_name: str,
    difficulty: str,
    neighbor_classes: Mapping[str, set[str]],
) -> int:
    """Return 0 for valid, 1 for ignored, and -1 for irrelevant GT."""
    gt_class = str(gt["class_name"])
    if gt_class in neighbor_classes.get(class_name, set()):
        return 1
    if gt_class != class_name:
        return -1

    difficult = (
        _bbox_height(gt) < MIN_HEIGHT[difficulty]
        or int(gt.get("occluded", 0)) > MAX_OCCLUSION[difficulty]
        or float(gt.get("truncated", 0.0)) > MAX_TRUNCATION[difficulty]
    )
    return 1 if difficult else 0


def _prediction_ignored(prediction: Mapping[str, Any], difficulty: str) -> bool:
    return _bbox_height(prediction) < MIN_HEIGHT[difficulty]


def interpolated_ap_r40(
    true_positives: Sequence[int],
    false_positives: Sequence[int],
    num_valid_gt: int,
) -> float:
    if num_valid_gt <= 0:
        return 0.0
    if not true_positives:
        return 0.0

    tp = np.cumsum(np.asarray(true_positives, dtype=np.float64))
    fp = np.cumsum(np.asarray(false_positives, dtype=np.float64))
    recalls = tp / float(num_valid_gt)
    precisions = tp / np.maximum(tp + fp, 1e-12)
    precisions = np.maximum.accumulate(precisions[::-1])[::-1]

    sampled = []
    for recall_target in np.arange(1, 41, dtype=np.float64) / 40.0:
        eligible = np.flatnonzero(recalls >= recall_target - 1e-12)
        sampled.append(float(precisions[eligible[0]]) if len(eligible) else 0.0)
    return 100.0 * float(np.mean(sampled))


def _evaluate_one(
    ground_truth: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    class_name: str,
    difficulty: str,
    metric: str,
    iou_threshold: float,
    neighbor_classes: Mapping[str, set[str]],
) -> EvalResult:
    overlap_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], float]
    overlap_fn = bev_iou if metric == "bev" else iou_3d

    gt_status_by_sample: Dict[str, List[int]] = {}
    num_valid_gt = 0
    for sample_id, sample_gt in ground_truth.items():
        statuses = [
            _gt_status(gt, class_name, difficulty, neighbor_classes)
            for gt in sample_gt
        ]
        gt_status_by_sample[sample_id] = statuses
        num_valid_gt += sum(status == 0 for status in statuses)

    ranked_predictions = []
    for sample_id, sample_predictions in predictions.items():
        for prediction in sample_predictions:
            if prediction["class_name"] == class_name:
                ranked_predictions.append((float(prediction["score"]), sample_id, prediction))
    ranked_predictions.sort(key=lambda item: item[0], reverse=True)

    matched: Dict[str, set[int]] = {sample_id: set() for sample_id in ground_truth}
    true_positives: List[int] = []
    false_positives: List[int] = []

    for _, sample_id, prediction in ranked_predictions:
        sample_gt = ground_truth.get(sample_id, ())
        statuses = gt_status_by_sample.get(sample_id, [])
        valid_candidates = []
        ignored_candidates = []

        for gt_index, (gt, status) in enumerate(zip(sample_gt, statuses)):
            if status < 0 or gt_index in matched.setdefault(sample_id, set()):
                continue
            overlap = overlap_fn(prediction, gt)
            if overlap + 1e-12 < iou_threshold:
                continue
            candidate = (overlap, gt_index)
            if status == 0:
                valid_candidates.append(candidate)
            else:
                ignored_candidates.append(candidate)

        if valid_candidates:
            _, gt_index = max(valid_candidates)
            matched[sample_id].add(gt_index)
            true_positives.append(1)
            false_positives.append(0)
        elif ignored_candidates:
            _, gt_index = max(ignored_candidates)
            matched[sample_id].add(gt_index)
        elif not _prediction_ignored(prediction, difficulty):
            true_positives.append(0)
            false_positives.append(1)

    return EvalResult(
        class_name=class_name,
        difficulty=difficulty,
        metric=metric,
        iou_threshold=iou_threshold,
        ap_r40=interpolated_ap_r40(true_positives, false_positives, num_valid_gt),
        num_valid_gt=num_valid_gt,
        num_scored_predictions=len(true_positives),
        num_true_positives=sum(true_positives),
        num_false_positives=sum(false_positives),
    )


def evaluate_kitti_r40(
    ground_truth: Mapping[str, Sequence[Mapping[str, Any]]],
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    classes: Iterable[str] = ("Car", "Pedestrian", "Cyclist"),
    iou_thresholds: Mapping[str, float] | None = None,
    neighbor_classes: Mapping[str, set[str]] | None = None,
) -> List[EvalResult]:
    iou_thresholds = IOU_THRESHOLDS if iou_thresholds is None else iou_thresholds
    neighbor_classes = (
        NEIGHBOR_CLASSES if neighbor_classes is None else neighbor_classes
    )
    results = []
    for class_name in classes:
        if class_name not in iou_thresholds:
            raise ValueError(f"No IoU threshold configured for {class_name}")
        for metric in ("bev", "3d"):
            for difficulty in DIFFICULTIES:
                results.append(
                    _evaluate_one(
                        ground_truth=ground_truth,
                        predictions=predictions,
                        class_name=class_name,
                        difficulty=difficulty,
                        metric=metric,
                        iou_threshold=float(iou_thresholds[class_name]),
                        neighbor_classes=neighbor_classes,
                    )
                )
    return results
