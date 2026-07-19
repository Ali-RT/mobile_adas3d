import unittest

import numpy as np

from data.kitti_r40 import (
    bev_iou,
    convex_polygon_intersection,
    evaluate_kitti_r40,
    interpolated_ap_r40,
    iou_3d,
    polygon_area,
)


def make_gt(height_px=50.0, location=(0.0, 1.5, 20.0)):
    return {
        "class_name": "Car",
        "truncated": 0.0,
        "occluded": 0,
        "bbox_2d": [100.0, 100.0, 200.0, 100.0 + height_px],
        "dimensions_3d": [1.5, 1.6, 4.0],
        "location_3d": list(location),
        "rotation_y": 0.0,
    }


def make_prediction(score=0.9, height_px=50.0, location=(0.0, 1.5, 20.0)):
    return {
        "class_name": "Car",
        "score": score,
        "bbox_2d": [100.0, 100.0, 200.0, 100.0 + height_px],
        "dimensions_3d_hwl": [1.5, 1.6, 4.0],
        "location_3d": list(location),
        "yaw": 0.0,
    }


class GeometryTests(unittest.TestCase):
    def test_convex_polygon_intersection(self):
        first = np.asarray([[0, 0], [2, 0], [2, 2], [0, 2]], dtype=float)
        second = np.asarray([[1, 1], [3, 1], [3, 3], [1, 3]], dtype=float)
        intersection = convex_polygon_intersection(first, second)
        self.assertAlmostEqual(polygon_area(intersection), 1.0, places=7)

    def test_identical_boxes_have_unit_iou(self):
        gt = make_gt()
        prediction = make_prediction()
        self.assertAlmostEqual(bev_iou(gt, prediction), 1.0, places=7)
        self.assertAlmostEqual(iou_3d(gt, prediction), 1.0, places=7)

    def test_vertical_disjoint_boxes_have_zero_3d_iou(self):
        gt = make_gt(location=(0.0, 1.5, 20.0))
        prediction = make_prediction(location=(0.0, 5.0, 20.0))
        self.assertAlmostEqual(bev_iou(gt, prediction), 1.0, places=7)
        self.assertAlmostEqual(iou_3d(gt, prediction), 0.0, places=7)


class APTests(unittest.TestCase):
    def test_r40_perfect_and_half_recall(self):
        self.assertAlmostEqual(interpolated_ap_r40([1], [0], 1), 100.0)
        self.assertAlmostEqual(interpolated_ap_r40([1], [0], 2), 50.0)

    def test_perfect_detection_scores_100(self):
        results = evaluate_kitti_r40(
            ground_truth={"000001": [make_gt()]},
            predictions={"000001": [make_prediction()]},
            classes=("Car",),
        )
        self.assertEqual(len(results), 6)
        for result in results:
            self.assertAlmostEqual(result.ap_r40, 100.0)
            self.assertEqual(result.num_true_positives, 1)

    def test_high_score_false_positive_reduces_ap(self):
        results = evaluate_kitti_r40(
            ground_truth={"000001": [make_gt()]},
            predictions={
                "000001": [make_prediction(score=0.8)],
                "000002": [make_prediction(score=0.9)],
            },
            classes=("Car",),
        )
        moderate_3d = next(
            result for result in results
            if result.metric == "3d" and result.difficulty == "moderate"
        )
        self.assertAlmostEqual(moderate_3d.ap_r40, 50.0)
        self.assertEqual(moderate_3d.num_false_positives, 1)

    def test_van_is_ignored_for_car_evaluation(self):
        van = make_gt()
        van["class_name"] = "Van"
        results = evaluate_kitti_r40(
            ground_truth={"000001": [van]},
            predictions={"000001": [make_prediction()]},
            classes=("Car",),
        )
        moderate_3d = next(
            result for result in results
            if result.metric == "3d" and result.difficulty == "moderate"
        )
        self.assertEqual(moderate_3d.num_valid_gt, 0)
        self.assertEqual(moderate_3d.num_scored_predictions, 0)
        self.assertEqual(moderate_3d.num_false_positives, 0)

    def test_difficulty_filter_uses_original_box_height(self):
        results = evaluate_kitti_r40(
            ground_truth={"000001": [make_gt(height_px=30.0)]},
            predictions={"000001": [make_prediction(height_px=30.0)]},
            classes=("Car",),
        )
        easy_3d = next(
            result for result in results
            if result.metric == "3d" and result.difficulty == "easy"
        )
        moderate_3d = next(
            result for result in results
            if result.metric == "3d" and result.difficulty == "moderate"
        )
        self.assertEqual(easy_3d.num_valid_gt, 0)
        self.assertEqual(moderate_3d.num_valid_gt, 1)
        self.assertAlmostEqual(moderate_3d.ap_r40, 100.0)


if __name__ == "__main__":
    unittest.main()
