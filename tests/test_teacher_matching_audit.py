import unittest

from scripts.audit_teacher_prediction_cache import (
    angle_error_degrees,
    audit_threshold,
    bbox_iou,
    select_recommendations,
)


def make_gt(depth=20.0):
    return {
        "class_name": "Car",
        "bbox_2d": [10.0, 10.0, 50.0, 50.0],
        "dimensions_3d": [1.5, 1.6, 4.0],
        "location_3d": [0.0, 1.5, depth],
        "rotation_y": 0.0,
    }


def make_prediction(score=0.9, bbox=None, depth=20.0):
    return {
        "class_name": "Car",
        "score": score,
        "bbox_2d": bbox or [10.0, 10.0, 50.0, 50.0],
        "dimensions_3d_hwl": [1.5, 1.6, 4.0],
        "location_3d": [0.0, 1.5, depth],
        "yaw": 0.0,
    }


class TeacherMatchingAuditTests(unittest.TestCase):
    def test_bbox_iou_and_wrapped_angle(self):
        self.assertAlmostEqual(bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)
        self.assertAlmostEqual(angle_error_degrees(3.13, -3.13), 1.328, places=2)

    def test_one_to_one_matching_counts_duplicate_as_false_positive(self):
        summary, distance_rows, matches = audit_threshold(
            predictions={
                "000001": [
                    make_prediction(score=0.9),
                    make_prediction(score=0.8),
                ]
            },
            ground_truth={"000001": [make_gt()]},
            score_threshold=0.5,
            match_iou_threshold=0.5,
        )
        self.assertEqual(summary["matched"], 1)
        self.assertEqual(summary["unmatched_teacher_predictions"], 1)
        self.assertEqual(summary["unmatched_ground_truth"], 0)
        self.assertAlmostEqual(summary["precision"], 0.5)
        self.assertAlmostEqual(summary["recall"], 1.0)
        self.assertEqual(len(matches), 1)
        self.assertEqual(sum(row["matched"] for row in distance_rows), 1)

    def test_threshold_recommendations_are_deterministic(self):
        rows = [
            {"score_threshold": 0.1, "precision": 0.7, "recall": 0.99, "f1": 0.82},
            {"score_threshold": 0.3, "precision": 0.9, "recall": 0.96, "f1": 0.93},
            {"score_threshold": 0.5, "precision": 0.95, "recall": 0.90, "f1": 0.92},
        ]
        selected = select_recommendations(rows)
        self.assertEqual(selected["max_f1"]["score_threshold"], 0.3)
        self.assertEqual(selected["high_recall_95"]["score_threshold"], 0.3)


if __name__ == "__main__":
    unittest.main()
