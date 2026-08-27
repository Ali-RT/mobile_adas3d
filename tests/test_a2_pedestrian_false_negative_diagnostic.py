from __future__ import annotations

import unittest

from scripts.diagnose_a2_pedestrian_false_negatives import diagnose, kitti_difficulty


def target(bbox, depth=15.0, occluded=0, truncated=0.0):
    return {
        "class_name": "Pedestrian",
        "source_class_name": "Pedestrian",
        "bbox_2d": list(bbox),
        "location_3d": [0.0, 1.5, depth],
        "occluded": occluded,
        "truncated": truncated,
    }


def prediction(class_name, bbox, score=0.5):
    return {"class_name": class_name, "bbox_2d": list(bbox), "score": score}


class A2PedestrianFalseNegativeDiagnosticTests(unittest.TestCase):
    def test_failure_modes_separate_classification_localization_and_missing(self):
        ground_truth = {
            "000001": [
                target([0, 0, 10, 50]),
                target([20, 0, 30, 50]),
                target([40, 0, 50, 50]),
                target([60, 0, 70, 50]),
            ]
        }
        predictions = {
            "000001": [
                prediction("Pedestrian", [0, 0, 10, 50], 0.9),
                prediction("Vehicle", [20, 0, 30, 50], 0.8),
                prediction("Pedestrian", [44, 0, 54, 50], 0.7),
            ]
        }
        report, rows, _ = diagnose(predictions, ground_truth)
        self.assertEqual(
            [row["failure_mode"] for row in rows],
            [
                "detected_pedestrian",
                "wrong_class_classification",
                "localization_failure",
                "missing_query",
            ],
        )
        self.assertEqual(report["near_pedestrian_ground_truth"], 4)
        self.assertIn("truncation_bucket", rows[0])

    def test_difficulty_uses_kitti_thresholds(self):
        self.assertEqual(kitti_difficulty(target([0, 0, 10, 40])), "easy")
        self.assertEqual(
            kitti_difficulty(target([0, 0, 10, 25], occluded=1, truncated=0.3)),
            "moderate",
        )
        self.assertEqual(
            kitti_difficulty(target([0, 0, 10, 25], occluded=2, truncated=0.5)),
            "hard",
        )
        self.assertEqual(kitti_difficulty(target([0, 0, 10, 24])), "outside_hard")


if __name__ == "__main__":
    unittest.main()
