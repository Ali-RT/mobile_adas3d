from __future__ import annotations

import unittest

from scripts.audit_product_prediction_geometry import audit


def target(class_name: str, bbox, depth: float, x: float = 0.0):
    return {
        "class_name": class_name,
        "bbox_2d": list(bbox),
        "dimensions_3d": [1.5, 1.6, 4.0] if class_name == "Vehicle" else [1.7, 0.6, 0.8],
        "location_3d": [x, 1.5, depth],
        "rotation_y": 0.1,
    }


def prediction(class_name: str, bbox, depth: float, score: float = 0.9, x: float = 0.0):
    return {
        "class_name": class_name,
        "bbox_2d": list(bbox),
        "dimensions_3d_hwl": [1.5, 1.6, 4.0] if class_name == "Vehicle" else [1.7, 0.6, 0.8],
        "location_3d": [x, 1.5, depth],
        "rotation_y": 0.1,
        "yaw": 0.1,
        "score": score,
    }


class ProductPredictionGeometryTests(unittest.TestCase):
    def test_near_recall_and_geometry_are_class_specific(self):
        ground_truth = {
            "000001": [
                target("Vehicle", [0, 0, 20, 20], 10.0),
                target("Vehicle", [40, 0, 60, 20], 50.0),
                target("Pedestrian", [80, 0, 90, 30], 15.0),
            ]
        }
        predictions = {
            "000001": [
                prediction("Vehicle", [0, 0, 20, 20], 11.0),
                prediction("Pedestrian", [80, 0, 90, 30], 15.5),
            ]
        }
        report, matched, false_negatives = audit(
            predictions, ground_truth, score_threshold=0.001, match_iou_threshold=0.5
        )
        vehicle = report["classes"]["Vehicle"]
        pedestrian = report["classes"]["Pedestrian"]
        self.assertEqual(vehicle["gt"], 2)
        self.assertEqual(vehicle["matched"], 1)
        self.assertAlmostEqual(vehicle["recall"], 0.5)
        self.assertAlmostEqual(vehicle["near_recall"], 1.0)
        self.assertTrue(vehicle["near_recall_gate_passed"])
        self.assertAlmostEqual(pedestrian["near_recall"], 1.0)
        self.assertTrue(report["all_near_recall_gates_passed"])
        self.assertEqual(len(matched), 2)
        self.assertEqual(len(false_negatives), 1)
        self.assertFalse(false_negatives[0]["near_field"])

    def test_score_and_iou_thresholds_create_false_negatives(self):
        ground_truth = {"000001": [target("Vehicle", [0, 0, 20, 20], 10.0)]}
        predictions = {
            "000001": [
                prediction("Vehicle", [100, 100, 120, 120], 10.0, score=0.9),
                prediction("Vehicle", [0, 0, 20, 20], 10.0, score=0.05),
            ]
        }
        report, matched, false_negatives = audit(
            predictions, ground_truth, score_threshold=0.1, match_iou_threshold=0.5
        )
        vehicle = report["classes"]["Vehicle"]
        self.assertEqual(vehicle["matched"], 0)
        self.assertEqual(vehicle["fp"], 1)
        self.assertEqual(vehicle["near_gt"], 1)
        self.assertEqual(vehicle["near_matched"], 0)
        self.assertFalse(vehicle["near_recall_gate_passed"])
        self.assertEqual(matched, [])
        self.assertEqual(len(false_negatives), 1)


if __name__ == "__main__":
    unittest.main()
