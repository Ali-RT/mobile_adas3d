from __future__ import annotations

import json
import math
from pathlib import Path
import unittest

import numpy as np

from scripts.audit_rtm3d_km3d_res18_onnx_fullval import (
    prediction_from_row,
    selected_detections,
    stable_sigmoid,
    validate_raw_fixture_report,
    validate_onnx_head,
)
from scripts.audit_rtm3d_km3d_res18_onnx_targeted import (
    compare_selected,
    selected_rows_with_indices,
    summarize_head_delta,
    wrapped_angle_delta_degrees,
)
from scripts.run_rtm3d_km3d_res18_smoke import HEADS


class FullValPredictionTests(unittest.TestCase):
    def test_colab_notebook_code_compiles_and_preserves_scope(self):
        notebook_path = Path(__file__).resolve().parents[1] / "notebooks" / "RTM3D_KM3D_ResNet18_ONNX_FullVal_Parity_Colab.ipynb"
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
        cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]
        self.assertGreaterEqual(len(cells), 4)
        for index, source in enumerate(cells):
            compile(source, f"{notebook_path}:cell-{index}", "exec")
        all_text = "\n".join(cells + ["".join(c.get("source", [])) for c in notebook["cells"]])
        self.assertIn("3769", all_text)
        self.assertIn("fullval_results.zip", all_text)
        self.assertIn("not an official KITTI leaderboard", all_text)
        self.assertIn("iPhone speed test", all_text)
        self.assertIn("audit_rtm3d_km3d_res18_onnx_targeted.py", all_text)
        self.assertIn("'006767','002646','006908','003529'", all_text)
        self.assertIn("full_val_rerun'] is False", all_text)

    def test_sigmoid_is_stable_for_large_logits(self):
        self.assertEqual(stable_sigmoid(1000.0), 1.0)
        self.assertEqual(stable_sigmoid(-1000.0), 0.0)

    def test_raw_tolerance_miss_does_not_block_finite_complete_fixture(self):
        fixture_ids = [f"{index:06d}" for index in range(16)]
        export = {"input": {"fixture_sample_ids": fixture_ids}}
        per_head = {
            name: {"finite": True, "passed": name != "hps"}
            for name in HEADS
        }
        raw = {
            "complete": False,
            "all_head_outputs_finite": True,
            "images_evaluated": 16,
            "fixture_sample_ids": fixture_ids,
            "parity_tolerance_max_abs": 0.0002,
            "per_head": per_head,
            "per_image": {
                sample_id: {
                    name: {"finite": True, "passed": name != "hps"}
                    for name in HEADS
                }
                for sample_id in fixture_ids
            },
        }
        status = validate_raw_fixture_report(raw, export)
        self.assertTrue(status["all_16_images_evaluated"])
        self.assertTrue(status["all_outputs_finite"])
        self.assertFalse(status["all_heads_within_raw_tolerance"])
        self.assertEqual(status["failed_heads"], ["hps"])

    def test_raw_fixture_report_still_rejects_nonfinite_outputs(self):
        fixture_ids = [f"{index:06d}" for index in range(16)]
        export = {"input": {"fixture_sample_ids": fixture_ids}}
        raw = {
            "all_head_outputs_finite": False,
            "images_evaluated": 16,
            "fixture_sample_ids": fixture_ids,
            "per_head": {name: {"finite": True} for name in HEADS},
            "per_image": {
                sample_id: {name: {"finite": True} for name in HEADS}
                for sample_id in fixture_ids
            },
        }
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            validate_raw_fixture_report(raw, export)

    def test_onnx_head_shape_failure_reports_runtime_and_input_shapes(self):
        input_array = np.zeros((1, 3, 384, 1280), dtype=np.float32)
        actual = np.zeros((1, 4, 96, 320), dtype=np.float32)
        with self.assertRaisesRegex(RuntimeError, r'"actual_shape": \[1, 4, 96, 320\]') as raised:
            validate_onnx_head(actual, "hm", "000001", [1, 3, 96, 320], [1, 4, 96, 320], input_array)
        self.assertIn('"input_shape": [1, 3, 384, 1280]', str(raised.exception))

    def test_onnx_head_nonfinite_failure_reports_count_and_indices(self):
        input_array = np.zeros((1, 3, 384, 1280), dtype=np.float32)
        actual = np.zeros((1, 3, 96, 320), dtype=np.float32)
        actual[0, 0, 1, 2] = np.nan
        with self.assertRaisesRegex(RuntimeError, '"nonfinite_values": 1') as raised:
            validate_onnx_head(actual, "hm", "000001", [1, 3, 96, 320], [1, 3, 96, 320], input_array)
        self.assertIn('"first_nonfinite_indices": [[0, 0, 1, 2]]', str(raised.exception))

    def test_kitti_serialization_matches_upstream_field_layout(self):
        row = np.zeros(41, dtype=np.float32)
        row[:5] = [10.0, 20.0, 100.0, 200.0, 0.8]
        row[23:32] = 0.9
        row[32:35] = [1.5, 1.6, 4.0]
        row[35] = 0.2
        row[36:39] = [1.0, 2.0, 30.0]
        row[39] = 0.0
        row[40] = 0.0

        record, line = prediction_from_row(row)
        fields = line.split()
        self.assertEqual(len(fields), 16)
        self.assertEqual(fields[0], "Car")
        self.assertEqual(fields[1:3], ["-1.00", "-1"])
        self.assertAlmostEqual(float(fields[8]), 1.5)
        self.assertAlmostEqual(float(fields[9]), 1.6)
        self.assertAlmostEqual(float(fields[10]), 4.0)
        self.assertAlmostEqual(float(fields[12]), 2.75)
        self.assertAlmostEqual(float(fields[15]), (0.8 + 0.5 + 0.9) / 3.0, places=6)
        self.assertEqual(record["location_3d"], [1.0, 2.75, 30.0])
        for actual, expected in zip(record["dimensions_3d_hwl"], [1.5, 1.6, 4.0]):
            self.assertAlmostEqual(actual, expected, places=5)

    def test_only_upstream_visibility_selected_rows_are_serialized(self):
        rows = np.zeros((2, 41), dtype=np.float32)
        rows[0, :5] = [1, 2, 10, 20, 0.31]
        rows[0, 23:32] = 0.8
        rows[0, 32:35] = [1.5, 1.6, 4.0]
        rows[0, 36:39] = [0.0, 1.0, 20.0]
        rows[0, 40] = 1
        rows[1, 4] = 0.3
        selected, records, lines = selected_detections(rows)
        self.assertEqual(selected.shape, (1, 41))
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["class_name"], "Pedestrian")
        self.assertEqual(len(lines), 1)

    def test_invalid_geometry_is_rejected(self):
        row = np.zeros(41, dtype=np.float32)
        row[:5] = [10, 20, 100, 200, 0.8]
        row[32:35] = [1.5, 1.6, 4.0]
        row[36:39] = [1.0, 2.0, 30.0]
        row[40] = 0
        row[35] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            prediction_from_row(row)

    def test_negative_depth_is_preserved_and_marked_invalid(self):
        row = np.zeros(41, dtype=np.float32)
        row[:5] = [10, 20, 100, 200, 0.8]
        row[23:32] = 0.9
        row[32:35] = [1.62, 1.56, 3.46]
        row[36:39] = [3.77, -0.67, -24.18]
        row[40] = 0

        record, line = prediction_from_row(row)
        self.assertFalse(record["valid_3d_geometry"])
        self.assertEqual(record["geometry_issues"], ["nonpositive_camera_depth"])
        self.assertAlmostEqual(float(line.split()[13]), -24.18, places=5)

    def test_nonpositive_dimensions_are_rejected(self):
        row = np.zeros(41, dtype=np.float32)
        row[:5] = [10, 20, 100, 200, 0.8]
        row[32:35] = [1.5, 0.0, 4.0]
        row[36:39] = [1.0, 2.0, 30.0]
        row[40] = 0
        with self.assertRaisesRegex(RuntimeError, "nonpositive 3D dimensions"):
            prediction_from_row(row)


class TargetedParityTests(unittest.TestCase):
    @staticmethod
    def detection_row(score: float, left: float = 10.0) -> np.ndarray:
        row = np.zeros(41, dtype=np.float32)
        row[:5] = [left, 20.0, left + 40.0, 70.0, score]
        row[23:32] = 0.8
        row[32:35] = [1.5, 1.6, 4.0]
        row[35] = 0.2
        row[36:39] = [1.0, 2.0, 30.0]
        row[40] = 0
        return row

    def test_raw_head_summary_reports_error_location_and_tolerance_count(self):
        reference = np.zeros((1, 1, 2, 2), dtype=np.float32)
        candidate = np.asarray([[[[0.0, 0.1], [0.2, 0.3]]]], dtype=np.float32)
        summary = summarize_head_delta(reference, candidate, tolerance=0.15)
        self.assertEqual(summary["shape"], [1, 1, 2, 2])
        self.assertAlmostEqual(summary["max_abs_delta"], 0.3, places=6)
        self.assertEqual(summary["elements_over_tolerance"], 2)
        self.assertEqual(summary["max_delta_index"], [0, 0, 1, 1])

    def test_selected_rows_keep_postprocess_index_and_match_details(self):
        reference = np.stack([
            self.detection_row(0.31),
            self.detection_row(0.30, 100.0),  # strict threshold excludes it
            self.detection_row(0.80, 200.0),
        ])
        candidate = np.stack([self.detection_row(0.32, 11.0)])
        ref_selected, ref_records = selected_rows_with_indices(reference)
        onnx_selected, onnx_records = selected_rows_with_indices(candidate)
        self.assertEqual(ref_selected.shape, (2, 41))
        self.assertEqual(ref_records[0]["postprocess_row_index"], 0)
        self.assertEqual(ref_records[1]["postprocess_row_index"], 2)
        comparison = compare_selected(ref_selected, onnx_selected, ref_records, onnx_records)
        self.assertEqual(comparison["matched_count"], 1)
        self.assertEqual(comparison["pytorch_unmatched_count"], 1)
        self.assertEqual(comparison["matched_predictions"][0]["pytorch_postprocess_row_index"], 0)
        self.assertEqual(comparison["matched_predictions"][0]["onnx_postprocess_row_index"], 0)

    def test_yaw_difference_wraps_at_pi(self):
        actual = wrapped_angle_delta_degrees(math.pi - 0.01, -math.pi + 0.01)
        self.assertAlmostEqual(actual, math.degrees(0.02), places=5)


if __name__ == "__main__":
    unittest.main()
