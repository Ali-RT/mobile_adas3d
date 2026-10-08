from __future__ import annotations

import json
from pathlib import Path
import unittest

import numpy as np

from scripts.audit_rtm3d_km3d_res18_onnx_fullval import (
    prediction_from_row,
    selected_detections,
    stable_sigmoid,
)


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

    def test_sigmoid_is_stable_for_large_logits(self):
        self.assertEqual(stable_sigmoid(1000.0), 1.0)
        self.assertEqual(stable_sigmoid(-1000.0), 0.0)

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


if __name__ == "__main__":
    unittest.main()
