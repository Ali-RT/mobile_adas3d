import ast
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.diagnose_monodgp_m59c_backbone import compare_backbone_tensors, first_failed


class MonoDGPM59cBackboneTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_scale_aware_gate_handles_large_backbone_values(self):
        reference = {"backbone_feat_2": np.array([[5.9e8]], dtype=np.float32)}
        prediction = {"backbone_feat_2": np.array([[5.9e8 - 848.0]], dtype=np.float32)}
        comparison = compare_backbone_tensors(reference, prediction, ("backbone_feat_2",))
        row = comparison["backbone_feat_2"]
        self.assertFalse(row["strict_passed"])
        self.assertTrue(row["scale_aware_passed"])
        self.assertEqual(first_failed(comparison, "strict_passed"), "backbone_feat_2")
        self.assertIsNone(first_failed(comparison, "scale_aware_passed"))

    def test_material_mismatch_fails_scale_aware_gate(self):
        reference = {"backbone_feat_0": np.ones((1, 2), dtype=np.float32)}
        prediction = {"backbone_feat_0": np.full((1, 2), 1.1, dtype=np.float32)}
        row = compare_backbone_tensors(reference, prediction, ("backbone_feat_0",))["backbone_feat_0"]
        self.assertFalse(row["scale_aware_passed"])

    def test_missing_and_shape_changed_fail_closed(self):
        reference = {
            "backbone_feat_0": np.zeros((1, 2), dtype=np.float32),
            "backbone_feat_1": np.zeros((1, 2), dtype=np.float32),
        }
        prediction = {"backbone_feat_1": np.zeros((1, 3), dtype=np.float32)}
        comparison = compare_backbone_tensors(
            reference, prediction, ("backbone_feat_0", "backbone_feat_1")
        )
        self.assertFalse(comparison["backbone_feat_0"]["scale_aware_passed"])
        self.assertFalse(comparison["backbone_feat_1"]["scale_aware_passed"])

    def test_script_is_standalone_and_fail_closed(self):
        script = self.ROOT / "scripts/diagnose_monodgp_m59c_backbone.py"
        ast.parse(script.read_text())
        self.assertIn("physical_device_testing_authorized", script.read_text())
        self.assertIn("first_material_diverging_tensor", script.read_text())
        json.loads("{}")


if __name__ == "__main__":
    unittest.main()
