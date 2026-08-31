from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_monodetr_r1_gate import is_eligible
from scripts.patch_monodetr_pedestrian_match_cost import replace_once
from scripts.prepare_monodetr_r1_gate import R0_EPOCH, R0_SHA256, VARIANTS


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDETR_R1_Pedestrian_Matcher_Gate_Colab.ipynb"


class MonoDETRR1GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_notebook_cells_parse(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_two_variants_are_frozen(self):
        self.assertEqual(VARIANTS, {"control_match_w1_0": 1.0, "ped_match_w2_0": 2.0})
        self.assertIn("--gate-epochs','5'", self.code)
        self.assertIn("--learning-rate','1e-5'", self.code)

    def test_default_seed_survives_monodetr_squared_seed(self):
        source = (ROOT / "scripts/prepare_monodetr_r1_gate.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("default=20268", source)
        self.assertLessEqual(20268**2, 2**32 - 1)

    def test_patch_weights_only_native_pedestrian_assignment_localization(self):
        source = (ROOT / "scripts/patch_monodetr_pedestrian_match_cost.py").read_text(encoding="utf-8")
        self.assertIn("tgt_ids == 0", source)
        self.assertIn("cost_bbox * target_localization_weights", source)
        self.assertIn("cost_giou * target_localization_weights", source)
        self.assertNotIn("cost_class * target_localization_weights", source)
        self.assertNotIn("cost_3dcenter * target_localization_weights", source)

    def test_gate_preserves_r0_provenance(self):
        self.assertEqual(R0_EPOCH, 185)
        self.assertEqual(
            R0_SHA256,
            "fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59",
        )
        self.assertIn("temperature_scaling_enabled'] is False", self.code)
        self.assertIn("distillation_enabled'] is False", self.code)

    def test_patch_is_idempotent_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "source.py"
            path.write_text("old\n", encoding="utf-8")
            replace_once(path, "old\n", "new\n", "test")
            replace_once(path, "old\n", "new\n", "test")
            self.assertEqual(path.read_text(), "new\n")
            with self.assertRaises(RuntimeError):
                replace_once(path, "missing", "also_missing", "test")

    def test_eligibility_requires_recall_gain_without_regression(self):
        control = {
            "vehicle_near_recall": 0.88,
            "pedestrian_3d_moderate": 7.0,
            "pedestrian_bev_moderate": 8.0,
        }
        row = {
            "variant": "ped_match_w2_0",
            "pedestrian_near_gain_vs_control": 0.02,
            "pedestrian_localization_reduction_vs_control": 0.02,
            "vehicle_near_recall": 0.87,
            "vehicle_3d_delta_vs_control": -0.15,
            "vehicle_bev_delta_vs_control": -0.15,
            "pedestrian_3d_moderate": 7.0,
            "pedestrian_bev_moderate": 8.0,
        }
        self.assertTrue(is_eligible(row, control))
        row["pedestrian_near_gain_vs_control"] = 0.019
        self.assertFalse(is_eligible(row, control))
        row["pedestrian_near_gain_vs_control"] = 0.02
        row["pedestrian_localization_reduction_vs_control"] = 0.019
        self.assertFalse(is_eligible(row, control))


if __name__ == "__main__":
    unittest.main()
