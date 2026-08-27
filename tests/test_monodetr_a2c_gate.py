from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_monodetr_a2c_gate import is_eligible
from scripts.patch_monodetr_class_positive_weight import replace_once
from scripts.prepare_monodetr_a2c_gate import A2_EPOCH, A2_SHA256, VARIANTS


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDETR_A2c_Pedestrian_Focal_Gate_Colab.ipynb"


class MonoDETRA2cGateTests(unittest.TestCase):
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

    def test_four_variants_are_frozen(self):
        self.assertEqual(
            VARIANTS,
            {
                "control_w1_0": 1.0,
                "pedcls_w1_5": 1.5,
                "pedcls_w2_0": 2.0,
                "pedcls_w2_5": 2.5,
            },
        )
        self.assertIn("--gate-epochs','5'", self.code)
        self.assertIn("--learning-rate','1e-5'", self.code)

    def test_pedestrian_is_native_class_zero(self):
        source = (ROOT / "scripts/prepare_monodetr_a2c_gate.py").read_text(encoding="utf-8")
        self.assertIn("[pedestrian_weight, 1.0, 1.0]", source)
        self.assertNotIn("[1.0, pedestrian_weight, 1.0]", source)

    def test_gate_preserves_a2_provenance(self):
        self.assertEqual(A2_EPOCH, 130)
        self.assertEqual(
            A2_SHA256,
            "ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4",
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
            "variant": "pedcls_w1_5",
            "pedestrian_near_gain_vs_control": 0.02,
            "vehicle_near_recall": 0.87,
            "vehicle_3d_delta_vs_control": -0.15,
            "vehicle_bev_delta_vs_control": -0.15,
            "pedestrian_3d_moderate": 7.0,
            "pedestrian_bev_moderate": 8.0,
        }
        self.assertTrue(is_eligible(row, control))
        row["pedestrian_near_gain_vs_control"] = 0.019
        self.assertFalse(is_eligible(row, control))


if __name__ == "__main__":
    unittest.main()
