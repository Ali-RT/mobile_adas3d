from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_monodetr_a2f_gate import is_eligible
from scripts.patch_monodetr_a2f_high_resolution import replace_once
from scripts.prepare_monodetr_a2f_gate import A2_EPOCH, A2_SHA256


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDETR_A2f_High_Resolution_Feature_Gate_Colab.ipynb"


class MonoDETRA2fGateTests(unittest.TestCase):
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

    def test_two_paired_variants_are_frozen(self):
        self.assertIn("[\u0027control_stride8\u0027,\u0027stride4_feature\u0027]", self.code)
        self.assertIn("--gate-epochs\u0027,\u00275\u0027", self.code)
        self.assertIn("--learning-rate\u0027,\u00271e-5\u0027", self.code)

    def test_default_seed_survives_monodetr_squared_seed(self):
        source = (ROOT / "scripts/prepare_monodetr_a2f_gate.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("default=20268", source)
        self.assertLessEqual(20268**2, 2**32 - 1)

    def test_patch_changes_only_configurable_feature_strides(self):
        source = (ROOT / "scripts/patch_monodetr_a2f_high_resolution.py").read_text(encoding="utf-8")
        self.assertIn("backbone_expected_strides", source)
        self.assertNotIn("loss_bbox", source)
        self.assertNotIn("depthaware_transformer.py", source)

    def test_preparer_adds_stride4_and_remaps_trained_projections(self):
        source = (ROOT / "scripts/prepare_monodetr_a2f_gate.py").read_text(encoding="utf-8")
        self.assertIn("backbone_out_indices\u0022] = [1, 2, 3, 4]", source)
        self.assertIn("backbone_expected_strides\u0022] = [4, 8, 16, 32]", source)
        self.assertIn("name.startswith(\u0022input_proj.0.\u0022)", source)
        self.assertIn("int(name.split(\u0022.\u0022)[1]) - 1", source)
        self.assertIn("\u0022losses_changed\u0022: False", source)
        self.assertIn("\u0022transformer_changed\u0022: False", source)

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
            "variant": "stride4_feature",
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
