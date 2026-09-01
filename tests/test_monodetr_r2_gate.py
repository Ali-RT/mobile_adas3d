from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_monodetr_r2_gate import is_eligible
from scripts.patch_monodetr_pedestrian_refinement import replace_once
from scripts.prepare_monodetr_r2_gate import R0_EPOCH, R0_SHA256, VARIANTS


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDETR_R2_Pedestrian_Refinement_Gate_Colab.ipynb"


class MonoDETRR2GateTests(unittest.TestCase):
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
        self.assertEqual(VARIANTS, {"control_refine_off": False, "ped_refine_stride4": True})
        self.assertIn("--gate-epochs','5'", self.code)
        self.assertIn("--learning-rate','1e-5'", self.code)
        self.assertIn("smoke_test_monodetr_r2_refinement.py", self.code)
        self.assertIn("initialization_max_abs_deltas", self.code)
        self.assertIn("monodetr_r2_pedestrian_refinement_gate", self.code)
        self.assertNotIn("monodetr_a2d_pedestrian_box_gate", self.code)
        self.assertIn("r2_refinement_smoke.log", self.code)

    def test_default_seed_survives_monodetr_squared_seed(self):
        source = (ROOT / "scripts/prepare_monodetr_r2_gate.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("default=20268", source)
        self.assertLessEqual(20268**2, 2**32 - 1)

    def test_patch_adds_only_local_residual_refinement(self):
        source = (ROOT / "scripts/patch_monodetr_pedestrian_refinement.py").read_text(encoding="utf-8")
        self.assertIn("backbone.layer1.register_forward_hook", source)
        self.assertIn("F.grid_sample", source)
        self.assertIn("pedestrian_probability = outputs_class[..., 0:1].sigmoid()", source)
        self.assertIn("pedestrian_refinement_head.layers[-1].weight, 0", source)
        self.assertIn("outputs_coord[..., 2:6] + self.pedestrian_refinement_scale", source)
        self.assertIn("upgrade_if_present", source)
        self.assertIn("outputs_coord[..., 0:2], refined_edges", source)
        self.assertNotIn("out['pred_depth'] =", source)
        self.assertNotIn("out['pred_angle'] =", source)

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
            "variant": "ped_refine_stride4",
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
