from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_monodetr_r2b_gate import is_eligible
from scripts.patch_monodetr_r2b_frozen_refinement import replace_once
from scripts.prepare_monodetr_r2b_gate import R0_EPOCH, R0_SHA256, VARIANTS


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDETR_R2b_Frozen_Refinement_Gate_Colab.ipynb"


class MonoDETRR2bGateTests(unittest.TestCase):
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

    def test_single_treatment_is_frozen(self):
        self.assertEqual(VARIANTS, {"ped_refine_frozen_hard": True})
        self.assertIn("--gate-epochs','10'", self.code)
        self.assertIn("--learning-rate','1e-4'", self.code)
        self.assertIn("smoke_test_monodetr_r2b_refinement.py", self.code)
        self.assertIn("monodetr_r2b_frozen_refinement_gate", self.code)

    def test_default_seed_survives_monodetr_squared_seed(self):
        source = (ROOT / "scripts/prepare_monodetr_r2_gate.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("default=20268", source)
        self.assertLessEqual(20268**2, 2**32 - 1)

    def test_patch_freezes_base_and_hard_gates_pedestrian(self):
        source = (ROOT / "scripts/patch_monodetr_r2b_frozen_refinement.py").read_text(encoding="utf-8")
        self.assertIn("parameter.requires_grad_(name.startswith(trainable_prefixes))", source)
        self.assertIn("outputs_class.argmax(dim=-1, keepdim=True) == 0", source)
        self.assertIn(".detach()", source)
        self.assertIn("pedestrian_refinement_proj.", source)
        self.assertIn("pedestrian_refinement_head.", source)

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

    def test_eligibility_requires_r0_gain_without_regression(self):
        row = {
            "pedestrian_near_gain_vs_r0": 0.02,
            "pedestrian_localization_reduction_vs_r0": 0.02,
            "vehicle_near_recall": 0.8724637112593173,
            "vehicle_3d_delta_vs_r0": -0.15,
            "vehicle_bev_delta_vs_r0": -0.15,
            "pedestrian_3d_moderate": 5.721371354710236,
            "pedestrian_bev_moderate": 6.596148868813419,
        }
        self.assertTrue(is_eligible(row))
        row["pedestrian_near_gain_vs_r0"] = 0.019
        self.assertFalse(is_eligible(row))
        row["pedestrian_near_gain_vs_r0"] = 0.02
        row["pedestrian_localization_reduction_vs_r0"] = 0.019
        self.assertFalse(is_eligible(row))


if __name__ == "__main__":
    unittest.main()
