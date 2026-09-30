import ast
import copy
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_m63_lower_lr_control import lower_lr_optimizer_config, checkpoint_for, VARIANT

class LowerLRTests(unittest.TestCase):
    def test_only_lr_changes_without_mutating_source(self):
        m = {"student": {"config": {"optimizer": {"type": "adamw", "lr": 1e-5, "weight_decay": 1e-4}}}}
        before = copy.deepcopy(m)
        cfg = lower_lr_optimizer_config(m)
        self.assertEqual(m, before)
        self.assertEqual(cfg, {"type": "adamw", "lr": 1e-6, "weight_decay": 1e-4})

    def test_wrong_source_optimizer_stops(self):
        for cfg in ({"type": "adamw", "lr": 1e-4}, {"type": "sgd", "lr": 1e-5}):
            with self.assertRaises(RuntimeError):
                lower_lr_optimizer_config({"student": {"config": {"optimizer": cfg}}})

    def test_isolated_single_epoch(self):
        m = {"output_dir": "/tmp/example"}
        self.assertIn("m63e_frozen_bn_lr", str(checkpoint_for(m, VARIANT, 1)))
        with self.assertRaises(ValueError):
            checkpoint_for(m, VARIANT, 2)
        with self.assertRaises(ValueError):
            checkpoint_for(m, "vehicle_kd", 1)

    def test_notebook_current_section(self):
        nb = json.loads((ROOT / "notebooks/MonoDETR_M63_R0_A2_Depth_Pilot_Colab.ipynb").read_text())
        code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        for cell, action in zip(code[-3:], ("--smoke", "--train", "--evaluate")):
            ast.parse(cell)
            self.assertIn("run_m63_lower_lr_control.py", cell)
            self.assertIn(action, cell)
        self.assertIn("m63e_results.zip", code[-1])

if __name__ == "__main__":
    unittest.main()
