from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDETR_R0_Two_Class_Reference_Colab.ipynb"


class MonoDETRR0NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_all_code_cells_parse(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_resume_selects_only_valid_epoch_checkpoints(self):
        self.assertIn("checkpoint_epoch_*.pth", self.code)
        self.assertIn("payload.get('model_state') is None", self.code)
        self.assertIn("payload.get('optimizer_state') is None", self.code)
        self.assertIn("latest = max(valid_checkpoints", self.code)
        self.assertIn("run_cfg['trainer'].pop('pretrain_model', None)", self.code)
        self.assertIn("run_cfg['trainer']['resume_model']", self.code)

    def test_completed_run_does_not_restart(self):
        self.assertIn("if START_EPOCH >= MAX_EPOCHS", self.code)

    def test_training_failure_preserves_output_and_diagnostics(self):
        self.assertIn("stderr=subprocess.STDOUT", self.code)
        self.assertIn("PYTHONUNBUFFERED", self.code)
        self.assertIn("Durable combined log:", self.code)
        self.assertIn("Last captured lines:", self.code)
        self.assertIn("subprocess.run(['nvidia-smi'])", self.code)
        self.assertIn("subprocess.run(['df', '-h'", self.code)

    def test_product_checkpoint_sweep_is_wired(self):
        self.assertIn("sweep_monodetr_r0_product_checkpoints.py", self.code)
        self.assertIn("r0_product_selection.json", self.code)
        self.assertIn("r0_product_checkpoint_sweep.csv", self.code)

    def test_locked_epoch185_qualification_is_wired(self):
        self.assertIn("Expected frozen R0 epoch 185", self.code)
        self.assertIn("fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59", self.code)
        self.assertIn("Expected 3769 prediction files", self.code)
        self.assertIn("audit_product_prediction_geometry.py", self.code)
        self.assertIn("evaluate_yaw_diagnostics.py", self.code)
        self.assertIn("diagnose_a2_pedestrian_false_negatives.py", self.code)
        self.assertIn("score-threshold\",\"0.001", self.code.replace(" ", ""))
        self.assertIn("match-iou-threshold\",\"0.5", self.code.replace(" ", ""))


if __name__ == "__main__":
    unittest.main()
