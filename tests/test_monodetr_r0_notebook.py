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


if __name__ == "__main__":
    unittest.main()
