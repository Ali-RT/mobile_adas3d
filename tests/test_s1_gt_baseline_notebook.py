from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MobileADAS3D_S1_GT_Baseline_Colab.ipynb"


class S1GTBaselineNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_code_cells_parse(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_gate_is_gt_only_and_resume_safe(self):
        self.assertIn("prepare_s1_gt_baseline.py", self.code)
        self.assertIn("manifest['distillation_enabled'] is False", self.code)
        self.assertIn("check_training_ready.py", self.code)
        self.assertIn("checkpoints/latest.pt", self.code)
        self.assertIn("--resume", self.code)

    def test_continuation_requires_explicit_authorization(self):
        self.assertIn("AUTHORIZE_CONTINUATION = False", self.code)
        self.assertIn("Review epoch-20 gate", self.code)

    def test_gate_requires_complete_product_evaluation(self):
        self.assertIn("gate_summary['complete_split']", self.code)
        self.assertIn("gate_summary['evaluated_images']==3769", self.code)


if __name__ == "__main__":
    unittest.main()
