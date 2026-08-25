from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_monodetr_a1_distillation import EXPECTED_A1, validate_a1_selection


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDETR_A1_Two_Class_Distillation_Colab.ipynb"
LOSS_TEMPLATE = ROOT / "third_party/monodetr/a1_distillation_loss.py"
PATCHER = ROOT / "scripts/patch_monodetr_a1_distillation.py"


class MonoDETRA1DistillationWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )
        cls.loss_source = LOSS_TEMPLATE.read_text(encoding="utf-8")
        cls.patch_source = PATCHER.read_text(encoding="utf-8")

    def test_all_python_sources_parse(self):
        ast.parse(self.loss_source)
        ast.parse(self.patch_source)
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_notebook_locks_paired_provenance_and_smoke_gate(self):
        self.assertIn("A1_MANIFEST", self.code)
        self.assertIn("A1_SELECTION", self.code)
        self.assertIn("prepare_monodetr_a1_distillation.py", self.code)
        self.assertIn("smoke_test_monodetr_a1_distillation.py", self.code)
        self.assertIn("approved_query_pairs", self.code)
        self.assertIn("optimizer_steps", self.code)
        self.assertIn("paired_gt_baseline_selection", self.code)

    def test_notebook_has_exact_resume_sweep_and_comparison(self):
        self.assertIn("checkpoint_epoch_*.pth", self.code)
        self.assertIn("payload.get('optimizer_state')", self.code)
        self.assertIn("latest = max(valid_checkpoints", self.code)
        self.assertIn("PYTHONUNBUFFERED", self.code)
        self.assertIn("sweep_monodetr_a1_product_checkpoints.py", self.code)
        self.assertIn("a1_distillation_vs_gt_comparison.json", self.code)

    def test_query_distillation_matches_teacher_and_student_through_gt(self):
        self.assertIn("student_indices = matcher", self.loss_source)
        self.assertIn("teacher_indices = matcher", self.loss_source)
        self.assertIn("teacher_by_gt", self.loss_source)
        self.assertIn("min_teacher_score", self.loss_source)
        self.assertIn("min_teacher_iou_2d", self.loss_source)
        for loss in ("logits", "boxes", "depth", "dims", "angles"):
            self.assertIn(f'"distill_{loss}"', self.loss_source)

    def test_patch_freezes_teacher_and_keeps_it_out_of_optimizer(self):
        self.assertIn("teacher_model.eval()", self.patch_source)
        self.assertIn("parameter.requires_grad_(False)", self.patch_source)
        self.assertIn("with torch.no_grad()", self.patch_source)
        self.assertIn("optimizer = build_optimizer(cfg['optimizer'], model)", self.patch_source)

    def test_a1_selection_validation_is_fail_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint_epoch_140.pth"
            checkpoint.touch()
            report = {
                "complete": True,
                "selected_epoch": 140,
                "selected_checkpoint": str(checkpoint),
                "metrics": {
                    key: value for key, value in EXPECTED_A1.items() if key != "selected_epoch"
                },
            }
            path = root / "a1_product_selection.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(validate_a1_selection(path), report)
            report["metrics"]["vehicle_3d_moderate"] += 0.01
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "A1 metric mismatch"):
                validate_a1_selection(path)


if __name__ == "__main__":
    unittest.main()
