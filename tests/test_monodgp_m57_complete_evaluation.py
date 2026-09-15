from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from scripts.evaluate_monodgp_m57_deformable_attention import (
    M57_MANIFEST_SHA256,
    M57_SMOKE_SHA256,
    changed_config_paths,
)


ROOT = Path(__file__).resolve().parents[1]


class MonoDGPM57CompleteEvaluationTests(unittest.TestCase):
    def test_reviewed_evidence_hashes_are_frozen(self):
        self.assertEqual(len(M57_MANIFEST_SHA256), 64)
        self.assertEqual(len(M57_SMOKE_SHA256), 64)
        self.assertNotEqual(M57_MANIFEST_SHA256, M57_SMOKE_SHA256)

    def test_runtime_config_diff_is_recursive(self):
        before = {"model_name": "old", "trainer": {"save_path": "old", "max_epoch": 0}}
        after = {"model_name": "new", "trainer": {"save_path": "new", "max_epoch": 0}}
        self.assertEqual(
            changed_config_paths(before, after),
            {"model_name", "trainer.save_path"},
        )

    def test_evaluator_is_fail_closed_and_portable_only(self):
        source = (
            ROOT / "scripts/evaluate_monodgp_m57_deformable_attention.py"
        ).read_text()
        for marker in (
            "validate_reviewed_pair",
            "M57_MANIFEST_SHA256",
            "M57_SMOKE_SHA256",
            'os.environ[PORTABLE_ENV] = "1"',
            "PREDICTION_FILES = 3769",
            "preservation_gate_results",
            '"direct_coreml_conversion_authorized": False',
            '"deployment_authorized": False',
            '"product_safety_qualified": False',
        ):
            self.assertIn(marker, source)
        self.assertIn(
            'changed_config_paths(base, runtime) != {"model_name", "trainer.save_path"}',
            source,
        )

    def test_continuation_notebook_is_standalone_and_does_not_repeat_smoke(self):
        path = ROOT / "notebooks/MonoDGP_M57_Complete_Validation_Colab.ipynb"
        notebook = json.loads(path.read_text())
        code_cells = [
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell.get("cell_type") == "code"
        ]
        for code in code_cells:
            ast.parse(code)
        code = "\n".join(code_cells)
        self.assertIn("evaluate_monodgp_m57_deformable_attention.py", code)
        self.assertIn("m57_portable_attention_gate.json", code)
        self.assertIn("m57_portable_attention_comparison.csv", code)
        self.assertIn(M57_MANIFEST_SHA256, code)
        self.assertIn(M57_SMOKE_SHA256, code)
        self.assertNotIn("smoke_test_monodgp_m57_deformable_attention.py", code)
        self.assertNotIn("prepare_monodgp_m57_deformable_attention.py", code)
        self.assertNotIn("['tools/train_val.py','--config'", code)


if __name__ == "__main__":
    unittest.main()
