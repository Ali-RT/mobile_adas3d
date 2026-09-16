import ast
import json
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_monodgp_m59b_macos_diagnostics import (
    compare_diagnostic_tensors,
    first_diverging_tensor,
)


class MonoDGPM59bDiagnosticTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_first_divergence_preserves_causal_output_order(self):
        names = ["backbone_feat_0", "det2d_hs_last", "det3d_hs_last", "final_pred_logits"]
        reference = {name: np.zeros((1, 2), dtype=np.float32) for name in names}
        prediction = {name: value.copy() for name, value in reference.items()}
        prediction["det3d_hs_last"][0, 1] = 0.01
        comparison = compare_diagnostic_tensors(reference, prediction, names)
        self.assertEqual(first_diverging_tensor(comparison), "det3d_hs_last")
        self.assertTrue(comparison["backbone_feat_0"]["passed"])
        self.assertFalse(comparison["det3d_hs_last"]["passed"])

    def test_missing_or_shape_changed_tensor_fails_closed(self):
        reference = {"tap": np.zeros((1, 2), dtype=np.float32)}
        self.assertFalse(
            compare_diagnostic_tensors(reference, {}, ["tap"])["tap"]["passed"]
        )
        changed = {"tap": np.zeros((1, 3), dtype=np.float32)}
        row = compare_diagnostic_tensors(reference, changed, ["tap"])["tap"]
        self.assertFalse(row["passed"])
        self.assertIn("expected_shape", row)

    def test_nan_fails_even_when_delta_is_small(self):
        reference = {"tap": np.zeros((1, 1), dtype=np.float32)}
        prediction = {"tap": np.array([[np.nan]], dtype=np.float32)}
        row = compare_diagnostic_tensors(reference, prediction, ["tap"])["tap"]
        self.assertFalse(row["passed"])

    def test_exporter_and_notebook_are_standalone_and_fail_closed(self):
        exporter = (self.ROOT / "scripts/export_monodgp_m59b_coreml_diagnostics.py").read_text()
        validator = (self.ROOT / "scripts/validate_monodgp_m59b_macos_diagnostics.py").read_text()
        ast.parse(exporter)
        ast.parse(validator)
        self.assertIn("first_diverging_tensor", validator)
        self.assertIn("physical_device_testing_authorized", validator)
        self.assertIn("fp16_or_quantization_authorized", validator)
        notebook = json.loads(
            (self.ROOT / "notebooks/MonoDGP_M59b_CoreML_Intermediate_Diagnostic_Colab.ipynb").read_text()
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell.get("source", [])))
        self.assertIn("export_monodgp_m59b_coreml_diagnostics.py", code)
        self.assertIn("m59b_coreml_diagnostic_export_gate.json", code)
        self.assertNotIn("MLModel.predict", code)
        self.assertNotIn("train_val.py", code)


if __name__ == "__main__":
    unittest.main()
