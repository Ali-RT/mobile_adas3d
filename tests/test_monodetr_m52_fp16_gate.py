from __future__ import annotations

import ast
import json
import unittest
from pathlib import Path

from scripts.evaluate_monodetr_m52_fp16_gate import MAX_LOCALIZATION, MINIMUMS, R0
from scripts.prepare_monodetr_m52_fp16_gate import R0_EPOCH, R0_SHA256


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDETR_M52_R0_FP16_Gate_Colab.ipynb"


class MonoDETRM52FP16GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join("".join(cell["source"]) for cell in cls.notebook["cells"] if cell["cell_type"] == "code")

    def test_notebook_cells_parse(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_frozen_provenance(self):
        self.assertEqual(R0_EPOCH, 185)
        self.assertEqual(R0_SHA256, "fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59")
        self.assertIn("m52_fp16_gate_manifest.json", self.code)
        self.assertIn("training_authorized'] is False", self.code)

    def test_contract_thresholds(self):
        self.assertAlmostEqual(MINIMUMS["vehicle_3d_moderate"], R0["vehicle_3d_moderate"] * 0.95)
        self.assertAlmostEqual(MINIMUMS["pedestrian_near_recall"], R0["pedestrian_near_recall"] - 0.01)
        self.assertAlmostEqual(MAX_LOCALIZATION, R0["pedestrian_localization_failure_rate"] + 0.01)

    def test_isolated_fp16_evaluation_only_workflow(self):
        self.assertIn("/compression/monodetr_m52_r0_fp16_gate", self.code)
        self.assertIn("patch_monodetr_m52_fp16_eval.py", self.code)
        self.assertIn("smoke_test_monodetr_m52_fp16.py", self.code)
        self.assertIn("evaluate_monodetr_m52_fp16_gate.py", self.code)
        self.assertNotIn("tools/train_val.py", self.code)

    def test_fp32_stability_islands_are_instrumented(self):
        source = (ROOT / "scripts/patch_monodetr_m52_fp16_eval.py").read_text(encoding="utf-8")
        self.assertIn("torch.is_autocast_enabled()", source)
        self.assertIn("features, pos = m52_fp32(self.backbone, images)", source)
        self.assertIn("m52_fp32(self.input_proj[l]", source)
        self.assertIn("last_backbone_feature_dtypes", source)
        self.assertIn("last_projected_feature_dtypes", source)
        self.assertIn("depth_features = [feature.float() for feature in srcs]", source)
        self.assertIn("last_depth_predictor_dtype", source)
        self.assertIn("M52 depth-predictor FP32 island", source)
        self.assertIn("query.float()", source)
        self.assertIn("last_kernel_dtype", source)
        self.assertIn("M52 deformable-attention FP32 islands", source)

    def test_cuda_smoke_precedes_complete_evaluation(self):
        smoke = self.code.index("smoke_test_monodetr_m52_fp16.py")
        evaluation = self.code.index("evaluate_monodetr_m52_fp16_gate.py")
        self.assertLess(smoke, evaluation)
        self.assertIn("depth_predictor_dtype", self.code)
        self.assertIn("backbone_feature_dtypes", self.code)
        self.assertIn("projected_feature_dtypes", self.code)
        self.assertIn("deformable_attention_kernel_dtypes", self.code)
        self.assertIn("torch.float32", self.code)

    def test_complete_gate_is_fail_closed(self):
        source = (ROOT / "scripts/evaluate_monodetr_m52_fp16_gate.py").read_text(encoding="utf-8")
        self.assertIn('len(list(prediction.glob("*.txt"))) == 3769', source)
        self.assertIn('"compression_rung_authorized": all(gates.values())', source)
        self.assertIn('"product_safety_qualified": False', source)


if __name__ == "__main__":
    unittest.main()
