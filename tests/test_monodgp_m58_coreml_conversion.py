import ast
import json
import unittest
from pathlib import Path

from scripts.export_monodgp_m58_coreml import (
    COMPUTE_PRECISION,
    COREMLTOOLS_VERSION,
    EXPECTED_OUTPUT_SHAPES,
    INPUT_SHAPES,
    M57_COMPARISON_SHA256,
    M57_GATE_SHA256,
    M57_PREDICTION_TREE_SHA256,
    MINIMUM_DEPLOYMENT_TARGET,
    OUTPUT_NAMES,
    REGION_OUTPUT_NAMES,
    SEMANTIC_OUTPUT_NAMES,
    flatten_export_outputs,
    output_family,
)


ROOT = Path(__file__).resolve().parents[1]


class MonoDGPM58CoreMLConversionTests(unittest.TestCase):
    def test_frozen_interface_and_evidence(self):
        self.assertEqual(COREMLTOOLS_VERSION, "9.0")
        self.assertEqual(MINIMUM_DEPLOYMENT_TARGET, "iOS17")
        self.assertEqual(COMPUTE_PRECISION, "FLOAT32")
        self.assertEqual(INPUT_SHAPES["image"], [1, 3, 384, 1280])
        self.assertEqual(len(SEMANTIC_OUTPUT_NAMES), 7)
        self.assertEqual(len(REGION_OUTPUT_NAMES), 4)
        self.assertEqual(len(OUTPUT_NAMES), 10)
        self.assertEqual(set(OUTPUT_NAMES), set(EXPECTED_OUTPUT_SHAPES))
        for value in (
            M57_GATE_SHA256,
            M57_COMPARISON_SHA256,
            M57_PREDICTION_TREE_SHA256,
        ):
            self.assertEqual(len(value), 64)

    def test_region_probability_pyramid_is_flattened_without_loss(self):
        values = {name: name for name in SEMANTIC_OUTPUT_NAMES[:-1]}
        values["pred_region_prob"] = [
            "region_0",
            "region_1",
            "region_2",
            "region_3",
        ]
        flattened = flatten_export_outputs(values)
        self.assertEqual(len(flattened), 10)
        self.assertEqual(flattened[-4:], tuple(values["pred_region_prob"]))
        self.assertEqual(
            [output_family(name) for name in REGION_OUTPUT_NAMES],
            ["pred_region_prob"] * 4,
        )

    def test_export_is_fixed_fp32_and_fail_closed(self):
        source = (ROOT / "scripts/export_monodgp_m58_coreml.py").read_text()
        ast.parse(source)
        self.assertIn('convert_to="mlprogram"', source)
        self.assertIn("minimum_deployment_target=ct.target.iOS17", source)
        self.assertIn("compute_precision=ct.precision.FLOAT32", source)
        self.assertIn("skip_model_load=True", source)
        self.assertIn('counts["resample"] > 0', source)
        self.assertIn('counts["custom"] == 0', source)
        self.assertIn("validate_export_outputs(source_outputs, torch)", source)
        self.assertIn("coreml_inplace_update_patch", source)
        self.assertIn('"deployment_authorized": False', source)
        self.assertIn('"product_safety_qualified": False', source)
        self.assertIn('"conversion_error"', source)

    def test_contract_preserves_all_later_barriers(self):
        contract = (ROOT / "MONODGP_M58_COREML_CONVERSION_CONTRACT.md").read_text()
        self.assertIn("Status: prepared", contract)
        self.assertIn("Stop point 1", contract)
        self.assertIn("macOS Core ML prediction-parity", contract)
        self.assertIn("physical-iPhone latency", contract)
        self.assertIn("FP16 Core ML conversion or quantization", contract)
        self.assertIn("the separate `0.80` target", contract)

    def test_notebook_is_standalone_and_stops_after_conversion(self):
        path = ROOT / "notebooks/MonoDGP_M58_CoreML_Conversion_Colab.ipynb"
        notebook = json.loads(path.read_text())
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell.get("source", [])))
        self.assertIn("coremltools==9.0", code)
        self.assertIn("patch_monodgp_m57_deformable_attention.py", code)
        self.assertIn("patch_monodgp_m58_coreml.py", code)
        self.assertIn("export_monodgp_m58_coreml.py", code)
        self.assertIn("m58_coreml_export_gate.json", code)
        self.assertNotIn("MLModel.predict", code)
        self.assertNotIn("train_val.py", code)


if __name__ == "__main__":
    unittest.main()
