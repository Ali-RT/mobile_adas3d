import ast
import unittest
from pathlib import Path

import numpy as np

from scripts.export_monodgp_m59d_2d_transformer import compare_tensors
from scripts.validate_monodgp_m59d_macos import compare_tensors as compare_macos


class MonoDGPM59dTransformerTests(unittest.TestCase):
    ROOT = Path(__file__).resolve().parents[1]

    def test_layer_order_and_first_divergence(self):
        names = ("enhanced_src_0", "det2d_encoder_layer_0", "det2d_decoder_layer_0", "det2d_hs_last")
        reference = {name: np.zeros((1, 2), dtype=np.float32) for name in names}
        candidate = {name: value.copy() for name, value in reference.items()}
        candidate["det2d_decoder_layer_0"][0, 1] = 0.01
        comparison = compare_tensors(reference, candidate, names)
        self.assertTrue(comparison["enhanced_src_0"]["passed"])
        self.assertTrue(comparison["det2d_encoder_layer_0"]["passed"])
        self.assertFalse(comparison["det2d_decoder_layer_0"]["passed"])

    def test_shape_and_missing_layer_fail_closed(self):
        reference = {"det2d_encoder_layer_0": np.zeros((1, 2), dtype=np.float32)}
        comparison = compare_tensors(reference, {}, tuple(reference))
        self.assertFalse(comparison["det2d_encoder_layer_0"]["passed"])
        changed = {"det2d_encoder_layer_0": np.zeros((1, 3), dtype=np.float32)}
        self.assertFalse(compare_macos(reference, changed, tuple(reference))["det2d_encoder_layer_0"]["passed"])

    def test_scripts_are_syntax_valid_and_deployment_closed(self):
        for relative in (
            "scripts/export_monodgp_m59d_2d_transformer.py",
            "scripts/validate_monodgp_m59d_macos.py",
        ):
            text = (self.ROOT / relative).read_text()
            ast.parse(text)
            self.assertIn("physical_device_testing_authorized", text)
            self.assertIn("fp16_or_quantization_authorized", text)


if __name__ == "__main__":
    unittest.main()
