import unittest

import numpy as np

from scripts.validate_monodgp_m58_macos_parity import (
    INPUT_SHAPES,
    OUTPUT_NAMES,
    OUTPUT_SHAPES,
    compare_candidates,
    compare_tensor_outputs,
    decode_candidates,
)


class MonoDGPM58MacOSParityTests(unittest.TestCase):
    def test_frozen_interface(self):
        self.assertEqual(INPUT_SHAPES["image"], [1, 3, 384, 1280])
        self.assertEqual(len(OUTPUT_NAMES), 10)
        self.assertEqual(set(OUTPUT_NAMES), set(OUTPUT_SHAPES))

    def test_raw_output_comparison_uses_region_family_limit(self):
        values = {
            name: np.zeros(shape, dtype=np.float32)
            for name, shape in OUTPUT_SHAPES.items()
        }
        prediction = {name: value.copy() for name, value in values.items()}
        prediction["pred_region_prob_2"][0, 0, 0, 0] = 0.009
        report = compare_tensor_outputs(values, prediction)
        self.assertTrue(report["pred_region_prob_2"]["passed"])
        prediction["pred_region_prob_2"][0, 0, 0, 0] = 0.011
        report = compare_tensor_outputs(values, prediction)
        self.assertFalse(report["pred_region_prob_2"]["passed"])

    def test_decoded_candidate_parity_is_deterministic(self):
        values = {
            "pred_logits": np.zeros((1, 50, 3), dtype=np.float32),
            "pred_boxes": np.zeros((1, 50, 6), dtype=np.float32),
            "pred_3d_dim": np.zeros((1, 50, 3), dtype=np.float32),
            "pred_depth": np.zeros((1, 50, 2), dtype=np.float32),
            "pred_angle": np.zeros((1, 50, 24), dtype=np.float32),
        }
        decoded = decode_candidates(values, topk=7)
        self.assertEqual(decoded.shape, (1, 7, 37))
        self.assertTrue(compare_candidates(decoded, decoded)["passed"])


if __name__ == "__main__":
    unittest.main()
