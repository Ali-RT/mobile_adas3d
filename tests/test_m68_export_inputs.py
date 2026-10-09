"""Exercise the exporter's direct KITTI NumPy input contract with real tensors."""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_monodetr_a2_onnx import input_tensors
from m68_onnx_common import INPUT_SHAPES


class M68ExportInputTests(unittest.TestCase):
    def sample(self, *, tensors=False):
        # MonoDETR normalizes HWC arrays then transposes to CHW. This array is
        # intentionally non-contiguous, just like direct KITTI_Dataset output.
        image = np.full((384, 1280, 3), 0.25, dtype=np.float64).transpose(2, 0, 1)
        calibration = np.arange(12, dtype=np.float64).reshape(3, 4)
        size = np.array([1242, 375], dtype=np.int64)
        if tensors:
            image, calibration, size = map(torch.as_tensor, (image, calibration, size))
        return image, calibration, {}, {"img_id": 1, "img_size": size}

    def test_numpy_dataset_inputs_are_batched_finite_fp32_tensors(self):
        sample = self.sample()
        self.assertFalse(sample[0].flags.c_contiguous)
        values = input_tensors([sample], 0, "000001", device="cpu")
        for name, tensor in values.items():
            self.assertIsInstance(tensor, torch.Tensor)
            self.assertEqual(tuple(tensor.shape), INPUT_SHAPES[name])
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tensor.device.type, "cpu")
            self.assertTrue(torch.isfinite(tensor).all().item())
        torch.testing.assert_close(values["image"], torch.full(INPUT_SHAPES["image"], 0.25))
        torch.testing.assert_close(values["calibration"][0], torch.arange(12, dtype=torch.float32).reshape(3, 4))
        torch.testing.assert_close(values["image_size"], torch.tensor([[1242., 375.]]))

    def test_already_tensor_inputs_follow_the_same_contract(self):
        arrays = input_tensors([self.sample()], 0, "000001", device="cpu")
        tensors = input_tensors([self.sample(tensors=True)], 0, "000001", device="cpu")
        for name in arrays:
            torch.testing.assert_close(tensors[name], arrays[name])

    def test_wrong_sample_order_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "Fixed validation order changed"):
            input_tensors([self.sample()], 0, "000002", device="cpu")

    def test_wrong_input_shape_is_rejected(self):
        image, calibration, targets, info = self.sample()
        with self.assertRaisesRegex(RuntimeError, "Unexpected calibration input shape"):
            input_tensors([(image, calibration[:, :3], targets, info)], 0, "000001", device="cpu")

    def test_nonfinite_input_is_rejected(self):
        image, calibration, targets, info = self.sample()
        calibration[0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Unexpected calibration input shape or values"):
            input_tensors([(image, calibration, targets, info)], 0, "000001", device="cpu")


if __name__ == "__main__":
    unittest.main()
