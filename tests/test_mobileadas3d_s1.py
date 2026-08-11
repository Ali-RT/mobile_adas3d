import unittest
from pathlib import Path

import torch

from models.build import build_model
from models.mobile_adas3d_s1 import S1_OUTPUT_NAMES, MobileADAS3DS1TupleWrapper
from tools.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "kitti_mobileadas3d_s1.yaml"


class MobileADAS3DS1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config(str(CONFIG_PATH))
        config["model"]["pretrained"] = False
        cls.model = build_model(config).eval()

    def test_locked_model_and_taxonomy(self):
        self.assertEqual(self.model.architecture_name, "MobileADAS3D-S1")
        self.assertEqual(self.model.num_classes, 2)
        self.assertEqual(self.model.output_stride, 8)
        self.assertEqual(self.model.fpn_channels, 96)

    def test_output_contract_is_stride_8(self):
        with torch.inference_mode():
            outputs = self.model(torch.rand(1, 3, 384, 1280))
        channels = {
            "cls_logits": 2,
            "quality": 1,
            "box2d": 4,
            "projected_center_offset": 2,
            "log_depth": 1,
            "depth_uncertainty": 1,
            "dim": 3,
            "yaw": 2,
            "yaw_axis": 2,
            "yaw_direction": 1,
            "loc_xy": 2,
        }
        self.assertEqual(set(outputs), set(S1_OUTPUT_NAMES) | {"yaw"})
        for name, count in channels.items():
            self.assertEqual(tuple(outputs[name].shape), (1, count, 48, 160))
        self.assertTrue(torch.all(outputs["box2d"] > 0))

    def test_parameter_gate_and_export_wrapper(self):
        parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertLessEqual(parameters, 10_000_000)
        wrapper = MobileADAS3DS1TupleWrapper(self.model).eval()
        with torch.inference_mode():
            outputs = wrapper(torch.rand(1, 3, 384, 1280))
        self.assertEqual(len(outputs), len(S1_OUTPUT_NAMES))

    def test_imagenet_normalization_is_embedded(self):
        self.assertEqual(tuple(self.model.input_mean.shape), (1, 3, 1, 1))
        self.assertAlmostEqual(float(self.model.input_mean[0, 0, 0, 0]), 0.485)


if __name__ == "__main__":
    unittest.main()
