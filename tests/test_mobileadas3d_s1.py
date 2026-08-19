import unittest
from pathlib import Path

import torch

from losses.mobile_adas3d_loss import (
    MobileADAS3DLoss,
    masked_weighted_yaw_cosine_loss,
    normalize_yaw_with_floor,
)
from models.build import build_model
from models.mobile_adas3d_s1 import (
    S1_OUTPUT_NAMES,
    S1_V2_OUTPUT_NAMES,
    MobileADAS3DS1TupleWrapper,
)
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
            "center_offset": 2,
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


class MobileADAS3DS1V2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config(
            str(PROJECT_ROOT / "configs" / "kitti_mobileadas3d_s1_v2_continuous_yaw.yaml")
        )
        config["model"]["pretrained"] = False
        cls.model = build_model(config).eval()

    def test_continuous_yaw_is_the_only_yaw_head(self):
        self.assertEqual(self.model.architecture_name, "MobileADAS3D-S1-V2")
        self.assertEqual(self.model.yaw_encoding, "continuous_sincos")
        with torch.inference_mode():
            outputs = self.model(torch.rand(1, 3, 384, 1280))
        self.assertEqual(set(outputs), set(S1_V2_OUTPUT_NAMES))
        self.assertNotIn("yaw_axis", outputs)
        self.assertNotIn("yaw_direction", outputs)
        self.assertTrue(torch.isfinite(outputs["yaw"]).all())

    def test_v2_export_wrapper_uses_ten_heads(self):
        wrapper = MobileADAS3DS1TupleWrapper(self.model).eval()
        with torch.inference_mode():
            outputs = wrapper(torch.rand(1, 3, 384, 1280))
        self.assertEqual(len(outputs), len(S1_V2_OUTPUT_NAMES))

    def test_v2_zero_yaw_corner_loss_has_finite_bounded_gradient(self):
        def leaf(channels, value=0.0):
            return torch.full(
                (1, channels, 1, 1),
                value,
                dtype=torch.float32,
                requires_grad=True,
            )

        yaw = leaf(2)
        outputs = {
            "cls_logits": leaf(2),
            "box2d": leaf(4, 1.0),
            "log_depth": leaf(1, 2.0),
            "dim": leaf(3),
            "yaw": yaw,
            "center_offset": leaf(2),
            "depth_uncertainty": leaf(1),
            "loc_xy": leaf(2),
        }
        targets = {
            "cls_target": torch.tensor([[[[1.0]], [[0.0]]]]),
            "box2d_target": torch.ones(1, 4, 1, 1),
            "log_depth_target": torch.full((1, 1, 1, 1), 2.0),
            "dim_target": torch.zeros(1, 3, 1, 1),
            "yaw_target": torch.tensor([[[[0.0]], [[1.0]]]]),
            "offset_target": torch.zeros(1, 2, 1, 1),
            "loc_xy_target": torch.zeros(1, 2, 1, 1),
            "location_xyz_target": torch.tensor(
                [[[[0.0]], [[0.0]], [[7.3891]]]]
            ),
            "valid_mask": torch.ones(1, 1, 1, 1),
            "loss_weight_target": torch.ones(1, 1, 1, 1),
        }
        criterion = MobileADAS3DLoss(
            input_height=384,
            input_width=1280,
            classes=["Vehicle", "Pedestrian"],
            class_mean_dims={
                "Vehicle": [1.5, 1.6, 3.9],
                "Pedestrian": [1.7, 0.6, 0.8],
            },
            yaw_weight=2.0,
            yaw_cosine_weight=1.0,
            corner3d_weight=0.5,
            detach_yaw_in_corner3d=True,
            yaw_pred_is_direct_sincos=True,
        )

        losses = criterion(outputs, targets)
        losses["total_loss"].backward()

        self.assertTrue(torch.isfinite(losses["total_loss"]))
        self.assertTrue(torch.isfinite(yaw.grad).all())
        self.assertLessEqual(float(yaw.grad.abs().max()), 30.0)

    def test_v2b_yaw_loss_is_bounded_across_vector_scales(self):
        target = torch.tensor([[[[0.0]], [[1.0]]]])
        mask = torch.ones(1, 1, 1, 1)
        losses = []
        for magnitude in (0.0, 1e-6, 0.1, 1.0, 100.0):
            raw = torch.tensor(
                [[[[magnitude]], [[0.0]]]], requires_grad=True
            )
            normalized = normalize_yaw_with_floor(raw, norm_floor=0.1)
            loss = masked_weighted_yaw_cosine_loss(
                normalized,
                target,
                mask,
                pred_is_normalized=True,
            )
            loss.backward()
            loss_value = float(loss.detach())
            self.assertGreaterEqual(loss_value, 0.0)
            self.assertLessEqual(loss_value, 2.0)
            self.assertTrue(torch.isfinite(raw.grad).all())
            self.assertLessEqual(float(raw.grad.abs().max()), 10.0)
            losses.append(loss_value)
        self.assertAlmostEqual(losses[-1], losses[-2], places=6)


if __name__ == "__main__":
    unittest.main()
