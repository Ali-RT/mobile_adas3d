import unittest
from pathlib import Path

import torch

from data.h1_query_targets import (
    build_h1_query_targets_for_sample,
    pad_h1_query_targets,
)
from losses.h1_set_loss import H1SetCriterion
from models.decode import decode_mobile_adas3d_outputs
from scripts.train_mobile_adas3d import build_criterion
from tools.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT_ROOT / "configs" / "kitti_mobileadas3d_h1_gt_gate.yaml"


class H1SetTrainingTests(unittest.TestCase):
    def test_query_target_geometry_and_padding(self):
        target = build_h1_query_targets_for_sample(
            objects=[{
                "class_name": "Vehicle",
                "bbox_2d": [100.0, 50.0, 300.0, 250.0],
                "dimensions_3d": [1.6, 1.7, 4.2],
                "location_3d": [1.0, 1.5, 20.0],
                "rotation_y": 0.25,
            }],
            original_width=1000,
            original_height=500,
            classes=["Vehicle", "Pedestrian"],
            class_mean_dims={"Vehicle": [1.6, 1.7, 4.2], "Pedestrian": [1.7, 0.6, 0.8]},
            P2=[[700.0, 0.0, 500.0, 0.0], [0.0, 700.0, 250.0, 0.0], [0.0, 0.0, 1.0, 0.0]],
            depth_bins=40,
            min_depth_m=1.0,
            max_depth_m=80.0,
        )
        padded = pad_h1_query_targets([target])
        self.assertEqual(padded["object_mask"].tolist(), [[True]])
        self.assertTrue(torch.allclose(padded["box2d"][0, 0], torch.tensor([0.2, 0.3, 0.2, 0.4])))
        self.assertTrue(torch.allclose(padded["dimensions"][0, 0], torch.zeros(3), atol=1e-6))
        self.assertTrue(padded["projected_center_valid"][0, 0])

    def test_empty_query_target_is_padded_safely(self):
        empty = build_h1_query_targets_for_sample(
            objects=[], original_width=1000, original_height=500,
            classes=["Vehicle", "Pedestrian"],
            class_mean_dims={"Vehicle": [1.6, 1.7, 4.2], "Pedestrian": [1.7, 0.6, 0.8]},
            P2=torch.eye(3, 4), depth_bins=40, min_depth_m=1.0, max_depth_m=80.0,
        )
        padded = pad_h1_query_targets([empty])
        self.assertEqual(tuple(padded["class_ids"].shape), (1, 1))
        self.assertFalse(padded["object_mask"].any())

    def test_hungarian_loss_is_finite_and_backpropagates(self):
        torch.manual_seed(3)
        outputs = {
            "class_logits": torch.randn(1, 4, 2, requires_grad=True),
            "box2d_cxcywh": torch.rand(1, 4, 4, requires_grad=True),
            "projected_center": torch.rand(1, 4, 2, requires_grad=True),
            "depth_logits": torch.randn(1, 4, 40, requires_grad=True),
            "depth_residual": torch.randn(1, 4, 1, requires_grad=True),
            "dimensions": torch.randn(1, 4, 3, requires_grad=True),
            "yaw": torch.randn(1, 4, 2, requires_grad=True),
            "location_xy": torch.randn(1, 4, 2, requires_grad=True),
            "quality": torch.randn(1, 4, 1, requires_grad=True),
        }
        targets = {
            "object_mask": torch.tensor([[True, False]]),
            "class_ids": torch.tensor([[0, 0]]),
            "box2d": torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.0, 0.0, 0.0, 0.0]]]),
            "projected_center": torch.tensor([[[0.5, 0.6], [0.0, 0.0]]]),
            "projected_center_valid": torch.tensor([[True, False]]),
            "depth_bin": torch.tensor([[20, 0]]),
            "depth_residual": torch.zeros(1, 2),
            "dimensions": torch.zeros(1, 2, 3),
            "yaw": torch.tensor([[[0.0, 1.0], [0.0, 0.0]]]),
            "location_xy": torch.zeros(1, 2, 2),
        }
        criterion = H1SetCriterion(2, [1.0, 2.5])
        losses = criterion(outputs, targets)
        self.assertTrue(torch.isfinite(losses["total_loss"]))
        losses["total_loss"].backward()
        self.assertTrue(all(value.grad is not None for value in outputs.values()))
        self.assertTrue(all(torch.isfinite(value.grad).all() for value in outputs.values()))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA regression runs in GPU CI/Colab")
    def test_cpu_constructed_criterion_accepts_cuda_tensors(self):
        device = torch.device("cuda")
        outputs = {
            "class_logits": torch.randn(1, 2, 2, device=device, requires_grad=True),
            "box2d_cxcywh": torch.rand(1, 2, 4, device=device, requires_grad=True),
            "projected_center": torch.rand(1, 2, 2, device=device, requires_grad=True),
            "depth_logits": torch.randn(1, 2, 40, device=device, requires_grad=True),
            "depth_residual": torch.randn(1, 2, 1, device=device, requires_grad=True),
            "dimensions": torch.randn(1, 2, 3, device=device, requires_grad=True),
            "yaw": torch.randn(1, 2, 2, device=device, requires_grad=True),
            "location_xy": torch.randn(1, 2, 2, device=device, requires_grad=True),
            "quality": torch.randn(1, 2, 1, device=device, requires_grad=True),
        }
        targets = {
            "object_mask": torch.tensor([[False]], device=device),
            "class_ids": torch.zeros(1, 1, dtype=torch.long, device=device),
            "box2d": torch.zeros(1, 1, 4, device=device),
            "projected_center": torch.zeros(1, 1, 2, device=device),
            "projected_center_valid": torch.tensor([[False]], device=device),
            "depth_bin": torch.zeros(1, 1, dtype=torch.long, device=device),
            "depth_residual": torch.zeros(1, 1, device=device),
            "dimensions": torch.zeros(1, 1, 3, device=device),
            "yaw": torch.zeros(1, 1, 2, device=device),
            "location_xy": torch.zeros(1, 1, 2, device=device),
        }
        criterion = H1SetCriterion(2, [1.0, 2.5])
        loss = criterion(outputs, targets)["total_loss"]
        self.assertEqual(loss.device.type, "cuda")
        self.assertTrue(torch.isfinite(loss))

    def test_gate_config_selects_gt_only_h1_criterion(self):
        config = load_config(str(CONFIG))
        self.assertEqual(config["model"]["name"], "MobileADAS3D-H1")
        self.assertFalse(config["distillation"]["enabled"])
        self.assertEqual(config["training"]["epochs"], 20)
        self.assertIsInstance(build_criterion(config), H1SetCriterion)

    def test_query_decoder_emits_kitti_geometry(self):
        outputs = {
            "class_logits": torch.tensor([[[8.0, -8.0]]]),
            "box2d_cxcywh": torch.tensor([[[0.5, 0.5, 0.2, 0.4]]]),
            "projected_center": torch.tensor([[[0.5, 0.6]]]),
            "depth_logits": torch.zeros(1, 1, 40),
            "depth_residual": torch.zeros(1, 1, 1),
            "dimensions": torch.zeros(1, 1, 3),
            "yaw": torch.tensor([[[0.0, 1.0]]]),
            "location_xy": torch.zeros(1, 1, 2),
            "quality": torch.tensor([[[8.0]]]),
        }
        outputs["depth_logits"][0, 0, 20] = 8.0
        decoded = decode_mobile_adas3d_outputs(
            outputs=outputs,
            classes=["Vehicle", "Pedestrian"],
            class_mean_dims={"Vehicle": [1.6, 1.7, 4.2], "Pedestrian": [1.7, 0.6, 0.8]},
            input_height=384,
            input_width=1280,
            score_threshold=0.01,
            topk=2,
            P2=torch.tensor([[700.0, 0.0, 640.0, 0.0], [0.0, 700.0, 192.0, 0.0], [0.0, 0.0, 1.0, 0.0]]),
        )[0]
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["class_name"], "Vehicle")
        self.assertAlmostEqual(decoded[0]["bbox_2d"][0], 512.0, places=3)
        self.assertGreater(decoded[0]["depth"], 1.0)
        self.assertEqual(decoded[0]["dimensions_3d_hwl"], [1.600000023841858, 1.7000000476837158, 4.199999809265137])


if __name__ == "__main__":
    unittest.main()
