import unittest

import torch

from data.target_builder import build_targets_for_sample
from losses.mobile_adas3d_loss import MobileADAS3DLoss


def make_dense_batch():
    def tensor(channels, value=0.0, requires_grad=False):
        return torch.full(
            (1, channels, 1, 1),
            value,
            dtype=torch.float32,
            requires_grad=requires_grad,
        )

    outputs = {
        "cls_logits": tensor(1, requires_grad=True),
        "box2d": tensor(4, requires_grad=True),
        "log_depth": tensor(1, 1.5, requires_grad=True),
        "dim": tensor(3, requires_grad=True),
        "yaw": torch.tensor(
            [[[[0.0]], [[1.0]]]], dtype=torch.float32, requires_grad=True
        ),
        "center_offset": tensor(2, requires_grad=True),
        "depth_uncertainty": tensor(1, requires_grad=True),
        "loc_xy": tensor(2, requires_grad=True),
    }
    targets = {
        "cls_target": tensor(1, 1.0),
        "box2d_target": tensor(4),
        "log_depth_target": tensor(1, 1.0),
        "dim_target": tensor(3),
        "yaw_target": torch.tensor([[[[0.0]], [[1.0]]]]),
        "offset_target": tensor(2),
        "loc_xy_target": tensor(2),
        "valid_mask": tensor(1, 1.0),
        "loss_weight_target": tensor(1, 1.0),
    }
    return outputs, targets


class TeacherDistillationIntegrationTests(unittest.TestCase):
    def test_disabled_path_is_numerically_identical(self):
        outputs, targets = make_dense_batch()
        baseline = MobileADAS3DLoss(384, 1280, classes=["Car"])
        explicitly_disabled = MobileADAS3DLoss(
            384,
            1280,
            classes=["Car"],
            distillation_enabled=False,
            teacher_depth_weight=99.0,
            teacher_dim_weight=99.0,
            teacher_loc_xy_weight=99.0,
            teacher_yaw_weight=99.0,
        )
        baseline_losses = baseline(outputs, targets)
        disabled_losses = explicitly_disabled(outputs, targets)
        self.assertEqual(set(baseline_losses), set(disabled_losses))
        for key in baseline_losses:
            self.assertTrue(torch.equal(baseline_losses[key], disabled_losses[key]))

    def test_enabled_teacher_losses_are_finite_and_backpropagate(self):
        outputs, targets = make_dense_batch()
        targets.update(
            {
                "teacher_valid_mask": torch.ones(1, 1, 1, 1),
                "teacher_score_target": torch.full((1, 1, 1, 1), 0.9),
                "teacher_log_depth_target": torch.full((1, 1, 1, 1), 2.0),
                "teacher_dim_target": torch.full((1, 3, 1, 1), 0.2),
                "teacher_loc_xy_target": torch.full((1, 2, 1, 1), 0.1),
                "teacher_yaw_target": torch.tensor([[[[1.0]], [[0.0]]]]),
            }
        )
        criterion = MobileADAS3DLoss(
            384,
            1280,
            classes=["Car"],
            distillation_enabled=True,
            teacher_depth_weight=0.25,
            teacher_dim_weight=0.25,
            teacher_loc_xy_weight=0.10,
            teacher_yaw_weight=0.25,
        )
        losses = criterion(outputs, targets)
        losses["total_loss"].backward()
        for key in (
            "teacher_depth_loss",
            "teacher_dim_loss",
            "teacher_loc_xy_loss",
            "teacher_yaw_loss",
        ):
            self.assertTrue(torch.isfinite(losses[key]))
            self.assertGreater(float(losses[key].detach()), 0.0)
        self.assertTrue(torch.isfinite(outputs["log_depth"].grad).all())
        self.assertTrue(torch.isfinite(outputs["dim"].grad).all())
        self.assertTrue(torch.isfinite(outputs["loc_xy"].grad).all())
        self.assertTrue(torch.isfinite(outputs["yaw"].grad).all())

    def test_target_builder_maps_teacher_to_gt_owned_cells(self):
        objects = [
            {
                "class_name": "Car",
                "bbox_2d": [0.0, 0.0, 16.0, 16.0],
                "location_3d": [0.0, 1.0, 10.0],
                "dimensions_3d": [1.5, 1.6, 3.9],
                "rotation_y": 0.0,
            }
        ]
        teacher = {
            "teacher_valid_mask": torch.tensor([True]),
            "teacher_score": torch.tensor([0.9]),
            "teacher_location_3d": torch.tensor([[0.5, 1.0, 12.0]]),
            "teacher_dimensions_3d": torch.tensor([[1.6, 1.7, 4.0]]),
            "teacher_yaw": torch.tensor([0.2]),
        }
        targets = build_targets_for_sample(
            objects,
            original_width=16,
            original_height=16,
            input_width=16,
            input_height=16,
            output_stride=16,
            classes=["Car"],
            class_mean_dims={"Car": [1.5, 1.6, 3.9]},
            center_sampling_radius=0,
            teacher_targets=teacher,
        )
        self.assertEqual(float(targets["teacher_valid_mask"][0, 0, 0]), 1.0)
        self.assertAlmostEqual(
            float(targets["teacher_log_depth_target"][0, 0, 0]),
            float(torch.log(torch.tensor(12.0))),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()
