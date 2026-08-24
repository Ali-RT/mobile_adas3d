import unittest
from pathlib import Path

import torch

from models.build import build_model
from models.mobile_adas3d_h1 import H1_OUTPUT_NAMES
from models.mobile_adas3d_h2 import (
    MobileADAS3DH2TupleWrapper,
    fixed_query_reference_grid,
)
from tools.config import load_config


ROOT = Path(__file__).resolve().parents[1]
H1_CONFIG = ROOT / "configs" / "kitti_mobileadas3d_h1.yaml"
H2_CONFIG = ROOT / "configs" / "kitti_mobileadas3d_h2.yaml"


class MobileADAS3DH2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config(str(H2_CONFIG))
        config["model"]["pretrained"] = False
        cls.model = build_model(config).eval()
        cls.image = torch.rand(1, 3, 384, 1280)

    def test_reference_grid_is_fixed_row_major_and_complete(self):
        grid = fixed_query_reference_grid()
        self.assertEqual(tuple(grid.shape), (1, 50, 2))
        self.assertTrue(torch.allclose(grid[0, 0], torch.tensor([0.05, 0.10])))
        self.assertTrue(torch.allclose(grid[0, 9], torch.tensor([0.95, 0.10])))
        self.assertTrue(torch.allclose(grid[0, 49], torch.tensor([0.95, 0.90])))
        self.assertEqual(torch.unique(grid[0], dim=0).shape[0], 50)

    def test_h2_preserves_export_contract_with_spatial_centers(self):
        with torch.inference_mode():
            outputs = self.model(self.image)
        self.assertEqual(self.model.architecture_name, "MobileADAS3D-H2")
        self.assertEqual(tuple(outputs), H1_OUTPUT_NAMES)
        expected_shapes = {
            "class_logits": (1, 50, 2), "box2d_cxcywh": (1, 50, 4),
            "projected_center": (1, 50, 2), "depth_logits": (1, 50, 40),
            "depth_residual": (1, 50, 1), "dimensions": (1, 50, 3),
            "yaw": (1, 50, 2), "location_xy": (1, 50, 2), "quality": (1, 50, 1),
        }
        for name, shape in expected_shapes.items():
            self.assertEqual(tuple(outputs[name].shape), shape)
            self.assertTrue(torch.isfinite(outputs[name]).all())
        references = self.model.query_reference_points
        center_error = (outputs["box2d_cxcywh"][..., :2] - references).abs()
        projected_error = (outputs["projected_center"] - references).abs()
        self.assertLessEqual(float(center_error.max()), self.model.center_offset_scale + 1e-6)
        self.assertLessEqual(float(projected_error.max()), self.model.center_offset_scale + 1e-6)
        self.assertGreater(torch.unique(outputs["box2d_cxcywh"][0, :, :2], dim=0).shape[0], 40)

    def test_tuple_wrapper_and_backward_are_finite(self):
        wrapper = MobileADAS3DH2TupleWrapper(self.model).eval()
        named = self.model(self.image)
        values = wrapper(self.image)
        for name, value in zip(H1_OUTPUT_NAMES, values):
            self.assertTrue(torch.equal(value, named[name]))
        self.model.zero_grad(set_to_none=True)
        sum(value.square().mean() for value in named.values()).backward()
        gradients = [p.grad for p in self.model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(g).all() for g in gradients))

    def test_h1_config_still_builds_unanchored_h1(self):
        config = load_config(str(H1_CONFIG))
        config["model"]["pretrained"] = False
        model = build_model(config)
        self.assertEqual(model.architecture_name, "MobileADAS3D-H1")
        self.assertFalse(hasattr(model, "query_reference_points"))


if __name__ == "__main__":
    unittest.main()
