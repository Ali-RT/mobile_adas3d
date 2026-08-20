import unittest
from pathlib import Path

import torch

from models.build import build_model
from models.mobile_adas3d_h1 import H1_OUTPUT_NAMES, MobileADAS3DH1TupleWrapper
from tools.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "kitti_mobileadas3d_h1.yaml"


class MobileADAS3DH1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config(str(CONFIG_PATH))
        config["model"]["pretrained"] = False
        cls.model = build_model(config).eval()
        cls.image = torch.rand(1, 3, 384, 1280)

    def test_locked_architecture_and_parameter_gate(self):
        self.assertEqual(self.model.architecture_name, "MobileADAS3D-H1")
        self.assertEqual(self.model.num_queries, 50)
        self.assertEqual(self.model.depth_bins, 40)
        self.assertEqual(self.model.transformer_width, 192)
        parameters = sum(parameter.numel() for parameter in self.model.parameters())
        self.assertLessEqual(parameters, 10_000_000)

    def test_fixed_query_output_contract(self):
        with torch.inference_mode():
            outputs = self.model(self.image)
        shapes = {
            "class_logits": (1, 50, 2),
            "box2d_cxcywh": (1, 50, 4),
            "projected_center": (1, 50, 2),
            "depth_logits": (1, 50, 40),
            "depth_residual": (1, 50, 1),
            "dimensions": (1, 50, 3),
            "yaw": (1, 50, 2),
            "location_xy": (1, 50, 2),
            "quality": (1, 50, 1),
        }
        self.assertEqual(tuple(outputs), H1_OUTPUT_NAMES)
        for name, shape in shapes.items():
            self.assertEqual(tuple(outputs[name].shape), shape)
            self.assertTrue(torch.isfinite(outputs[name]).all())
        self.assertTrue(torch.all((outputs["box2d_cxcywh"] >= 0.0) & (outputs["box2d_cxcywh"] <= 1.0)))
        self.assertTrue(torch.all((outputs["projected_center"] >= 0.0) & (outputs["projected_center"] <= 1.0)))

    def test_query_heads_use_neutral_initialization(self):
        for name in H1_OUTPUT_NAMES:
            head_name = {
                "class_logits": "class_head",
                "box2d_cxcywh": "box2d_head",
                "projected_center": "projected_center_head",
                "depth_logits": "query_depth_head",
                "depth_residual": "depth_residual_head",
                "dimensions": "dimensions_head",
                "location_xy": "location_xy_head",
            }.get(name, f"{name}_head")
            head = getattr(self.model, head_name)
            self.assertTrue(torch.equal(head.bias, torch.zeros_like(head.bias)))
            self.assertLess(float(head.weight.detach().std()), 0.0012)

    def test_tuple_wrapper_matches_named_outputs(self):
        wrapper = MobileADAS3DH1TupleWrapper(self.model).eval()
        with torch.inference_mode():
            named = self.model(self.image)
            values = wrapper(self.image)
        self.assertEqual(len(values), len(H1_OUTPUT_NAMES))
        for name, value in zip(H1_OUTPUT_NAMES, values):
            self.assertTrue(torch.equal(value, named[name]))

    def test_full_graph_backward_is_finite(self):
        model = self.model
        model.zero_grad(set_to_none=True)
        outputs = model(self.image)
        loss = sum(value.square().mean() for value in outputs.values())
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        self.assertGreater(len(gradients), 0)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


if __name__ == "__main__":
    unittest.main()
