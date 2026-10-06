"""CPU tests for same-object pairing and honest M65 gradient evidence."""
import copy
import sys
from pathlib import Path
import unittest
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_m65_preservation_regression as d


def row(sample="000001", depth=10., matched=True, **changes):
    result = dict(sample_id=sample, class_name="Pedestrian", gt_depth_m=depth,
                  near_field=True, distance_bucket="00_20m", matched=matched,
                  gt_yaw_rad=0., size_bucket="medium_h_32_96px")
    result.update({name: 0. for name in d.FIELDS})
    result.update(score=.5, iou_2d=.8, iou_3d=.6, iou_bev=.7)
    result.update(changes)
    return result


class RegressionDiagnosticTests(unittest.TestCase):
    def test_pair_by_gt_not_row_order(self):
        a = [row(), row(depth=20.)]
        b = [copy.deepcopy(a[1]), copy.deepcopy(a[0])]
        left, right, coverage = d.pair_records(a, b)
        self.assertEqual(left, right)
        self.assertEqual(coverage["paired_gt"], 2)

    def test_ambiguous_depth_excluded_even_if_one_is_missed(self):
        a = [row(), row(matched=False)]
        b = [row(matched=False), row()]
        left, right, coverage = d.pair_records(a, b)
        self.assertEqual(left, {})
        self.assertEqual(right, {})
        self.assertEqual(coverage["excluded_gt"], 2)

    def test_gt_universe_and_metadata_must_agree(self):
        with self.assertRaisesRegex(ValueError, "multisets"):
            d.pair_records([row()], [row(depth=11.)])
        with self.assertRaisesRegex(ValueError, "metadata"):
            d.pair_records([row()], [row(gt_yaw_rad=1.)])

    def test_misses_and_geometry_crossings_remain_distinct(self):
        a = [row(), row(depth=20.), row(depth=30., matched=False)]
        b = [row(iou_3d=.4), row(depth=20., matched=False), row(depth=30., iou_3d=.7)]
        left, right, _ = d.pair_records(a, b)
        report = d.summarize_pairs(left, right, "Pedestrian")
        self.assertEqual(report["common_matched"], 1)
        self.assertEqual(report["lost_2d_matches"], 1)
        self.assertEqual(report["gained_2d_matches"], 1)
        crossing = report["overlap_threshold_crossings"]["iou_3d"]
        self.assertEqual(crossing["common_lost"], 1)
        self.assertEqual(crossing["common_gained"], 0)
        self.assertEqual(crossing["baseline_success_lost_to_2d_miss"], 1)
        self.assertEqual(crossing["control_success_gained_from_2d_miss"], 1)

    def test_empty_population_is_unavailable_not_zero(self):
        report = d.summarize_pairs({}, {}, "Vehicle")
        self.assertIsNone(report["geometry_on_common_objects"]["score"]["delta_mean"])

    def test_combined_norm_does_not_invent_component_norms(self):
        result = d.logged_gradients("grad_norm=10.0\ngrad_norm=20.0")
        self.assertEqual(result["combined_preclip_norm_median"], 15.)
        self.assertFalse(result["component_gradient_norms_recorded"])
        self.assertFalse(result["gt_preservation_gradient_cosine_recorded"])

    def test_nonfinite_and_invalid_boolean_rejected(self):
        with self.assertRaises(ValueError):
            d.number("nan")
        with self.assertRaises(ValueError):
            d.boolean("yes")

    def test_native_ranking_score_can_exceed_one(self):
        source = {k: str(v) for k, v in row(score=3.5).items() if k != "matched"}
        source["pred_yaw_rad"] = "0.0"
        source["split"] = "val"
        self.assertEqual(d.records([source], [])[0]["score"], 3.5)
        source["iou_3d"] = "1.01"
        with self.assertRaises(ValueError):
            d.records([source], [])

    def test_notebook_is_self_contained_and_cannot_train(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root / "notebooks/MonoDETR_A2_M65b_Gradient_Diagnostic_Colab.ipynb").read_text())
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code), 3)
        for index, source in enumerate(code):
            compile(source, f"m65b_cell_{index + 1}", "exec")
        self.assertIn("from collections import deque", code[0])
        self.assertIn("import json, os, shlex, subprocess, sys, runpy, zipfile", code[0])
        self.assertIn("https://github.com/ZrrSkywalker/MonoDETR.git", code[1])
        self.assertNotIn("git','reset", "".join(code))
        self.assertNotIn("workflow('train'", "".join(code))
        self.assertIn("optimizer_steps'] == 0", code[2])


class GradientProbeTests(unittest.TestCase):
    def setUp(self):
        try:
            import torch
        except ImportError:
            self.skipTest("Tensor tests require the project PyTorch runtime")
        self.torch = torch
        import probe_m65_preservation_gradients as probe
        self.probe = probe

    def test_aggregate_gradient_norm_and_cosine(self):
        a = [self.torch.tensor([3., 0.]), self.torch.tensor([4.])]
        b = [self.torch.tensor([-3., 0.]), self.torch.tensor([-4.])]
        report = self.probe.gradient_stats(["backbone.weight", "depth_embed.weight"], a, b)
        self.assertEqual(report["all"]["gt_norm"], 5.)
        self.assertEqual(report["all"]["cosine"], -1.)
        self.assertEqual(report["all"]["preservation_to_gt_norm_ratio"], 1.)

    def test_zero_preservation_has_no_defined_cosine(self):
        report = self.probe.gradient_stats(["angle_embed.weight"],
            [self.torch.tensor([1.])], [None])["all"]
        self.assertEqual(report["preservation_norm"], 0.)
        self.assertIsNone(report["cosine"])

    def test_autograd_probe_does_not_populate_parameter_grad_or_update(self):
        parameter = self.torch.nn.Parameter(self.torch.tensor([2., 3.]))
        before = parameter.detach().clone()
        grads = self.probe.cpu_gradients(parameter.square().sum(), [parameter])
        self.assertTrue(self.torch.equal(grads[0], self.torch.tensor([4., 6.])))
        self.assertTrue(self.torch.equal(parameter, before))
        self.assertIsNone(parameter.grad)


if __name__ == "__main__":
    unittest.main()
