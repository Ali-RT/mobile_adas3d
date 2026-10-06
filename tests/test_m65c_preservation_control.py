"""CPU-only checks: these do not establish CUDA performance or accuracy."""
from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import m65c_a2_preservation_control as control
from third_party.monodetr.m65c_preservation import preservation, SCALES, WEIGHTS
from test_m65_preservation_control import fixture, matcher, completed_summary


def calibration_fixture(m):
    rows = [dict(image_ids=m["calibration_ids"][i:i + 4], gt_norm=10.0,
                 preservation_norm=2.0) for i in range(0, 64, 4)]
    counts = {key: {"0": 8, "1": 8} for key in WEIGHTS}
    coefficient, median = control.choose_coefficient(rows, counts)
    report = dict(complete=True, manifest_sha256=m["manifest_sha256"], optimizer_steps=0,
                  parameters_unchanged=True, buffers_unchanged=True, anchor_unchanged=True,
                  calibration_ids=m["calibration_ids"], calibration_endpoint_sha256=m["calibration_endpoint_sha256"],
                  rows=rows, eligible_pairs=counts, coefficient=coefficient,
                  median_raw_gradient_ratio=median)
    report["report_sha256"] = control.q.signature(report)
    return report


class ScaledLossTests(unittest.TestCase):
    def test_equality_counts_and_finite_gradients(self):
        outputs, ref, targets = fixture()
        losses, counts = preservation(outputs, ref, targets, matcher)
        self.assertLess(float(losses["total"].detach()), control.INITIAL_LOSS_TOLERANCE)
        self.assertEqual(counts["uncertainty"], {0: 1, 1: 1})
        losses["total"].backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in outputs.values()))

    def test_uncertainty_only_has_nonzero_second_channel_gradient(self):
        outputs, ref, targets = fixture()
        outputs["pred_depth"] = (outputs["pred_depth"].detach() + torch.tensor([[[0, .1], [0, .1]]])).requires_grad_()
        losses, _ = preservation(outputs, ref, targets, matcher)
        self.assertAlmostEqual(float(losses["uncertainty"].detach()), .5, places=5)
        losses["total"].backward()
        self.assertEqual(float(outputs["pred_depth"].grad[..., 0].abs().sum()), 0)
        self.assertGreater(float(outputs["pred_depth"].grad[..., 1].abs().sum()), 0)

    def test_two_percent_depth_shift_has_unit_scaled_smooth_l1(self):
        outputs, ref, targets = fixture()
        outputs["pred_depth"] = outputs["pred_depth"] + torch.tensor([[[.2, 0], [.2, 0]]])
        losses, _ = preservation(outputs, ref, targets, matcher)
        self.assertAlmostEqual(float(losses["depth"].detach()), .5, places=5)

    def test_unreliable_depth_excludes_uncertainty(self):
        outputs, ref, targets = fixture()
        ref["pred_depth"][0, 0, 0] = 20.0
        _, counts = preservation(outputs, ref, targets, matcher)
        self.assertEqual(counts["depth"], {0: 0, 1: 1})
        self.assertEqual(counts["uncertainty"], {0: 0, 1: 1})

    def test_reference_is_detached(self):
        outputs, ref, targets = fixture()
        ref = {k: v.clone().requires_grad_() for k, v in ref.items()}
        outputs["pred_depth"] = outputs["pred_depth"] + .1
        losses, _ = preservation(outputs, ref, targets, matcher)
        losses["total"].backward()
        self.assertTrue(all(v.grad is None for v in ref.values()))

    def test_query_permutation_is_associated_via_gt(self):
        outputs, ref, targets = fixture()
        outputs = {k: v[:, [1, 0]].detach().requires_grad_() for k, v in outputs.items()}
        losses, counts = preservation(outputs, ref, targets, matcher)
        self.assertLess(float(losses["total"].detach()), 1e-5)
        self.assertEqual(counts["classification"], {0: 1, 1: 1})

    def test_vehicle_frequency_does_not_change_class_mean(self):
        values = []
        for labels in ((0, 1), (0, 1, 1, 1, 1, 1)):
            outputs, ref, targets = fixture(labels)
            outputs["pred_depth"] = outputs["pred_depth"] + torch.tensor([[[.4, .2]] + [[.2, .1]] * (len(labels) - 1)])
            losses, _ = preservation(outputs, ref, targets, matcher)
            values.append(float(losses["uncertainty"].detach()))
        self.assertAlmostEqual(*values, places=5)

    def test_excluded_class_and_box_do_not_contribute(self):
        for kind in ("class", "box"):
            outputs, ref, targets = fixture((0,))
            if kind == "class":
                ref["pred_logits"][0, 0] = torch.tensor([-7., 7., -7.])
            else:
                ref["pred_boxes"][0, 0, 0] += .5
            losses, counts = preservation(outputs, ref, targets, matcher)
            self.assertEqual(counts["uncertainty"], {0: 0, 1: 0})
            self.assertEqual(float(losses["total"].detach()), 0)

    def test_background_only_is_safe(self):
        outputs, ref, _ = fixture((0,))
        _, _, targets = fixture(())
        losses, _ = preservation(outputs, ref, targets, matcher)
        self.assertEqual(float(losses["total"].detach()), 0)
        losses["total"].backward()

    def test_nonfinite_and_missing_uncertainty_rejected(self):
        outputs, ref, targets = fixture()
        ref["pred_depth"][0, 0, 1] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            preservation(outputs, ref, targets, matcher)
        outputs, ref, targets = fixture()
        ref["pred_depth"] = ref["pred_depth"][..., :1]
        with self.assertRaisesRegex(RuntimeError, "log uncertainty"):
            preservation(outputs, ref, targets, matcher)


class CalibrationAndWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.m = dict(manifest_sha256="new-run", calibration_ids=[f"{i:06d}" for i in range(64)],
                      calibration_endpoint_sha256="old-endpoint")

    def test_selection_is_deterministic_stratified_train_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            train = [f"{i:06d}" for i in range(100)]
            for i, sample in enumerate(train):
                (labels / f"{sample}.txt").write_text("Pedestrian 0\n" if i < 50 else "Car 0\n")
            ids = control.calibration_ids(train, labels)
            self.assertEqual(ids, control.calibration_ids(train, labels))
            self.assertEqual(len(set(ids)), 64)
            self.assertTrue(set(ids) <= set(train))
            for i in range(0, 64, 4):
                self.assertTrue(all(int(x) < 50 for x in ids[i:i + 2]))
                self.assertTrue(all(int(x) >= 50 for x in ids[i + 2:i + 4]))

    def test_insufficient_or_duplicate_train_selection_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels = Path(tmp)
            (labels / "000000.txt").write_text("Car 0\n")
            with self.assertRaisesRegex(RuntimeError, "Need 32"):
                control.calibration_ids(["000000"], labels)
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                control.calibration_ids(["000000", "000000"], labels)

    def test_coefficient_uses_median_per_batch_ratios(self):
        report = calibration_fixture(self.m)
        self.assertEqual(report["coefficient"], 1.25)
        # medians of norms would yield a different answer for this distribution.
        rows = [dict(gt_norm=1., preservation_norm=1.)] * 8 + [dict(gt_norm=100., preservation_norm=10.)] * 8
        coefficient, median = control.choose_coefficient(rows, report["eligible_pairs"])
        self.assertAlmostEqual(median, .55)
        self.assertAlmostEqual(coefficient, .25 / .55)

    def test_no_zero_nonfinite_or_out_of_range_calibration_fallback(self):
        report = calibration_fixture(self.m)
        for value in (0., float("nan"), 1e-10, 1e10):
            rows = copy.deepcopy(report["rows"])
            for row in rows:
                row["preservation_norm"] = value
            with self.assertRaises(RuntimeError):
                control.choose_coefficient(rows, report["eligible_pairs"])

    def test_calibration_requires_both_class_geometry_coverage_and_all_batches(self):
        report = calibration_fixture(self.m)
        counts = copy.deepcopy(report["eligible_pairs"])
        counts["uncertainty"]["0"] = 7
        with self.assertRaisesRegex(RuntimeError, "Insufficient"):
            control.choose_coefficient(report["rows"], counts)
        with self.assertRaisesRegex(RuntimeError, "all16"):
            control.choose_coefficient(report["rows"][:-1], report["eligible_pairs"])

    def test_signed_calibration_rejects_scalar_row_or_state_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = dict(self.m, output=tmp)
            p = Path(tmp) / "m65c_calibration.json"
            report = calibration_fixture(m)
            control.q.write_json(p, report)
            self.assertEqual(control.validated_calibration(m)["coefficient"], 1.25)
            for key, value in (("coefficient", 2.), ("optimizer_steps", 1), ("anchor_unchanged", False)):
                mutated = dict(report, **{key: value})
                control.q.write_json(p, mutated)
                with self.assertRaises(RuntimeError):
                    control.validated_calibration(m)

    def test_calibration_rows_match_frozen_ids_even_after_resigning(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = dict(self.m, output=tmp)
            report = calibration_fixture(m)
            report["rows"][0]["image_ids"][0] = "999999"
            report["report_sha256"] = control.q.signature({k: v for k, v in report.items() if k != "report_sha256"})
            control.q.write_json(Path(tmp) / "m65c_calibration.json", report)
            with self.assertRaisesRegex(RuntimeError, "rows differ"):
                control.validated_calibration(m)

    def test_completed_checkpoint_reuses_calibration_and_recovers_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = dict(self.m, output=tmp)
            calibration = calibration_fixture(m)
            control.q.write_json(Path(tmp) / "m65c_calibration.json", calibration)
            summary = dict(completed_summary(m), calibration_sha256=calibration["report_sha256"], preservation_coefficient=1.25)
            checkpoint = control.c.checkpoint_path(m)
            payload = dict(epoch=1, model_state={"weight": torch.ones(1)},
                           m65c_manifest_sha256=m["manifest_sha256"], training_summary=summary)
            control.c.atomic_checkpoint(checkpoint, payload, m)
            checkpoint.with_suffix(".json").unlink()
            smoke = dict(complete=True, passed=True, optimizer_steps=0, external_kd_enabled=False,
                         checks=dict(parameters_unchanged=True, buffers_unchanged=True, anchor_unchanged=True,
                                     finite_nonzero_uncertainty_gradients=True, finite_gt_gradients=True),
                         manifest_sha256=m["manifest_sha256"],
                         calibration_sha256=calibration["report_sha256"])
            control.q.write_json(Path(tmp) / "m65c_training_smoke.json", smoke)
            with patch.object(control, "baseline"), patch.object(control.c, "training_runtime", side_effect=AssertionError("must not update")):
                control.train(m)
            self.assertTrue(checkpoint.with_suffix(".json").exists())
            self.assertEqual(control.q.read_json(Path(tmp) / "m65c_training_summary.json"), summary)

    def test_wrong_calibration_binding_rejected(self):
        calibration = calibration_fixture(self.m)
        summary = dict(completed_summary(self.m), calibration_sha256="different", preservation_coefficient=1.25)
        with self.assertRaisesRegex(RuntimeError, "calibration differs"):
            control.validate_summary(self.m, summary, calibration)

    def test_failed_baseline_prevents_runtime_or_optimizer_creation(self):
        with patch.object(control, "baseline", side_effect=RuntimeError("baseline failed")), patch.object(control.c, "training_runtime") as runtime:
            with self.assertRaisesRegex(RuntimeError, "baseline failed"):
                control.train(self.m)
            runtime.assert_not_called()

    def test_frozen_recipe_and_review_hash(self):
        self.assertEqual(control.q.sha(control.REVIEW), control.REVIEW_SHA)
        self.assertEqual(control.c.implementation_hash(), "6d7223fa24b27b2c5bf1769406fe2c412d9648cf759221121bb2d2748fc13f88")
        self.assertEqual(control.RECIPE["epochs"], 1)
        self.assertFalse(control.RECIPE["external_teacher"])
        self.assertEqual(control.CALIBRATION["target_gradient_ratio"], .25)
        self.assertEqual(SCALES["log_uncertainty"], .1)

    def test_absolute_launch_without_pythonpath_resolves_repository(self):
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        script = ROOT / "scripts/m65c_a2_preservation_control.py"
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run([sys.executable, script, "--help"], cwd=tmp, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, script, "train", "--manifest", Path(tmp) / "missing.json"], cwd=tmp, env=env, capture_output=True, text=True)
            self.assertIn("FileNotFoundError", result.stderr)
            self.assertNotIn("ModuleNotFoundError", result.stderr)

    def test_notebook_is_self_contained_and_fixed_order(self):
        nb = json.loads((ROOT / "notebooks/MonoDETR_A2_M65c_Scaled_Preservation_Colab.ipynb").read_text())
        cells = ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(cells), 6)
        for cell in cells:
            ast.parse(cell)
        self.assertIn(control.REVISION, cells[0])
        self.assertIn("def run_logged(", cells[0])
        self.assertIn("def workflow(", cells[0])
        self.assertIn("from google.colab import drive", cells[0])
        self.assertIn("TRAIN_READY = TRAIN_COMPLETE = False", cells[3])
        self.assertIn("workflow('calibrate')", cells[3])
        self.assertIn("workflow('smoke')", cells[3])
        self.assertIn("if not TRAIN_READY", cells[4])
        self.assertIn("finally:", cells[5])
        self.assertIn("m65c_results.zip", cells[5])
        text = "\n".join(cells)
        self.assertIn("https://github.com/ZrrSkywalker/MonoDETR.git", text)
        for forbidden in ("PuFanqi23/MonoDETR", "git reset", "MonoPRIO", "m62_manifest", "TEACHER_SOURCE"):
            self.assertNotIn(forbidden, text)
        self.assertTrue(all(cell.get("outputs", []) == [] for cell in nb["cells"]))


if __name__ == "__main__":
    unittest.main()
