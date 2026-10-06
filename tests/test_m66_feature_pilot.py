"""CPU regressions for M66. No CUDA execution or improved AP is claimed."""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import m66_r0_a2_feature_pilot as pilot
from third_party.monodetr.m66_vehicle_feature_kd import DepthFeatureTap, configure_trainable, feature_kd
from test_m65_preservation_control import fixture, matcher


def kd_fixture(labels=(0, 1)):
    outputs, reference, targets = fixture(labels)
    teacher = {k: v.clone().requires_grad_() for k, v in reference.items()}
    reference["pred_depth"][..., 0] = 11.2
    with torch.no_grad():
        teacher["pred_depth"][..., 0] = 10.3
    generator = torch.Generator().manual_seed(17)
    features = torch.randn(1, len(labels), 256, generator=generator).requires_grad_()
    teacher_features = torch.randn(1, len(labels), 256, generator=generator).requires_grad_()
    return outputs, teacher, reference, features, teacher_features, targets


class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(256, 256), nn.Linear(256, 2)])

    def forward(self, x):
        return self.layers[1](torch.relu(self.layers[0](x)))


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(3, 256)
        self.class_embed = nn.Linear(256, 3)
        self.depth_embed = nn.ModuleList([Head(), Head(), Head()])


def summary_fixture(m, role="kd"):
    gt, keep, kd = 1.0, 0.2, 0.3
    return dict(complete=True, role=role, manifest_sha256=m["manifest_sha256"],
                completed_epochs=1, optimizer_steps=928, external_kd_enabled=role == "kd",
                original_checkpoint_sha256=pilot.q.A2_SHA, running_buffers_unchanged=True,
                frozen_parameters_unchanged=True, anchor_unchanged=True, teacher_unchanged=True,
                inference_graph_unchanged=True, trainable_parameter_names=["depth_embed.2.layers.0.weight"],
                loss_means=dict(gt=gt, preservation=keep, feature_kd=kd,
                                total=gt + .10 * keep + (.10 * kd if role == "kd" else 0)),
                eligible_kd_pairs=dict(Vehicle=8, Pedestrian=0), batch_input_sha256=["a" * 64] * 928)


class FeatureLossTests(unittest.TestCase):
    def test_vehicle_only_and_no_external_gradients(self):
        args = kd_fixture()
        loss, counts = feature_kd(*args, matcher)
        self.assertEqual(counts, dict(Vehicle=1, Pedestrian=0, vehicle_near=1, vehicle_far=0))
        loss.backward()
        self.assertEqual(float(args[3].grad[0, 0].abs().sum()), 0)
        self.assertGreater(float(args[3].grad[0, 1].abs().sum()), 0)
        self.assertIsNone(args[4].grad)
        self.assertTrue(all(v.grad is None for v in args[1].values()))
        self.assertIsNone(args[0]["pred_depth"].grad)  # not direct external depth KD

    def test_hungarian_alignment_not_query_index(self):
        outputs, teacher, anchor, _, tf, targets = kd_fixture()
        outputs = {k: v[:, [1, 0]].detach() for k, v in outputs.items()}
        sf = tf[:, [1, 0]].detach().clone().requires_grad_()
        loss, counts = feature_kd(outputs, teacher, anchor, sf, tf, targets, matcher)
        self.assertLess(float(loss.detach()), 1e-6)
        self.assertEqual(counts["Vehicle"], 1)

    def test_teacher_must_be_more_accurate_than_frozen_a2(self):
        for z in (11.2, 11.15, 12.0):
            args = list(kd_fixture((1,)))
            with torch.no_grad():
                args[1]["pred_depth"][..., 0] = z
            loss, counts = feature_kd(*args, matcher)
            self.assertEqual(counts["Vehicle"], 0)
            self.assertEqual(float(loss.detach()), 0)
            loss.backward()
            self.assertTrue(torch.isfinite(args[3].grad).all())

    def test_anchor_already_accurate_not_distilled(self):
        args = list(kd_fixture((1,)))
        args[2]["pred_depth"][..., 0] = 10.01
        _, counts = feature_kd(*args, matcher)
        self.assertEqual(counts["Vehicle"], 0)

    def test_teacher_confidence_class_overlap_and_geometry_masks(self):
        for condition in ("class", "confidence", "overlap", "negative_depth", "range", "accuracy"):
            args = list(kd_fixture((1,)))
            with torch.no_grad():
                if condition == "class":
                    args[1]["pred_logits"][0, 0] = torch.tensor([7., -7., -7.])
                elif condition == "confidence":
                    args[1]["pred_logits"][0, 0] = torch.tensor([-7., -2., -7.])
                elif condition == "overlap":
                    args[1]["pred_boxes"][..., 0] += .4
                elif condition == "negative_depth":
                    args[1]["pred_depth"][..., 0] = -1
                elif condition == "range":
                    args[-1][0]["depth"][...] = 60.0
                else:
                    args[1]["pred_depth"][..., 0] = 8.4
                    args[2]["pred_depth"][..., 0] = 13.0
            _, counts = feature_kd(*args, matcher)
            self.assertEqual(counts["Vehicle"], 0, condition)

    def test_pedestrian_only_and_empty_gt_are_safe(self):
        for labels in ((0,), ()):
            args = kd_fixture(labels)
            loss, counts = feature_kd(*args, matcher)
            self.assertEqual(counts["Vehicle"], 0)
            loss.backward()
            self.assertTrue(torch.isfinite(args[3].grad).all())

    def test_nonfinite_and_mismatched_features_rejected(self):
        for value in ("nan", "width", "queries"):
            args = list(kd_fixture())
            if value == "nan":
                args[4] = args[4] * float("nan")
            elif value == "width":
                args[3] = args[3][..., :128]
            else:
                args[3] = args[3][:, :1]
            with self.assertRaisesRegex(RuntimeError, "feature"):
                feature_kd(*args, matcher)

    def test_zero_norm_eligible_feature_rejected(self):
        args = list(kd_fixture((1,)))
        args[4] = torch.zeros_like(args[4])
        with self.assertRaisesRegex(RuntimeError, "Zero-norm"):
            feature_kd(*args, matcher)


class ScopeAndTapTests(unittest.TestCase):
    def test_actual_paired_loss_changes_head_gradients_only_in_kd_arm(self):
        class Pipeline(ToyModel):
            def __init__(self, base_depth):
                super().__init__()
                self.base_depth = base_depth

            def forward(self, images, calibs, targets, sizes, dn_args=None):
                _, ref, _ = fixture()
                hidden = self.backbone(images)
                depth = self.depth_embed[-1](hidden)
                ref["pred_depth"] = torch.stack((self.base_depth + .01 * depth[..., 0],
                                                  .01 * depth[..., 1]), dim=-1)
                return ref

        class Criterion(nn.Module):
            weight_dict = {"loss_depth": 1.0}
            matcher = staticmethod(matcher)

            def forward(self, outputs, targets, extra):
                return {"loss_depth": torch.nn.functional.smooth_l1_loss(
                    outputs["pred_depth"][..., 0], targets[0]["depth"].reshape(1, 2))}

        torch.manual_seed(42)
        original = Pipeline(11.2)
        anchor = copy.deepcopy(original).eval().requires_grad_(False)
        teacher = Pipeline(10.3).eval().requires_grad_(False)
        _, _, targets = fixture()
        raw = {k: v[None] for k, v in targets[0].items()}
        raw.update(boxes=raw["boxes_3d"][..., :4], calibs=torch.eye(3)[None, None].repeat(1, 2, 1, 1),
                   mask_2d=torch.ones(1, 2, dtype=torch.bool), img_size=torch.tensor([[1280., 384.]]))
        batch = (torch.ones(1, 2, 3), torch.eye(3)[None], raw, {})
        grads = []
        for role in ("control", "kd"):
            student = copy.deepcopy(original)
            configure_trainable(student)
            taps = DepthFeatureTap(student), DepthFeatureTap(teacher)
            with patch.object(torch.Tensor, "cuda", lambda value: value):
                total, gt, keep, kd, kp, kc = pilot.losses(student, anchor, teacher, Criterion(), batch, *taps, role)
            self.assertEqual(kc["Vehicle"], 1)
            self.assertEqual(kc["Pedestrian"], 0)
            self.assertLess(float(keep["total"].detach()), 1e-3)
            expected = gt + .1 * keep["total"] + (.1 * kd if role == "kd" else 0)
            self.assertTrue(torch.allclose(total, expected))
            total.backward()
            grads.append(student.depth_embed[-1].layers[0].weight.grad.clone())
            self.assertIsNone(student.backbone.weight.grad)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            self.assertTrue(all(p.grad is None for p in anchor.parameters()))
            for tap in taps:
                tap.close()
        self.assertGreater(float((grads[0] - grads[1]).abs().sum()), 0)

    def test_only_depth_parameters_update_and_tap_is_differentiable(self):
        model = ToyModel()
        names = configure_trainable(model)
        self.assertTrue(all(n.startswith("depth_embed.") for n in names))
        frozen_before = pilot.frozen_state(model)
        tap = DepthFeatureTap(model)
        x = model.backbone(torch.ones(1, 2, 3))
        model.depth_embed[-1](x)
        features = tap.take()
        self.assertTrue(features.requires_grad)
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
        loss = (1 - torch.nn.functional.cosine_similarity(features, torch.ones_like(features), dim=-1)).mean()
        loss.backward()
        self.assertGreater(float(model.depth_embed[-1].layers[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(model.backbone.weight.grad)
        optimizer.step()
        self.assertEqual(frozen_before, pilot.frozen_state(model))
        tap.close()
        self.assertFalse(model.depth_embed[-1].layers[-1]._forward_pre_hooks)

    def test_tap_rejects_missing_and_duplicate_calls(self):
        model = ToyModel()
        tap = DepthFeatureTap(model)
        with self.assertRaisesRegex(RuntimeError, "Missing"):
            tap.take()
        model.depth_embed[-1](torch.ones(1, 2, 256))
        with self.assertRaisesRegex(RuntimeError, "twice"):
            model.depth_embed[-1](torch.ones(1, 2, 256))
        tap.reset()
        model.depth_embed[-1](torch.ones(1, 2, 256))
        self.assertEqual(tuple(tap.take().shape), (1, 2, 256))
        tap.close()

    def test_feature_width_guard(self):
        model = ToyModel()
        model.depth_embed[-1].layers[-1] = nn.Linear(128, 2)
        with self.assertRaisesRegex(RuntimeError, "layout"):
            DepthFeatureTap(model)

    def test_r0_config_is_independent_and_resnet(self):
        original = dict(model=dict(backbone_source="timm", backbone="mobilenetv4_conv_medium.e500_r256_in1k",
                                   backbone_out_indices=[2, 3, 4], backbone_pretrained=False), dataset=dict(meanshape=False))
        result = pilot.teacher_config(original)
        self.assertEqual(result["model"]["backbone"], "resnet50")
        self.assertNotIn("backbone_out_indices", result["model"])
        self.assertEqual(original["model"]["backbone_source"], "timm")
        self.assertFalse(result["dataset"]["meanshape"])


class WorkflowTests(unittest.TestCase):
    def test_signed_smoke_needs_real_feature_gradients_and_full_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = dict(output=tmp, manifest_sha256="test")
            report = dict(complete=True, manifest_sha256="test", optimizer_steps=0,
                          checked_train_images=64, eligible_vehicle_pairs=16, passed=True,
                          checks=dict(states_unchanged=True),
                          rows=[dict(pedestrian_pairs=0, finite_gt_gradient=True,
                                     vehicle_pairs=1, finite_kd_gradient=True)] * 16)
            def save(value):
                value.pop("report_sha256", None)
                value["report_sha256"] = pilot.q.signature(value)
                pilot.q.write_json(Path(tmp) / "m66_training_smoke.json", value)
            save(copy.deepcopy(report))
            pilot.validated_smoke(m)
            for mutation in ("steps", "rows", "kd", "pedestrian", "signature"):
                broken = copy.deepcopy(report)
                broken["rows"] = [dict(r) for r in report["rows"]]
                if mutation == "steps":
                    broken["optimizer_steps"] = 1
                elif mutation == "rows":
                    broken["rows"].pop()
                elif mutation == "kd":
                    for row in broken["rows"]:
                        row["finite_kd_gradient"] = False
                elif mutation == "pedestrian":
                    broken["rows"][0]["pedestrian_pairs"] = 1
                save(broken)
                if mutation == "signature":
                    broken["report_sha256"] = "changed"
                    pilot.q.write_json(Path(tmp) / "m66_training_smoke.json", broken)
                with self.assertRaisesRegex(RuntimeError, "smoke"):
                    pilot.validated_smoke(m)

    def test_review_rejects_different_augmented_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = dict(output=tmp, manifest_sha256="test")
            for role in ("control", "kd"):
                report = summary_fixture(m, role)
                if role == "kd":
                    report["batch_input_sha256"][12] = "b" * 64
                pilot.q.write_json(Path(tmp) / role / "training_summary.json", report)
            source = dict(product=dict(pilot.BASELINE), nearby=dict(pilot.NEARBY))
            with patch.object(pilot, "baseline", return_value=source), \
                 patch.object(pilot, "validated_metrics", return_value=source):
                with self.assertRaisesRegex(RuntimeError, "different training inputs"):
                    pilot.review(m)

    def test_acceptance_requires_improvement_over_original_and_control(self):
        source = dict(product=dict(pilot.BASELINE), nearby=dict(pilot.NEARBY))
        control = copy.deepcopy(source)
        kd = copy.deepcopy(source)
        kd["product"]["vehicle_3d_moderate"] += .2
        self.assertTrue(all(pilot.acceptance(source, control, kd).values()))
        control["product"]["vehicle_3d_moderate"] += .1
        self.assertFalse(pilot.acceptance(source, control, kd)["matched_control_vehicle_3d_gain"])

    def test_pedestrian_and_nearby_regression_cannot_hide_in_vehicle_gain(self):
        source = dict(product=dict(pilot.BASELINE), nearby=dict(pilot.NEARBY))
        kd = copy.deepcopy(source)
        kd["product"]["vehicle_3d_moderate"] += .3
        kd["product"]["pedestrian_3d_moderate"] -= .16
        kd["nearby"]["Pedestrian"] -= .006
        checks = pilot.acceptance(source, source, kd)
        self.assertFalse(checks["original_a2_pedestrian_3d_moderate"])
        self.assertFalse(checks["original_a2_Pedestrian_nearby"])

    def test_failed_control_is_not_a_prerequisite_for_kd_acceptance(self):
        source = dict(product=dict(pilot.BASELINE), nearby=dict(pilot.NEARBY))
        control = copy.deepcopy(source)
        control["product"]["pedestrian_3d_moderate"] -= .5
        kd = copy.deepcopy(source)
        kd["product"]["vehicle_3d_moderate"] += .2
        self.assertTrue(all(pilot.acceptance(source, control, kd).values()))

    def test_summary_role_scope_counts_and_loss_arithmetic(self):
        m = dict(manifest_sha256="test")
        for role in ("control", "kd"):
            report = summary_fixture(m, role)
            pilot.validate_summary(m, role, report)
            for key, value in (("optimizer_steps", 927), ("external_kd_enabled", role != "kd"),
                               ("frozen_parameters_unchanged", False), ("teacher_unchanged", False),
                               ("trainable_parameter_names", ["backbone.weight"]), ("batch_input_sha256", [])):
                broken = copy.deepcopy(report)
                broken[key] = value
                with self.assertRaises(RuntimeError):
                    pilot.validate_summary(m, role, broken)
            report["loss_means"]["total"] += .1
            with self.assertRaisesRegex(RuntimeError, "arithmetic"):
                pilot.validate_summary(m, role, report)

    def test_completed_checkpoint_recovery_never_takes_more_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = dict(output=tmp, manifest_sha256="test")
            report = summary_fixture(m, "control")
            payload = dict(model_state={"weight": torch.ones(1)}, epoch=1,
                           m66_manifest_sha256="test", training_summary=report)
            pilot.atomic_checkpoint(m, "control", payload)
            path = pilot.checkpoint_path(m, "control")
            digest = pilot.q.sha(path)
            path.with_suffix(".json").unlink()
            self.assertTrue(pilot.recover_completed(m, "control"))
            self.assertEqual(digest, pilot.q.sha(path))
            self.assertEqual(pilot.q.read_json(path.with_suffix(".json"))["role"], "control")
            self.assertEqual(pilot.q.read_json(Path(tmp) / "control/training_summary.json"), report)
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                pilot.atomic_checkpoint(m, "control", payload)
            with patch.object(pilot, "baseline"), patch.object(pilot, "validated_smoke"), \
                 patch.object(pilot, "training_runtime", side_effect=AssertionError("must not initialize")):
                pilot.train(m, "control")

    def test_wrong_arm_or_changed_sidecar_recovery_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = dict(output=tmp, manifest_sha256="test")
            payload = dict(model_state={"w": torch.ones(1)}, epoch=1,
                           m66_manifest_sha256="test", training_summary=summary_fixture(m, "control"))
            pilot.atomic_checkpoint(m, "kd", payload)
            with self.assertRaisesRegex(RuntimeError, "wrong-arm"):
                pilot.recover_completed(m, "kd")
            pilot.atomic_checkpoint(m, "control", payload)
            path = pilot.checkpoint_path(m, "control").with_suffix(".json")
            receipt = pilot.q.read_json(path)
            receipt["checkpoint_sha256"] = "changed"
            pilot.q.write_json(path, receipt)
            with self.assertRaisesRegex(RuntimeError, "receipt"):
                pilot.recover_completed(m, "control")

    def test_bundle_excludes_raw_weights_and_predictions(self):
        import zipfile
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            (output / "m66_manifest.json").write_text("{}")
            (output / "checkpoint.pth").write_text("not bundled")
            (output / "predictions").mkdir()
            (output / "predictions/cache.json").write_text("{}")
            pilot.bundle(output)
            with zipfile.ZipFile(output / "m66_results.zip") as archive:
                self.assertEqual(archive.namelist(), ["m66_manifest.json"])

    def test_script_launch_outside_repository(self):
        result = subprocess.run([sys.executable, ROOT / "scripts/m66_r0_a2_feature_pilot.py", "--help"],
                                cwd=tempfile.gettempdir(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_notebook_six_cells_revision_paths_and_restart_guards(self):
        path = ROOT / "notebooks/MonoDETR_A2_M66_R0_Vehicle_Feature_KD_Colab.ipynb"
        n = json.loads(path.read_text())
        cells = ["".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(cells), 6)
        for cell in cells:
            ast.parse(cell)
            compile(cell, str(path), "exec")
        self.assertIn(pilot.REVISION, cells[0])
        self.assertIn("from collections import deque", cells[0])
        self.assertIn("import ast, json, os, runpy, shlex, subprocess, sys", cells[0])
        self.assertIn("ZrrSkywalker/MonoDETR.git", cells[1])
        self.assertIn("build_m64_attention.py", cells[1])
        self.assertIn("--r0-selection", cells[2])
        self.assertIn("workflow('smoke')", cells[3])
        self.assertIn("for role in ('control','kd')", cells[4])
        self.assertIn("if not TRAIN_READY", cells[4])
        self.assertIn("finally:", cells[5])
        self.assertNotIn("reset", "\n".join(cells).replace("do not reset", ""))
        self.assertNotIn("MonoPRIO", "\n".join(cells))


if __name__ == "__main__":
    unittest.main()
