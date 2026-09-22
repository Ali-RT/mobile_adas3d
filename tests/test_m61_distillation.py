from __future__ import annotations

import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from audit_m61_teacher import alpha, audit_sample, center_xyz
from evaluate_m61_pilot import checkpoint_for, decide
from m61_common import (AP_GATES, COMPONENTS, NATIVE_CLASSES, array_hash,
                        check_split, json_hash, npz_read, npz_write, write_json)
from prepare_m61_distillation import make_config
from setup_m61_sources import patch_active_architecture
from train_m61_student import restore_checkpoint, save_checkpoint
from third_party.monodetr.m61_distillation_loss import geometry_distillation


def sample(error):
    boxes = np.array([[.5, .5, .1, .1, .1, .1], [.3, .4, .05, .05, .1, .1]], np.float32)
    angles = np.zeros((2, 24), np.float32)
    angles[:, 0] = 5
    angles[:, 12] = error / 10
    return dict(input_sha256=np.asarray("same-image"), target_sha256=np.asarray("same-gt"),
                manifest_sha256=np.asarray("same-manifest"),
                tgt_labels=np.array([1, 0], np.int8), tgt_boxes_3d=boxes,
                tgt_depth=np.array([[10.], [15.]], np.float32),
                tgt_size_3d=np.array([[1.6, 1.7, 4.], [1.7, .6, .8]], np.float32),
                tgt_heading_bin=np.zeros((2, 1), np.int64), tgt_heading_res=np.zeros((2, 1), np.float32),
                out_pred_logits=np.array([[0., 6., -5.], [6., 0., -5.]], np.float32),
                out_pred_boxes=boxes + np.array([error / 100, 0, 0, 0, 0, 0], np.float32),
                out_pred_depth=np.array([[10. + error, 0], [15. + error, 0]], np.float32),
                out_pred_3d_dim=np.array([[1.6, 1.7, 4.], [1.7, .6, .8]], np.float32) + error / 10,
                out_pred_angle=angles, query_indices=np.array([0, 1]), gt_indices=np.array([0, 1]),
                calibration=np.array([[700., 0, 640, -350], [0, 700, 192, 0], [0, 0, 1, 0]], np.float32),
                image_size=np.array([1280, 384]))


POLICY = dict(min_score=.3, min_iou=.5, max_depth_exclusive=60., individual_error_ratio_max=.95)


class M61GeometryTests(unittest.TestCase):
    def test_native_class_order(self):
        self.assertEqual(NATIVE_CLASSES, {"Pedestrian": 0, "Car": 1, "Cyclist": 2})

    def test_teacher_quality_approves_vehicle_only(self):
        _, masks, rows = audit_sample(sample(.1), sample(2), POLICY)
        self.assertTrue(masks[0].all())
        self.assertFalse(masks[1].any())
        self.assertEqual(len(rows), 1)

    def test_worse_teacher_not_approved(self):
        _, masks, _ = audit_sample(sample(1), sample(.1), POLICY)
        self.assertFalse(masks.any())

    def test_same_query_number_is_not_an_association(self):
        teacher = sample(.1)
        for key in list(teacher):
            if key.startswith("out_"):
                teacher[key] = teacher[key][[1, 0]]
        teacher["query_indices"] = np.array([1, 0])
        aligned, masks, _ = audit_sample(teacher, sample(2), POLICY)
        self.assertAlmostEqual(float(aligned["pred_depth"][0, 0]), 10.1, places=5)
        self.assertTrue(masks[0].all())
        self.assertFalse(masks[1].any())

    def test_cache_view_or_gt_mismatch_stops(self):
        for key in ("input_sha256", "target_sha256", "manifest_sha256"):
            s = sample(2)
            s[key] = np.asarray("different")
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                audit_sample(sample(.1), s, POLICY)

    def test_low_confidence_wrong_class_and_far_targets_rejected(self):
        for condition in ("low_score", "wrong_class", "far", "negative_dimensions"):
            t = sample(.1)
            if condition == "low_score":
                t["out_pred_logits"][0] = [-3, -2, -3]
            elif condition == "wrong_class":
                t["out_pred_logits"][0] = [9, 6, -3]
            elif condition == "far":
                t["tgt_depth"][0, 0] = 60.
            else:
                t["out_pred_3d_dim"][0, 0] = -1.
            _, masks, _ = audit_sample(t, sample(2), POLICY)
            self.assertFalse(masks.any(), condition)

    def test_component_approval_is_independent(self):
        teacher = sample(.1)
        teacher["out_pred_depth"][0, 0] = 30
        _, masks, _ = audit_sample(teacher, sample(2), POLICY)
        self.assertFalse(masks[0, 0])
        self.assertTrue(masks[0, 1])
        self.assertTrue(masks[0, 3])

    def test_projection_includes_camera_translation(self):
        s = sample(0)
        xyz = center_xyz(s["tgt_boxes_3d"][0], 10., s["calibration"], s["image_size"])
        np.testing.assert_allclose(xyz, [.5, 0, 10])

    def test_heading_wrap_is_periodic(self):
        t, s = sample(.1), sample(2)
        t["out_pred_angle"][0, :12] = 0
        t["out_pred_angle"][0, 11] = 5
        t["out_pred_angle"][0, 23] = np.pi / 6 - .01
        self.assertAlmostEqual(alpha(t["out_pred_angle"][0]), 2 * np.pi - .01, places=5)
        _, masks, rows = audit_sample(t, s, POLICY)
        self.assertLess(rows[0]["teacher"]["angle"], 1.)
        self.assertTrue(masks[0, 3])

    def test_loss_finite_and_no_pedestrian_or_teacher_gradients(self):
        teacher, student = sample(.1), sample(2)
        aligned, masks, _ = audit_sample(teacher, student, POLICY)
        approved = {k: torch.tensor(v, requires_grad=True) for k, v in aligned.items()}
        approved["masks"] = masks
        approved["image_size"] = teacher["image_size"]
        outputs = {k[4:]: torch.tensor(v[None], requires_grad=True) for k, v in student.items() if k.startswith("out_")}
        targets = [{"labels": torch.tensor([1, 0]), "depth": torch.tensor(student["tgt_depth"]),
                    "calibs": torch.tensor(np.repeat(student["calibration"][None], 2, axis=0)),
                    "size_3d": torch.tensor(student["tgt_size_3d"])}]
        assignments = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        losses, counts = geometry_distillation(outputs, targets, assignments, [approved], COMPONENTS)
        self.assertGreater(float(losses["total"].detach()), 0)
        losses["total"].backward()
        self.assertEqual(counts, {c: 1 for c in COMPONENTS})
        self.assertIsNone(outputs["pred_logits"].grad)
        for key in ("pred_depth", "pred_3d_dim", "pred_boxes", "pred_angle"):
            self.assertTrue(torch.isfinite(outputs[key].grad).all())
            self.assertTrue((outputs[key].grad[0, 1] == 0).all())
            self.assertIsNone(approved[key].grad)
        self.assertTrue((outputs["pred_boxes"].grad[..., 2:] == 0).all())

    def test_empty_approved_set_has_differentiable_zero(self):
        o = {"pred_depth": torch.ones(1, 2, 2, requires_grad=True)}
        losses, counts = geometry_distillation(o, [{"labels": torch.tensor([1, 0])}],
                                             [(torch.tensor([0, 1]), torch.tensor([0, 1]))],
                                             [{"masks": np.zeros((2, 4), bool)}], COMPONENTS)
        losses["total"].backward()
        self.assertEqual(float(losses["total"].detach()), 0)
        self.assertEqual(sum(counts.values()), 0)

    def test_pedestrian_mask_cannot_be_silently_used(self):
        with self.assertRaisesRegex(ValueError, "native class 1"):
            geometry_distillation({"pred_depth": torch.ones(1, 1, 2)},
                                  [{"labels": torch.tensor([0])}],
                                  [(torch.tensor([0]), torch.tensor([0]))],
                                  [{"masks": np.ones((1, 4), bool)}], COMPONENTS)


class M61WorkflowTests(unittest.TestCase):
    def test_resume_restores_model_optimizer_and_recovers_missing_sidecar(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
            model(torch.ones(1, 2)).sum().backward()
            optimizer.step()
            saved = copy.deepcopy(model.state_dict())
            binding = {"variant": "control", "manifest_sha256": "frozen"}
            path = directory / "checkpoint_epoch_3.pth"
            save_checkpoint(path, dict(epoch=3, model_state=model.state_dict(),
                                       optimizer_state=optimizer.state_dict(), history=[{"epoch": 3}], m61=binding))
            with torch.no_grad():
                model.weight.zero_()
            start, history = restore_checkpoint(directory, binding, model, optimizer)
            self.assertEqual(start, 3)
            self.assertEqual(history, [{"epoch": 3}])
            self.assertTrue(path.with_suffix(".sha256").exists())
            self.assertTrue(torch.equal(model.weight, saved["weight"]))
            self.assertTrue(optimizer.state)
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                restore_checkpoint(directory, {**binding, "variant": "vehicle_kd"}, model, optimizer)
            path.with_suffix(".sha256").write_text("wrong")
            with self.assertRaisesRegex(RuntimeError, "checkpoint transaction"):
                restore_checkpoint(directory, binding, model, optimizer)

    def test_array_identity_includes_dtype_shape_and_name(self):
        a = np.ones((2, 3), np.float32)
        self.assertNotEqual(array_hash({"a": a}), array_hash({"a": a.reshape(3, 2)}))
        self.assertNotEqual(array_hash({"a": a}), array_hash({"a": a.astype(np.float64)}))
        self.assertNotEqual(array_hash({"a": a}), array_hash({"b": a}))

    def test_npz_is_pickle_free_and_roundtrips(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.npz"
            values = dict(value=np.ones((2, 3), np.float32), input_sha256=np.asarray("hash"))
            npz_write(path, values)
            restored = npz_read(path)
            for k in values:
                np.testing.assert_array_equal(restored[k], values[k])
            self.assertFalse(path.with_suffix(".tmp").exists())

    def test_noncanonical_or_tiny_split_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "train.txt"
            path.write_text("000001\n" * 3712)
            with self.assertRaisesRegex(RuntimeError, "frozen Chen"):
                check_split(path, "train")

    def test_config_does_not_depend_on_old_runtime_yaml(self):
        base = dict(dataset={}, model={}, optimizer={}, lr_scheduler={}, trainer={}, tester={})
        cfg = make_config(base, Path("/data"), "student", Path("/out"), Path("/repo"))
        self.assertEqual(cfg["model"]["backbone"], "mobilenetv4_conv_medium.e500_r256_in1k")
        self.assertFalse(cfg["model"]["backbone_pretrained"])
        self.assertEqual(cfg["dataset"]["class_mapping"]["Truck"], "Car")
        self.assertEqual(cfg["dataset"]["random_mixup3d"], 0.)
        self.assertFalse(cfg["dataset"]["aug_pd"])
        self.assertEqual(cfg["trainer"]["max_epoch"], 10)
        self.assertEqual(base["model"], {})

    def test_gate_does_not_trade_away_pedestrians(self):
        source = dict(vehicle_3d_moderate=15.4573, pedestrian_3d_moderate=7.5328,
                      mean_3d_moderate=11.495, vehicle_bev_moderate=21.375,
                      pedestrian_bev_moderate=8.489, vehicle_near_recall=.8829,
                      pedestrian_near_recall=.69224)
        good = {**source, "vehicle_3d_moderate": 16., "mean_3d_moderate": 11.77}
        self.assertTrue(all(decide(source, source, good).values()))
        for key in ("pedestrian_3d_moderate", "pedestrian_bev_moderate", "pedestrian_near_recall"):
            bad = {**good, key: source[key] - .0001}
            self.assertFalse(all(decide(source, source, bad).values()), key)
        self.assertFalse(all(decide(source, source, source).values()))

    def test_epoch_five_cannot_replace_predeclared_decision(self):
        m = dict(student={"checkpoint": "/a2.pth"}, evaluation_epochs=[5, 10], variants={"control": {"run_dir": "/run"}})
        self.assertEqual(checkpoint_for(m, "baseline", 0), Path("/a2.pth"))
        with self.assertRaises(ValueError):
            checkpoint_for(m, "control", 15)
        source = (ROOT / "scripts/evaluate_m61_pilot.py").read_text()
        self.assertIn('r["epoch"] == 10', source)
        self.assertIn("full_run_authorized=False", source)

    def test_cuda_architecture_patch_is_idempotent(self):
        flags = ["-arch=sm_60", "-gencode=arch=compute_60,code=sm_60", "-gencode=arch=compute_61,code=sm_61",
                 "-gencode=arch=compute_70,code=sm_70", "-gencode=arch=compute_75,code=sm_75"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "setup.py"
            path.write_text('args = [\n' + ''.join(f'            "{f}",\n' for f in flags) + ']\n')
            patch_active_architecture(path)
            first = path.read_bytes()
            patch_active_architecture(path)
            self.assertEqual(path.read_bytes(), first)
            self.assertNotIn("sm_60", path.read_text())

    def test_notebook_syntax_and_self_contained_order(self):
        path = ROOT / "notebooks/MonoDGP_to_MonoDETR_M61_Vehicle_Distillation_Colab.ipynb"
        notebook = json.loads(path.read_text())
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(cells), 12)
        for cell in cells:
            ast.parse(cell)
        code = "\n".join(cells)
        self.assertIn("M61-2026-09-22-r1", code)
        self.assertNotIn("git reset", code)
        self.assertNotIn("run_streamed", code)
        self.assertLess(code.index("def pilot("), code.index("pilot('cache_m61"))
        self.assertLess(code.index("pilot('audit_m61"), code.index("'--variant', 'control'"))
        self.assertIn("'--variant', 'vehicle_kd'", code)
        self.assertIn("m61_results.zip", code)

    def test_all_sources_parse(self):
        for path in (ROOT / "scripts").glob("*m61*.py"):
            ast.parse(path.read_text())


if __name__ == "__main__":
    unittest.main()
