"""CPU checks; these do not establish M65 CUDA training or accuracy."""
from __future__ import annotations

import ast
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import audit_m65_prior_provenance as audit
import m65_a2_preservation_control as control
from third_party.monodetr.m65_preservation import preservation, WEIGHTS


def matcher(outputs, targets, group_num):
    assert group_num == 1
    result = []
    for i, t in enumerate(targets):
        cost = torch.cdist(outputs["pred_boxes"][i].detach(), t["boxes_3d"], p=1).cpu().numpy()
        a, b = linear_sum_assignment(cost)
        result.append((torch.tensor(a), torch.tensor(b)))
    return result


def fixture(labels=(0, 1)):
    n = len(labels)
    boxes = torch.tensor([[0.1 + i * 0.03, 0.5, 0.01, 0.01, 0.1, 0.1] for i in range(n)]).reshape(n, 6)
    logits = torch.full((n, 3), -7.0)
    for i, label in enumerate(labels):
        logits[i, label] = 7.0
    angle = torch.zeros(n, 24)
    angle[:, 0] = 7.0
    reference = dict(pred_logits=logits.unsqueeze(0), pred_boxes=boxes.unsqueeze(0),
                      pred_depth=torch.tensor([[10.0, 0.0]] * n).reshape(1, n, 2),
                      pred_3d_dim=torch.tensor([[1.0, 1.0, 4.0]] * n).reshape(1, n, 3),
                      pred_angle=angle.unsqueeze(0))
    targets = [dict(labels=torch.tensor(labels), boxes_3d=boxes.clone(),
                    depth=torch.full((n, 1), 10.0), size_3d=torch.tensor([[1.0, 1.0, 4.0]] * n).reshape(n, 3),
                    heading_bin=torch.zeros(n, 1, dtype=torch.long), heading_res=torch.zeros(n, 1))]
    outputs = {k: v.clone().requires_grad_() for k, v in reference.items()}
    return outputs, reference, targets


def completed_summary(m):
    return dict(complete=True, manifest_sha256=m["manifest_sha256"], completed_epochs=1,
                optimizer_steps=928, running_buffers_unchanged=True, anchor_unchanged=True,
                external_kd_enabled=False, original_checkpoint_sha256=control.q.A2_SHA,
                loss_means=dict(gt=1.0, preservation=0.01, total=1.01))


class PreservationTests(unittest.TestCase):
    def test_identical_models_zero_loss_and_complete_class_counts(self):
        outputs, ref, targets = fixture()
        losses, counts = preservation(outputs, ref, targets, matcher)
        self.assertLess(float(losses["total"].detach()), 1e-5)
        self.assertEqual(counts["classification"], {0: 1, 1: 1})
        losses["total"].backward()
        self.assertTrue(all(torch.isfinite(v.grad).all() for v in outputs.values()))

    def test_queries_are_associated_by_gt_not_index(self):
        outputs, ref, targets = fixture()
        outputs = {k: v[:, [1, 0]].detach().requires_grad_() for k, v in outputs.items()}
        losses, counts = preservation(outputs, ref, targets, matcher)
        self.assertLess(float(losses["total"].detach()), 1e-5)
        self.assertEqual(counts["box"], {0: 1, 1: 1})

    def test_output_perturbation_has_finite_nonzero_gradients(self):
        outputs, ref, targets = fixture()
        outputs["pred_logits"] = (outputs["pred_logits"].detach() + 0.1).requires_grad_()
        outputs["pred_boxes"] = (outputs["pred_boxes"].detach() + 0.005).requires_grad_()
        losses, _ = preservation(outputs, ref, targets, matcher)
        self.assertGreater(float(losses["total"].detach()), 0)
        losses["total"].backward()
        self.assertGreater(float(outputs["pred_boxes"].grad.abs().sum()), 0)
        self.assertTrue(torch.isfinite(outputs["pred_logits"].grad).all())

    def test_reference_never_receives_gradients(self):
        outputs, ref, targets = fixture()
        ref = {k: v.clone().requires_grad_() for k, v in ref.items()}
        outputs["pred_depth"] = (outputs["pred_depth"].detach() + 0.1).requires_grad_()
        losses, _ = preservation(outputs, ref, targets, matcher)
        losses["total"].backward()
        self.assertTrue(all(v.grad is None for v in ref.values()))

    def test_geometry_masks_do_not_freeze_unreliable_a2_errors(self):
        outputs, ref, targets = fixture()
        ref["pred_depth"][0, 0, 0] = 20
        ref["pred_3d_dim"][0, 1] *= 2
        ref["pred_angle"][0, 1, 0] = 0
        ref["pred_angle"][0, 1, 1] = 7
        _, counts = preservation(outputs, ref, targets, matcher)
        self.assertEqual(counts["depth"], {0: 0, 1: 1})
        self.assertEqual(counts["dimensions"], {0: 1, 1: 0})
        self.assertEqual(counts["angle"], {0: 1, 1: 0})
        self.assertEqual(counts["classification"], {0: 1, 1: 1})

    def test_wrong_class_and_low_iou_are_not_preserved(self):
        # One GT per case avoids making a second query the valid Hungarian
        # match when testing two different exclusion conditions together.
        for condition in ("wrong_class", "low_iou"):
            with self.subTest(condition=condition):
                outputs, ref, targets = fixture((0,))
                if condition == "wrong_class":
                    ref["pred_logits"][0, 0] = torch.tensor([-7.0, 7.0, -7.0])
                else:
                    ref["pred_boxes"][0, 0, 0] += 0.5
                losses, counts = preservation(outputs, ref, targets, matcher)
                self.assertEqual(counts["classification"], {0: 0, 1: 0})
                self.assertEqual(float(losses["total"].detach()), 0)

    def test_class_balancing_does_not_scale_with_vehicle_count(self):
        outputs, ref, targets = fixture((0, 1))
        outputs["pred_depth"] = outputs["pred_depth"] + torch.tensor([[[2.0, 0], [1.0, 0]]])
        one, _ = preservation(outputs, ref, targets, matcher)
        outputs2, ref2, targets2 = fixture((0, 1, 1, 1, 1, 1, 1, 1))
        offset = torch.tensor([[[2.0, 0]] + [[1.0, 0]] * 7])
        outputs2["pred_depth"] = outputs2["pred_depth"] + offset
        many, _ = preservation(outputs2, ref2, targets2, matcher)
        self.assertAlmostEqual(float(one["depth"].detach()), float(many["depth"].detach()), places=6)

    def test_background_only_batch_is_safe(self):
        outputs, ref, _ = fixture((0,))
        _, _, targets = fixture(())
        losses, counts = preservation(outputs, ref, targets, matcher)
        self.assertEqual(float(losses["total"].detach()), 0)
        losses["total"].backward()
        self.assertEqual(counts["classification"], {0: 0, 1: 0})

    def test_nonfinite_outputs_refused(self):
        outputs, ref, targets = fixture()
        ref["pred_depth"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "Non-finite"):
            preservation(outputs, ref, targets, matcher)

    def test_batchnorm_affine_buffers_and_attention_dropout_frozen(self):
        model = torch.nn.Sequential(torch.nn.BatchNorm1d(4), torch.nn.Dropout(.5),
                                     torch.nn.MultiheadAttention(4, 1, dropout=.5))
        control.training_mode(model)
        self.assertFalse(model[0].training)
        self.assertTrue(all(not p.requires_grad for p in model[0].parameters()))
        self.assertFalse(model[1].training)
        self.assertEqual(model[2].dropout, 0)

    def test_recipe_one_epoch_no_external_teacher(self):
        self.assertEqual(control.RECIPE["epochs"], 1)
        self.assertEqual(control.RECIPE["learning_rate"], 1e-6)
        self.assertFalse(control.RECIPE["external_teacher"])
        self.assertFalse(control.RECIPE["mixup"])

    def test_completed_checkpoint_recovers_sidecar_without_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            m = dict(output=tmp, manifest_sha256="test-run")
            checkpoint = control.checkpoint_path(m)
            payload = dict(epoch=1, model_state={"weight": torch.ones(1)},
                           m65_manifest_sha256=m["manifest_sha256"], training_summary=completed_summary(m))
            control.atomic_checkpoint(checkpoint, payload, m)
            checkpoint.with_suffix(".json").unlink()
            control.q.write_json(Path(tmp)/"m65_training_smoke.json",
                                 dict(passed=True, manifest_sha256=m["manifest_sha256"]))
            with patch.object(control, "baseline"), patch.object(control, "training_runtime", side_effect=AssertionError("must not train")):
                control.train(m)
            receipt = control.q.read_json(checkpoint.with_suffix(".json"))
            self.assertEqual(receipt["checkpoint_sha256"], control.q.sha(checkpoint))
            self.assertEqual(control.q.read_json(Path(tmp)/"m65_training_summary.json"), payload["training_summary"])

    def test_completed_checkpoint_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"checkpoint.pth"
            m = dict(manifest_sha256="test-run")
            control.atomic_checkpoint(path, dict(model_state={"weight": torch.ones(1)}), m)
            before = control.q.sha(path)
            with self.assertRaisesRegex(RuntimeError, "overwrite"):
                control.atomic_checkpoint(path, dict(model_state={"weight": torch.zeros(1)}), m)
            self.assertEqual(before, control.q.sha(path))

    def test_completed_summary_rejects_partial_or_nonfinite_state(self):
        m = dict(manifest_sha256="test-run")
        summary = completed_summary(m)
        control.validate_training_summary(m, summary)
        summary["optimizer_steps"] = 927
        with self.assertRaisesRegex(RuntimeError, "one-epoch"):
            control.validate_training_summary(m, summary)
        summary = completed_summary(m)
        summary["loss_means"]["total"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "loss summary"):
            control.validate_training_summary(m, summary)

    def test_manifest_refuses_rewritten_preservation_rules(self):
        from third_party.monodetr.m65_preservation import RULES
        m = dict(revision=control.REVISION, recipe=control.RECIPE, limits=control.CONTROL_LIMITS,
                 kd_authorized=False, preservation_rules=dict(RULES, score=0.1), preservation_weights=WEIGHTS)
        m["manifest_sha256"] = control.q.signature(m)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"manifest.json"
            control.q.write_json(path, m)
            with self.assertRaisesRegex(RuntimeError, "Fixed control recipe"):
                control.load(path)

    def test_preservation_gate_is_not_historical_product_gate(self):
        source = dict(product=dict(vehicle_3d_moderate=15.45, pedestrian_3d_moderate=7.53),
                      nearby=dict(Vehicle=.8829, Pedestrian=.6927))
        candidate = copy.deepcopy(source)
        self.assertTrue(all(control.control_checks(source, candidate).values()))
        candidate["product"]["pedestrian_3d_moderate"] -= .151
        candidate["nearby"]["Vehicle"] -= .0051
        checks = control.control_checks(source, candidate)
        self.assertFalse(checks["pedestrian_3d_moderate"])
        self.assertFalse(checks["Vehicle_nearby"])

    def test_reviewed_m64_rejects_missing_or_changed_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("m64_manifest.json", "m64_teacher_qualification.json"):
                (root / name).write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "exact reviewed"):
                control.reviewed_m64(root)

    def test_notebook_has_self_contained_ordered_sections_and_compiles(self):
        notebook = json.loads((ROOT / "notebooks/MonoDETR_A2_M65_Preservation_Control_Colab.ipynb").read_text())
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        for cell in cells:
            ast.parse(cell)
        text = "\n".join(cells)
        self.assertIn(control.REVISION, text)
        self.assertIn("def run_logged(", cells[0])
        self.assertIn("def workflow(", cells[0])
        self.assertIn("from google.colab import drive", cells[0])
        self.assertIn("https://github.com/ZrrSkywalker/MonoDETR.git", text)
        self.assertNotIn("PuFanqi23/MonoDETR", text)
        self.assertNotIn("git reset", text)
        self.assertNotIn("m62_manifest", text)
        tree = ast.parse(cells[0])
        initialization = [n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == "TRAIN_READY" for t in n.targets)]
        self.assertEqual(len(initialization), 1)
        self.assertIs(ast.literal_eval(initialization[0].value), False)
        self.assertIn("if not TRAIN_READY", text)
        self.assertIn("m65_results.zip", text)
        constants = [n.value for n in ast.walk(ast.parse(cells[0]))
                     if isinstance(n, ast.Constant) and isinstance(n.value, str)]
        self.assertIn("\nCOMMAND: ", constants)
        self.assertIn("SETUP_READY = MANIFEST_READY = TRAIN_READY = TRAIN_COMPLETE = False", cells[1])


class PriorAuditTests(unittest.TestCase):
    def test_literal_default_recipe_parsed_without_importing_builder(self):
        text = 'def default_class_cfgs():\n    return {"Car": ClassPriorCfg(k_geo=5,k_vis=4), "Pedestrian": ClassPriorCfg(k_geo=4,k_vis=2), "Cyclist": ClassPriorCfg(k_geo=4,k_vis=2)}'
        self.assertEqual(audit.default_maxima(text), {"Car": 20, "Pedestrian": 8, "Cyclist": 8})

    def test_released_counts_incompatible_not_claimed_as_leakage(self):
        bank = {k: np.zeros((35, 3)) for k in ("visual", "mu", "sigma", "mu_log", "V_log", "inv_std_log")}
        bank.update(class_counts=np.array([13, 10, 12]), class_offsets=np.array([0, 13, 23, 35]),
                    class_names=np.array(["Pedestrian", "Car", "Cyclist"]))
        r = audit.inspect(bank, {"Pedestrian": 8, "Car": 20, "Cyclist": 8})
        self.assertFalse(r["default_recipe_compatible"])
        self.assertFalse(r["validation_label_leakage_proven"])
        self.assertFalse(r["kd_authorized"])
        self.assertFalse(r["construction_identifiers_present"])

    def test_default_compatible_bank_still_does_not_prove_split(self):
        bank = {k: np.zeros((3, 3)) for k in ("visual", "mu", "sigma", "mu_log", "V_log", "inv_std_log")}
        bank.update(class_counts=np.array([1, 1, 1]), class_offsets=np.array([0, 1, 2, 3]),
                    class_names=np.array(["Pedestrian", "Car", "Cyclist"]))
        r = audit.inspect(bank, {"Pedestrian": 8, "Car": 20, "Cyclist": 8})
        self.assertTrue(r["default_recipe_compatible"])
        self.assertFalse(r["prior_training_ids_independently_verified"])
        self.assertFalse(r["kd_authorized"])


if __name__ == "__main__":
    unittest.main()
