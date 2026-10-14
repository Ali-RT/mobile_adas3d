"""Single-coefficient KD, preserved GT, resumable training and audit guards."""
import ast
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts"), str(ROOT / "tests")]
import recover_a2_lite_balanced_geometry as balanced
from models.a2_lite_balanced_geometry_kd import balanced_geometry_kd_loss, COMPONENT_WEIGHTS
from models.a2_lite_geometry_kd import geometry_kd_loss
from models.a2_lite_recovery_loss import gt_loss, teacher_matches
from test_a2_lite_recovery import ToyStudent, target, outputs
from test_a2_lite_full_recovery import evaluated

engine = balanced.engine


class BalancedTests(unittest.TestCase):
    def test_only_teacher_angle_part_is_scaled_without_renormalization(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}
        teacher = outputs((42, 47), True); mapping = teacher_matches(teacher, target())
        _, original, old_counts = geometry_kd_loss(student, teacher, target(), mapping)
        loss, parts, counts = balanced_geometry_kd_loss(student, teacher, target(), mapping)
        self.assertEqual(counts, old_counts)
        for name in ("dimensions", "depth", "classification", "boxes"):
            self.assertTrue(torch.equal(parts[name], original[name]))
        self.assertTrue(torch.equal(parts["angle"], original["angle"] * .01))
        self.assertTrue(torch.equal(loss, original["dimensions"] + original["depth"] + original["angle"] * .01))

    def test_orientation_gradient_is_exactly_one_hundredth_on_same_coordinates(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}; teacher = outputs((42, 47), True)
        mapping = teacher_matches(teacher, target())
        _, old, _ = geometry_kd_loss(student, teacher, target(), mapping)
        _, new, _ = balanced_geometry_kd_loss(student, teacher, target(), mapping)
        old_gradient = torch.autograd.grad(old["angle"], student["pred_angle"], retain_graph=True)[0]
        new_gradient = torch.autograd.grad(new["angle"], student["pred_angle"], retain_graph=True)[0]
        self.assertGreater(float(old_gradient.abs().sum()), 0.)
        self.assertTrue(torch.allclose(new_gradient, old_gradient * .01, atol=1e-7, rtol=1e-5))

    def test_all_gt_losses_and_angle_learning_remain_enabled(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}
        before, parts, _ = gt_loss(student, target())
        with balanced.recipe_context():
            after, bound_parts, _ = engine.gt_loss(student, target())
        self.assertTrue(torch.equal(before, after))
        self.assertEqual(balanced.RECIPE["gt_weights"], engine.GT_WEIGHTS)
        self.assertEqual(balanced.RECIPE["gt_weights"]["loss_angle"], 1.)
        self.assertEqual(parts.keys(), bound_parts.keys())
        after.backward()
        for name in ("pred_logits", "pred_boxes", "pred_angle", "pred_depth", "pred_3d_dim"):
            self.assertGreater(float(student[name].grad.abs().sum()), 0.)

    def test_selective_masks_and_teacher_detachment_are_unchanged(self):
        teacher = {k: v.requires_grad_() for k, v in outputs((42, 47), True).items()}
        teacher["pred_depth"].data[0, 42, 0] = 100.
        student = {k: v.requires_grad_() for k, v in outputs().items()}; mapping = teacher_matches(teacher, target())
        _, _, old_counts = geometry_kd_loss(student, teacher, target(), mapping)
        loss, _, counts = balanced_geometry_kd_loss(student, teacher, target(), mapping)
        self.assertEqual(counts, old_counts); self.assertEqual(counts["depth"]["Pedestrian"], 0)
        self.assertEqual(counts["depth"]["Vehicle"], 1)
        loss.backward(); self.assertTrue(all(v.grad is None for v in teacher.values()))
        self.assertIsNone(student["pred_boxes"].grad)
        self.assertTrue(student["pred_logits"].grad is None or float(student["pred_logits"].grad.abs().sum()) == 0.)

    def test_no_matches_give_differentiable_zero(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}
        loss, _, counts = balanced_geometry_kd_loss(student, outputs(perfect=True), target(), torch.tensor([-1, -1]))
        self.assertEqual(float(loss.detach()), 0.); loss.backward()
        self.assertTrue(all(sum(v.values()) == 0 for v in counts.values()))

    def test_training_recipe_and_guardrails_preserve_every_old_setting(self):
        for key, value in balanced.ORIGINAL_RECIPE.items(): self.assertEqual(balanced.RECIPE[key], value, key)
        self.assertEqual(balanced.RECIPE["teacher_component_weights"], dict(dimensions=1., depth=1., angle=.01))
        self.assertEqual(balanced.RECIPE["train_images"] * balanced.RECIPE["epochs"], 2560)
        self.assertEqual(balanced.RECIPE["kd_max_weight"] * COMPONENT_WEIGHTS["angle"], .0025)
        self.assertEqual(balanced.RECIPE["temperature"], 2.)
        self.assertEqual(balanced.GUARDRAILS, balanced.geometry.GUARDRAILS)
        self.assertFalse(balanced.RECIPE["architecture_changed"])

    def test_scoped_adapters_restore_all_original_globals_after_error(self):
        g = balanced.geometry
        old = g.RECIPE, g.REVISION, g.geometry_kd_loss, g.implementation, engine.RECIPE, engine.REVISION, engine.kd_loss
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with balanced.recipe_context():
                self.assertIs(engine.kd_loss, balanced_geometry_kd_loss)
                self.assertIs(engine.RECIPE, balanced.RECIPE)
                self.assertIs(g.implementation, balanced.implementation)
                self.assertEqual(g.REVISION, balanced.REVISION)
                raise RuntimeError("synthetic")
        self.assertEqual((g.RECIPE, g.REVISION, g.geometry_kd_loss, g.implementation,
                          engine.RECIPE, engine.REVISION, engine.kd_loss), old)

    def test_wrong_gradient_artifact_bytes_are_rejected(self):
        with patch.object(engine, "read_record", return_value=b"{}"):
            with self.assertRaisesRegex(RuntimeError, "exact reviewed gradient"):
                balanced.reviewed_audit(Path("synthetic.zip"))

    def test_prepare_rejects_changed_subset_before_any_model_or_data_work(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); old = dict(reference_run_signature="same", pilot_ids=["000001"],
                                     data_file_sha256={"000001": {}}, source_sha256="same")
            pilot = dict(signature_sha256="same", pilot_ids=["000002"], data_file_sha256=old["data_file_sha256"], source_sha256="same")
            a = SimpleNamespace(audit=root / "audit", pilot=root / "pilot", output=root / "new")
            with patch.object(balanced, "reviewed_audit", return_value=(old, {}, dict(cpu_environment={}), {})), \
                 patch.object(balanced.reference, "reviewed_pilot", return_value=(pilot, {}, b"", b"")), \
                 patch.object(balanced.geometry, "require_reference_environment"), \
                 patch.object(engine, "runtime_environment", return_value={}), \
                 patch.object(balanced.geometry, "prepare", side_effect=AssertionError("Must not stage data")):
                with self.assertRaisesRegex(RuntimeError, "same original512"):
                    balanced.prepare(a)

    def prepared_inputs(self, root, output):
        old = dict(output=str(root / "old_geometry"), reference_run_signature="same", pilot_ids=["000001"],
                   data_file_sha256={"000001": {}}, source_sha256="same", source=str(root / "old_source"), dataset=str(root / "old_data"))
        pilot = {**old, "signature_sha256": "same", "output": str(root / "old_pilot")}
        audit = dict(output=str(root / "old_audit"), cpu_environment={},
                     view=dict(source=str(root / "audit_source"), dataset=str(root / "audit_data")))
        a = SimpleNamespace(audit=root / "audit.zip", pilot=root / "pilot.zip", compact=root / "compact.zip",
                            output=output, initialization=root / "weights/init.pth", repo=root / "new_source", dataset=root / "new_data")
        return a, old, pilot, audit

    def test_prepare_cannot_write_inside_completed_history(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); output = root / "old_geometry/new_run"
            a, old, pilot, audit = self.prepared_inputs(root, output)
            with patch.object(balanced, "reviewed_audit", return_value=(old, {}, audit, {"gradient_manifest.json": b"proof"})), \
                 patch.object(balanced.reference, "reviewed_pilot", return_value=(pilot, {}, b"", b"")), \
                 patch.object(balanced.geometry, "require_reference_environment"), patch.object(engine, "runtime_environment", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "read-only"):
                    balanced.prepare(a)
            self.assertFalse(output.exists())

    def test_prepare_reuses_adapter_in_new_context_and_preserves_proof_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); a, old, pilot, audit = self.prepared_inputs(root, root / "new_output")
            observed = []
            def prepare(args):
                observed.append((copy.deepcopy(engine.RECIPE), engine.kd_loss, args.initialization))
            with patch.object(balanced, "reviewed_audit", return_value=(old, {}, audit, {"gradient_manifest.json": b"proof"})), \
                 patch.object(balanced.reference, "reviewed_pilot", return_value=(pilot, {}, b"", b"")), \
                 patch.object(balanced.geometry, "require_reference_environment"), patch.object(engine, "runtime_environment", return_value={}), \
                 patch.object(balanced.geometry, "prepare", side_effect=prepare):
                balanced.prepare(a)
            self.assertEqual(observed, [(balanced.RECIPE, balanced_geometry_kd_loss, a.initialization)])
            self.assertEqual((a.output / "reviewed_gradient/gradient_manifest.json").read_bytes(), b"proof")
            self.assertNotEqual(engine.REVISION, balanced.REVISION)
            self.assertFalse(Path(old["output"]).exists())

    def test_load_rejects_resigned_recipe_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); recipe = copy.deepcopy(balanced.RECIPE)
            recipe["teacher_component_weights"]["angle"] = .02
            m = dict(revision=balanced.REVISION, recipe=recipe)
            m["signature_sha256"] = engine.signature(m)
            engine.write_json(root / "geometry_manifest.json", m)
            with patch.object(balanced, "reviewed_audit", return_value=({}, {}, {}, {})):
                with self.assertRaisesRegex(RuntimeError, "code/packages/recipe changed"):
                    balanced.load(root)

    def make_cache(self, root, m):
        values = outputs((42, 47), True)
        arrays = {name: values[key].tolist() for name, key in zip(engine.OUTPUT_NAMES, engine.OUTPUT_KEYS)}
        hashes = {}
        for sample in m["pilot_ids"]:
            row = dict(run_signature=m["signature_sha256"], sample_id=sample, source_sha256=m["data_file_sha256"][sample],
                       target_signature=engine.target_signature(target()), outputs=arrays,
                       object_to_teacher_query=[42, 47], target_labels=[0, 1])
            row["record_sha256"] = engine.signature(row)
            path = root / "teacher_cache" / f"{sample}.json"; engine.write_json(path, row); hashes[sample] = engine.sha256(path)
        engine.save_report(root / "teacher_cache_complete.json", dict(complete=True, run_signature=m["signature_sha256"],
                           kd_pilot_authorized=True, cache_sha256=hashes))

    def test_rebalanced_real_loss_smoke_is_nonzero_with_no_updates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = dict(signature_sha256="synthetic", pilot_ids=["000001", "000002"],
                                    data_file_sha256={s: {"image_2": "same"} for s in ("000001", "000002")})
            self.make_cache(root, m)
            class Dataset:
                idx_list = m["pilot_ids"]
            model = ToyStudent(); before = copy.deepcopy(model.state_dict())
            with balanced.recipe_context(), patch.object(engine, "build_model", return_value=model), \
                 patch.object(engine, "native_dataset", return_value=Dataset()), patch.object(engine, "verify_files"), \
                 patch.object(engine, "item", side_effect=lambda *a: ({}, target(), {})), \
                 patch.object(engine, "forward", side_effect=lambda model, inputs: model.outputs()):
                balanced.geometry.kd_smoke(SimpleNamespace(output=root, device="cpu"), m)
            for key, value in before.items(): self.assertTrue(torch.equal(value, model.state_dict()[key]))
            report = engine.checked_report(root / "geometry_kd_smoke.json", m)
            self.assertTrue(report["passed"]); self.assertEqual(report["optimizer_steps"], 0)
            self.assertTrue(all(report["component_gradient_abs_sum"].values()))
            for key in balanced.geometry.DISABLED_COMPONENTS: self.assertEqual(sum(report["active_counts"][key].values()), 0)

    def test_training_resume_keeps_rebalanced_objective_and_disabled_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = dict(signature_sha256="synthetic", output=str(root), pilot_ids=["000001", "000002"],
                                    data_file_sha256={s: {"image_2": "same"} for s in ("000001", "000002")})
            self.make_cache(root, m)
            engine.save_report(root / "geometry_kd_smoke.json", dict(passed=True, run_signature="synthetic"))
            class Dataset:
                idx_list = m["pilot_ids"]
                def __len__(self): return len(self.idx_list)
            with balanced.recipe_context(), patch.dict(balanced.RECIPE, epochs=2, kd_warmup_steps=1), \
                 patch.object(engine, "tiny_gate"), patch.object(engine, "build_model", side_effect=lambda *a: ToyStudent()), \
                 patch.object(engine, "native_dataset", return_value=Dataset()), patch.object(engine, "verify_files"), \
                 patch.object(engine, "item", side_effect=lambda *a: ({}, target(), {})), \
                 patch.object(engine, "forward", side_effect=lambda model, inputs: model.outputs()):
                a = SimpleNamespace(output=root, device="cpu", role="kd")
                balanced.geometry.train(a, m); first = (root / "kd/training_complete.json").read_bytes()
                balanced.geometry.train(a, m); self.assertEqual(first, (root / "kd/training_complete.json").read_bytes())
            report = engine.checked_report(root / "kd/training_complete.json", m)
            self.assertEqual(report["optimizer_steps"], 4); self.assertFalse(report["tiny_checkpoint_used"])
            for key in balanced.geometry.DISABLED_COMPONENTS: self.assertEqual(sum(report["active_kd_counts"][key].values()), 0)
            self.assertTrue(all(sum(report["active_kd_counts"][key].values()) > 0 for key in COMPONENT_WEIGHTS))

    def test_four_way_comparison_preserves_recall_gate_and_no_deployment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); endpoint = root / "kd/last.pth"; endpoint.parent.mkdir(); endpoint.write_bytes(b"rebalanced")
            digest = engine.sha256(endpoint)
            m = dict(signature_sha256="synthetic", reference_results=dict(control=evaluated(ap=1.), kd=evaluated(ap=1.2, pedestrian=.53)))
            engine.save_report(root / "kd/training_complete.json", dict(complete=True, run_signature="synthetic", train_images=512,
                               optimizer_steps=2560, fixed_endpoint_epoch=5, same_initialization_sha256=engine.resolution.CHECKPOINT_SHA,
                               tiny_checkpoint_used=False, external_teacher_used=True, architecture_changed=False, checkpoint_sha256=digest,
                               active_kd_counts={k: {"Pedestrian": int(k in COMPONENT_WEIGHTS), "Vehicle": 0} for k in engine.COMPONENTS}))
            result = evaluated(ap=1.3, pedestrian=.55)
            result.update(complete=True, evaluated_images=3769, run_signature="synthetic", checkpoint_sha256=digest)
            engine.write_json(root / "evaluation/kd/accuracy.json", result)
            old = dict(results=dict(geometry_only=evaluated(ap=.7, pedestrian=.49)))
            with patch.object(balanced, "reviewed_audit", return_value=({}, old, {}, {})):
                balanced.compare(SimpleNamespace(output=root), m)
                value = engine.checked_report(root / "balanced_geometry_comparison.json", m)
                self.assertEqual(set(value["results"]), {"label_only", "all_heads", "geometry_angle1", "geometry_angle001"})
                self.assertFalse(value["benefit_guardrail_passed"])
                self.assertFalse(value["comparisons"]["label_only"]["nearby_recall_guardrails"]["Pedestrian"])
                self.assertFalse(value["deployment_qualified"]); self.assertFalse(value["full_training_authorized"])
                endpoint.write_bytes(b"changed")
                with self.assertRaises(RuntimeError): balanced.compare(SimpleNamespace(output=root), m)

    def test_bundle_keeps_evidence_but_not_weights_or_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "assets").mkdir(); (root / "assets/package.json").write_text("{}")
            (root / "balanced_geometry_comparison.json").write_text("{}"); (root / "last.pth").write_bytes(b"weights")
            balanced.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root / "a2_lite_balanced_geometry_results.zip") as z:
                self.assertEqual(z.namelist(), ["balanced_geometry_comparison.json"])

    def test_notebook_standalone_one_arm_and_optional_verified_archive(self):
        nb = json.loads((ROOT / "notebooks/MonoDETR_A2_Lite_512_Balanced_Geometry_KD_Colab.ipynb").read_text())
        codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(codes), 8)
        for i, code in enumerate(codes): compile(code, f"balanced-{i}", "exec")
        text = "\n".join(codes); self.assertIn(balanced.REVISION, text)
        for stage in ("prepare", "smoke", "tiny", "cache", "kd-smoke", "train", "evaluate", "compare", "bundle"):
            self.assertIn(f"stage('{stage}'", text)
        self.assertEqual(text.count("stage('train')"), 1)
        for forbidden in ("setup.py", "venv", "coremltools", "reset --hard", "--role"):
            self.assertNotIn(forbidden, text)
        self.assertLess(text.index("'pull','--ff-only'"), text.index("SCRIPT.is_file()"))
        self.assertIn("'--audit',AUDIT", text); self.assertIn("'--pilot',PILOT", text)
        self.assertIn("CACHE/'kitti_input_cache.json'", text)
        self.assertIn("'restore'", text); self.assertIn("'--workers','4'", text)
        self.assertIn("balanced_geometry_comparison.json", text)
        functions = {n.name for n in ast.parse(codes[0]).body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"stage", "run_logged"} <= functions)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code": self.assertEqual(cell["outputs"], [])


if __name__ == "__main__": unittest.main()
