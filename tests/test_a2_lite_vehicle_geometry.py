"""Vehicle contribution preservation, both-class GT and pilot guardrails."""
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
import recover_a2_lite_vehicle_geometry as vehicle
from models.a2_lite_vehicle_geometry_kd import vehicle_geometry_kd_loss
from models.a2_lite_balanced_geometry_kd import balanced_geometry_kd_loss
from models.a2_lite_recovery_loss import gt_loss, teacher_matches
from test_a2_lite_recovery import ToyStudent, target, outputs
import test_a2_lite_balanced_geometry as balanced_test_helpers
from test_a2_lite_full_recovery import evaluated

engine = vehicle.engine


class VehicleTests(unittest.TestCase):
    def test_vehicle_contribution_keeps_original_half_share_on_mixed_frames(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}; teacher = outputs((42, 47), True)
        mapping = teacher_matches(teacher, target()); filtered = mapping.clone(); filtered[0] = -1
        _, class_parts, _ = balanced_geometry_kd_loss(student, teacher, target(), filtered)
        total, parts, counts = vehicle_geometry_kd_loss(student, teacher, target(), mapping)
        for name in vehicle.balanced.COMPONENT_WEIGHTS:
            self.assertTrue(torch.equal(parts[name], class_parts[name] / 2))
            self.assertEqual(counts[name], dict(Pedestrian=0, Vehicle=1))
        self.assertTrue(torch.equal(total, sum(class_parts[k] / 2 for k in vehicle.balanced.COMPONENT_WEIGHTS)))
        old_gradient = torch.autograd.grad(class_parts["angle"], student["pred_angle"], retain_graph=True)[0]
        new_gradient = torch.autograd.grad(parts["angle"], student["pred_angle"], retain_graph=True)[0]
        self.assertTrue(torch.allclose(new_gradient, old_gradient / 2))

    def test_normalization_uses_original_component_masks_not_frame_class_presence(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}; teacher = outputs((42, 47), True)
        teacher["pred_depth"][0, 42, 0] = 100.
        mapping = teacher_matches(teacher, target()); filtered = mapping.clone(); filtered[0] = -1
        _, expected, _ = balanced_geometry_kd_loss(student, teacher, target(), filtered)
        _, parts, _ = vehicle_geometry_kd_loss(student, teacher, target(), mapping)
        self.assertTrue(torch.equal(parts["depth"], expected["depth"]))
        self.assertTrue(torch.equal(parts["angle"], expected["angle"] / 2))

    def test_unequal_object_counts_do_not_change_the_two_class_denominator(self):
        t = {k: torch.cat((v, v[1:]), dim=0) for k, v in target().items()}
        student = outputs(); teacher = outputs((42, 47), True)
        for k in student: student[k][0, 2] = student[k][0, 1]
        for k in teacher: teacher[k][0, 48] = teacher[k][0, 47]
        student = {k: v.requires_grad_() for k, v in student.items()}
        mapping = teacher_matches(teacher, t); filtered = mapping.clone(); filtered[t["labels"] == 0] = -1
        _, per_vehicle, _ = balanced_geometry_kd_loss(student, teacher, t, filtered)
        _, parts, counts = vehicle_geometry_kd_loss(student, teacher, t, mapping)
        for name in vehicle.balanced.COMPONENT_WEIGHTS:
            self.assertEqual(counts[name]["Vehicle"], 2)
            self.assertTrue(torch.equal(parts[name], per_vehicle[name] / 2))

    def test_vehicle_only_frame_retains_full_original_class_share(self):
        t = {k: v[1:] for k, v in target().items()}; teacher = outputs((42, 47), True)
        student = {k: v.requires_grad_() for k, v in outputs().items()}; mapping = teacher_matches(teacher, t)
        old, old_parts, old_counts = balanced_geometry_kd_loss(student, teacher, t, mapping)
        new, parts, counts = vehicle_geometry_kd_loss(student, teacher, t, mapping)
        self.assertTrue(torch.equal(old, new)); self.assertEqual(counts, old_counts)
        for k in old_parts: self.assertTrue(torch.equal(old_parts[k], parts[k]))

    def test_pedestrian_only_and_no_matches_have_differentiable_zero_teacher_loss(self):
        for t, mapping in (({k: v[:1] for k, v in target().items()}, torch.tensor([42])), (target(), torch.tensor([-1, -1]))):
            student = {k: v.requires_grad_() for k, v in outputs().items()}
            loss, _, counts = vehicle_geometry_kd_loss(student, outputs((42, 47), True), t, mapping)
            self.assertEqual(float(loss.detach()), 0.); loss.backward()
            self.assertTrue(all(sum(v.values()) == 0 for v in counts.values()))

    def test_no_direct_pedestrian_teacher_geometry_gradient_and_vehicle_nonzero(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}
        teacher = {k: v.requires_grad_() for k, v in outputs((42, 47), True).items()}
        loss, _, counts = vehicle_geometry_kd_loss(student, teacher, target(), teacher_matches(teacher, target()))
        loss.backward()
        self.assertTrue(all(v.grad is None for v in teacher.values()))
        for name in ("pred_depth", "pred_3d_dim", "pred_angle"):
            self.assertEqual(float(student[name].grad[0, 0].abs().sum()), 0.)
            self.assertGreater(float(student[name].grad[0, 1].abs().sum()), 0.)
        self.assertTrue(all(v["Pedestrian"] == 0 for v in counts.values()))

    def test_bad_vehicle_component_is_not_replaced_by_pedestrian_guidance(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}; teacher = outputs((42, 47), True)
        teacher["pred_depth"][0, 47, 0] = 100.
        _, parts, counts = vehicle_geometry_kd_loss(student, teacher, target(), teacher_matches(teacher, target()))
        self.assertEqual(float(parts["depth"].detach()), 0.)
        self.assertEqual(counts["depth"], dict(Pedestrian=0, Vehicle=0))
        self.assertFalse(vehicle.valid_vehicle_counts(counts))

    def test_ground_truth_losses_and_pedestrian_learning_are_unchanged(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}
        before, parts, _ = gt_loss(student, target())
        with vehicle.recipe_context(): after, actual, _ = engine.gt_loss(student, target())
        self.assertTrue(torch.equal(before, after)); self.assertEqual(parts.keys(), actual.keys())
        after.backward()
        for name in ("pred_logits", "pred_boxes", "pred_depth", "pred_3d_dim", "pred_angle"):
            self.assertGreater(float(student[name].grad[0, 0].abs().sum()), 0.)

    def test_recipe_changes_only_teacher_class_contribution(self):
        for key, value in vehicle.balanced.RECIPE.items():
            if key != "only_objective_change": self.assertEqual(vehicle.RECIPE[key], value, key)
        self.assertEqual(vehicle.RECIPE["teacher_class_weights"], dict(Pedestrian=0., Vehicle=1.))
        self.assertEqual(vehicle.RECIPE["teacher_component_weights"]["angle"], .01)
        self.assertEqual(vehicle.RECIPE["train_images"] * vehicle.RECIPE["epochs"], 2560)
        self.assertEqual(vehicle.RECIPE["gt_weights"], engine.GT_WEIGHTS)
        self.assertFalse(vehicle.RECIPE["gt_class_weighting_changed"])
        self.assertEqual(vehicle.GUARDRAILS, vehicle.balanced.GUARDRAILS)

    def test_context_restores_all_frozen_adapters_after_exception(self):
        g = vehicle.geometry
        old = g.RECIPE, g.REVISION, g.geometry_kd_loss, g.implementation, engine.RECIPE, engine.REVISION, engine.kd_loss
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with vehicle.recipe_context():
                self.assertIs(engine.kd_loss, vehicle_geometry_kd_loss)
                self.assertIs(engine.RECIPE, vehicle.RECIPE)
                raise RuntimeError("synthetic")
        self.assertEqual((g.RECIPE, g.REVISION, g.geometry_kd_loss, g.implementation,
                          engine.RECIPE, engine.REVISION, engine.kd_loss), old)

    def test_parent_reader_resolves_wrapped_root_not_nested_older_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "parent.zip"
            with zipfile.ZipFile(path, "w") as z:
                z.writestr("run/balanced_geometry_comparison.json", b"{}")
                z.writestr("run/geometry_manifest.json", b"correct")
                z.writestr("run/reviewed_gradient/reviewed_geometry/geometry_manifest.json", b"older")
            self.assertEqual(vehicle.parent_record(path, "geometry_manifest.json"), b"correct")

    def test_unreviewed_parent_bytes_stop_before_training(self):
        with patch.object(vehicle, "parent_record", return_value=b"{}"):
            with self.assertRaisesRegex(RuntimeError, "exact reviewed balanced"):
                vehicle.reviewed_balanced(Path("synthetic.zip"))

    def test_load_rejects_resigned_teacher_class_recipe_drift(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); recipe = copy.deepcopy(vehicle.RECIPE); recipe["teacher_class_weights"]["Pedestrian"] = .1
            m = dict(revision=vehicle.REVISION, recipe=recipe); m["signature_sha256"] = engine.signature(m)
            engine.write_json(root / "geometry_manifest.json", m)
            with patch.object(vehicle, "reviewed_balanced", return_value=({}, {}, {})):
                with self.assertRaisesRegex(RuntimeError, "code/packages/recipe changed"):
                    vehicle.load(root)

    def prepare_fixture(self, root, output):
        previous = dict(output=str(root / "old_balanced"), source=str(root / "old_source"), dataset=str(root / "old_data"),
                        reference_run_signature="same", cpu_environment={}, config={}, pilot_ids=["000001"],
                        data_file_sha256={"000001": {}}, source_sha256="same")
        pilot = {**previous, "signature_sha256": "same", "output": str(root / "old_pilot")}
        raw = {"reviewed_gradient/gradient_manifest.json": json.dumps(dict(output=str(root / "old_audit"))).encode(),
               "reviewed_gradient/reviewed_geometry/geometry_manifest.json": json.dumps(dict(output=str(root / "old_geometry"))).encode()}
        args = SimpleNamespace(balanced=root / "parent.zip", pilot=root / "pilot.zip", compact=root / "compact.zip",
                               initialization=root / "weights/init.pth", output=output, repo=root / "new_source", dataset=root / "new_data")
        return args, previous, pilot, raw

    def test_prepare_rejects_historical_output_before_any_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args, previous, pilot, raw = self.prepare_fixture(root, root / "old_balanced/child")
            with patch.object(vehicle, "reviewed_balanced", return_value=(previous, {}, raw)), \
                 patch.object(vehicle.reference, "reviewed_pilot", return_value=(pilot, {}, b"", b"")), \
                 patch.object(vehicle.geometry, "require_reference_environment"), patch.object(engine, "runtime_environment", return_value={}):
                with self.assertRaisesRegex(RuntimeError, "read-only"): vehicle.prepare(args)
            self.assertFalse(args.output.exists())

    def test_prepare_binds_only_vehicle_loss_and_preserves_parent_proof(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args, previous, pilot, raw = self.prepare_fixture(root, root / "new_output")
            observed = []
            def prepare(a): observed.append((copy.deepcopy(engine.RECIPE), engine.kd_loss, a.initialization))
            with patch.object(vehicle, "reviewed_balanced", return_value=(previous, {}, raw)), \
                 patch.object(vehicle.reference, "reviewed_pilot", return_value=(pilot, {}, b"", b"")), \
                 patch.object(vehicle.geometry, "require_reference_environment"), patch.object(engine, "runtime_environment", return_value={}), \
                 patch.object(vehicle.geometry, "prepare", side_effect=prepare): vehicle.prepare(args)
            self.assertEqual(observed, [(vehicle.RECIPE, vehicle_geometry_kd_loss, args.initialization)])
            for name, value in raw.items(): self.assertEqual((args.output / "reviewed_balanced" / name).read_bytes(), value)
            self.assertFalse(Path(previous["output"]).exists())

    def test_smoke_is_vehicle_only_finite_and_zero_update(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = dict(signature_sha256="synthetic", pilot_ids=["000001", "000002"],
                                    data_file_sha256={s: {"image_2": "same"} for s in ("000001", "000002")})
            balanced_test_helpers.BalancedTests().make_cache(root, m)
            class Dataset:
                idx_list = m["pilot_ids"]
            model = ToyStudent(); before = copy.deepcopy(model.state_dict())
            with vehicle.recipe_context(), patch.object(engine, "build_model", return_value=model), \
                 patch.object(engine, "native_dataset", return_value=Dataset()), patch.object(engine, "verify_files"), \
                 patch.object(engine, "item", side_effect=lambda *a: ({}, target(), {})), \
                 patch.object(engine, "forward", side_effect=lambda model, inputs: model.outputs()):
                args = SimpleNamespace(output=root, device="cpu")
                vehicle.kd_smoke(args, m); vehicle.kd_smoke(args, m)
            for key, value in before.items(): self.assertTrue(torch.equal(value, model.state_dict()[key]))
            report = engine.checked_report(root / "vehicle_kd_smoke.json", m)
            self.assertTrue(report["passed"]); self.assertEqual(report["optimizer_steps"], 0)
            self.assertTrue(report["weights_and_buffers_unchanged"])
            self.assertTrue(vehicle.valid_vehicle_counts(report["active_counts"]))
            self.assertTrue(report["vehicle_weight_not_redistributed"])

    def test_training_resumes_with_zero_pedestrian_teacher_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = dict(signature_sha256="synthetic", output=str(root), pilot_ids=["000001", "000002"],
                                    data_file_sha256={s: {"image_2": "same"} for s in ("000001", "000002")})
            balanced_test_helpers.BalancedTests().make_cache(root, m)
            counts = {k: dict(Pedestrian=0, Vehicle=int(k in vehicle.balanced.COMPONENT_WEIGHTS)) for k in engine.COMPONENTS}
            engine.save_report(root / "vehicle_kd_smoke.json", dict(passed=True, run_signature="synthetic", active_counts=counts))
            class Dataset:
                idx_list = m["pilot_ids"]
                def __len__(self): return len(self.idx_list)
            with vehicle.recipe_context(), patch.dict(vehicle.RECIPE, epochs=2, kd_warmup_steps=1), \
                 patch.object(engine, "tiny_gate"), patch.object(engine, "build_model", side_effect=lambda *a: ToyStudent()), \
                 patch.object(engine, "native_dataset", return_value=Dataset()), patch.object(engine, "verify_files"), \
                 patch.object(engine, "item", side_effect=lambda *a: ({}, target(), {})), \
                 patch.object(engine, "forward", side_effect=lambda model, inputs: model.outputs()):
                args = SimpleNamespace(output=root, device="cpu", role="kd")
                vehicle.train(args, m); first = (root / "kd/training_complete.json").read_bytes()
                vehicle.train(args, m); self.assertEqual(first, (root / "kd/training_complete.json").read_bytes())
            report = engine.checked_report(root / "kd/training_complete.json", m)
            self.assertEqual(report["optimizer_steps"], 4); self.assertFalse(report["tiny_checkpoint_used"])
            self.assertTrue(vehicle.valid_vehicle_counts(report["active_kd_counts"]))

    def test_comparison_has_five_arms_recall_guard_and_explicit_stop(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); endpoint = root / "kd/last.pth"; endpoint.parent.mkdir(); endpoint.write_bytes(b"vehicle")
            digest = engine.sha256(endpoint); control = evaluated(ap=1.); all_heads = evaluated(ap=1.2, pedestrian=.53)
            m = dict(signature_sha256="synthetic", reference_results=dict(control=control, kd=all_heads))
            counts = {k: dict(Pedestrian=0, Vehicle=int(k in vehicle.balanced.COMPONENT_WEIGHTS)) for k in engine.COMPONENTS}
            engine.save_report(root / "kd/training_complete.json", dict(complete=True, run_signature="synthetic", train_images=512,
                               optimizer_steps=2560, fixed_endpoint_epoch=5, same_initialization_sha256=engine.resolution.CHECKPOINT_SHA,
                               tiny_checkpoint_used=False, external_teacher_used=True, architecture_changed=False, checkpoint_sha256=digest, active_kd_counts=counts))
            current = evaluated(ap=1.3, pedestrian=.55)
            current.update(complete=True, evaluated_images=3769, run_signature="synthetic", checkpoint_sha256=digest)
            engine.write_json(root / "evaluation/kd/accuracy.json", current)
            parent = dict(results=dict(label_only=control, all_heads=all_heads, geometry_angle1=evaluated(ap=.7, pedestrian=.49),
                                       geometry_angle001=evaluated(ap=.9, pedestrian=.55)))
            with patch.object(vehicle, "reviewed_balanced", return_value=({}, parent, {})):
                vehicle.compare(SimpleNamespace(output=root), m)
                report = engine.checked_report(root / "vehicle_geometry_comparison.json", m)
                self.assertEqual(len(report["results"]), 5); self.assertFalse(report["benefit_guardrail_passed"])
                self.assertIn("Stop this KD recipe", report["next_review"])
                self.assertFalse(report["full_training_authorized"]); self.assertFalse(report["deployment_qualified"])
                endpoint.write_bytes(b"changed")
                with self.assertRaises(RuntimeError): vehicle.compare(SimpleNamespace(output=root), m)

    def test_pedestrian_teacher_count_cannot_pass_train_gate(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = dict(signature_sha256="synthetic")
            counts = {k: dict(Pedestrian=0, Vehicle=int(k in vehicle.balanced.COMPONENT_WEIGHTS)) for k in engine.COMPONENTS}
            counts["depth"]["Pedestrian"] = 1
            engine.save_report(root / "vehicle_kd_smoke.json", dict(passed=True, run_signature="synthetic", active_counts=counts))
            with patch.object(engine, "train", side_effect=AssertionError("Must not train")):
                with self.assertRaisesRegex(RuntimeError, "Vehicle-only gradient"):
                    vehicle.train(SimpleNamespace(output=root), m)

    def test_bundle_excludes_large_assets_and_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "assets").mkdir(); (root / "assets/package.json").write_text("{}")
            (root / "vehicle_geometry_comparison.json").write_text("{}"); (root / "last.pth").write_bytes(b"weights")
            vehicle.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root / "a2_lite_vehicle_geometry_results.zip") as z:
                self.assertEqual(z.namelist(), ["vehicle_geometry_comparison.json"])

    def test_notebook_self_contained_one_new_arm_and_fast_inputs(self):
        nb = json.loads((ROOT / "notebooks/MonoDETR_A2_Lite_512_Vehicle_Geometry_KD_Colab.ipynb").read_text())
        codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(codes), 8)
        for i, code in enumerate(codes): compile(code, f"vehicle-{i}", "exec")
        text = "\n".join(codes); self.assertIn(vehicle.REVISION, text)
        for name in ("prepare", "smoke", "tiny", "cache", "kd-smoke", "train", "evaluate", "compare", "bundle"):
            self.assertIn(f"stage('{name}'", text)
        self.assertEqual(text.count("stage('train')"), 1)
        for forbidden in ("setup.py", "venv", "coremltools", "reset --hard", "--role"):
            self.assertNotIn(forbidden, text)
        self.assertLess(text.index("'pull','--ff-only'"), text.index("SCRIPT.is_file()"))
        self.assertIn("'--balanced',BALANCED", text); self.assertIn("'--pilot',PILOT", text)
        self.assertIn("vehicle_kd_smoke.json", text)
        self.assertIn("teacher_class_weights", text)
        self.assertIn("kitti_a2_lite_accuracy_angle001_a2_lite_512_angle001_kd_r1", text)
        self.assertIn("'--workers','4'", text)
        functions = {n.name for n in ast.parse(codes[0]).body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"stage", "run_logged"} <= functions)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code": self.assertEqual(cell["outputs"], [])


if __name__ == "__main__": unittest.main()
