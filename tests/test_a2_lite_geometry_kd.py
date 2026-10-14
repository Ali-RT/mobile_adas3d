"""Geometry-only KD gradients, frozen reference scope and three-way comparison."""
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
import recover_a2_lite_geometry as geometry
import recover_a2_lite_640 as engine
from models.a2_lite_geometry_kd import geometry_kd_loss, ENABLED_COMPONENTS, DISABLED_COMPONENTS
from models.a2_lite_recovery_loss import kd_loss, gt_loss, teacher_matches
from test_a2_lite_recovery import ToyStudent, target, outputs
from test_a2_lite_full_recovery import evaluated


class GeometryTests(unittest.TestCase):
    def test_only_three_direct_teacher_terms_contribute(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}
        teacher = outputs((42, 47), True); mapping = teacher_matches(teacher, target())
        _, all_parts, _ = kd_loss(student, teacher, target(), mapping)
        loss, parts, counts = geometry_kd_loss(student, teacher, target(), mapping)
        self.assertTrue(torch.equal(loss, sum(all_parts[k] for k in ENABLED_COMPONENTS)))
        for name in DISABLED_COMPONENTS:
            self.assertEqual(float(parts[name].detach()), 0.)
            self.assertEqual(sum(counts[name].values()), 0)
        loss.backward()
        self.assertIsNone(student["pred_logits"].grad)
        self.assertIsNone(student["pred_boxes"].grad)
        for name in ("pred_depth", "pred_3d_dim", "pred_angle"):
            self.assertGreater(float(student[name].grad.abs().sum()), 0.)

    def test_gt_still_trains_classification_and_boxes(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}
        loss, parts, _ = gt_loss(student, target()); loss.backward()
        for name in ("pred_logits", "pred_boxes"):
            self.assertGreater(float(student[name].grad.abs().sum()), 0.)
        for name in ("loss_ce", "loss_center", "loss_bbox", "loss_giou"):
            self.assertIn(name, parts)

    def test_teacher_detached_and_worse_depth_not_copied(self):
        teacher = {k: v.requires_grad_() for k, v in outputs((42, 47), True).items()}
        teacher["pred_depth"].data[0, 42, 0] = 100.
        student = {k: v.requires_grad_() for k, v in outputs().items()}
        loss, _, counts = geometry_kd_loss(student, teacher, target(), teacher_matches(teacher, target()))
        self.assertEqual(counts["depth"]["Pedestrian"], 0)
        self.assertEqual(counts["depth"]["Vehicle"], 1)
        loss.backward(); self.assertTrue(all(v.grad is None for v in teacher.values()))

    def test_empty_mapping_is_differentiable_zero(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}
        loss, _, counts = geometry_kd_loss(student, outputs(perfect=True), target(), torch.tensor([-1, -1]))
        self.assertEqual(float(loss.detach()), 0.); loss.backward()
        self.assertTrue(all(sum(v.values()) == 0 for v in counts.values()))

    def test_teacher_classification_and_boxes_still_filter_object_matching(self):
        teacher = outputs((42, 47), True)
        teacher["pred_logits"][0, 42] = torch.tensor([-10., 10., -10.])
        self.assertEqual(int(teacher_matches(teacher, target())[0]), -1)
        teacher = outputs((42, 47), True); teacher["pred_boxes"][0, 42] = torch.tensor([.99, .99, .01, .01, .01, .01])
        self.assertEqual(int(teacher_matches(teacher, target())[0]), -1)

    def test_same_budget_and_recipe_only_changes_teacher_components(self):
        self.assertEqual(geometry.RECIPE["train_images"], 512)
        self.assertEqual(geometry.RECIPE["epochs"], 5)
        self.assertEqual(geometry.RECIPE["batch_size"], 1)
        self.assertEqual(geometry.RECIPE["train_images"] * geometry.RECIPE["epochs"], 2560)
        for key, value in engine.RECIPE.items(): self.assertEqual(geometry.RECIPE[key], value, key)
        self.assertEqual(geometry.RECIPE["kd_components"], list(ENABLED_COMPONENTS))
        self.assertFalse(geometry.RECIPE["gt_loss_changed"])

    def test_context_restores_original_loss_and_recipe_after_exception(self):
        old = engine.RECIPE, engine.REVISION, engine.kd_loss
        with self.assertRaises(RuntimeError):
            with geometry.recipe_context():
                self.assertIs(engine.kd_loss, geometry_kd_loss)
                self.assertIs(engine.RECIPE, geometry.RECIPE)
                raise RuntimeError("synthetic")
        self.assertEqual((engine.RECIPE, engine.REVISION, engine.kd_loss), old)

    def test_exact_original512_ids_and_no_validation(self):
        train = [f"{n:06d}" for n in range(3712)]
        val = [f"{n:06d}" for n in range(3712, 7481)]
        old = dict(train_ids=train, val_ids=val, pilot_ids=train[::7][:512])
        self.assertEqual(geometry.exact_subset(old, train, val), old["pilot_ids"])
        bad = copy.deepcopy(old); bad["pilot_ids"][0] = val[0]
        with self.assertRaises(RuntimeError): geometry.exact_subset(bad, train, val)
        bad = copy.deepcopy(old); bad["pilot_ids"] = bad["pilot_ids"][:-1]
        with self.assertRaises(RuntimeError): geometry.exact_subset(bad, train, val)

    def test_software_mismatch_does_not_silently_compare_or_restore_cuda(self):
        geometry.require_reference_environment({"torch": "2.11"}, {"torch": "2.11"})
        with self.assertRaisesRegex(RuntimeError, "fresh matched references"):
            geometry.require_reference_environment({"torch": "2.11"}, {"torch": "2.12"})

    def test_resigned_reused_teacher_values_cannot_change(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); old = root / "old"; new = root / "new"
            m = dict(output=str(new), signature_sha256="new", pilot_ids=["000001"],
                     data_file_sha256={"000001": {"image_2": "same"}})
            row = dict(run_signature=geometry.reference.PILOT_SIGNATURE, sample_id="000001",
                       source_sha256=m["data_file_sha256"]["000001"], outputs={"depth": [[20.]]})
            row["record_sha256"] = engine.signature(row)
            source = old / "teacher_cache/000001.json"; engine.write_json(source, row)
            report = old / "teacher_cache_complete.json"
            engine.write_json(report, dict(cache_sha256={"000001": engine.sha256(source)}))
            row.pop("record_sha256"); row.update(run_signature="new", reused_from_run_signature=geometry.reference.PILOT_SIGNATURE)
            row["record_sha256"] = engine.signature(row)
            destination = new / "teacher_cache/000001.json"; engine.write_json(destination, row)
            with patch.object(geometry.reference, "PILOT_CACHE_SHA", engine.sha256(report)):
                geometry.verify_reused_values(old, m)
                row.pop("record_sha256"); row["outputs"]["depth"][0][0] = 21.
                row["record_sha256"] = engine.signature(row); engine.write_json(destination, row)
                with self.assertRaisesRegex(RuntimeError, "differ from original512"):
                    geometry.verify_reused_values(old, m)

    def make_cache(self, root, m):
        teacher = outputs((42, 47), True)
        raw = {name: teacher[key].tolist() for name, key in zip(engine.OUTPUT_NAMES, engine.OUTPUT_KEYS)}
        hashes = {}
        for sample in m["pilot_ids"]:
            row = dict(run_signature=m["signature_sha256"], sample_id=sample, source_sha256=m["data_file_sha256"][sample],
                       target_signature=engine.target_signature(target()), outputs=raw,
                       object_to_teacher_query=[42, 47], target_labels=[0, 1])
            row["record_sha256"] = engine.signature(row)
            path = root / "teacher_cache" / f"{sample}.json"; engine.write_json(path, row); hashes[sample] = engine.sha256(path)
        engine.save_report(root / "teacher_cache_complete.json", dict(complete=True, run_signature=m["signature_sha256"],
                           kd_pilot_authorized=True, cache_sha256=hashes))

    def test_real_loss_gradient_smoke_has_no_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = dict(signature_sha256="synthetic", pilot_ids=["000001", "000002"],
                                    data_file_sha256={s: {"image_2": "same"} for s in ("000001", "000002")})
            self.make_cache(root, m)
            class Dataset:
                idx_list = m["pilot_ids"]
            model = ToyStudent(); before = copy.deepcopy(model.state_dict())
            with patch.object(engine, "build_model", return_value=model), patch.object(engine, "native_dataset", return_value=Dataset()), \
                 patch.object(engine, "verify_files"), patch.object(engine, "item", side_effect=lambda *a: ({}, target(), {})), \
                 patch.object(engine, "forward", side_effect=lambda model, inputs: model.outputs()):
                a = SimpleNamespace(output=root, device="cpu")
                geometry.kd_smoke(a, m); geometry.kd_smoke(a, m)
            for key, value in before.items(): self.assertTrue(torch.equal(value, model.state_dict()[key]))
            report = engine.checked_report(root / "geometry_kd_smoke.json", m)
            self.assertTrue(report["passed"]); self.assertEqual(report["optimizer_steps"], 0)
            self.assertTrue(all(report["component_gradient_abs_sum"].values()))
            for key in DISABLED_COMPONENTS: self.assertEqual(sum(report["active_counts"][key].values()), 0)

    def test_training_resumes_and_retains_zero_disabled_teacher_counts(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = dict(signature_sha256="synthetic", output=str(root), pilot_ids=["000001", "000002"],
                                    data_file_sha256={s: {"image_2": "same"} for s in ("000001", "000002")})
            self.make_cache(root, m)
            engine.save_report(root / "geometry_kd_smoke.json", dict(passed=True, run_signature="synthetic"))
            class Dataset:
                idx_list = m["pilot_ids"]
                def __len__(self): return len(self.idx_list)
            with geometry.recipe_context(), patch.dict(geometry.RECIPE, epochs=2, kd_warmup_steps=1), \
                 patch.object(engine, "tiny_gate"), patch.object(engine, "build_model", side_effect=lambda *a: ToyStudent()), \
                 patch.object(engine, "native_dataset", return_value=Dataset()), patch.object(engine, "verify_files"), \
                 patch.object(engine, "item", side_effect=lambda *a: ({}, target(), {})), \
                 patch.object(engine, "forward", side_effect=lambda model, inputs: model.outputs()):
                a = SimpleNamespace(output=root, device="cpu", role="kd")
                geometry.train(a, m); before = (root / "kd/training_complete.json").read_bytes()
                geometry.train(a, m); self.assertEqual(before, (root / "kd/training_complete.json").read_bytes())
            report = engine.checked_report(root / "kd/training_complete.json", m)
            self.assertEqual(report["optimizer_steps"], 4); self.assertFalse(report["tiny_checkpoint_used"])
            for key in DISABLED_COMPONENTS: self.assertEqual(sum(report["active_kd_counts"][key].values()), 0)
            self.assertTrue(all(sum(report["active_kd_counts"][key].values()) > 0 for key in ENABLED_COMPONENTS))

    def test_three_way_comparison_is_not_automatic_deployment(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); endpoint = root / "kd/last.pth"; endpoint.parent.mkdir(); endpoint.write_bytes(b"geometry")
            digest = engine.sha256(endpoint)
            m = dict(signature_sha256="synthetic", reference_run_signature="old",
                     reference_results=dict(control=evaluated(ap=1.), kd=evaluated(ap=1.2, pedestrian=.53)))
            engine.save_report(root / "kd/training_complete.json", dict(complete=True, run_signature="synthetic",
                               train_images=512, optimizer_steps=2560, fixed_endpoint_epoch=5,
                               same_initialization_sha256=engine.resolution.CHECKPOINT_SHA, tiny_checkpoint_used=False,
                               external_teacher_used=True, architecture_changed=False, checkpoint_sha256=digest,
                               active_kd_counts={k: {"Pedestrian": int(k in ENABLED_COMPONENTS), "Vehicle": 0} for k in engine.COMPONENTS}))
            result = evaluated(ap=1.3, pedestrian=.61)
            result.update(complete=True, evaluated_images=3769, run_signature="synthetic", checkpoint_sha256=digest)
            engine.write_json(root / "evaluation/kd/accuracy.json", result)
            geometry.compare(SimpleNamespace(output=root), m)
            value = engine.checked_report(root / "geometry_comparison.json", m)
            self.assertEqual(set(value["results"]), {"label_only", "all_heads", "geometry_only"})
            self.assertTrue(value["geometry_benefit_guardrail_passed"])
            self.assertFalse(value["deployment_qualified"]); self.assertFalse(value["full_training_authorized"])
            self.assertTrue(value["historical_references"])
            endpoint.write_bytes(b"changed")
            with self.assertRaises(RuntimeError): geometry.compare(SimpleNamespace(output=root), m)

    def test_bundle_keeps_results_not_large_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "assets").mkdir(); (root / "assets/manifest.json").write_text("{}")
            (root / "last.pth").write_bytes(b"large"); (root / "geometry_comparison.json").write_text("{}")
            geometry.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root / "a2_lite_geometry_results.zip") as z:
                self.assertEqual(z.namelist(), ["geometry_comparison.json"])

    def test_notebook_is_standalone_and_trains_one_new_arm(self):
        nb = json.loads((ROOT / "notebooks/MonoDETR_A2_Lite_512_Geometry_KD_Colab.ipynb").read_text())
        codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(codes), 8)
        for i, code in enumerate(codes): compile(code, f"geometry-{i}", "exec")
        text = "\n".join(codes); self.assertIn(geometry.REVISION, text)
        for name in ("prepare", "smoke", "tiny", "cache", "kd-smoke", "train", "evaluate", "compare", "bundle"):
            self.assertIn(f"stage('{name}'", text)
        self.assertEqual(text.count("stage('train')"), 1)
        self.assertNotIn("--role", text)
        for forbidden in ("setup.py", "venv", "coremltools", "reset --hard"):
            self.assertNotIn(forbidden, text)
        self.assertLess(text.index("'pull','--ff-only'"), text.index("SCRIPT.is_file()"))
        self.assertIn("'--pilot',PILOT", text)
        self.assertIn("geometry_manifest.json", text)
        functions = {n.name for n in ast.parse(codes[0]).body if isinstance(n, ast.FunctionDef)}
        self.assertTrue({"stage", "run_logged"} <= functions)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code": self.assertFalse(cell["outputs"])


if __name__ == "__main__": unittest.main()
