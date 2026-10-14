"""Loss-preserving class decomposition and zero-update gradient audit."""
import copy
import json
import math
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
import audit_a2_lite_gradient_balance as audit
from test_a2_lite_recovery import ToyStudent, target, outputs
from models.a2_lite_recovery_loss import teacher_matches


class GradientTests(unittest.TestCase):
    def test_gt_and_weighted_class_kd_decomposition_preserve_original_losses(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}; teacher = outputs((42, 47), True)
        values, counts = audit.losses(student, teacher, target(), teacher_matches(teacher, target()))
        gt, _, _ = audit.gt_loss(student, target())
        _, original, _ = audit.geometry.geometry_kd_loss(student, teacher, target(), teacher_matches(teacher, target()))
        self.assertTrue(torch.allclose(values["gt_detection"] + values["gt_geometry"], gt))
        for term in audit.GEOMETRY_TERMS:
            self.assertEqual(sum(counts[term].values()), 2)
            self.assertTrue(torch.allclose(sum(values[f"{term}/{c}"] for c in ("Pedestrian", "Vehicle")), original[term] * .25))

    def test_single_eligible_class_keeps_full_component_coefficient(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}; teacher = outputs((42, 47), True)
        teacher["pred_depth"][0, 42, 0] = 100.
        mapping = teacher_matches(teacher, target())
        values, counts = audit.losses(student, teacher, target(), mapping)
        _, original, _ = audit.geometry.geometry_kd_loss(student, teacher, target(), mapping)
        self.assertEqual(counts["depth"]["Pedestrian"], 0)
        self.assertEqual(float(values["depth/Pedestrian"].detach()), 0.)
        self.assertTrue(torch.allclose(values["depth/Vehicle"], original["depth"] * .25))

    def test_teacher_detached_and_no_direct_teacher_detection_loss(self):
        student = {k: v.requires_grad_() for k, v in outputs().items()}
        teacher = {k: v.requires_grad_() for k, v in outputs((42, 47), True).items()}
        values, _ = audit.losses(student, teacher, target(), teacher_matches(teacher, target()))
        loss = sum(v for k, v in values.items() if not k.startswith("gt_")); loss.backward()
        self.assertTrue(all(v.grad is None for v in teacher.values()))
        self.assertTrue(student["pred_logits"].grad is None or float(student["pred_logits"].grad.abs().sum()) == 0)
        self.assertIsNone(student["pred_boxes"].grad)

    def test_full_vector_norm_cosines_and_combination_are_correct(self):
        vectors = {k: [None] for k in audit.TERMS}
        vectors["gt_detection"] = [torch.tensor([1., 0.])]
        vectors["gt_geometry"] = [torch.tensor([0., 1.])]
        vectors["dimensions/Pedestrian"] = [torch.tensor([1., 0.])]
        vectors["depth/Pedestrian"] = [torch.tensor([-1., 0.])]
        vectors["angle/Pedestrian"] = [torch.tensor([0., 2.])]
        gram = audit.gram_by_group(["shared_other"], vectors)
        result = audit.measurements(gram["all"])
        self.assertEqual(gram["all"], gram["shared"])
        self.assertAlmostEqual(result["norms"]["gt"], math.sqrt(2))
        self.assertAlmostEqual(result["norms"]["kd_all"], 2.)
        self.assertAlmostEqual(result["teacher"]["depth"]["detection_gt_cosine"], -1.)
        self.assertAlmostEqual(result["teacher"]["depth"]["gt_cosine"], -1 / math.sqrt(2))
        self.assertAlmostEqual(result["gt_plus_kd_norm"], math.sqrt(10))

    def test_zero_norm_cosine_is_null_and_nonfinite_gradients_rejected(self):
        values = {k: [torch.zeros(2)] for k in audit.TERMS}
        result = audit.measurements(audit.gram_by_group(["shared_other"], values)["all"])
        self.assertIsNone(result["teacher"]["kd_all"]["gt_cosine"])
        self.assertIsNone(result["teacher"]["kd_all"]["kd_to_gt_norm_ratio"])
        values["gt_detection"] = [torch.tensor([float("nan"), 1.])]
        with self.assertRaises(RuntimeError): audit.gram_by_group(["shared_other"], values)

    def test_coordinate_shapes_and_term_contracts_rejected(self):
        values = {k: [None] for k in audit.TERMS}; values["gt_detection"] = [torch.ones(2)]
        values["gt_geometry"] = [torch.ones(3)]
        with self.assertRaises(RuntimeError): audit.gram_by_group(["shared_other"], values)
        values.pop("angle/Vehicle")
        with self.assertRaises(ValueError): audit.gram_by_group(["shared_other"], values)

    def test_parameter_grouping_uses_module_identity_for_refinement_aliases(self):
        model = torch.nn.Module(); model.depthaware_transformer = torch.nn.Module()
        model.depthaware_transformer.decoder = torch.nn.Module()
        model.depthaware_transformer.decoder.bbox_embed = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        model.bbox_embed = model.depthaware_transformer.decoder.bbox_embed
        model.class_embed = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        model.backbone = torch.nn.Linear(2, 2); model.backbone.bias.requires_grad_(False)
        names, parameters, groups = audit.parameter_groups(model)
        self.assertEqual(len(parameters), len({id(p) for p in parameters}))
        for name, group in zip(names, groups):
            if "bbox_embed" in name: self.assertEqual(group, "box_head")
        self.assertNotIn("backbone.bias", names)
        self.assertIn("shared_backbone", groups)

    def test_train_only_first32_and_frame_strata_fixed(self):
        ids = [f"{n:06d}" for n in range(512)]
        m = dict(pilot_ids=ids, train_ids=ids, val_ids=["999999"])
        rows = {s: dict(target_labels=[0] if i % 2 == 0 else [1]) for i, s in enumerate(ids[:32])}
        self.assertEqual(audit.selected_ids(m, rows), ids[:32])
        bad = copy.deepcopy(m); bad["val_ids"] = [ids[0]]
        with self.assertRaises(RuntimeError): audit.selected_ids(bad, rows)
        rows[ids[0]]["target_labels"] = [1]
        with self.assertRaises(RuntimeError): audit.selected_ids(m, rows)

    def test_reviewed_artifacts_are_immutable_and_failed_geometry_is_required(self):
        m = dict(signature_sha256=audit.GEOMETRY_SIGNATURE, implementation={"same": "source"}, recipe=audit.geometry.RECIPE)
        # Use a synthetic signed identity for unit tests, not a real run rewrite.
        m.pop("signature_sha256"); m["signature_sha256"] = audit.engine.signature(m)
        records = {"geometry_manifest.json": m,
                   "geometry_comparison.json": dict(run_signature=m["signature_sha256"], geometry_benefit_guardrail_passed=False),
                   "kd/training_complete.json": dict(run_signature=m["signature_sha256"], checkpoint_sha256=audit.ENDPOINT_SHA),
                   "teacher_cache_complete.json": dict(run_signature=m["signature_sha256"])}
        for name, value in records.items():
            if name != "geometry_manifest.json": value["report_sha256"] = audit.engine.signature(value)
        raw = {k: json.dumps(v).encode() for k, v in records.items()}
        hashes = {k: audit.engine.accuracy.sha_bytes(v) for k, v in raw.items()}
        with patch.object(audit, "REVIEWED", hashes), patch.object(audit, "GEOMETRY_SIGNATURE", m["signature_sha256"]), \
             patch.object(audit.geometry, "implementation", return_value={"same": "source"}), \
             patch.object(audit.engine, "read_record", side_effect=lambda source, name: raw[name]):
            self.assertEqual(audit.reviewed(Path("synthetic"))[0], m)
            raw["geometry_comparison.json"] += b" "
            with self.assertRaises(RuntimeError): audit.reviewed(Path("synthetic"))

    def test_zero_update_probe_and_resume_preserve_all_weights_and_buffers(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); ids = [f"{n:06d}" for n in range(32)]
            m = dict(signature_sha256="synthetic", view=dict(pilot_ids=ids))
            teacher = outputs((42, 47), True)
            arrays = {name: teacher[key].tolist() for name, key in zip(audit.engine.OUTPUT_NAMES, audit.engine.OUTPUT_KEYS)}
            for sample in ids:
                audit.engine.write_json(root / "teacher_subset" / f"{sample}.json", dict(target_signature=audit.engine.target_signature(target()),
                                        outputs=arrays, object_to_teacher_query=[42, 47]))
            class Dataset:
                idx_list = ids
            model = ToyStudent(); before = copy.deepcopy(model.state_dict())
            with patch.object(audit.engine, "build_model", return_value=model), patch.object(audit.engine, "native_dataset", return_value=Dataset()), \
                 patch.object(audit.engine, "verify_files"), patch.object(audit.engine, "item", side_effect=lambda *a: ({}, target(), {})), \
                 patch.object(audit.engine, "forward", side_effect=lambda model, data: model.outputs()) as forward, \
                 patch.object(audit.engine, "optimizer", side_effect=AssertionError("No optimizer allowed")):
                args = SimpleNamespace(output=root, device="cpu", role="initial")
                audit.probe(args, m); count = forward.call_count
                report = (root / "initial_gradient_report.json").read_bytes()
                audit.probe(args, m)
                self.assertEqual(forward.call_count, count)
                self.assertEqual(report, (root / "initial_gradient_report.json").read_bytes())
            for key, value in before.items(): self.assertTrue(torch.equal(value, model.state_dict()[key]))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            result = audit.engine.checked_report(root / "initial_gradient_report.json", m)
            self.assertEqual(result["frames"], 32); self.assertEqual(result["optimizer_steps"], 0)
            self.assertTrue(result["weights_and_buffers_unchanged"])

    def test_wrong_endpoint_payload_rejected_before_gradients(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path = root / "wrong.pth"
            torch.save(dict(run_signature=audit.GEOMETRY_SIGNATURE, role="kd", step=1), path)
            m = dict(view=dict(pilot_ids=[]), weights=dict(geometry_epoch5=dict(path=str(path))))
            with patch.object(audit.engine, "build_model", return_value=ToyStudent()), patch.object(audit.engine, "native_dataset"), \
                 patch.object(audit.engine, "forward", side_effect=AssertionError("Must stop before forward")):
                with self.assertRaisesRegex(RuntimeError, "endpoint payload"):
                    audit.probe(SimpleNamespace(output=root, device="cpu", role="geometry_epoch5"), m)

    def test_summary_inactive_values_are_not_reported_as_conflicts(self):
        zero = [[0.] * len(audit.TERMS) for _ in audit.TERMS]
        result = audit.summarize([dict(gradients=dict(all=audit.measurements(zero)))])
        row = next(r for r in result if r["term"] == "depth")
        self.assertEqual(row["active_frames"], 0)
        self.assertIsNone(row["negative_gt_cosine_fraction"])
        self.assertIsNone(row["dominance_signal_fraction"])

    def test_bundle_excludes_weights_and_images(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "gradient_manifest.json").write_text("{}")
            (root / "last.pth").write_bytes(b"weights"); (root / "image.png").write_bytes(b"pixels")
            audit.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root / "a2_lite_gradient_results.zip") as z:
                self.assertEqual(z.namelist(), ["gradient_manifest.json"])

    def test_notebook_standalone_no_training_no_validation_no_runtime_migration(self):
        nb = json.loads((ROOT / "notebooks/MonoDETR_A2_Lite_Geometry_Gradient_Audit_Colab.ipynb").read_text())
        codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(codes), 5)
        for i, code in enumerate(codes): compile(code, f"gradient-{i}", "exec")
        text = "\n".join(codes); self.assertIn(audit.REVISION, text)
        self.assertIn("def run_logged(", codes[0]); self.assertIn("def stage(", codes[0])
        for term in ("stage('prepare'", "stage('probe'", "stage('bundle'"): self.assertIn(term, text)
        for forbidden in ("stage('train'", "stage('evaluate'", "setup.py", "venv", "coremltools", "reset --hard"):
            self.assertNotIn(forbidden, text)
        self.assertIn("('initial','geometry_epoch5')", text)
        self.assertIn("'--weights-dir',WEIGHTS", text)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code": self.assertEqual(cell["outputs"], [])


if __name__ == "__main__": unittest.main()
