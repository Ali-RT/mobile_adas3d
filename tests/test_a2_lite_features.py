"""Foreground feature alignment, teacher isolation and matched/resumable pilot."""
import copy
import importlib.util
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
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import recover_a2_lite_features as r
from models.a2_lite_feature_kd import (EncoderFeatureTap, object_features, feature_kd_loss,
                                     STUDENT_SHAPES, TEACHER_SHAPES, CHANNELS, ROI_SIZE)

# File-based helper import avoids third-party packages called "tests" in Colab.
spec = importlib.util.spec_from_file_location("feature_recovery_helpers", ROOT / "tests/test_a2_lite_recovery.py")
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)
target, outputs = helpers.target, helpers.outputs


def memory(shapes, coordinate=False):
    rows = []
    for level, (h, w) in enumerate(shapes):
        if coordinate:
            yy, xx = torch.meshgrid((torch.arange(h) + .5) / h, (torch.arange(w) + .5) / w, indexing="ij")
            values = xx[..., None] + 2 * yy[..., None] + torch.arange(CHANNELS)[None, None, :] / 100 + level
            rows.append(values.reshape(1, h * w, CHANNELS))
        else:
            rows.append(torch.randn(1, h * w, CHANNELS))
    return torch.cat(rows, dim=1)


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.depthaware_transformer = torch.nn.Module()
        self.depthaware_transformer.encoder = torch.nn.Linear(CHANNELS, CHANNELS, bias=False)
        with torch.no_grad():
            self.depthaware_transformer.encoder.weight.copy_(torch.eye(CHANNELS))
        self.register_buffer("tokens", memory(STUDENT_SHAPES, True))
        for name, value in outputs().items():
            self.register_parameter(name, torch.nn.Parameter(value.clone()))

    def result(self):
        value = self.depthaware_transformer.encoder(self.tokens)
        return {name: p + .001 * value.mean() for name, p in self.named_parameters() if name.startswith("pred_")}


class Dataset:
    idx_list = ["000001", "000002"]
    def __len__(self):
        return len(self.idx_list)


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(123)

    def manifest(self, root):
        return dict(signature_sha256="synthetic", output=str(root), pilot_ids=Dataset.idx_list,
                    data_file_sha256={s: {"image_2": "frozen"} for s in Dataset.idx_list},
                    teacher_checkpoint_sha256="teacher", local_feature_cache=str(root / "local"))

    def make_cache(self, root, m):
        hashes = {}
        for sample in Dataset.idx_list:
            value = dict(**r.feature_identity(m, sample), target_signature=r.engine.target_signature(target()),
                         labels=target()["labels"], eligible=torch.tensor([True, True]),
                         object_to_teacher_query=torch.tensor([42, 47]),
                         features=torch.randn(2, 4, CHANNELS, ROI_SIZE, ROI_SIZE))
            path = root / "teacher_features" / f"{sample}.pt"
            r.engine.atomic_checkpoint(path, value)
            hashes[sample] = r.engine.sha256(path)
        report = dict(complete=True, run_signature=m["signature_sha256"], images=2,
                      kd_pilot_authorized=True, cache_sha256=hashes)
        r.engine.save_report(root / "feature_cache_complete.json", report)
        return r.local_cache(m)

    def test_coordinate_alignment_preserves_normalized_box_geometry_across_resolutions(self):
        student = object_features(memory(STUDENT_SHAPES, True), target(), STUDENT_SHAPES)
        teacher = object_features(memory(TEACHER_SHAPES, True), target(), TEACHER_SHAPES)
        self.assertEqual(student.shape, (2, 4, 256, 3, 3))
        self.assertTrue(torch.allclose(student, teacher, atol=2e-6, rtol=1e-6))
        # Small and large boxes receive the same point/level budget.
        self.assertEqual(student[0].numel(), student[1].numel())

    def test_foreground_sampling_has_student_gradient_but_no_background_pixel_loss(self):
        source = memory(STUDENT_SHAPES).requires_grad_()
        feature = object_features(source, target(), STUDENT_SHAPES)
        loss, counts = feature_kd_loss(feature, torch.randn_like(feature), target()["labels"], torch.ones(2, dtype=torch.bool))
        loss.backward()
        self.assertGreater(float(source.grad.abs().sum()), 0)
        self.assertEqual(float(source.grad[0, 0].abs().sum()), 0.)
        self.assertEqual(float(source.grad[0, -1].abs().sum()), 0.)
        self.assertEqual(counts, dict(Pedestrian=1, Vehicle=1))

    def test_teacher_is_detached_and_equal_features_have_zero_loss(self):
        student = torch.randn(2, 4, 256, 3, 3, requires_grad=True)
        teacher = torch.randn_like(student, requires_grad=True)
        loss, _ = feature_kd_loss(student, teacher, target()["labels"], torch.ones(2, dtype=torch.bool))
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertGreater(float(student.grad.abs().sum()), 0.)
        same, _ = feature_kd_loss(student, student.detach(), target()["labels"], torch.ones(2, dtype=torch.bool))
        self.assertLess(float(same.detach()), 1e-6)

    def test_present_classes_are_equal_weight_not_object_frequency(self):
        student = torch.zeros(4, 4, 256, 3, 3)
        teacher = torch.zeros_like(student)
        student[:, :, 0] = 1
        teacher[:3, :, 0] = 1
        teacher[3, :, 1] = 1
        loss, counts = feature_kd_loss(student, teacher, torch.tensor([1, 1, 1, 0]), torch.ones(4, dtype=torch.bool))
        self.assertAlmostEqual(float(loss), .5)
        self.assertEqual(counts, dict(Pedestrian=1, Vehicle=3))

    def test_ineligible_objects_and_empty_targets_have_differentiable_zero_loss(self):
        for n in (0, 2):
            student = torch.randn(n, 4, 256, 3, 3, requires_grad=True)
            loss, counts = feature_kd_loss(student, torch.randn_like(student), target()["labels"][:n], torch.zeros(n, dtype=torch.bool))
            self.assertEqual(float(loss.detach()), 0.)
            loss.backward()
            self.assertEqual(float(student.grad.abs().sum()), 0.)
            self.assertEqual(sum(counts.values()), 0)
        empty = {k: v[:0] for k, v in target().items()}
        sampled = object_features(memory(STUDENT_SHAPES).requires_grad_(), empty, STUDENT_SHAPES)
        self.assertEqual(sampled.shape, (0, 4, 256, 3, 3))

    def test_invalid_shapes_nonfinite_classes_and_zero_norm_stop(self):
        feature = torch.randn(2, 4, 256, 3, 3)
        for changed in (feature.double(), feature[:, :3], feature * float("nan"), torch.zeros_like(feature)):
            with self.assertRaises(ValueError):
                feature_kd_loss(changed, feature, target()["labels"], torch.ones(2, dtype=torch.bool))
        with self.assertRaises(ValueError):
            feature_kd_loss(feature, feature, torch.tensor([0, 2]), torch.ones(2, dtype=torch.bool))
        bad = target(); bad["boxes_3d"][0, 2:] = 0
        with self.assertRaisesRegex(ValueError, "visible area"):
            object_features(memory(STUDENT_SHAPES), bad, STUDENT_SHAPES)

    def test_hook_does_not_change_forward_state_and_is_removed(self):
        model = ToyModel()
        baseline = model.result()
        before = copy.deepcopy(model.state_dict())
        with EncoderFeatureTap(model) as tap:
            actual = model.result()
            self.assertTrue(all(torch.equal(actual[k], baseline[k]) for k in actual))
            self.assertTrue(tap.take().requires_grad)
            with self.assertRaisesRegex(RuntimeError, "did not fire"):
                tap.take()
            model.result()
            with self.assertRaisesRegex(RuntimeError, "consumed"):
                model.result()
        self.assertEqual(len(model.depthaware_transformer.encoder._forward_hooks), 0)
        self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))

    def test_recipe_is_fresh_matched_five_epoch_feature_only_with_original_gt(self):
        self.assertEqual(r.RECIPE["gt_weights"], r.engine.GT_WEIGHTS)
        self.assertEqual(r.RECIPE["train_images"] * r.RECIPE["epochs"], 2560)
        self.assertEqual(r.RECIPE["prediction_kd_components"], [])
        self.assertFalse(r.RECIPE["feature_adapter"])
        self.assertFalse(r.RECIPE["background_kd"])
        self.assertFalse(r.RECIPE["gt_loss_changed"])
        self.assertFalse(r.RECIPE["architecture_changed"])
        self.assertTrue(r.RECIPE["fresh_matched_control"])
        self.assertEqual(r.RECIPE["feature_gradient_target_ratio"], .05)
        before = r.engine.RECIPE, r.engine.REVISION
        with self.assertRaisesRegex(RuntimeError, "synthetic"):
            with r.recipe_context():
                self.assertIs(r.engine.RECIPE, r.RECIPE)
                raise RuntimeError("synthetic")
        self.assertEqual((r.engine.RECIPE, r.engine.REVISION), before)

    def test_cache_rejects_validation_corruption_and_wrong_target(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = self.manifest(root)
            local, report = self.make_cache(root, m)
            sample = Dataset.idx_list[0]; path = local / f"{sample}.pt"
            r.read_feature(path, m, sample, target(), report["cache_sha256"][sample])
            with self.assertRaisesRegex(RuntimeError, "TRAIN-only"):
                r.read_feature(path, m, "999999")
            changed = target(); changed["depth"][0] += 1
            with self.assertRaisesRegex(RuntimeError, "GT encoding"):
                r.read_feature(path, m, sample, changed)
            path.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "bytes changed"):
                r.local_cache(m)

    def test_local_feature_cache_reuses_verified_files_without_drive_rereads(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = self.manifest(root)
            local, _ = self.make_cache(root, m)
            for path in (root / "teacher_features").glob("*.pt"):
                path.unlink()
            self.assertEqual(r.local_cache(m)[0], local)

    def test_feature_smoke_is_finite_both_classes_and_zero_update(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = self.manifest(root)
            self.make_cache(root, m)
            model = ToyModel(); before = copy.deepcopy(model.state_dict())
            with patch.object(r.engine, "tiny_gate"), patch.object(r.engine, "build_model", return_value=model), \
                 patch.object(r.engine, "native_dataset", return_value=Dataset()), patch.object(r.engine, "verify_files"), \
                 patch.object(r.engine, "item", side_effect=lambda *a: ({}, target(), {})), \
                 patch.object(r.engine, "forward", side_effect=lambda model, inputs: model.result()):
                r.feature_smoke(SimpleNamespace(output=root, device="cpu"), m)
                r.feature_smoke(SimpleNamespace(output=root, device="cpu"), m)
            report = r.engine.checked_report(root / "feature_kd_smoke.json", m)
            self.assertTrue(report["passed"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertEqual(report["active_counts"], dict(Pedestrian=2, Vehicle=2))
            self.assertTrue(all(v["encoder_kd_gradient_norm"] > 0 for v in report["checks"]))
            self.assertTrue(all(torch.equal(v, model.state_dict()[k]) for k, v in before.items()))

    def test_control_matches_frozen_gt_engine_bit_for_bit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = self.manifest(root)
            old, new = ToyModel(), ToyModel()
            new.load_state_dict(old.state_dict())
            with r.recipe_context(), patch.object(r.engine, "verify_files"), \
                 patch.object(r.engine, "item", side_effect=lambda *a: ({}, target(), {})), \
                 patch.object(r.engine, "forward", side_effect=lambda model, inputs: model.result()):
                r.engine.run_steps(m, old, r.engine.optimizer(old), Dataset(), 4, root / "old.pth", "control")
                r.run_steps(m, new, r.engine.optimizer(new), Dataset(), 4, root / "new.pth", "control", "cpu")
            self.assertTrue(all(torch.equal(v, new.state_dict()[k]) for k, v in old.state_dict().items()))

    def test_feature_training_resume_matches_uninterrupted_and_teacher_is_not_optimized(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = self.manifest(root)
            local, report = self.make_cache(root, m)
            full, interrupted = ToyModel(), ToyModel()
            interrupted.load_state_dict(full.state_dict())
            cache_bytes = {p.name: p.read_bytes() for p in local.glob("*.pt")}
            with r.recipe_context(), patch.object(r.engine, "verify_files"), \
                 patch.object(r.engine, "item", side_effect=lambda *a: ({}, target(), {})), \
                 patch.object(r.engine, "forward", side_effect=lambda model, inputs: model.result()):
                r.run_steps(m, full, r.engine.optimizer(full), Dataset(), 4, root / "full.pth", "kd", "cpu", local, report)
                r.run_steps(m, interrupted, r.engine.optimizer(interrupted), Dataset(), 2, root / "resume.pth", "kd", "cpu", local, report)
                fresh = ToyModel()
                step, _, counts, _ = r.run_steps(m, fresh, r.engine.optimizer(fresh), Dataset(), 4, root / "resume.pth", "kd", "cpu", local, report)
                r.run_steps(m, fresh, r.engine.optimizer(fresh), Dataset(), 4, root / "resume.pth", "kd", "cpu", local, report)
                with patch.object(r, "device_identity", return_value={"device": "cpu", "gpu": "changed"}):
                    with self.assertRaisesRegex(RuntimeError, "Resume identity"):
                        r.run_steps(m, fresh, r.engine.optimizer(fresh), Dataset(), 4, root / "resume.pth", "kd", "cpu", local, report)
            self.assertEqual(step, 4)
            self.assertEqual(counts, dict(Pedestrian=4, Vehicle=4))
            self.assertTrue(all(torch.equal(v, fresh.state_dict()[k]) for k, v in full.state_dict().items()))
            self.assertEqual(cache_bytes, {p.name: p.read_bytes() for p in local.glob("*.pt")})
            self.assertEqual(len(fresh.depthaware_transformer.encoder._forward_hooks), 0)

    def test_historical_outputs_are_rejected_before_any_writes(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            a = SimpleNamespace(pilot=root / "pilot.zip", initialization=root / "init/weights.pth",
                                teacher_checkpoint=root / "teacher/weights.pth", compact=root / "compact.zip",
                                teacher_export=root / "teacher.zip", output=root / "old_run/child")
            with patch.object(r.reference, "reviewed_pilot", return_value=(dict(output=str(root / "old_run")), {}, b"", b"")):
                with self.assertRaisesRegex(RuntimeError, "read-only"):
                    r.prepare(a)
            self.assertFalse(a.output.exists())

    def test_prepare_records_fresh_runtime_and_preserves_original_subset_and_weights(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            init = root / "init/weights.pth"; init.parent.mkdir(); init.write_bytes(b"init")
            teacher = root / "teacher/epoch130.pth"; teacher.parent.mkdir(); teacher.write_bytes(b"teacher")
            train = [f"{i:06d}" for i in range(3712)]
            val = [f"{i:06d}" for i in range(3712, 7481)]
            ids = train[:512]; files = {i: dict(image_2="image", label_2="label", calib="calib") for i in train + val}
            old = dict(output=str(root / "old_output"), source=str(root / "old_source"), dataset=str(root / "old_data"),
                       source_sha256="frozen_source", train_ids=train, val_ids=val, pilot_ids=ids,
                       data_file_sha256=files, teacher_model_sha256="onnx", config=dict(model={}, dataset={}),
                       signature_sha256="old_signature", cpu_environment=dict(torch="older"))
            a = SimpleNamespace(pilot=root / "pilot.zip", compact=root / "compact.zip", initialization=init,
                                teacher_checkpoint=teacher, teacher_export=root / "teacher.zip", output=root / "new_output",
                                repo=root / "new_source", dataset=root / "new_data", split_dir=root / "splits",
                                candidate=[], device="cpu")
            original_sha = r.engine.sha256
            def digest(path):
                if Path(path) == init: return r.engine.resolution.CHECKPOINT_SHA
                if Path(path) == teacher: return r.engine.base.q.A2_SHA
                return original_sha(path)
            package = dict(model=dict(sha256="onnx"), fixtures=[dict(sample_id=i) for i in r.engine.resolution.IDS])
            with patch.object(r.reference, "reviewed_pilot", return_value=(old, dict(results={}), b"old_m", b"old_c")), \
                 patch.object(r.engine, "sha256", side_effect=digest), patch.object(r, "implementation", return_value={}), \
                 patch.object(r.engine.base, "prepare_repo"), patch.object(r.engine.base.q, "source_hash", return_value="frozen_source"), \
                 patch.object(r.engine.base.q, "split_ids", side_effect=lambda p, kind: train if kind == "train" else val), \
                 patch.object(r.engine.accuracy, "stage_data", return_value=files), \
                 patch.object(r.engine, "choose_ids", return_value=ids), \
                 patch.object(r.engine.accuracy, "checked_package", return_value=(root / "student_package", {})), \
                 patch.object(r.engine, "materialize_export", return_value=root / "teacher_package"), \
                 patch.object(r.engine, "A2Teacher", return_value=SimpleNamespace(package=package)), \
                 patch.object(r.engine, "runtime_environment", return_value=dict(torch="current")):
                r.prepare(a)
            m = json.loads((a.output / "feature_manifest.json").read_text())
            self.assertEqual(m["pilot_ids"], ids)
            self.assertEqual(m["cpu_environment"], dict(torch="current"))
            self.assertEqual(m["historical_environment"], dict(torch="older"))
            self.assertFalse(m["historical_environment_match_required"])
            self.assertTrue(m["fresh_control_required"])
            self.assertEqual(m["teacher_package_signature"], r.engine.signature(package))
            self.assertEqual((a.output / "reviewed_pilot/recovery_manifest.json").read_bytes(), b"old_m")
            self.assertEqual(init.read_bytes(), b"init")
            self.assertEqual(teacher.read_bytes(), b"teacher")

    def test_teacher_cache_resumes_with_frozen_weights_and_no_validation_features(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = self.manifest(root)
            package = dict(fixtures=[dict(sample_id="fixture", inputs={n: {} for n in r.engine.INPUT_NAMES},
                                         cpu_reference_outputs={n: {} for n in r.engine.OUTPUT_NAMES})])
            m.update(teacher_export="teacher_export", teacher_package_signature=r.engine.signature(package), dataset="data")
            teacher = ToyModel().eval().requires_grad_(False)
            teacher.tokens = memory(TEACHER_SHAPES, True)
            before = copy.deepcopy(teacher.state_dict())
            def forward(model, inputs):
                model.result()  # Fire the real encoder hook without changing the output contract.
                return outputs((42, 47), True)
            original = dict(calibration=torch.zeros(1, 3, 4).numpy())
            with patch.object(r.engine, "tiny_gate"), patch.object(r, "build_teacher", return_value=teacher), \
                 patch.object(r.engine, "A2Teacher", return_value=SimpleNamespace(package=package, root=root)), \
                 patch.object(r.engine, "read_f32", return_value=None), \
                 patch.object(r.engine, "compare_outputs", return_value=dict(passed=True)), \
                 patch.object(r.engine, "native_dataset", return_value=Dataset()), patch.object(r.engine, "verify_files"), \
                 patch.object(r.engine, "item", side_effect=lambda *a: ({}, target(), original)), \
                 patch.object(r.engine, "teacher_inputs", return_value=original), patch.object(r.engine, "preprocessing_comparison"), \
                 patch.object(r.engine, "forward", side_effect=forward), \
                 patch.dict(r.RECIPE, minimum_teacher_matches_per_class=1):
                a = SimpleNamespace(output=root, device="cpu")
                r.cache(a, m)
                data = {p.name: p.read_bytes() for p in (root / "teacher_features").glob("*.pt")}
                r.cache(a, m)
                self.assertEqual(data, {p.name: p.read_bytes() for p in (root / "teacher_features").glob("*.pt")})
            report = r.engine.checked_report(root / "feature_cache_complete.json", m)
            self.assertTrue(report["kd_pilot_authorized"])
            self.assertTrue(report["teacher_weights_buffers_unchanged"])
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertEqual(report["qualified_teacher_matches"], dict(Pedestrian=2, Vehicle=2))
            self.assertEqual(set(report["cache_sha256"]), set(Dataset.idx_list))
            self.assertTrue(all(torch.equal(v, teacher.state_dict()[k]) for k, v in before.items()))

    def test_teacher_builder_strict_loads_full_epoch130_and_freezes_every_parameter(self):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(1))
        model.depthaware_transformer = torch.nn.Module()
        for name in ("encoder", "decoder"):
            module = torch.nn.Module(); module.num_layers = 3
            setattr(model.depthaware_transformer, name, module)
        model.attention = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])
        for module in model.attention:
            module.coreml_spatial_shapes = TEACHER_SHAPES
            module.coreml_export = True
        m = dict(source="/tmp/source", config=dict(model={}), teacher_checkpoint="teacher.pth")
        state = copy.deepcopy(model.state_dict())
        with patch.object(r.engine.base, "inference_model", return_value=model), \
             patch.object(r.torch, "load", return_value=dict(epoch=130, model_state=state)) as load:
            built = r.build_teacher(m, "cpu")
            self.assertFalse(built.training)
            self.assertTrue(all(not p.requires_grad for p in built.parameters()))
            self.assertTrue(load.call_args.kwargs["weights_only"])
        with patch.object(r.engine.base, "inference_model", return_value=model), \
             patch.object(r.torch, "load", return_value=dict(epoch=129, model_state=state)):
            with self.assertRaisesRegex(RuntimeError, "epoch130"):
                r.build_teacher(m, "cpu")

    def comparison_fixture(self, root, pedestrian_drop=False):
        m = dict(signature_sha256="synthetic")
        checks = [dict(max_weight_kd_to_gt_gradient_ratio=.1)]
        calibration = r.calibrate_feature_weight(checks)
        r.engine.save_report(root / "feature_kd_smoke.json", dict(complete=True, passed=True,
            run_signature="synthetic", optimizer_steps=0, weights_buffers_unchanged=True,
            prediction_kd_components=[], active_counts=dict(Pedestrian=1, Vehicle=1),
            checks=checks, calibration=calibration))
        for role in ("control", "kd"):
            checkpoint = root / role / "last.pth"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(role.encode())
            digest = r.engine.sha256(checkpoint)
            r.engine.save_report(root / role / "training_complete.json", dict(complete=True,
                run_signature="synthetic", role=role, checkpoint_sha256=digest, train_images=512,
                optimizer_steps=2560, fixed_endpoint_epoch=5, tiny_checkpoint_used=False,
                architecture_changed=False, same_initialization_sha256=r.engine.resolution.CHECKPOINT_SHA,
                external_teacher_used=role == "kd", prediction_kd_components=[],
                device_identity=dict(device="cpu", gpu=None),
                feature_loss_scale=calibration["loss_scale"],
                active_feature_counts=dict(Pedestrian=5, Vehicle=5) if role == "kd" else {}))
            metrics = {k: 1. if role == "control" else 1.2 for k in r.engine.accuracy.GATES}
            if pedestrian_drop and role == "kd":
                metrics["pedestrian_3d_moderate"] = .5
                metrics["mean_3d_moderate"] = .85
            value = dict(complete=True, run_signature="synthetic", evaluated_images=3769,
                         checkpoint_sha256=digest, metrics=metrics,
                         nearby=dict(classes={c: dict(near_recall=.7) for c in ("Vehicle", "Pedestrian")}))
            r.engine.write_json(root / "evaluation" / role / "accuracy.json", value)
        return m

    def test_calibration_uses_only_fixed_train_gradient_units_and_is_bounded(self):
        result = r.calibrate_feature_weight([dict(max_weight_kd_to_gt_gradient_ratio=v) for v in (.01, .02, .99)])
        self.assertEqual(result["loss_scale"], 2.5)
        self.assertAlmostEqual(result["calibrated_median_weighted_gradient_ratio"], .05)
        self.assertFalse(result["validation_used"])
        self.assertTrue(result["fixed_for_entire_pilot"])
        self.assertFalse(r.calibrate_feature_weight([dict(max_weight_kd_to_gt_gradient_ratio=1e-12)])["within_scale_limit"])
        for value in (0, float("nan"), float("inf")):
            with self.assertRaises(RuntimeError):
                r.calibrate_feature_weight([dict(max_weight_kd_to_gt_gradient_ratio=value)])

    def test_resigned_calibration_drift_cannot_authorize_training(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); self.comparison_fixture(root)
            path = root / "feature_kd_smoke.json"
            m = dict(output=str(root), signature_sha256="synthetic")
            value = r.engine.checked_report(path, m)
            value.pop("report_sha256"); value["calibration"]["loss_scale"] *= 2
            path.unlink(); r.engine.save_report(path, value)
            with self.assertRaisesRegex(RuntimeError, "calibration changed"):
                r.checked_calibration(m)

    def test_comparison_requires_two_fresh_endpoints_and_class_preservation(self):
        for drop in (False, True):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d); m = self.comparison_fixture(root, drop); m["output"] = str(root)
                r.compare(SimpleNamespace(output=root), m)
                report = r.engine.checked_report(root / "feature_comparison.json", m)
                self.assertEqual(report["benefit_guardrail_passed"], not drop)
                self.assertTrue(report["fresh_matched_control"])
                self.assertFalse(report["full_training_authorized"])
                self.assertFalse(report["deployment_qualified"])
                (root / "control/last.pth").write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "fresh matched"):
                    r.compare(SimpleNamespace(output=root), m)

    def test_comparison_cannot_mix_gpus_or_add_prediction_kd(self):
        for key, value in (("device_identity", dict(device="cuda", gpu="other")),
                           ("prediction_kd_components", ["depth"])):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d); m = self.comparison_fixture(root); m["output"] = str(root)
                path = root / "kd/training_complete.json"
                report = r.engine.checked_report(path, m)
                report.pop("report_sha256"); report[key] = value
                path.unlink(); r.engine.save_report(path, report)
                with self.assertRaises(RuntimeError):
                    r.compare(SimpleNamespace(output=root), m)

    def test_failed_feature_smoke_cannot_authorize_either_training_arm(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); m = self.manifest(root)
            r.engine.save_report(root / "feature_kd_smoke.json", dict(passed=False, run_signature="synthetic"))
            with patch.object(r.engine, "tiny_gate"), patch.object(r.engine, "build_model") as build:
                for role in ("control", "kd"):
                    with self.assertRaisesRegex(RuntimeError, "Passed feature smoke"):
                        r.train(SimpleNamespace(output=root, device="cpu", role=role), m)
            build.assert_not_called()

    def test_bundle_excludes_feature_tensors_models_images_and_assets(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "assets").mkdir(); (root / "assets/manifest.json").write_text("{}")
            for name in ("features.pt", "last.pth", "model.onnx", "image.png"):
                (root / name).write_bytes(b"large")
            (root / "feature_comparison.json").write_text("{}")
            r.bundle(SimpleNamespace(output=root))
            with zipfile.ZipFile(root / "a2_lite_feature_results.zip") as archive:
                self.assertEqual(archive.namelist(), ["feature_comparison.json"])

    def test_notebook_is_self_contained_fresh_pair_and_uses_fast_input_staging(self):
        nb = json.loads((ROOT / "notebooks/MonoDETR_A2_Lite_512_Feature_KD_Colab.ipynb").read_text())
        codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(codes), 8)
        for index, code in enumerate(codes):
            compile(code, f"feature-notebook-{index}", "exec")
        text = "\n".join(codes)
        self.assertIn(r.REVISION, text)
        self.assertLess(text.index("'pull','--ff-only'"), text.index("SCRIPT.is_file()"))
        for stage in ("prepare", "smoke", "tiny", "cache", "feature-smoke", "compare", "bundle"):
            self.assertIn(f"stage('{stage}'", text)
        for role in ("control", "kd"):
            self.assertIn(f"stage('train','--role','{role}')", text)
            self.assertIn(f"stage('evaluate','--role','{role}')", text)
        for helper in ("cache_a2_lite_inputs.py", "stage_a2_lite_data.py", "'--workers','4'"):
            self.assertIn(helper, text)
        for forbidden in ("setup.py", "system_site_packages", "coremltools", "reset --hard", "setup_m67_runtime"):
            self.assertNotIn(forbidden, text)


if __name__ == "__main__":
    unittest.main()
