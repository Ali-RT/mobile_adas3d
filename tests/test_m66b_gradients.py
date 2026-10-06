"""CPU regressions; native CUDA execution remains a separate Colab check."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import copy
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import probe_m66_kd_gradients as probe
from test_m66_feature_pilot import ToyModel, summary_fixture
from test_m65_preservation_control import fixture, matcher

NAMES = [f"depth_embed.{head}.layers.{layer}.{kind}"
         for head in range(3) for layer in range(2) for kind in ("weight", "bias")]
CHECKS = dict(model_parameters_and_buffers_unchanged=True, anchor_parameters_and_buffers_unchanged=True,
              teacher_parameters_and_buffers_unchanged=True, parameter_grad_buffers_empty=True,
              historical_files_unchanged=True, no_pedestrian_external_targets=True)


def report_fixture(role="original", active=True):
    ids = [f"{i:06d}" for i in range(64)]
    identity = dict(revision=probe.REVISION, sample_ids=ids)
    gradients = probe.gradient_measurements(NAMES, {key: [torch.ones(1)] * 12 for key in probe.TERMS})
    rows = [dict(batch=i + 1, sample_ids=ids[4 * i:4 * i + 4], input_sha256="a" * 64,
                 weighted_losses=dict(gt=1., preservation=.1, kd=.01), gt_counts=dict(Vehicle=4, Pedestrian=1),
                 kd_pairs=dict(Vehicle=int(active), Pedestrian=0, vehicle_near=int(active), vehicle_far=0),
                 gradients=copy.deepcopy(gradients))
            for i in range(16)]
    return probe.signed(dict(complete=True, identity=identity, role=role, optimizer_steps=0,
        checked_train_images=64, trainable_parameter_names=NAMES, checks=CHECKS, rows=rows,
        training_authorized=False, checkpoint_promotion_authorized=False, deployment_authorized=False))


class GradientMathTests(unittest.TestCase):
    def stats(self, g, s, k):
        values = [torch.tensor(x, dtype=torch.float64) if x is not None else None for x in (g, s, k)]
        return probe.gradient_measurements([NAMES[0]], dict(zip(probe.TERMS, ([v] for v in values))))["all"]

    def test_parallel_weighted_norms(self):
        result = self.stats([3., 4.], [0.3, 0.4], [0.03, 0.04])
        self.assertAlmostEqual(result["gt_norm"], 5.)
        self.assertAlmostEqual(result["preservation_to_gt_norm_ratio"], .1)
        self.assertAlmostEqual(result["kd_to_gt_norm_ratio"], .01)
        self.assertAlmostEqual(result["kd_to_control_norm_ratio"], .05 / 5.5)
        self.assertAlmostEqual(result["total_with_kd_norm"], 5.55)
        self.assertAlmostEqual(result["kd_control_cosine"], 1.)
        self.assertLess(result["total_deflection_degrees"], 1e-5)

    def test_orthogonal_deflection(self):
        result = self.stats([1., 0.], None, [0., 1.])
        self.assertEqual(result["kd_gt_cosine"], 0)
        self.assertAlmostEqual(result["total_deflection_degrees"], 45.)
        self.assertAlmostEqual(result["total_with_kd_norm"], math.sqrt(2))

    def test_opposing_and_cancellation(self):
        result = self.stats([1.], None, [-2.])
        self.assertEqual(result["kd_control_cosine"], -1.)
        self.assertEqual(result["total_deflection_degrees"], 180.)
        result = self.stats([1.], None, [-1.])
        self.assertEqual(result["total_with_kd_norm"], 0.)
        self.assertIsNone(result["total_deflection_degrees"])

    def test_zero_is_connected_none_is_disconnected(self):
        result = self.stats(None, [0., 0.], [0., 0.])
        self.assertIsNone(result["kd_to_gt_norm_ratio"])
        self.assertIsNone(result["kd_control_cosine"])
        self.assertEqual(result["reach"]["gt"]["connected_tensors"], 0)
        self.assertEqual(result["reach"]["kd"]["connected_tensors"], 1)
        self.assertEqual(result["reach"]["kd"]["nonzero_tensors"], 0)

    def test_vector_dot_not_mean_of_tensor_cosines(self):
        result = probe.gradient_measurements(NAMES[:2], dict(
            gt=[torch.tensor([100.]), torch.tensor([1.])], preservation=[None, None],
            kd=[torch.tensor([100.]), torch.tensor([-1.])]))["all"]
        self.assertAlmostEqual(result["kd_gt_cosine"], 9999 / 10001)

    def test_bad_shapes_lengths_and_names_rejected(self):
        with self.assertRaises(ValueError):
            probe.gradient_measurements(NAMES[:1], dict(gt=[], preservation=[None], kd=[None]))
        with self.assertRaises(ValueError):
            self.stats([1.], None, [1., 2.])
        with self.assertRaises(ValueError):
            probe.parameter_group("backbone.conv.weight")
        with self.assertRaises(ValueError):
            probe.parameter_group("depth_embed.3.layers.0.weight")

    def test_nonfinite_rejected(self):
        with self.assertRaises(RuntimeError):
            self.stats([float("nan")], None, [1.])
        with self.assertRaises(RuntimeError):
            self.stats([float("inf")], None, [1.])

    def test_unused_and_nondifferentiable_loss(self):
        a, b = nn.Parameter(torch.ones(2)), nn.Parameter(torch.ones(1))
        values = probe.cpu_gradients((a * .1).sum(), (a, b))
        self.assertTrue(torch.allclose(values[0], torch.full_like(a, .1)))
        self.assertIsNone(values[1])
        self.assertEqual(probe.cpu_gradients(torch.tensor(0.), (a, b)), [None, None])
        self.assertIsNone(a.grad)
        self.assertIsNone(b.grad)


class ActualLossIntegrationTests(unittest.TestCase):
    def test_unchanged_m66_loss_weighting_and_kd_layer_reach(self):
        class Pipeline(ToyModel):
            def __init__(self, depth):
                super().__init__()
                self.depth = depth

            def forward(self, images, calibs, targets, sizes, dn_args=None):
                _, ref, _ = fixture()
                values = self.depth_embed[-1](self.backbone(images))
                ref["pred_depth"] = torch.stack((self.depth + .01 * values[..., 0], .01 * values[..., 1]), -1)
                return ref

        class Criterion(nn.Module):
            matcher = staticmethod(matcher)
            weight_dict = {"loss_depth": 2.}

            def forward(self, outputs, targets, extra):
                return dict(loss_depth=nn.functional.smooth_l1_loss(
                    outputs["pred_depth"][..., 0], targets[0]["depth"].reshape(1, 2)))

        torch.manual_seed(42)
        model = Pipeline(11.2)
        anchor = copy.deepcopy(model).eval().requires_grad_(False)
        teacher = Pipeline(10.3).eval().requires_grad_(False)
        names = probe.p.configure_trainable(model)
        parameters = [value for value in model.parameters() if value.requires_grad]
        states = [probe.q.tensor_hash(value.state_dict()) for value in (model, anchor, teacher)]
        _, _, targets = fixture()
        raw = {key: value[None] for key, value in targets[0].items()}
        raw.update(boxes=raw["boxes_3d"][..., :4], calibs=torch.eye(3)[None, None].repeat(1, 2, 1, 1),
                   mask_2d=torch.ones(1, 2, dtype=torch.bool), img_size=torch.tensor([[1280., 384.]]))
        batch = (torch.ones(1, 2, 3), torch.eye(3)[None], raw, dict(img_id=torch.tensor([1])))
        taps = probe.p.DepthFeatureTap(model), probe.p.DepthFeatureTap(teacher)
        try:
            with patch.object(torch.Tensor, "cuda", lambda value: value), \
                 patch.object(torch.Tensor, "backward", side_effect=AssertionError("No backward()")):
                row = probe.measure_batch(model, anchor, teacher, Criterion(), batch, taps, names, parameters)
        finally:
            for tap in taps:
                tap.close()
        self.assertEqual(row["kd_pairs"]["Vehicle"], 1)
        self.assertEqual(row["kd_pairs"]["Pedestrian"], 0)
        self.assertAlmostEqual(row["weighted_losses"]["kd"], .1 * row["raw_losses"]["feature_kd"], places=7)
        self.assertGreater(row["gradients"]["depth_head_2/hidden"]["kd_norm"], 0.)
        self.assertEqual(row["gradients"]["depth_head_2/output"]["reach"]["kd"]["connected_tensors"], 0)
        self.assertGreater(row["gradients"]["depth_head_2/output"]["gt_norm"], 0.)
        self.assertEqual(row["gradients"]["depth_head_0/hidden"]["kd_norm"], 0.)
        self.assertEqual(states, [probe.q.tensor_hash(value.state_dict()) for value in (model, anchor, teacher)])
        self.assertTrue(all(value.grad is None for value in model.parameters()))
        self.assertTrue(all(value.grad is None for value in teacher.parameters()))
        self.assertEqual(len(model.depth_embed[-1].layers[-1]._forward_pre_hooks), 0)

    def test_complete_cpu_probe_reuses_report_and_keeps_all_state(self):
        from argparse import Namespace

        class Pipeline(ToyModel):
            def __init__(self, depth):
                super().__init__()
                self.depth = depth

            def forward(self, images, calibs, targets, sizes, dn_args=None):
                _, ref, _ = fixture()
                count = len(images)
                ref = {key: value.repeat(count, 1, 1) for key, value in ref.items()}
                values = self.depth_embed[-1](self.backbone(images))
                ref["pred_depth"] = torch.stack((self.depth + .01 * values[..., 0], .01 * values[..., 1]), -1)
                return ref

        class Criterion(nn.Module):
            matcher = staticmethod(matcher)
            weight_dict = {"loss_depth": 1.}

            def forward(self, outputs, targets, extra):
                depths = torch.stack([target["depth"].reshape(2) for target in targets])
                return dict(loss_depth=nn.functional.smooth_l1_loss(outputs["pred_depth"][..., 0], depths))

        class Dataset(torch.utils.data.Dataset):
            def __len__(self):
                return 64

            def __getitem__(self, index):
                _, _, targets = fixture()
                raw = targets[0]
                raw.update(boxes=raw["boxes_3d"][..., :4], calibs=torch.eye(3)[None].repeat(2, 1, 1),
                           mask_2d=torch.ones(2, dtype=torch.bool), img_size=torch.tensor([1280., 384.]))
                return torch.ones(2, 3), torch.eye(3), raw, dict(img_id=index)

        torch.manual_seed(42)
        model = Pipeline(11.2)
        anchor = copy.deepcopy(model).eval().requires_grad_(False)
        teacher = Pipeline(10.3).eval().requires_grad_(False)
        names = probe.p.configure_trainable(model)
        identity = dict(revision=probe.REVISION, sample_ids=[f"{i:06d}" for i in range(64)], inputs={})
        runtime = (model, anchor, teacher, Criterion(), Dataset(), names)
        with tempfile.TemporaryDirectory() as directory:
            args = Namespace(role="original", report=Path(directory) / "original.json")
            with patch.object(probe, "verify_inputs", return_value=({}, {}, identity, {})), \
                 patch.object(probe.p, "training_runtime", return_value=runtime) as constructor, \
                 patch.object(torch.Tensor, "cuda", lambda value: value), redirect_stdout(io.StringIO()):
                probe.run_probe(args)
                first = args.report.read_bytes()
                probe.run_probe(args)
                self.assertEqual(constructor.call_count, 1)
                self.assertEqual(first, args.report.read_bytes())
            report = probe.q.read_json(args.report)
            self.assertEqual(len(report["rows"]), 16)
            self.assertEqual(report["optimizer_steps"], 0)
            self.assertTrue(all(report["checks"].values()))
            self.assertEqual(sum(row["kd_pairs"]["Vehicle"] for row in report["rows"]), 64)
            self.assertEqual(len(model.depth_embed[-1].layers[-1]._forward_pre_hooks), 0)


class ReportTests(unittest.TestCase):
    def test_summary_uses_only_active_kd_batches(self):
        reports = {role: report_fixture(role) for role in probe.ROLES}
        for role in reports:
            report = reports[role]
            report.pop("report_sha256")
            report["rows"][0]["kd_pairs"]["Vehicle"] = 0
            report["rows"][0]["kd_pairs"]["vehicle_near"] = 0
            report["rows"][0]["gradients"]["all"]["kd_control_cosine"] = -1.
            reports[role] = probe.signed(report)
        result = probe.summarize_reports(reports)
        self.assertEqual(result["optimizer_steps"], 0)
        self.assertTrue(result["paired_augmented_inputs_identical"])
        self.assertEqual(result["roles"]["kd"]["active_batches"], 15)
        self.assertEqual(result["roles"]["kd"]["active_batch_kd_control_opposing_rate"], 0.)
        self.assertFalse(result["training_authorized"])

    def test_zero_coverage_is_reported_not_authorized(self):
        result = probe.summarize_reports({role: report_fixture(role, False) for role in probe.ROLES})
        self.assertFalse(result["roles"]["kd"]["sufficient_diagnostic_coverage"])
        self.assertIsNone(result["roles"]["kd"]["active_batch_statistics"]["kd_to_gt_norm_ratio"]["median"])

    def test_changed_identity_or_augmented_input_rejected(self):
        for field in ("identity", "fingerprint"):
            reports = {role: report_fixture(role) for role in probe.ROLES}
            bad = reports["kd"]
            bad.pop("report_sha256")
            if field == "identity":
                bad["identity"]["different_gpu"] = True
            else:
                bad["rows"][0]["input_sha256"] = "b" * 64
            reports["kd"] = probe.signed(bad)
            with self.assertRaisesRegex(RuntimeError, "different"):
                probe.summarize_reports(reports)

    def test_bad_signature_scope_checks_and_sample_selection_rejected(self):
        for change in ("signature", "step", "check", "ids"):
            report = report_fixture()
            report.pop("report_sha256")
            if change == "step": report["optimizer_steps"] = 1
            elif change == "check": report["checks"]["historical_files_unchanged"] = False
            elif change == "ids": report["rows"][0]["sample_ids"][0] = "999999"
            report = probe.signed(report)
            if change == "signature": report["optimizer_steps"] = 1
            with self.assertRaises(RuntimeError):
                probe.validate_probe(report, "original")

    def test_missing_head_measurement_rejected(self):
        report = report_fixture()
        report.pop("report_sha256")
        report["rows"][0]["gradients"].pop("depth_head_2/output")
        with self.assertRaisesRegex(RuntimeError, "Missing component"):
            probe.validate_probe(probe.signed(report), "original")

    def test_inconsistent_eligible_counts_rejected(self):
        report = report_fixture()
        report.pop("report_sha256")
        report["rows"][0]["kd_pairs"]["vehicle_far"] = 100
        with self.assertRaisesRegex(RuntimeError, "coverage counts"):
            probe.validate_probe(probe.signed(report), "original")

    def test_write_once_preserves_existing_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.json"
            probe.write_once(path, dict(complete=True))
            before = path.read_bytes()
            probe.write_once(path, dict(complete=True))
            with self.assertRaisesRegex(RuntimeError, "Preserve"):
                probe.write_once(path, dict(complete=False))
            self.assertEqual(before, path.read_bytes())

    def test_bundle_supports_partial_failure_excludes_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "new"
            output.mkdir()
            (output / "diagnostic.log").write_text("native probe failed")
            (output / "never_include.pth").write_text("weights")
            old = root / "old"
            old.mkdir()
            manifest = old / "m66_manifest.json"
            manifest.write_text(json.dumps(dict(output=str(old))))
            before = manifest.read_bytes()
            probe.bundle(output, manifest)
            with zipfile.ZipFile(output / "m66b_gradient_results.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"diagnostic.log", "historical_inputs/m66_manifest.json"})
                self.assertIsNone(archive.testzip())
            self.assertEqual(before, manifest.read_bytes())


class InputIdentityTests(unittest.TestCase):
    def test_fresh_runtime_view_rebinds_paths_without_historical_environment_match(self):
        from argparse import Namespace
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "MonoDETR_M66B_TEST"
            ops = repo / "lib/models/monodetr/ops"
            ops.mkdir(parents=True)
            binary = ops / "MultiScaleDeformableAttention.so"
            binary.write_bytes(b"mock binary")
            build = dict(import_verified=True, binary=str(binary), binary_sha256=probe.q.sha(binary),
                         identity=dict(torch="2.10.0+cu130", python=sys.version))
            probe.q.write_json(ops / "m64_build_receipt.json", build)
            receipt = root / "runtime.json"
            probe.q.write_json(receipt, dict(complete=True, python=sys.executable))
            dataset = root / "data"
            (dataset / "ImageSets").mkdir(parents=True)
            original = dict(environment=dict(gpu="historical"), data_identity=dict(train="old", val="old"),
                models=dict(a2=dict(repo="missing historical checkout", patched_source_sha256="same",
                                    config=dict(dataset=dict(root_dir="historical")))),
                training_config=dict(dataset=dict(root_dir="historical")),
                r0=dict(config=dict(dataset=dict(root_dir="historical"))))
            before = copy.deepcopy(original)
            env = dict(gpu="prospective", cuda="13.0", packages=dict(torch="2.10.0+cu130"))
            args = Namespace(manifest=root / "historical.json", repo=repo, dataset_root=dataset,
                             runtime_receipt=receipt)
            with patch.object(probe, "reviewed_inputs", return_value=(original, {}, {})), \
                 patch.object(probe.subprocess, "check_output", side_effect=[probe.q.A2_COMMIT, "https://github.com/ZrrSkywalker/MonoDETR.git"]), \
                 patch.object(probe.q, "source_hash", return_value="same"), \
                 patch.object(probe.c, "data_identity", return_value=original["data_identity"]), \
                 patch.object(probe.q, "environment", return_value=env), \
                 patch.object(probe.q, "split_ids", return_value=[f"{i:06d}" for i in range(3712)]):
                old, view, identity, summaries = probe.verify_inputs(args)
            self.assertEqual(original, before)
            self.assertEqual(view["models"]["a2"]["repo"], str(repo.resolve()))
            self.assertEqual(view["training_config"]["dataset"]["root_dir"], str(dataset.resolve()))
            self.assertFalse(identity["historical_runtime_equivalence_claimed"])
            self.assertEqual(identity["environment"], env)
            self.assertEqual(identity["sample_ids"], [f"{i:06d}" for i in range(64)])

    def historical_fixture(self, root):
        m = dict(revision=probe.p.REVISION, manifest_sha256=probe.MANIFEST_SHA,
                 recipe=probe.p.RECIPE, implementation_sha256=probe.IMPLEMENTATION_SHA,
                 output=str(root), models=dict(a2=dict(checkpoint=str(root / "original.pth"))),
                 r0=dict(checkpoint=str(root / "teacher.pth")))
        files = {root / "m66_manifest.json": m}
        hashes = {root / "original.pth": probe.q.A2_SHA, root / "teacher.pth": probe.p.R0_SHA,
                  root / "m66_paired_gate.json": probe.GATE_SHA, root / "m66_training_smoke.json": probe.SMOKE_SHA}
        files[root / "m66_paired_gate.json"] = dict(complete=True, manifest_sha256=probe.MANIFEST_SHA, pilot_passed=False)
        for role in probe.ENDPOINTS:
            path = root / role / "checkpoint_epoch_1.pth"
            hashes[path] = probe.ENDPOINTS[role]
            files[path.with_suffix(".json")] = dict(complete=True, manifest_sha256=probe.MANIFEST_SHA,
                checkpoint_sha256=probe.ENDPOINTS[role], completed_control_epoch=1, role=role)
            files[root / role / "training_summary.json"] = summary_fixture(m, role)
        for path in {*files, *hashes}:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture")
        return files, hashes

    def test_historical_receipts_are_read_only_and_hashes_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, hashes = self.historical_fixture(root)
            with patch.object(probe.q, "read_json", side_effect=lambda path: files[Path(path)]), \
                 patch.object(probe.q, "sha", side_effect=lambda path: hashes.get(Path(path), "mock hash")), \
                 patch.object(probe.q, "signature", return_value=probe.MANIFEST_SHA), \
                 patch.object(probe.p, "implementation_hash", return_value=probe.IMPLEMENTATION_SHA), \
                 patch.object(probe.p, "validated_smoke"), \
                 patch.object(probe.q, "write_json", side_effect=AssertionError("No historical writes")):
                m, paths, summaries = probe.reviewed_inputs(root / "m66_manifest.json")
                self.assertEqual(set(summaries), {"control", "kd"})
                self.assertIn("kd_receipt", paths)
                hashes[root / "kd/checkpoint_epoch_1.pth"] = "changed"
                with self.assertRaisesRegex(RuntimeError, "bytes differ"):
                    probe.reviewed_inputs(root / "m66_manifest.json")

    def test_wrong_sidecar_and_missing_raw_weights_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            files, hashes = self.historical_fixture(root)
            with patch.object(probe.q, "read_json", side_effect=lambda path: files[Path(path)]), \
                 patch.object(probe.q, "sha", side_effect=lambda path: hashes.get(Path(path), "mock hash")), \
                 patch.object(probe.q, "signature", return_value=probe.MANIFEST_SHA), \
                 patch.object(probe.p, "implementation_hash", return_value=probe.IMPLEMENTATION_SHA), \
                 patch.object(probe.p, "validated_smoke"):
                files[root / "control/checkpoint_epoch_1.json"]["complete"] = False
                with self.assertRaisesRegex(RuntimeError, "receipt"):
                    probe.reviewed_inputs(root / "m66_manifest.json")
                (root / "teacher.pth").unlink()
                with self.assertRaisesRegex(FileNotFoundError, "ZIP does not contain weights"):
                    probe.reviewed_inputs(root / "m66_manifest.json")


class PackagingTests(unittest.TestCase):
    def test_absolute_cli_outside_repo(self):
        result = subprocess.run([sys.executable, ROOT / "scripts/probe_m66_kd_gradients.py", "--help"],
                                cwd=tempfile.gettempdir(), capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("summarize", result.stdout)

    def test_historical_implementation_stays_frozen(self):
        self.assertEqual(probe.p.implementation_hash(), probe.IMPLEMENTATION_SHA)

    def test_notebook_has_three_self_contained_code_sections(self):
        path = ROOT / "notebooks/MonoDETR_A2_M66b_Gradient_Diagnostic_Colab.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["metadata"]["m66b_revision"], probe.REVISION)
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code), 3)
        for index, text in enumerate(code):
            ast.parse(text)
            compile(text, f"notebook_section_{index}", "exec")
        self.assertIn(probe.REVISION, code[0])
        for name in ("run_logged", "checkout", "READY", "RUNTIME_RECEIPT", "SCRIPT"):
            self.assertIn(name, code[0])
        text = "\n".join(code)
        self.assertIn("https://github.com/ZrrSkywalker/MonoDETR.git", text)
        for forbidden in ("PuFanqi23/MonoDETR", "train_val.py", "AdamW", "reset', '--hard", "setup_m63_isolated_runtime"):
            self.assertNotIn(forbidden, text)
        self.assertIn("'--ff-only'", text)
        self.assertIn("finally:", code[2])
        self.assertIn("'--runtime-receipt',RUNTIME_RECEIPT", code[2])

    def test_probe_does_not_call_mutating_historical_helpers(self):
        source = (ROOT / "scripts/probe_m66_kd_gradients.py").read_text()
        ast.parse(source)
        for forbidden in ("recover_completed(", "p.load(", "optimizer.step(", ".backward(", "clip_grad_norm_("):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
