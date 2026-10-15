"""Cold-reset restore order, early runtime selection and preservation guards."""
from contextlib import contextmanager, redirect_stdout
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import preflight_a2_lite_feature_session as session


class SessionTests(unittest.TestCase):
    def fixture(self, root):
        a = SimpleNamespace(run_id="feature_r1", output=root / "runs/feature_r1",
                            repo=root / "MonoDETR_A2LITE_FEATURE_feature_r1",
                            dataset=root / "kitti_a2_lite_accuracy_feature_feature_r1",
                            initialization=root / "weights/init.pth", teacher_checkpoint=root / "weights/teacher.pth",
                            pilot=root / "reviewed.zip", report=root / "plan.json", device="cuda",
                            allow_pretraining_restart=True)
        a.initialization.parent.mkdir(); a.initialization.write_bytes(b"init"); a.teacher_checkpoint.write_bytes(b"teacher")
        baseline = dict(config={}, source_sha256="frozen_source", train_ids=["000001"], val_ids=["000002"],
                        pilot_ids=["000001"], data_file_sha256={"000001": {"image_2": "image"}})
        runtime = dict(cpu_environment=dict(torch="2.11.0", numpy="2.1.3"),
                       execution_identity=dict(device="cuda", gpu="A100", cuda_build="13.0"))
        return a, baseline, runtime

    @contextmanager
    def dependencies(self, a, baseline, runtime):
        with patch.object(session.pilot.reference, "reviewed_pilot", return_value=(baseline, {}, b"", b"")), \
             patch.object(session.pilot, "implementation", return_value={"frozen.py": "hash"}), \
             patch.object(session.pilot.engine.resolution, "CHECKPOINT_SHA", session.pilot.engine.sha256(a.initialization)), \
             patch.object(session.pilot.engine.base.q, "A2_SHA", session.pilot.engine.sha256(a.teacher_checkpoint)), \
             patch.object(session, "current_runtime", return_value=runtime), redirect_stdout(io.StringIO()):
            yield

    def save_manifest(self, a, baseline, runtime, output=None, repo=None, dataset=None):
        output, repo, dataset = output or a.output, repo or a.repo, dataset or a.dataset
        m = dict(**copy.deepcopy(baseline), **copy.deepcopy(runtime), revision=session.pilot.REVISION,
                 recipe=copy.deepcopy(session.pilot.RECIPE), guardrails=copy.deepcopy(session.pilot.GUARDRAILS),
                 implementation={"frozen.py": "hash"}, output=str(output), source=str(repo), dataset=str(dataset),
                 initialization=str(a.initialization), teacher_checkpoint=str(a.teacher_checkpoint),
                 initialization_sha256=session.pilot.engine.sha256(a.initialization),
                 teacher_checkpoint_sha256=session.pilot.engine.sha256(a.teacher_checkpoint))
        m["signature_sha256"] = session.pilot.engine.signature(m)
        session.pilot.engine.write_json(output / "feature_manifest.json", m)
        return m

    def test_fresh_run_needs_no_historical_runtime_or_ephemeral_files(self):
        with tempfile.TemporaryDirectory() as d:
            a, baseline, runtime = self.fixture(Path(d))
            with self.dependencies(a, baseline, runtime), patch.object(session.pilot, "load") as frozen_load:
                value = session.plan(a)
            self.assertEqual(value["action"], "new_run")
            self.assertEqual(value["output"], str(a.output))
            self.assertFalse(a.repo.exists()); self.assertFalse(a.dataset.exists())
            frozen_load.assert_not_called()

    def test_cold_reset_plan_does_not_validate_missing_source_before_restore(self):
        with tempfile.TemporaryDirectory() as d:
            a, baseline, runtime = self.fixture(Path(d))
            self.save_manifest(a, baseline, runtime)
            old = (a.output / "feature_manifest.json").read_bytes(); calls = []
            with self.dependencies(a, baseline, runtime), patch.object(session.pilot, "load") as frozen_load:
                value = session.plan(a)
                with patch.object(session.pilot.engine.base, "prepare_repo", side_effect=lambda p: (calls.append("restore"), p.mkdir())), \
                     patch.object(session.pilot.engine.base.q, "source_hash", side_effect=lambda p: (calls.append("verify"), "frozen_source")[1]):
                    session.restore_source(SimpleNamespace(plan=a.report, device="cuda"))
            self.assertEqual(value["action"], "resume_matching_runtime")
            self.assertEqual(calls, ["restore", "verify"])
            self.assertTrue(a.repo.is_dir())
            self.assertEqual((a.output / "feature_manifest.json").read_bytes(), old)
            frozen_load.assert_not_called()

    def test_changed_gpu_before_training_gets_stable_separate_run_and_paths(self):
        with tempfile.TemporaryDirectory() as d:
            a, baseline, current = self.fixture(Path(d)); previous = copy.deepcopy(current)
            previous["execution_identity"]["gpu"] = "L4"
            self.save_manifest(a, baseline, previous)
            old = (a.output / "feature_manifest.json").read_bytes()
            with self.dependencies(a, baseline, current):
                first = session.plan(a); second = session.plan(a)
            self.assertEqual(first, second)
            self.assertEqual(first["action"], "new_runtime_run_before_main_training")
            self.assertIn("execution_identity.gpu", first["runtime_changes"])
            self.assertNotEqual(first["output"], str(a.output))
            self.assertNotEqual(first["repo"], str(a.repo))
            self.assertNotEqual(first["dataset"], str(a.dataset))
            self.assertTrue(first["run_id"].startswith(a.run_id + "_restart_"))
            self.assertEqual((a.output / "feature_manifest.json").read_bytes(), old)

    def test_current_software_change_before_training_is_not_mixed_with_old_tiny_fit(self):
        with tempfile.TemporaryDirectory() as d:
            a, baseline, current = self.fixture(Path(d)); previous = copy.deepcopy(current)
            previous["cpu_environment"]["torch"] = "older"
            self.save_manifest(a, baseline, previous)
            (a.output / "tiny_fit.json").write_bytes(b"old diagnostic")
            with self.dependencies(a, baseline, current):
                value = session.plan(a)
            self.assertIn("cpu_environment.torch", value["runtime_changes"])
            self.assertNotEqual(value["output"], str(a.output))
            self.assertEqual((a.output / "tiny_fit.json").read_bytes(), b"old diagnostic")

    def test_training_started_blocks_automatic_runtime_switch(self):
        for marker in ("control/last.pth", "kd/last.pth", "control/last.tmp", "evaluation/control/result.json"):
            with tempfile.TemporaryDirectory() as d:
                a, baseline, current = self.fixture(Path(d)); previous = copy.deepcopy(current)
                previous["execution_identity"]["gpu"] = "L4"
                self.save_manifest(a, baseline, previous)
                path = a.output / marker; path.parent.mkdir(parents=True); path.write_bytes(b"preserve")
                with self.dependencies(a, baseline, current):
                    with self.assertRaisesRegex(RuntimeError, "after main training started"):
                        session.plan(a)
                self.assertFalse(a.report.exists()); self.assertEqual(path.read_bytes(), b"preserve")

    def test_matching_runtime_resumes_existing_main_training_without_new_run(self):
        with tempfile.TemporaryDirectory() as d:
            a, baseline, current = self.fixture(Path(d)); self.save_manifest(a, baseline, current)
            path = a.output / "control/last.pth"; path.parent.mkdir(); path.write_bytes(b"checkpoint")
            with self.dependencies(a, baseline, current):
                value = session.plan(a)
            self.assertEqual(value["action"], "resume_matching_runtime")
            self.assertEqual(value["run_id"], a.run_id)

    def test_existing_matching_restart_output_is_reused_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d:
            a, baseline, current = self.fixture(Path(d)); previous = copy.deepcopy(current)
            previous["execution_identity"]["gpu"] = "L4"
            self.save_manifest(a, baseline, previous)
            with self.dependencies(a, baseline, current):
                first = session.plan(a)
                self.save_manifest(a, baseline, current, Path(first["output"]), Path(first["repo"]), Path(first["dataset"]))
                second = session.plan(a)
            self.assertEqual(first["run_id"], second["run_id"])
            self.assertIsNotNone(second["selected_manifest_sha256"])

    def test_changed_recipe_or_path_or_missing_checkpoint_stops_before_staging(self):
        with tempfile.TemporaryDirectory() as d:
            a, baseline, current = self.fixture(Path(d)); m = self.save_manifest(a, baseline, current)
            m["recipe"]["epochs"] = 20
            m["signature_sha256"] = session.pilot.engine.signature({k: v for k, v in m.items() if k != "signature_sha256"})
            session.pilot.engine.write_json(a.output / "feature_manifest.json", m)
            with self.dependencies(a, baseline, current):
                with self.assertRaisesRegex(RuntimeError, "identity/code"):
                    session.plan(a)
            self.assertFalse(a.report.exists())

    def test_pretraining_restart_requires_explicit_policy(self):
        with tempfile.TemporaryDirectory() as d:
            a, baseline, current = self.fixture(Path(d)); previous = copy.deepcopy(current)
            previous["execution_identity"]["gpu"] = "L4"
            self.save_manifest(a, baseline, previous); a.allow_pretraining_restart = False
            with self.dependencies(a, baseline, current):
                with self.assertRaisesRegex(RuntimeError, "pretraining-restart policy"):
                    session.plan(a)

    def test_plan_tampering_or_runtime_change_blocks_source_restore(self):
        with tempfile.TemporaryDirectory() as d:
            a, baseline, current = self.fixture(Path(d))
            with self.dependencies(a, baseline, current):
                value = session.plan(a)
                value["repo"] = str(Path(d) / "other")
                session.pilot.engine.write_json(a.report, value)
                with patch.object(session.pilot.engine.base, "prepare_repo") as restore:
                    with self.assertRaisesRegex(RuntimeError, "plan/runtime changed"):
                        session.restore_source(SimpleNamespace(plan=a.report, device="cuda"))
                restore.assert_not_called()

    def test_bad_source_hash_stops_without_reset_or_manifest_edits(self):
        with tempfile.TemporaryDirectory() as d:
            a, baseline, current = self.fixture(Path(d)); self.save_manifest(a, baseline, current)
            original = (a.output / "feature_manifest.json").read_bytes()
            with self.dependencies(a, baseline, current):
                session.plan(a)
                with patch.object(session.pilot.engine.base, "prepare_repo"), patch.object(session.pilot.engine.base.q, "source_hash", return_value="wrong"):
                    with self.assertRaisesRegex(RuntimeError, "Restored portable source differs"):
                        session.restore_source(SimpleNamespace(plan=a.report, device="cuda"))
            self.assertEqual((a.output / "feature_manifest.json").read_bytes(), original)

    def test_notebook_restores_source_before_saved_run_prepare_and_handles_bootstrap_failure(self):
        n = json.loads((ROOT / "notebooks/MonoDETR_A2_Lite_512_Feature_KD_Colab.ipynb").read_text())
        codes = ["".join(c["source"]) for c in n["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(codes), 8)
        for i, text in enumerate(codes): compile(text, f"session-cell-{i}", "exec")
        all_code = "\n".join(codes)
        self.assertIn(session.REVISION, all_code)
        self.assertIn("'--allow-pretraining-restart'", all_code)
        self.assertLess(all_code.index("'restore-source'"), all_code.index("stage('prepare'"))
        self.assertIn("OUTPUT_SELECTED=True", all_code)
        self.assertIn("a2_lite_feature_startup_failure.zip", all_code)
        for forbidden in ("reset --hard", "weights_only=False", "setup_m67_runtime", "venv"):
            self.assertNotIn(forbidden, all_code)


if __name__ == "__main__":
    unittest.main()
