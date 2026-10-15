"""Restricted teacher metadata load and preservation of a prepared r1 run."""
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
import run_a2_lite_features_safe as safe

UNEXPECTED_EXECUTIONS = []


def unexpected_global():
    UNEXPECTED_EXECUTIONS.append(True)
    return "unsafe"


class UntrustedMetadata:
    def __reduce__(self):
        return unexpected_global, ()


def full_teacher():
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.zeros(2))
    model.depthaware_transformer = torch.nn.Module()
    for name in ("encoder", "decoder"):
        module = torch.nn.Module(); module.num_layers = 3
        setattr(model.depthaware_transformer, name, module)
    model.attention = torch.nn.ModuleList([torch.nn.Identity() for _ in range(6)])
    for module in model.attention:
        module.coreml_export = True
        module.coreml_spatial_shapes = safe.pilot.TEACHER_SHAPES
    return model


class CheckpointTests(unittest.TestCase):
    def checkpoint(self, root, extra=None):
        path = root / "teacher.pth"
        payload = dict(epoch=130, model_state=dict(weight=torch.tensor([1., 2.])),
                       best_result=np.float64(15.4573), auxiliary_score=np.float32(7.5328))
        payload.update(extra or {})
        torch.save(payload, path)
        digest = safe.pilot.engine.sha256(path)
        m = dict(signature_sha256="prepared-r1", teacher_checkpoint=str(path),
                 teacher_checkpoint_sha256=digest, source=str(root / "source"), config=dict(model={}),
                 implementation={"scripts/recover_a2_lite_features.py": safe.FROZEN_SCRIPT_SHA})
        return path, digest, m

    def test_reproduces_error_then_loads_numpy_metadata_with_weights_only_and_no_global_leak(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); path, digest, m = self.checkpoint(root)
            before = set(torch.serialization.get_safe_globals())
            with self.assertRaises(pickle.UnpicklingError):
                torch.load(path, map_location="cpu", weights_only=True)
            model = full_teacher(); original = safe.pilot.build_teacher
            with patch.object(safe.pilot.engine.base.q, "A2_SHA", digest), \
                 patch.object(safe.pilot.engine.base, "inference_model", return_value=model), \
                 patch.object(torch, "load", wraps=torch.load) as load:
                with safe.restricted_teacher_loader():
                    result = safe.pilot.build_teacher(m, "cpu")
                self.assertTrue(load.call_args.kwargs["weights_only"])
                self.assertEqual(load.call_args.kwargs["map_location"], "cpu")
            self.assertIs(safe.pilot.build_teacher, original)
            self.assertEqual(set(torch.serialization.get_safe_globals()), before)
            self.assertTrue(torch.equal(result.weight, torch.tensor([1., 2.])))
            self.assertFalse(result.training)
            self.assertTrue(all(not p.requires_grad for p in result.parameters()))

    def test_unknown_pickle_global_remains_blocked_and_context_restores_on_error(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _, digest, m = self.checkpoint(root, dict(untrusted=UntrustedMetadata()))
            before = set(torch.serialization.get_safe_globals()); original = safe.pilot.build_teacher
            UNEXPECTED_EXECUTIONS.clear()
            with patch.object(safe.pilot.engine.base.q, "A2_SHA", digest), \
                 patch.object(safe.pilot.engine.base, "inference_model", return_value=full_teacher()):
                with self.assertRaises(pickle.UnpicklingError):
                    with safe.restricted_teacher_loader():
                        safe.pilot.build_teacher(m, "cpu")
            self.assertEqual(UNEXPECTED_EXECUTIONS, [])
            self.assertEqual(set(torch.serialization.get_safe_globals()), before)
            self.assertIs(safe.pilot.build_teacher, original)

    def test_existing_numpy_allowlist_entries_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _, digest, m = self.checkpoint(root)
            entry = safe.numpy_scalar_globals()[0]
            with torch.serialization.safe_globals([entry]):
                before = set(torch.serialization.get_safe_globals())
                with patch.object(safe.pilot.engine.base.q, "A2_SHA", digest), \
                     patch.object(safe.pilot.engine.base, "inference_model", return_value=full_teacher()):
                    with safe.restricted_teacher_loader():
                        safe.pilot.build_teacher(m, "cpu")
                self.assertEqual(set(torch.serialization.get_safe_globals()), before)

    def test_wrong_checkpoint_hash_stops_before_build_or_unpickling(self):
        with tempfile.TemporaryDirectory() as d:
            _, digest, m = self.checkpoint(Path(d)); m["teacher_checkpoint_sha256"] = "changed"
            with patch.object(safe.pilot.engine.base.q, "A2_SHA", digest), \
                 patch.object(safe.pilot.engine.base, "inference_model") as build, patch.object(torch, "load") as load:
                with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                    with safe.restricted_teacher_loader():
                        safe.pilot.build_teacher(m, "cpu")
            build.assert_not_called(); load.assert_not_called()

    def test_receipt_preserves_manifest_smoke_and_tiny_then_reuses_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _, digest, m = self.checkpoint(root)
            files = {"feature_manifest.json": b"original manifest", "smoke.json": b"passed smoke",
                     "tiny_fit.json": b"passed tiny fit", "tiny/last.pth": b"disposable tiny weights"}
            for name, content in files.items():
                path = root / name; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
            with patch.object(safe.pilot.engine.base.q, "A2_SHA", digest), \
                 patch.object(safe.pilot, "load", return_value=m):
                safe.ensure_receipt(root)
                receipt = (root / safe.RECEIPT).read_bytes()
                (root / "teacher_features").mkdir(); (root / "teacher_features/000001.pt").write_bytes(b"later cache")
                safe.ensure_receipt(root)
                self.assertEqual((root / safe.RECEIPT).read_bytes(), receipt)
            self.assertEqual(files, {name: (root / name).read_bytes() for name in files})
            report = safe.pilot.engine.checked_report(root / safe.RECEIPT, m)
            self.assertTrue(report["weights_only"])
            self.assertFalse(report["unsafe_pickle_fallback"])
            self.assertFalse(report["manifest_rewritten"])
            self.assertFalse(report["training_objective_changed"])

    def test_cannot_retroactively_attach_loader_after_feature_or_training_start(self):
        for marker in ("teacher_features/000001.pt", "feature_cache_complete.json", "control/last.pth", "kd/training_complete.json"):
            with tempfile.TemporaryDirectory() as d:
                root = Path(d); _, digest, m = self.checkpoint(root)
                path = root / marker; path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"prior data")
                with patch.object(safe.pilot.engine.base.q, "A2_SHA", digest), patch.object(safe.pilot, "load", return_value=m):
                    with self.assertRaisesRegex(RuntimeError, "after feature caching/training"):
                        safe.ensure_receipt(root)
                self.assertFalse((root / safe.RECEIPT).exists())

    def test_receipt_drift_or_original_implementation_drift_stops_without_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); _, digest, m = self.checkpoint(root)
            with patch.object(safe.pilot.engine.base.q, "A2_SHA", digest), patch.object(safe.pilot, "load", return_value=m):
                safe.ensure_receipt(root)
                original = (root / safe.RECEIPT).read_bytes()
                with patch.object(safe, "receipt_value", return_value={"changed": True}):
                    with self.assertRaisesRegex(RuntimeError, "receipt changed"):
                        safe.ensure_receipt(root)
                self.assertEqual((root / safe.RECEIPT).read_bytes(), original)
            with patch.object(safe, "FROZEN_SCRIPT_SHA", "wrong"), patch.object(safe.pilot, "load") as load:
                with self.assertRaisesRegex(RuntimeError, "original published"):
                    safe.ensure_receipt(root)
                load.assert_not_called()

    def test_wrapper_receipt_precedes_resume_and_original_dispatch_is_restored(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); events = []; original = safe.pilot.build_teacher
            with patch.object(sys, "argv", ["safe.py", "cache", "--output", str(root), "--device", "cpu"]), \
                 patch.object(safe, "ensure_receipt", side_effect=lambda p: events.append("receipt")), \
                 patch.object(safe.pilot, "main", side_effect=lambda: events.append("stage")):
                safe.main()
            self.assertEqual(events, ["receipt", "stage"])
            self.assertIs(safe.pilot.build_teacher, original)

    def test_bundle_does_not_require_successful_loader_or_manifest_validation(self):
        with tempfile.TemporaryDirectory() as d:
            with patch.object(sys, "argv", ["safe.py", "bundle", "--output", d]), \
                 patch.object(safe, "ensure_receipt") as receipt, patch.object(safe.pilot, "main") as main:
                safe.main()
            receipt.assert_not_called(); main.assert_called_once()

    def test_prepare_registers_receipt_after_new_manifest_is_created(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); events = []
            with patch.object(sys, "argv", ["safe.py", "prepare", "--output", str(root)]), \
                 patch.object(safe, "ensure_receipt", side_effect=lambda p: events.append("receipt")), \
                 patch.object(safe.pilot, "main", side_effect=lambda: events.append("prepare")):
                safe.main()
            self.assertEqual(events, ["prepare", "receipt"])

    def test_notebook_uses_safe_launcher_and_keeps_original_run_id(self):
        notebook = json.loads((ROOT / "notebooks/MonoDETR_A2_Lite_512_Feature_KD_Colab.ipynb").read_text())
        text = "\n".join("".join(c["source"]) for c in notebook["cells"])
        self.assertIn(safe.REVISION, text)
        self.assertIn("scripts/run_a2_lite_features_safe.py", text)
        self.assertIn("test_a2_lite_feature_checkpoint.py", text)
        self.assertIn("RUN_ID='a2_lite_512_foreground_feature_kd_r1'", text)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile("".join(cell["source"]), "safe-notebook-cell", "exec")
        self.assertEqual(safe.pilot.engine.sha256(safe.pilot.__file__), safe.FROZEN_SCRIPT_SHA)


if __name__ == "__main__":
    unittest.main()
