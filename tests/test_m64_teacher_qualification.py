"""CPU regression checks; these do not claim CUDA execution or teacher accuracy."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m64_teacher_qualification as q
import patch_m64_inference_sources as source_patch
import setup_m64_runtime as setup


class UnapprovedObject:
    pass


class M64Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_normalized_split_accepts_only_exact_ids(self):
        path = self.root / "val.txt"
        path.write_bytes(b"000003\r\n000007\r\n")
        digest = hashlib.sha256(b"000003\n000007\n").hexdigest()
        with patch.dict(q.SPLITS, val=(2, digest)):
            self.assertEqual(q.split_ids(path, "val"), ["000003", "000007"])
            path.write_text("000007\n000003\n")
            with self.assertRaises(RuntimeError):
                q.split_ids(path, "val")

    def test_split_duplicates_rejected(self):
        path = self.root / "val.txt"
        path.write_text("000003\n000003\n")
        with patch.dict(q.SPLITS, val=(2, hashlib.sha256(path.read_bytes()).hexdigest())):
            with self.assertRaises(RuntimeError):
                q.split_ids(path, "val")

    def test_source_hash_rejects_absent_repository(self):
        with self.assertRaisesRegex(RuntimeError, "Missing model source"):
            q.source_hash(self.root)

    def test_lazy_imports_move_into_eval_and_are_idempotent(self):
        text = ("from lib.datasets.kitti.kitti_eval_python.eval import get_official_eval_result\n"
                "from lib.datasets.kitti.kitti_eval_python.eval import get_distance_eval_result\n"
                "import lib.datasets.kitti.kitti_eval_python.kitti_common as kitti\n"
                "class Dataset:\n    def eval(self, results_dir, logger):\n        return kitti\n")
        result = source_patch.lazy_evaluator(text)
        self.assertEqual(result, source_patch.lazy_evaluator(result))
        tree = ast.parse(result)
        self.assertFalse(any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in tree.body))
        self.assertEqual(sum(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(tree)), 3)

    def test_lazy_patch_unknown_layout_refused(self):
        with self.assertRaises(RuntimeError):
            source_patch.lazy_evaluator("class Dataset:\n    pass\n")

    def test_teacher_docstring_syntax_fix_is_narrow(self):
        text = ('class MonoPRIO(nn.Module):\n  """ This is the MonoDGP module that performs '
                'monocualr 3D object detection """\n    def forward(self):\n        return 1\n')
        result = source_patch.teacher_model_syntax(text)
        ast.parse(result)
        self.assertEqual(result, source_patch.teacher_model_syntax(result))
        self.assertEqual(result.splitlines()[2:], text.splitlines()[2:])

    def test_teacher_full_checkpoint_does_not_download_initial_weights(self):
        text = "def build():\n    return backbone(pretrained=is_main_process(), norm_layer=norm_layer)\n"
        result = source_patch.teacher_backbone(text)
        self.assertIn("pretrained=False", result)
        self.assertEqual(result, source_patch.teacher_backbone(result))

    def test_patches_refuse_historical_checkout(self):
        with patch.object(source_patch.subprocess, "check_output", return_value=source_patch.A2_COMMIT):
            with self.assertRaisesRegex(RuntimeError, "historical"):
                source_patch.patch(self.root / "MonoDETR_M62", "a2")

    def test_patches_refuse_wrong_commit_before_writing(self):
        with patch.object(source_patch.subprocess, "check_output", return_value="wrong"):
            with self.assertRaisesRegex(RuntimeError, "Wrong"):
                source_patch.patch(self.root, "teacher")
        self.assertEqual(list(self.root.iterdir()), [])

    def test_driver_paths_preserved_but_stubs_excluded(self):
        with patch.dict(os.environ, {"LD_LIBRARY_PATH": "/user/driver:/cuda/stubs:/other/libs",
                                     "PYTHONPATH": "/host/packages", "PYTHONHOME": "/host"}):
            env = setup.runtime_env(Path("/cuda13"))
        self.assertIn("/user/driver", env["LD_LIBRARY_PATH"])
        self.assertIn("/other/libs", env["LD_LIBRARY_PATH"])
        self.assertNotIn("/stubs", env["LD_LIBRARY_PATH"])
        self.assertNotIn("PYTHONPATH", env)
        self.assertNotIn("PYTHONHOME", env)
        self.assertEqual(env["CUDA_HOME"], "/cuda13")

    def test_runtime_uses_virtualenv_not_ensurepip_or_kernel_pip(self):
        text = (ROOT / "scripts/setup_m64_runtime.py").read_text()
        self.assertIn('"virtualenv==20.35.4"', text)
        self.assertNotIn("venv.EnvBuilder", text)
        self.assertIn('str(python), "-m", "pip", "install"', text)
        self.assertEqual(setup.TORCH_PACKAGES, ["torch==2.10.0", "torchvision==0.25.0"])
        self.assertTrue(setup.TORCH_INDEX.endswith("/cu130"))

    def test_atomic_cache_resume_and_tamper_detection(self):
        path = self.root / "000001.json"
        identity = {"role": "teacher", "input_sha256": "exact"}
        value = dict(identity=identity, texts={"product": "Car 0.0 0\n", "published": ""})
        value["record_sha256"] = q.signature(value)
        q.write_json(path, value)
        resumed = q.cached_prediction(path, identity)
        self.assertEqual(resumed["texts"], value["texts"])
        self.assertFalse(path.with_suffix(".json.tmp").exists())
        with self.assertRaises(RuntimeError):
            q.cached_prediction(path, dict(identity, input_sha256="different"))
        value["texts"]["product"] = "changed"
        q.write_json(path, value)
        with self.assertRaises(RuntimeError):
            q.cached_prediction(path, identity)

    def test_incomplete_cache_is_not_silently_recomputed(self):
        path = self.root / "000001.json"
        path.write_text('{"identity":')
        with self.assertRaises(json.JSONDecodeError):
            q.cached_prediction(path, {})

    def test_metric_summary_must_cover_full_validation(self):
        with self.assertRaises(RuntimeError):
            q.metric_rows(dict(complete_split=False, evaluated_images=3769, metrics=[]))
        with self.assertRaises(RuntimeError):
            q.metric_rows(dict(complete_split=True, evaluated_images=16, metrics=[]))

    def test_metric_summary_rejects_nonfinite(self):
        row = dict(class_name="Vehicle", difficulty="moderate", metric="3d", ap_r40=float("nan"))
        with self.assertRaises(RuntimeError):
            q.metric_rows(dict(complete_split=True, evaluated_images=3769, metrics=[row]))

    def test_native_evaluator_identity_does_not_initialize_torch(self):
        m = dict(revision=q.REVISION, implementation_sha256="fixed", models={}, asset_receipt={},
                 environment=dict(python="test", packages={}), dataset_root=str(self.root), data_identity={})
        m["manifest_sha256"] = q.signature(m)
        path = self.root / "manifest.json"
        q.write_json(path, m)
        with patch.object(q, "implementation_hash", return_value="fixed"), \
             patch.object(q, "environment", side_effect=AssertionError("Torch CUDA must not initialize")), \
             patch.object(q, "package_environment", return_value=dict(python="test", packages={})), \
             patch.dict(q.SPLITS, {}, clear=True):
            self.assertEqual(q.load(path, native_evaluator=True)["manifest_sha256"], m["manifest_sha256"])

    def test_manifest_tampering_rejected_before_runtime(self):
        value = dict(revision=q.REVISION, optimizer_steps=0)
        value["manifest_sha256"] = q.signature(value)
        value["optimizer_steps"] = 1
        path = self.root / "manifest.json"
        q.write_json(path, value)
        with self.assertRaisesRegex(RuntimeError, "manifest changed"):
            q.load(path)

    def test_restricted_checkpoint_accepts_tensor_state_and_numpy_metadata(self):
        import numpy as np
        import torch
        path = self.root / "model.pth"
        torch.save(dict(model_state={"weight": torch.ones(2)}, epoch=130, best=np.float64(1.0)), path)
        payload = q.safe_payload(path)
        self.assertEqual(payload["epoch"], 130)
        self.assertTrue(torch.equal(payload["model_state"]["weight"], torch.ones(2)))

    def test_restricted_checkpoint_refuses_unapproved_pickle(self):
        import torch
        path = self.root / "model.pth"
        torch.save(dict(model_state={"weight": torch.ones(2)}, extra=UnapprovedObject()), path)
        with self.assertRaises(pickle.UnpicklingError):
            q.safe_payload(path)

    def test_restricted_checkpoint_refuses_non_tensor_state(self):
        import torch
        path = self.root / "model.pth"
        torch.save(dict(model_state={"weight": "wrong"}), path)
        with self.assertRaisesRegex(RuntimeError, "Non-tensor"):
            q.safe_payload(path)

    def test_native_format_and_classes_match_release_decoder(self):
        row = [1, .1234, 1, 2, 3, 4, 1.5, 1.7, 4.3, 5, 6, 7, .4567, .9]
        text = q.prediction_text([row], ["Pedestrian", "Car", "Cyclist"])
        self.assertTrue(text.startswith("Car 0.0 0 0.12"))
        self.assertEqual(len(text.split()), 16)

    def test_model_configs_keep_a2_and_teacher_taxonomies_separate(self):
        import yaml
        (self.root / "configs").mkdir()
        config = dict(dataset={}, model=dict(use_monoprio=True, num_classes=3))
        for name in ("monodetr", "monoprio"):
            (self.root / f"configs/{name}.yaml").write_text(yaml.safe_dump(config))
        a2 = q.model_config(self.root, "a2", Path("/data"), Path("/prior.npz"))
        teacher = q.model_config(self.root, "teacher", Path("/data"), Path("/prior.npz"))
        self.assertEqual(a2["model"]["backbone"], "mobilenetv4_conv_medium.e500_r256_in1k")
        self.assertFalse(a2["model"]["backbone_pretrained"])
        self.assertEqual(a2["dataset"]["class_mapping"]["Truck"], "Car")
        self.assertNotIn("class_mapping", teacher["dataset"])
        self.assertEqual(teacher["dataset"]["writelist"], ["Car", "Pedestrian", "Cyclist"])

    def test_comparison_config_can_disable_inherited_mapping(self):
        sys.path.insert(0, str(ROOT))
        from tools.config import load_config
        import yaml
        path = self.root / "native.yaml"
        path.write_text(yaml.safe_dump(dict(base_config=str(ROOT / "configs/kitti_mobileadas3d_s1.yaml"),
                       dataset=dict(classes=["Car", "Pedestrian"], class_mapping=None))))
        cfg = load_config(str(path))
        self.assertIsNone(cfg["dataset"]["class_mapping"])
        self.assertEqual(cfg["dataset"]["classes"], ["Car", "Pedestrian"])

    def test_bundle_excludes_weights_and_per_image_predictions(self):
        (self.root / "predictions/a2").mkdir(parents=True)
        (self.root / "assets").mkdir()
        q.write_json(self.root / "report.json", {"kd_authorized": False})
        q.write_json(self.root / "predictions/a2/cache.json", {})
        (self.root / "assets/model.pth").write_bytes(b"weights")
        (self.root / "assets/reference.log").write_text("official")
        q.bundle(self.root)
        with zipfile.ZipFile(self.root / "m64_results.zip") as archive:
            self.assertEqual(set(archive.namelist()), {"report.json", "assets/reference.log"})

    def test_notebook_has_six_compilable_ordered_sections(self):
        notebook = json.loads((ROOT / "notebooks/MonoDETR_A2_M64_Teacher_Qualification_Colab.ipynb").read_text())
        cells = ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]
        self.assertEqual(len(cells), 6)
        self.assertEqual(notebook["metadata"]["m64_revision"], q.REVISION)
        for i, cell in enumerate(cells):
            compile(cell, f"M64 section {i + 1}", "exec")
            self.assertNotIn("git reset", cell)
            self.assertNotIn("shutil.rmtree", cell)
            self.assertNotIn("[https://", cell)
        self.assertIn("def run_logged", cells[0])
        self.assertIn("def qualify", cells[0])
        self.assertIn("SETUP_READY = MANIFEST_READY = SMOKE_READY = EVALUATION_READY = False", cells[0])
        for i, flag in ((2, "SETUP_READY"), (3, "MANIFEST_READY"), (4, "SMOKE_READY")):
            self.assertIn(f"if not {flag}:", cells[i])
        self.assertIn("allow_failure=True", cells[4])
        self.assertIn("qualify('bundle')", cells[5])

    def test_no_training_loop_or_unsafe_load_in_workflow(self):
        text = Path(q.__file__).read_text()
        tree = ast.parse(text)
        calls = [ast.unparse(node.func) for node in ast.walk(tree) if isinstance(node, ast.Call)]
        self.assertNotIn("optimizer.step", calls)
        self.assertNotIn("torch.optim.AdamW", calls)
        self.assertIn("weights_only=True", text)
        self.assertNotIn("weights_only=False", text)
        self.assertIn("teacher_selected=False", text)
        self.assertIn("prior_training_ids_independently_verified=False", text)

    def test_helper_imports_do_not_load_model_or_numba(self):
        probe = ("import sys;sys.path.insert(0,sys.argv[1]);import m64_teacher_qualification;"
                 "import patch_m64_inference_sources,setup_m64_runtime;"
                 "assert 'torch' not in sys.modules and 'numba' not in sys.modules")
        subprocess.run([sys.executable, "-c", probe, str(ROOT / "scripts")], check=True)


if __name__ == "__main__":
    unittest.main()
