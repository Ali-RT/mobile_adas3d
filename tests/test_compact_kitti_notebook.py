"""Notebook structure/contract tests require only the standard library."""
import ast
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/Compact_MobileNetV4_KITTI_GT_Baseline_Colab.ipynb"


class CompactNotebookTests(unittest.TestCase):
    def setUp(self):
        self.notebook = json.loads(NOTEBOOK.read_text())
        self.code = ["".join(c["source"]) for c in self.notebook["cells"] if c["cell_type"] == "code"]
        self.text = "\n".join("".join(c["source"]) for c in self.notebook["cells"])

    def test_every_code_cell_compiles_and_is_output_free(self):
        for index, code in enumerate(self.code):
            compile(code, f"compact-cell-{index+1}", "exec")
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])

    def test_all_helpers_defined_in_first_code_cell(self):
        tree = ast.parse(self.code[0])
        names = {node.name for node in tree.body if isinstance(node, ast.FunctionDef)}
        self.assertEqual(names, {"run_logged", "stage"})
        for name in ("Path", "json", "os", "subprocess", "sys", "deque", "shlex"):
            self.assertIn(name, self.code[0])
        self.assertIn("stderr=subprocess.STDOUT", self.code[0])
        self.assertIn("log.flush()", self.code[0])

    def test_updates_main_before_checking_scripts_without_reset(self):
        setup = self.code[1]
        self.assertIn("https://github.com/Ali-RT/mobile_adas3d.git", setup)
        self.assertLess(setup.index("'pull','--ff-only'"), setup.index("for required in"))
        self.assertIn("'status','--porcelain'", setup)
        self.assertNotIn("reset", setup)
        self.assertNotIn("git checkout", self.text)
        self.assertNotIn("[https", "\n".join(self.code))

    def test_no_old_runtime_or_native_build_and_no_teacher(self):
        code = "\n".join(self.code)
        self.assertNotIn("setup_m67_runtime", code)
        self.assertNotIn("setup.py", code)
        self.assertNotIn("MultiScaleDeformableAttention", code)
        self.assertNotIn("pip','install','torch", code)
        self.assertIn("no teacher", self.text.lower())
        self.assertIn("RGB", self.text)
        self.assertIn("BGR", self.text)

    def test_gate_order_and_resume_bundle_are_explicit(self):
        code = "\n".join(self.code)
        self.assertLess(code.index("stage('smoke')"), code.index("stage('tiny'"))
        self.assertLess(code.index("stage('tiny'"), code.index("stage('train'"))
        self.assertIn("'--resume'", self.code[4])
        self.assertIn("stage('bundle')", self.code[5])
        self.assertIn("checkpoints/latest.pt", self.code[4])
        self.assertIn("not AP accuracy", self.text)
        revision = self.notebook["metadata"]["compact_training_revision"]
        self.assertIn(revision, self.code[0])

    def test_embedded_cuda_probe_compiles(self):
        tree = ast.parse(self.code[1])
        probe = next(node.value.value for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "probe" for target in node.targets))
        compile(probe, "cuda-probe", "exec")

    def test_streaming_helper_persists_stdout_stderr_and_failure(self):
        tree = ast.parse(self.code[0])
        definition = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_logged")
        with tempfile.TemporaryDirectory() as directory:
            namespace = dict(LOGS=Path(directory), datetime=datetime, timezone=timezone,
                             os=os, subprocess=subprocess, deque=deque, shlex=shlex)
            exec(compile(ast.Module(body=[definition], type_ignores=[]), "log-helper", "exec"), namespace)
            code, path = namespace["run_logged"](
                [sys.executable, "-c", "import sys; print('stdout marker'); print('stderr marker',file=sys.stderr); sys.exit(3)"],
                "failure", allow_failure=True)
            self.assertEqual(code, 3)
            text = path.read_text()
            for marker in ("stdout marker", "stderr marker", "EXIT CODE: 3"):
                self.assertIn(marker, text)
            with self.assertRaisesRegex(RuntimeError, "full log"):
                namespace["run_logged"]([sys.executable, "-c", "raise RuntimeError('expected')"], "raise")


if __name__ == "__main__":
    unittest.main()
