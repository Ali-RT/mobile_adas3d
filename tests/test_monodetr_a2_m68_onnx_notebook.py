from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDETR_A2_M68_ONNX_Runtime_iPhone_Colab.ipynb"


class M68NotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = ["".join(c.get("source", [])) for c in cls.notebook["cells"]]
        cls.code = ["".join(c.get("source", [])) for c in cls.notebook["cells"]
                    if c["cell_type"] == "code"]

    def test_all_code_cells_compile_and_revision_matches(self):
        self.assertEqual(self.notebook["metadata"]["m68_revision"],
                         "M68-A2-STANDALONE-ONNX-RUNTIME-2026-10-09-r2")
        for index, cell in enumerate(self.code):
            compile(cell, f"m68-cell-{index}", "exec")

    def test_frozen_a2_then_fullval_then_phone_gate_order(self):
        notebook_text = "\n".join(self.cells)
        self.assertIn("m68_a2_onnx_cpu_r2", notebook_text)
        self.assertIn("m68_export", notebook_text)
        self.assertIn("m68_fullval", notebook_text)
        self.assertIn("quality_gate_passed", notebook_text)
        self.assertIn("if FULLVAL_READY", notebook_text)
        self.assertIn("m68_a2_onnx_phone_bundle.zip", notebook_text)
        self.assertNotIn("training performed", notebook_text.lower())
        self.assertIn("no training, distillation, quantization", notebook_text.lower())

    def test_standalone_setup_precedes_export_and_preserves_driver_environment(self):
        notebook_text = "\n".join(self.cells)
        self.assertIn("prepare_m68_a2_onnx.py", notebook_text)
        self.assertLess(notebook_text.index("m68_prepare"), notebook_text.index("m68_export"))
        self.assertNotIn("m67_a2_fp32_export_r1_cuda130_venv", notebook_text)
        self.assertNotIn("restore_m67_ephemeral_for_m68.py", notebook_text)
        self.assertIn("env=ENV", notebook_text)
        self.assertIn("from setup_m64_runtime import runtime_env", notebook_text)
        self.assertIn("--manifest", notebook_text)

    def test_existing_mobile_app_uses_cpu_only_and_keeps_m60_option(self):
        runner = (ROOT / "ios/M60Benchmark/A2ONNXRunner.swift").read_text()
        app = (ROOT / "ios/M60Benchmark/M60App.swift").read_text()
        self.assertIn('providers == ["CPUExecutionProvider"]', runner)
        self.assertIn('"coreml_execution_provider_enabled": false', runner)
        self.assertIn('manifest.onnxruntime_version == "1.30.0"', runner)
        self.assertIn('"runtime_version": manifest.onnxruntime_version', runner)
        self.assertIn("ORTSession(env: env", runner)
        self.assertIn("Run A2 MonoDETR — ONNX Runtime CPU", app)
        self.assertIn("Run MonoDGP M60 — Core ML", app)
        self.assertIn("camera capture", runner.lower())
        self.assertIn("deployment_qualified", runner)


if __name__ == "__main__":
    unittest.main()
