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
                         "M68-A2-ONNX-ORT-CPU-IPHONE-FEASIBILITY-2026-10-08-r1")
        for index, cell in enumerate(self.code):
            compile(cell, f"m68-cell-{index}", "exec")

    def test_frozen_a2_then_fullval_then_phone_gate_order(self):
        notebook_text = "\n".join(self.cells)
        self.assertIn("m67_a2_fp32_export_r1_cuda130_venv", notebook_text)
        self.assertIn("m68_export", notebook_text)
        self.assertIn("m68_fullval", notebook_text)
        self.assertIn("quality_gate_passed", notebook_text)
        self.assertIn("if FULLVAL_READY", notebook_text)
        self.assertIn("m68_a2_onnx_phone_bundle.zip", notebook_text)
        self.assertNotIn("training performed", notebook_text.lower())
        self.assertIn("no training, distillation, quantization", notebook_text.lower())

    def test_colab_reset_rebuilds_only_ephemeral_m67_inputs_safely(self):
        notebook_text = "\n".join(self.cells)
        recovery = (ROOT / "scripts/restore_m67_ephemeral_for_m68.py").read_text()
        self.assertIn("restore_m67_ephemeral_for_m68.py", notebook_text)
        self.assertIn("m68_restore_ephemeral_m67", notebook_text)
        self.assertLess(notebook_text.index("m68_restore_ephemeral_m67"),
                        notebook_text.index("Frozen M67 manifest:"))
        self.assertIn("signature_sha256", recovery)
        self.assertIn("runtime_receipt_sha256", recovery)
        self.assertIn("current_environment != m67[\"environment\"]", recovery)
        self.assertIn("build_m64_attention.py", recovery)
        self.assertIn("audit_m67_a2_coreml.py", recovery)
        self.assertNotIn('["git", "reset"', recovery)
        self.assertNotIn("shutil.rmtree", recovery)

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
