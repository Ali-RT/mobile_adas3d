from __future__ import annotations

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "scripts/export_rtm3d_km3d_res18_onnx.py"
AUDITOR = ROOT / "scripts/audit_rtm3d_km3d_res18_onnx.py"
NOTEBOOK = ROOT / "notebooks/RTM3D_KM3D_ResNet18_ONNX_Export_Colab.ipynb"


class RTM3DONNXExportTests(unittest.TestCase):
    def test_exporter_and_auditor_compile_and_freeze_scope(self):
        for path in (EXPORTER, AUDITOR):
            compile(path.read_text(), str(path), "exec")
        exporter = EXPORTER.read_text()
        auditor = AUDITOR.read_text()
        self.assertIn("EXPECTED_CHECKPOINT_SHA256", exporter)
        self.assertIn("EXPECTED_VAL_SPLIT_SHA256", exporter)
        self.assertIn("OPSET_VERSION = 17", exporter)
        self.assertIn("onnx.checker.check_model(onnx_model, full_check=True)", exporter)
        self.assertIn("dynamo=False", exporter)
        self.assertIn("onnxruntime_cpu_parity_passed", exporter)
        self.assertIn("PARITY_TOLERANCE = 2e-4", auditor)
        self.assertIn("allow_pickle=False", auditor)
        self.assertIn('"per_image": per_image', auditor)
        self.assertIn('"geometry_decode_audited": False', auditor)
        self.assertIn('"iphone_performance_measured": False', auditor)

    def test_onnx_cpu_audit_keeps_gate_and_reports_per_image(self):
        spec = importlib.util.spec_from_file_location("rtm3d_onnx_audit_test", AUDITOR)
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        with tempfile.TemporaryDirectory() as temporary:
            artifact_dir = Path(temporary) / "artifacts"
            artifact_dir.mkdir()
            model_path = artifact_dir / "RTM3D_KM3D_ResNet18.onnx"
            model_path.write_bytes(b"fixture model")
            images = np.zeros((2, 3, 384, 1280), dtype=np.float32)
            images[1, 0, 0, 0] = 1.0
            reference = np.zeros((2, 1, 1, 1), dtype=np.float32)
            fixture_path = artifact_dir / "parity_reference.npz"
            np.savez_compressed(fixture_path, image=images, hm=reference)
            export_report = {
                "export_complete": True,
                "onnx_checker_passed": True,
                "onnx_sha256": audit.sha256_file(model_path),
                "parity_fixture_sha256": audit.sha256_file(fixture_path),
                "input": {"name": "image", "fixture_sample_ids": ["first", "second"]},
                "outputs": {"hm": {"shape": [1, 1, 1, 1]}},
            }
            (artifact_dir / "rtm3d_onnx_export.json").write_text(json.dumps(export_report))

            class FakeSession:
                def __init__(self, _path, providers):
                    self.providers = providers

                def get_inputs(self):
                    return [types.SimpleNamespace(name="image")]

                def get_outputs(self):
                    return [types.SimpleNamespace(name="hm")]

                def get_providers(self):
                    return self.providers

                def run(self, _names, inputs):
                    value = float(inputs["image"][0, 0, 0, 0]) * 0.00025
                    return [np.array([[[[value]]]], dtype=np.float32)]

            fake_ort = types.ModuleType("onnxruntime")
            fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
            fake_ort.InferenceSession = FakeSession
            output = Path(temporary) / "ort_report.json"
            with mock.patch.dict(sys.modules, {"onnxruntime": fake_ort}):
                with redirect_stdout(io.StringIO()):
                    report = audit.run_audit(artifact_dir, output)

            self.assertFalse(report["complete"])
            self.assertEqual(report["images_evaluated"], 2)
            self.assertTrue(report["per_image"]["first"]["hm"]["passed"])
            self.assertFalse(report["per_image"]["second"]["hm"]["passed"])
            self.assertEqual(report["per_head"]["hm"]["elements_over_tolerance"], 1)
            self.assertAlmostEqual(report["per_head"]["hm"]["max_abs_delta"], 0.00025)

    def test_colab_notebook_code_compiles_and_explains_runtime_boundary(self):
        notebook = json.loads(NOTEBOOK.read_text())
        code = []
        all_text = []
        for index, cell in enumerate(notebook["cells"]):
            source = "".join(cell.get("source", []))
            all_text.append(source)
            if cell["cell_type"] == "code":
                compile(source, f"rtm3d-onnx-cell-{index}", "exec")
                code.append(source)
        all_code = "\n".join(code)
        self.assertIn("export_rtm3d_km3d_res18_onnx.py", all_code)
        self.assertIn("onnxruntime_cpu_parity.json", all_code)
        self.assertIn("ONNX Runtime CPU parity passed", all_code)
        self.assertIn("Core ML execution provider", "\n".join(all_text))
        self.assertIn("2026-10-07-r1", notebook["metadata"]["rtm3d_onnx_revision"])
        self.assertIn("No phone connection needed", "\n".join(all_text))


if __name__ == "__main__":
    unittest.main()
