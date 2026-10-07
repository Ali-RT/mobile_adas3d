from __future__ import annotations

import json
import importlib.util
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path
from contextlib import redirect_stdout
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
EXPORTER = ROOT / "scripts/export_rtm3d_km3d_res18_coreml.py"
AUDITOR = ROOT / "scripts/audit_rtm3d_km3d_res18_coreml.py"
NOTEBOOK = ROOT / "notebooks/RTM3D_KM3D_ResNet18_CoreML_Export_Colab.ipynb"


class RTM3DCoreMLExportTests(unittest.TestCase):
    def test_export_and_audit_compile_and_keep_scope_gates(self):
        for path in (EXPORTER, AUDITOR):
            compile(path.read_text(), str(path), "exec")
        exporter = EXPORTER.read_text()
        auditor = AUDITOR.read_text()
        self.assertIn('OUTPUT_NAMES = tuple(HEADS)', exporter)
        self.assertIn('return tuple(outputs[name] for name in OUTPUT_NAMES)', exporter)
        self.assertIn('EXPECTED_CHECKPOINT_SHA256 = "5fa355845f79c1afeffab427de32933758e5b4c1e7c9ec19a94a13737691d05b"', exporter)
        self.assertIn("importlib.util.spec_from_file_location", exporter)
        self.assertIn("def build_pinned_resnet18", exporter)
        self.assertIn("build_pinned_resnet18(repo, state, torch)", exporter)
        self.assertIn("FIXTURE_SAMPLE_COUNT = 16", exporter)
        self.assertIn("np.linspace(0, len(ids) - 1, num=sample_count, dtype=np.int64)", exporter)
        self.assertIn('"fixture_sample_ids": fixture_sample_ids', exporter)
        self.assertIn("MobileADAS3D has a", exporter)
        self.assertNotIn("from models.networks.msra_resnet", exporter)
        self.assertNotIn("from utils.image import", exporter)
        self.assertIn("skip_model_load=True", exporter)
        self.assertIn('"geometry_decode_included": False', exporter)
        self.assertIn('"iphone_performance_measured": False', exporter)
        self.assertIn("safe_load_state", exporter)
        self.assertIn("strict=True", exporter)
        self.assertIn("allow_pickle=False", auditor)
        self.assertIn("for image_index, sample_id in enumerate(sample_ids)", auditor)
        self.assertIn('"images_evaluated": len(sample_ids)', auditor)
        self.assertIn('"per_image": per_image', auditor)
        self.assertIn('platform.system() != "Darwin"', auditor)

    def test_mac_audit_reports_each_fixture_and_keeps_strict_gate(self):
        import numpy as np

        spec = importlib.util.spec_from_file_location("rtm3d_coreml_audit_test", AUDITOR)
        audit = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(audit)
        with tempfile.TemporaryDirectory() as temporary:
            artifact_dir = Path(temporary) / "artifacts"
            package = artifact_dir / "RTM3D_KM3D_ResNet18.mlpackage"
            package.mkdir(parents=True)
            (package / "model.stub").write_bytes(b"fixture model")
            images = np.zeros((2, 3, 384, 1280), dtype=np.float32)
            images[1, 0, 0, 0] = 1.0
            references = np.zeros((2, 1, 1, 1), dtype=np.float32)
            fixture_path = artifact_dir / "parity_reference.npz"
            np.savez_compressed(fixture_path, image=images, hm=references)
            report = {
                "complete": True,
                "mil_has_custom_op": False,
                "mlpackage_sha256": audit.sha256_tree(package),
                "parity_fixture_sha256": audit.sha256_file(fixture_path),
                "outputs": {"hm": {"shape": [1, 1, 1, 1]}},
                "input": {"fixture_sample_ids": ["first", "second"]},
            }
            (artifact_dir / "rtm3d_coreml_export.json").write_text(json.dumps(report))

            class FakeModel:
                def predict(self, inputs):
                    value = float(inputs["image"][0, 0, 0, 0])
                    return {"hm": np.array([[[[value * 0.00025]]]], dtype=np.float32)}

            fake_coremltools = types.ModuleType("coremltools")
            fake_coremltools.__version__ = "test"
            fake_coremltools.ComputeUnit = types.SimpleNamespace(CPU_ONLY="cpu_only")
            fake_coremltools.models = types.SimpleNamespace(
                MLModel=lambda _path, compute_units: FakeModel()
            )
            output_path = Path(temporary) / "native_parity.json"
            argv = ["audit", "--artifact-dir", str(artifact_dir), "--output", str(output_path)]
            with mock.patch.dict(sys.modules, {"coremltools": fake_coremltools}):
                    with mock.patch.object(audit.platform, "system", return_value="Darwin"):
                        with mock.patch.object(sys, "argv", argv):
                            with redirect_stdout(io.StringIO()):
                                with self.assertRaisesRegex(RuntimeError, "parity failed"):
                                    audit.main()

            result = json.loads(output_path.read_text())
            self.assertFalse(result["complete"])
            self.assertEqual(result["images_evaluated"], 2)
            self.assertTrue(result["per_image"]["first"]["hm"]["passed"])
            self.assertFalse(result["per_image"]["second"]["hm"]["passed"])
            self.assertAlmostEqual(result["per_head"]["hm"]["max_abs_delta"], 0.00025)

    def test_export_notebook_cells_compile_and_explain_boundaries(self):
        notebook = json.loads(NOTEBOOK.read_text())
        code = []
        for index, cell in enumerate(notebook["cells"]):
            source = "".join(cell.get("source", []))
            if cell["cell_type"] == "code":
                compile(source, f"rtm3d-coreml-cell-{index}", "exec")
                code.append(source)
        all_code = "\n".join(code)
        all_text = "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])
        self.assertIn("export_rtm3d_km3d_res18_coreml.py", all_code)
        self.assertIn("stderr=subprocess.STDOUT", all_code)
        self.assertIn("full combined log: {EXPORT_LOG}", all_code)
        self.assertIn("calibration-dependent 3d decoder", all_text.lower())
        self.assertIn("Do not connect/use the iPhone yet", all_text)
        self.assertIn("2026-10-07-r3", notebook["metadata"]["rtm3d_coreml_revision"])
        self.assertIn("16 evenly spaced", all_text)


if __name__ == "__main__":
    unittest.main()
