from __future__ import annotations

import json
import unittest
from pathlib import Path

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
        self.assertIn("skip_model_load=True", exporter)
        self.assertIn('"geometry_decode_included": False', exporter)
        self.assertIn('"iphone_performance_measured": False', exporter)
        self.assertIn("safe_load_state", exporter)
        self.assertIn("strict=True", exporter)
        self.assertIn("allow_pickle=False", auditor)
        self.assertIn('platform.system() != "Darwin"', auditor)

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
        self.assertIn("calibration-dependent 3d decoder", all_text.lower())
        self.assertIn("Do not connect/use the iPhone yet", all_text)
        self.assertIn("2026-10-07-r1", notebook["metadata"]["rtm3d_coreml_revision"])


if __name__ == "__main__":
    unittest.main()
