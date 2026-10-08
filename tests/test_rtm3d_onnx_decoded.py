from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
import warnings
import zipfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts/audit_rtm3d_km3d_res18_onnx_decoded.py"
NOTEBOOK = ROOT / "notebooks/RTM3D_KM3D_ResNet18_ONNX_Decoded_Parity_Colab.ipynb"


spec = importlib.util.spec_from_file_location("rtm3d_onnx_decoded_audit_test", AUDITOR)
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


class RTM3DONNXDecodedTests(unittest.TestCase):
    def test_iou_and_same_class_greedy_match(self):
        left = np.asarray([0, 0, 10, 10], dtype=np.float32)
        self.assertEqual(audit.box_iou_xyxy(left, left), 1.0)
        reference = np.zeros((2, 41), dtype=np.float32)
        onnx = np.zeros((2, 41), dtype=np.float32)
        reference[:, :4] = [[0, 0, 10, 10], [20, 20, 30, 30]]
        onnx[:, :4] = [[1, 1, 11, 11], [20, 20, 30, 30]]
        reference[:, 40] = [0, 1]
        onnx[:, 40] = [0, 0]
        matches, unmatched_reference, unmatched_onnx = audit.greedy_match(reference, onnx)
        self.assertEqual([(ref, pred) for ref, pred, _ in matches], [(0, 0)])
        self.assertEqual(unmatched_reference, [1])
        self.assertEqual(unmatched_onnx, [1])

    def test_selected_rows_drops_nonfinite_and_keeps_strict_score_threshold(self):
        rows = np.zeros((3, 41), dtype=np.float32)
        rows[:, 4] = [0.3, 0.30001, 0.9]
        rows[2, 35] = np.nan
        selected = audit.selected_rows(rows)
        self.assertEqual(selected.shape, (1, 41))
        self.assertAlmostEqual(float(selected[0, 4]), 0.30001, places=6)

    def test_report_metrics_include_decoded_hps_effects(self):
        source = AUDITOR.read_text(encoding="utf-8")
        for metric in (
            "keypoints_max_abs_px",
            "keypoints_mean_abs_px",
            "keypoints_rmse_px",
            "location_l2_m",
            "yaw_abs_deg",
            "interpretation_only_no_deployment_gate",
        ):
            self.assertIn(metric, source)
        self.assertIn('"training_performed": False', source)
        self.assertIn('"iphone_performance_measured": False', source)

    def test_bundle_extracts_only_expected_artifacts_and_rejects_duplicate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle.zip"
            with zipfile.ZipFile(bundle, "w") as archive:
                for name in audit.BUNDLE_FILES:
                    archive.writestr(name, name.encode())
                archive.writestr("ignored.txt", b"not extracted")
            extracted = root / "extracted"
            extracted.mkdir()
            audit.extract_bundle(bundle, extracted)
            self.assertEqual({path.name for path in extracted.iterdir()}, audit.BUNDLE_FILES)

            duplicate = root / "duplicate.zip"
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                with zipfile.ZipFile(duplicate, "w") as archive:
                    for name in audit.BUNDLE_FILES:
                        archive.writestr(name, name.encode())
                    archive.writestr("rtm3d_onnx_export.json", b"duplicate")
            with self.assertRaisesRegex(RuntimeError, "duplicate"):
                audit.extract_bundle(duplicate, root / "duplicate_extracted")

    def test_notebook_code_compiles_and_keeps_diagnostic_scope_clear(self):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        all_text = []
        for index, cell in enumerate(notebook["cells"]):
            source = "".join(cell.get("source", []))
            all_text.append(source)
            if cell["cell_type"] == "code":
                compile(source, f"rtm3d-onnx-decoded-cell-{index}", "exec")
        text = "\n".join(all_text)
        self.assertIn("audit_rtm3d_km3d_res18_onnx_decoded.py", text)
        self.assertIn("No phone connection is needed yet", text)
        self.assertIn("hps", text)
        self.assertIn("not an accuracy gate", text)
        self.assertEqual(notebook["metadata"]["rtm3d_onnx_decoded_revision"], "2026-10-08-r1")


if __name__ == "__main__":
    unittest.main()
