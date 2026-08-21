import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = PROJECT_ROOT / "notebooks" / "MobileADAS3D_H1_GT_Gate_Colab.ipynb"


class H1GTGateNotebookTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )

    def test_notebook_is_resumable_gt_only_and_stops_at_twenty(self):
        self.assertIn("prepare_h1_gt_gate.py", self.source)
        self.assertIn("distillation_enabled'] is False", self.source)
        self.assertIn("--resume", self.source)
        self.assertIn("payload.get('epoch')!=20", self.source)
        self.assertIn("evaluate_kitti_r40.py", self.source)
        self.assertNotIn("AUTHORIZE_CONTINUATION", self.source)

    def test_prepare_script_freezes_edge_and_r0_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = {
                "complete": True,
                "evaluated_checkpoints": 39,
                "selected": {
                    "epoch": 185,
                    "checkpoint": "/content/checkpoint_epoch_185.pth",
                    "checkpoint_sha256": "fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59",
                    "vehicle_3d_moderate": 17.634769196266316,
                    "pedestrian_3d_moderate": 5.721371354710236,
                },
            }
            edge = {
                "complete": True,
                "architecture": "MobileADAS3D-H1",
                "coreml_max_abs_delta": 0.001941,
                "device_gate": {"p95_ms": 5.804},
            }
            selection_path = root / "selection.json"
            edge_path = root / "edge.json"
            selection_path.write_text(json.dumps(selection))
            edge_path.write_text(json.dumps(edge))
            output = root / "output"
            configs = root / "configs"
            subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "prepare_h1_gt_gate.py"),
                    "--base-config", str(PROJECT_ROOT / "configs" / "kitti_mobileadas3d_h1_gt_gate.yaml"),
                    "--r0-selection", str(selection_path),
                    "--edge-evidence", str(edge_path),
                    "--output-dir", str(output),
                    "--config-dir", str(configs),
                    "--skip-checkpoint-hash",
                ],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((output / "h1_gt_gate_manifest.json").read_text())
            self.assertEqual(manifest["architecture"], "MobileADAS3D-H1")
            self.assertFalse(manifest["distillation_enabled"])
            self.assertEqual(manifest["criterion"], "h1_hungarian_set")
            self.assertEqual(manifest["epochs"], 20)


if __name__ == "__main__":
    unittest.main()
