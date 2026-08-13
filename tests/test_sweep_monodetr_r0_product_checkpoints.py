from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sweep_monodetr_r0_product_checkpoints.py"
SPEC = importlib.util.spec_from_file_location("r0_sweep", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class MonoDETRR0ProductSweepTests(unittest.TestCase):
    def test_collects_epoch_checkpoints_in_numeric_order(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (
                "checkpoint_epoch_10.pth",
                "checkpoint_epoch_5.pth",
                "checkpoint_best.pth",
                "checkpoint_epoch_bad.pth",
            ):
                (root / name).write_bytes(b"checkpoint")
            self.assertEqual(
                [(epoch, path.name) for epoch, path in MODULE.collect_checkpoints(root)],
                [(5, "checkpoint_epoch_5.pth"), (10, "checkpoint_epoch_10.pth")],
            )

    def test_extracts_one_moderate_product_metric(self):
        summary = {
            "metrics": [
                {
                    "metric": "3d",
                    "class_name": "Vehicle",
                    "difficulty": "moderate",
                    "ap_r40": 25.0,
                }
            ]
        }
        self.assertEqual(MODULE.metric_value(summary, "3d", "Vehicle"), 25.0)
        with self.assertRaises(RuntimeError):
            MODULE.metric_value(summary, "3d", "Pedestrian")


if __name__ == "__main__":
    unittest.main()
