from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_s1_gt_baseline.py"
SPEC = importlib.util.spec_from_file_location("prepare_s1", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareS1GTBaselineTests(unittest.TestCase):
    def test_accepts_exact_frozen_r0_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "evaluated_checkpoints": 39,
                        "selected": {
                            "epoch": MODULE.EXPECTED_R0_EPOCH,
                            "checkpoint": "/content/checkpoint_epoch_185.pth",
                            "checkpoint_sha256": MODULE.EXPECTED_R0_SHA256,
                            "vehicle_3d_moderate": MODULE.EXPECTED_VEHICLE_3D,
                            "pedestrian_3d_moderate": MODULE.EXPECTED_PEDESTRIAN_3D,
                        },
                    }
                ),
                encoding="utf-8",
            )
            selection = MODULE.validate_r0_selection(path, verify_checkpoint=False)
            self.assertEqual(selection["selected"]["epoch"], 185)

    def test_rejects_changed_r0_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selection.json"
            path.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "evaluated_checkpoints": 39,
                        "selected": {
                            "epoch": MODULE.EXPECTED_R0_EPOCH,
                            "checkpoint": "/content/checkpoint_epoch_185.pth",
                            "checkpoint_sha256": "bad",
                            "vehicle_3d_moderate": MODULE.EXPECTED_VEHICLE_3D,
                            "pedestrian_3d_moderate": MODULE.EXPECTED_PEDESTRIAN_3D,
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(RuntimeError):
                MODULE.validate_r0_selection(path, verify_checkpoint=False)


if __name__ == "__main__":
    unittest.main()
