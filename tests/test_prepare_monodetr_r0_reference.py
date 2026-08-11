from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import torch
    import yaml
except ModuleNotFoundError:  # Minimal host Python; Colab installs both.
    torch = None
    yaml = None


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/prepare_monodetr_r0_reference.py"


class PrepareMonoDETRR0ReferenceTests(unittest.TestCase):
    @unittest.skipUnless(torch is not None and yaml is not None, "requires torch and yaml")
    def test_prepares_frozen_two_class_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / "MonoDETR"
            dataset = root / "kitti"
            output = root / "outputs"
            (repo / "configs").mkdir(parents=True)
            (dataset / "ImageSets").mkdir(parents=True)
            (dataset / "ImageSets/train.txt").write_text("000001\n")
            (dataset / "ImageSets/val.txt").write_text("000002\n")
            base = {
                "model_name": "monodetr",
                "dataset": {"writelist": ["Car"]},
                "model": {"num_classes": 3},
                "trainer": {"save_path": "outputs"},
            }
            (repo / "configs/monodetr.yaml").write_text(
                yaml.safe_dump(base), encoding="utf-8"
            )
            checkpoint = root / "checkpoint_best.pth"
            torch.save({"model_state": {"weight": torch.ones(1)}}, checkpoint)

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--monodetr-repo",
                    str(repo),
                    "--dataset-root",
                    str(dataset),
                    "--official-checkpoint",
                    str(checkpoint),
                    "--output-root",
                    str(output),
                    "--max-epochs",
                    "20",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            config = yaml.safe_load(
                (repo / "configs/monodetr_r0_vehicle_pedestrian.yaml").read_text()
            )
            self.assertEqual(config["dataset"]["writelist"], ["Car", "Pedestrian"])
            self.assertEqual(config["dataset"]["class_mapping"]["Truck"], "Car")
            self.assertEqual(
                config["dataset"]["class_mapping"]["Person_sitting"],
                "Pedestrian",
            )
            self.assertEqual(config["model"]["num_classes"], 3)
            self.assertEqual(config["trainer"]["max_epoch"], 20)
            self.assertTrue(Path(config["trainer"]["pretrain_model"]).is_file())


if __name__ == "__main__":
    unittest.main()
