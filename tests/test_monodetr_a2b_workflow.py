from __future__ import annotations

import ast
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.prepare_monodetr_a2b_student import (
    R0_CHECKPOINT_SHA256,
    create_pedestrian_balanced_split,
    resolve_r0_selection,
)
from scripts.sweep_monodetr_a2b_product_checkpoints import GATES


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks/MonoDETR_A2b_Pedestrian_Balanced_Two_Class_GT_Colab.ipynb"


class MonoDETRA2bWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.code = "\n".join(
            "".join(cell["source"])
            for cell in cls.notebook["cells"]
            if cell["cell_type"] == "code"
        )

    def test_all_notebook_code_cells_parse(self):
        for cell in self.notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))

    def test_notebook_has_frozen_two_class_gt_only_contract(self):
        self.assertIn("prepare_monodetr_a2b_student.py", self.code)
        self.assertIn("R0_SELECTION", self.code)
        self.assertIn("monodetr_a2b_mnv4_pedestrian_balanced_gt", self.code)
        self.assertIn("timm.__version__ == '1.0.20'", self.code)
        self.assertIn("mobilenetv4_conv_medium.e500_r256_in1k", self.code)
        self.assertNotIn("teacher_cache", self.code)
        self.assertIn("manifest[\x27distillation_enabled\x27] is False", self.code)

    def test_notebook_has_durable_exact_resume_and_sweep(self):
        self.assertIn("checkpoint_epoch_*.pth", self.code)
        self.assertIn("payload.get('optimizer_state') is None", self.code)
        self.assertIn("latest = max(valid_checkpoints", self.code)
        self.assertIn("run_cfg['trainer'].pop('pretrain_model', None)", self.code)
        self.assertIn("PYTHONUNBUFFERED", self.code)
        self.assertIn("Durable combined log:", self.code)
        self.assertIn("sweep_monodetr_a2b_product_checkpoints.py", self.code)
        self.assertIn("a2b_product_selection.json", self.code)

    def test_only_controlled_change_is_pedestrian_sampling(self):
        self.assertIn("--pedestrian-repeat", self.code)
        self.assertIn("'2'", self.code)
        self.assertIn(
            "train_a2b_source.txt",
            (ROOT / "scripts/prepare_monodetr_a2b_student.py").read_text(),
        )
        self.assertIn("manifest[\x27temperature_scaling_enabled\x27] is False", self.code)
        self.assertIn("manifest[\x27distillation_enabled\x27] is False", self.code)

    def test_balanced_split_repeats_only_pedestrian_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "ImageSets").mkdir()
            (root / "training/label_2").mkdir(parents=True)
            (root / "ImageSets/train.txt").write_text("000001\n000002\n000003\n")
            (root / "training/label_2/000001.txt").write_text("Car 0 0 0\n")
            (root / "training/label_2/000002.txt").write_text("Pedestrian 0 0 0\n")
            (root / "training/label_2/000003.txt").write_text("Person_sitting 0 0 0\n")
            split_name, report = create_pedestrian_balanced_split(root, 2)
            self.assertEqual(split_name, "train")
            lines = (root / "ImageSets/train.txt").read_text().splitlines()
            self.assertEqual(lines, ["000001", "000002", "000003", "000002", "000003"])
            split_name_2, report_2 = create_pedestrian_balanced_split(root, 2)
            self.assertEqual(split_name_2, "train")
            self.assertEqual(
                (root / "ImageSets/train.txt").read_text().splitlines(), lines
            )
            self.assertEqual(report_2["source_split_sha256"], report["source_split_sha256"])
            self.assertEqual(report["source_samples"], 3)
            self.assertEqual(report["pedestrian_source_samples"], 2)
            self.assertEqual(report["balanced_samples"], 5)

    def test_all_five_accuracy_gates_are_frozen(self):
        self.assertEqual(
            set(GATES),
            {
                "vehicle_3d_moderate",
                "pedestrian_3d_moderate",
                "mean_3d_moderate",
                "vehicle_bev_moderate",
                "pedestrian_bev_moderate",
            },
        )
        self.assertAlmostEqual(GATES["vehicle_3d_moderate"], 15.8713)
        self.assertAlmostEqual(GATES["pedestrian_bev_moderate"], 5.9365)

    def test_sweep_uses_a2b_source_provenance(self):
        source = (
            ROOT / "scripts/sweep_monodetr_a2b_product_checkpoints.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"--source-name-prefix"', source)
        self.assertIn('"MobileMonoDETR_A2b"', source)

    def test_r0_selection_rejects_wrong_epoch_or_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "epoch185.pth"
            checkpoint.write_bytes(b"checkpoint")
            selection_path = root / "selection.json"
            base = {
                "complete": True,
                "selected": {
                    "epoch": 180,
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": R0_CHECKPOINT_SHA256,
                },
            }
            selection_path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Expected R0 epoch"):
                resolve_r0_selection(selection_path)
            base["selected"]["epoch"] = 185
            base["selected"]["checkpoint_sha256"] = "wrong"
            selection_path.write_text(json.dumps(base), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "Expected frozen R0 SHA-256"):
                resolve_r0_selection(selection_path)

    @mock.patch("scripts.prepare_monodetr_a2b_student.sha256_file")
    def test_r0_selection_accepts_exact_frozen_provenance(self, hash_file):
        hash_file.return_value = R0_CHECKPOINT_SHA256
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            checkpoint = root / "checkpoint_epoch_185.pth"
            checkpoint.touch()
            selection_path = root / "selection.json"
            selection_path.write_text(
                json.dumps(
                    {
                        "complete": True,
                        "selected": {
                            "epoch": 185,
                            "checkpoint": str(checkpoint),
                            "checkpoint_sha256": R0_CHECKPOINT_SHA256,
                        },
                    }
                ),
                encoding="utf-8",
            )
            resolved, _ = resolve_r0_selection(selection_path)
            self.assertEqual(resolved, checkpoint.resolve())


if __name__ == "__main__":
    unittest.main()
