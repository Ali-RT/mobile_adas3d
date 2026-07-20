import json
import tempfile
import unittest
from pathlib import Path

import torch

from models.build import build_model
from scripts.stage_colab_kitti import (
    MANIFEST_NAME,
    collect_counts,
    is_complete,
    write_manifest,
)
from tools.config import apply_runtime_overrides, load_config
from tools.run_manager import resume_run_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "kitti_mnv4_conv_small_baseline.yaml"
NOTEBOOK_PATH = (
    PROJECT_ROOT
    / "notebooks"
    / "MobileADAS3D_MobileNetV4_Colab_Baseline.ipynb"
)


class MobileNetV4BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = load_config(str(CONFIG_PATH))
        config["model"]["pretrained"] = False
        config["model"]["fpn_channels"] = 32
        config["model"]["head_channels"] = 32
        cls.model = build_model(config).eval()

    def test_output_contract_is_stride_16(self):
        with torch.inference_mode():
            outputs = self.model(torch.rand(1, 3, 384, 1280))
        expected_channels = {
            "cls_logits": 3,
            "box2d": 4,
            "log_depth": 1,
            "dim": 3,
            "yaw": 2,
            "center_offset": 2,
            "depth_uncertainty": 1,
            "loc_xy": 2,
        }
        self.assertEqual(set(outputs), set(expected_channels))
        for name, channels in expected_channels.items():
            self.assertEqual(tuple(outputs[name].shape), (1, channels, 24, 80))

    def test_imagenet_normalization_is_inside_model(self):
        self.assertTrue(self.model.normalize_imagenet)
        self.assertEqual(tuple(self.model.input_mean.shape), (1, 3, 1, 1))
        self.assertAlmostEqual(float(self.model.input_mean[0, 0, 0, 0]), 0.485)

    def test_runtime_drive_overrides(self):
        config = apply_runtime_overrides(
            load_config(str(CONFIG_PATH)),
            profile="colab_drive",
            dataset_root="/content/kitti",
            split_dir="/content/drive/splits",
            output_dir="/content/drive/outputs",
        )
        self.assertEqual(
            config["dataset"]["profiles"]["colab_drive"]["root_dir"],
            "/content/kitti",
        )
        self.assertEqual(
            config["dataset"]["splits"]["profile_split_dirs"]["colab_drive"],
            "/content/drive/splits",
        )
        self.assertEqual(config["outputs"]["runs_dir"], "/content/drive/outputs/runs")

    def test_resume_directory_must_reuse_checkpoint_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "run" / "checkpoints" / "latest.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            directories = resume_run_dir(checkpoint)
            self.assertEqual(directories["run_dir"], checkpoint.parent.parent.resolve())

    def test_notebook_is_valid_json_with_training_and_evaluation(self):
        notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        source = "\n".join(
            "".join(cell.get("source", [])) for cell in notebook["cells"]
        )
        self.assertIn("train_mobile_adas3d.py", source)
        self.assertIn("evaluate_kitti_r40.py", source)
        self.assertIn("AUTO_RESUME", source)
        self.assertIn("copy_subdir_with_progress", source)
        self.assertIn("validate_kitti_root", source)
        self.assertIn("STAGE_MANIFEST", source)

    def test_stage_manifest_marks_complete_copy(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "kitti"
            for subdir, suffix in {
                "training/image_2": ".png",
                "training/label_2": ".txt",
                "training/calib": ".txt",
            }.items():
                directory = root / subdir
                directory.mkdir(parents=True)
                (directory / f"000000{suffix}").write_text("x", encoding="utf-8")

            counts = collect_counts(root)
            self.assertEqual(set(counts.values()), {1})
            self.assertFalse(is_complete(root, expected_count=1))

            write_manifest(
                root,
                source=root,
                expected_count=1,
                complete=True,
                source_counts=counts,
                destination_counts=counts,
            )

            self.assertTrue((root / MANIFEST_NAME).is_file())
            self.assertTrue(is_complete(root, expected_count=1))


if __name__ == "__main__":
    unittest.main()
