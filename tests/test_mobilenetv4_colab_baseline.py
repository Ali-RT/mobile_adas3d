import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.target_builder import build_targets_for_sample, encode_yaw_axis_direction
from losses.mobile_adas3d_loss import (
    MobileADAS3DLoss,
    masked_weighted_yaw_cosine_loss,
)
from models.build import build_model
from models.decode import decode_mobile_adas3d_outputs
from models.mobile_adas3d import decode_yaw_axis_direction
from scripts.check_training_ready import parse_args as parse_training_ready_args
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
AP_V1_CONFIG_PATH = PROJECT_ROOT / "configs" / "kitti_mnv4_conv_small_ap_v1.yaml"
V2_CONFIG_PATH = PROJECT_ROOT / "configs" / "kitti_mnv4_calibrated_geometry_v2.yaml"
V3_CONFIG_PATH = PROJECT_ROOT / "configs" / "kitti_mnv4_quality_scoring_v3.yaml"
V4_CONFIG_PATH = PROJECT_ROOT / "configs" / "kitti_mnv4_angular_yaw_v4.yaml"
V5_CONFIG_PATH = PROJECT_ROOT / "configs" / "kitti_mnv4_axis_direction_v5.yaml"
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
        self.assertIn("validate_kitti_source", source)
        self.assertIn("source_mapping", source)
        self.assertIn("training/image_02", source)
        self.assertIn("training/label_02", source)
        self.assertIn("PREFER_ARCHIVE_STAGE", source)
        self.assertIn("try_stage_from_archives", source)
        self.assertIn("DRIVE_ARCHIVE_DIR", source)
        self.assertIn("STAGE_MANIFEST", source)
        self.assertIn("run_streamed", source)
        self.assertIn("PYTHONUNBUFFERED", source)
        self.assertIn("colab_logs", source)
        self.assertIn("mnv4_v1_long80_no_earlystop", source)
        self.assertIn("AUTO_RESUME_MATCH_RUN_NAME", source)
        self.assertIn("load_checkpoint_summary", source)
        self.assertIn("already reached epoch", source)
        self.assertIn("mnv4_v2_calibrated_geometry_quality", source)
        self.assertIn("mnv4_v5_axis_direction", source)
        self.assertIn("configs/kitti_mnv4_axis_direction_v5.yaml", source)
        self.assertIn("sweep_kitti_r40_checkpoints.py", source)
        self.assertIn("checkpoint_ap_summary.csv", source)
        self.assertIn("kitti_r40_latest", source)

    def test_ap_v1_config_has_distinct_run_policy(self):
        config = load_config(str(AP_V1_CONFIG_PATH))
        self.assertEqual(
            config["logging"]["run_name"],
            "mnv4_v1_long80_no_earlystop",
        )
        self.assertFalse(config["early_stopping"]["enabled"])
        self.assertEqual(config["training"]["epochs"], 80)
        self.assertEqual(config["training"]["save_interval"], 5)

    def test_v2_config_enables_calibrated_projected_center_head(self):
        config = load_config(str(V2_CONFIG_PATH))
        config["model"]["pretrained"] = False
        config["model"]["fpn_channels"] = 32
        config["model"]["head_channels"] = 32

        self.assertEqual(
            config["logging"]["run_name"],
            "mnv4_v2_calibrated_geometry_quality",
        )
        self.assertEqual(config["model"]["location_source"], "projected_center")
        self.assertTrue(config["model"]["heads"]["projected_center_offset"])

        model = build_model(config).eval()
        with torch.inference_mode():
            outputs = model(torch.rand(1, 3, 384, 1280))

        self.assertIn("projected_center_offset", outputs)
        self.assertEqual(tuple(outputs["projected_center_offset"].shape), (1, 2, 24, 80))

    def test_v3_config_enables_quality_head_and_scoring(self):
        config = load_config(str(V3_CONFIG_PATH))
        config["model"]["pretrained"] = False
        config["model"]["fpn_channels"] = 32
        config["model"]["head_channels"] = 32

        self.assertEqual(config["logging"]["run_name"], "mnv4_v3_quality_scoring")
        self.assertEqual(config["model"]["location_source"], "projected_center")
        self.assertEqual(config["model"]["score_mode"], "class_quality")
        self.assertEqual(config["inference"]["quality_score_power"], 0.0)
        self.assertTrue(config["model"]["heads"]["quality"])
        self.assertGreater(config["loss"]["quality_weight"], 0.0)

        model = build_model(config).eval()
        with torch.inference_mode():
            outputs = model(torch.rand(1, 3, 384, 1280))

        self.assertIn("projected_center_offset", outputs)
        self.assertIn("quality", outputs)
        self.assertEqual(tuple(outputs["quality"].shape), (1, 1, 24, 80))

    def test_v4_config_uses_angular_yaw_loss_and_class_scoring(self):
        config = load_config(str(V4_CONFIG_PATH))

        self.assertEqual(config["logging"]["run_name"], "mnv4_v4_angular_yaw")
        self.assertEqual(config["model"]["location_source"], "projected_center")
        self.assertFalse(config["model"]["use_quality"])
        self.assertEqual(config["inference"]["score_mode"], "class")
        self.assertGreater(config["loss"]["yaw_cosine_weight"], 0.0)

    def test_yaw_cosine_loss_penalizes_front_back_flip(self):
        target = torch.tensor([[[[0.0]], [[1.0]]]])
        mask = torch.ones(1, 1, 1, 1)

        aligned = masked_weighted_yaw_cosine_loss(target, target, mask)
        orthogonal = masked_weighted_yaw_cosine_loss(
            torch.tensor([[[[1.0]], [[0.0]]]]),
            target,
            mask,
        )
        flipped = masked_weighted_yaw_cosine_loss(-target, target, mask)

        self.assertAlmostEqual(float(aligned), 0.0, places=6)
        self.assertAlmostEqual(float(orthogonal), 1.0, places=6)
        self.assertAlmostEqual(float(flipped), 2.0, places=6)

    def test_v5_axis_direction_preserves_exported_yaw_contract(self):
        config = load_config(str(V5_CONFIG_PATH))
        config["model"]["pretrained"] = False
        config["model"]["fpn_channels"] = 32
        config["model"]["head_channels"] = 32

        self.assertEqual(config["logging"]["run_name"], "mnv4_v5_axis_direction")
        self.assertTrue(config["model"]["use_yaw_axis_direction"])
        self.assertGreater(config["loss"]["yaw_direction_weight"], 0.0)

        model = build_model(config).eval()
        with torch.inference_mode():
            outputs = model(torch.rand(1, 3, 384, 1280))

        self.assertEqual(tuple(outputs["yaw"].shape), (1, 2, 24, 80))
        self.assertEqual(tuple(outputs["yaw_axis"].shape), (1, 2, 24, 80))
        self.assertEqual(tuple(outputs["yaw_direction"].shape), (1, 1, 24, 80))

    def test_yaw_axis_direction_separates_front_and_back(self):
        front_axis, front_direction = encode_yaw_axis_direction(0.0)
        back_axis, back_direction = encode_yaw_axis_direction(torch.pi)

        self.assertAlmostEqual(front_axis[0], back_axis[0], places=6)
        self.assertAlmostEqual(front_axis[1], back_axis[1], places=6)
        self.assertEqual(front_direction, 0.0)
        self.assertEqual(back_direction, 1.0)

        axis_tensor = torch.tensor([[[[0.0]], [[1.0]]]])
        reconstructed = decode_yaw_axis_direction(
            axis_tensor.repeat(2, 1, 1, 1),
            torch.tensor([[[[-1.0]]], [[[1.0]]]]),
        )
        self.assertAlmostEqual(float(reconstructed[0, 0, 0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(reconstructed[0, 1, 0, 0]), 1.0, places=6)
        self.assertAlmostEqual(float(reconstructed[1, 0, 0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(reconstructed[1, 1, 0, 0]), -1.0, places=6)

    def test_v5_zero_axis_has_finite_corner_loss_backward(self):
        def leaf(channels, value=0.0):
            return torch.full(
                (1, channels, 1, 1),
                value,
                dtype=torch.float32,
                requires_grad=True,
            )

        yaw_axis = leaf(2)
        yaw_direction = leaf(1)
        outputs = {
            "cls_logits": leaf(1),
            "box2d": leaf(4, 1.0),
            "log_depth": leaf(1, 2.0),
            "dim": leaf(3),
            "yaw": decode_yaw_axis_direction(yaw_axis, yaw_direction),
            "yaw_axis": yaw_axis,
            "yaw_direction": yaw_direction,
            "center_offset": leaf(2),
            "depth_uncertainty": leaf(1),
            "loc_xy": leaf(2),
        }
        targets = {
            "cls_target": torch.ones(1, 1, 1, 1),
            "box2d_target": torch.ones(1, 4, 1, 1),
            "log_depth_target": torch.full((1, 1, 1, 1), 2.0),
            "dim_target": torch.zeros(1, 3, 1, 1),
            "yaw_target": torch.tensor([[[[0.0]], [[1.0]]]]),
            "yaw_axis_target": torch.tensor([[[[0.0]], [[1.0]]]]),
            "yaw_direction_target": torch.zeros(1, 1, 1, 1),
            "offset_target": torch.zeros(1, 2, 1, 1),
            "loc_xy_target": torch.zeros(1, 2, 1, 1),
            "location_xyz_target": torch.tensor([[[[0.0]], [[0.0]], [[7.3891]]]]),
            "valid_mask": torch.ones(1, 1, 1, 1),
            "loss_weight_target": torch.ones(1, 1, 1, 1),
        }
        criterion = MobileADAS3DLoss(
            input_height=384,
            input_width=1280,
            classes=["Car"],
            class_mean_dims={"Car": [1.5, 1.6, 3.9]},
            yaw_weight=2.0,
            yaw_cosine_weight=1.0,
            yaw_direction_weight=1.0,
            corner3d_weight=0.5,
        )

        losses = criterion(outputs, targets)
        losses["total_loss"].backward()

        self.assertTrue(torch.isfinite(losses["total_loss"]))
        self.assertTrue(torch.isfinite(yaw_axis.grad).all())
        self.assertTrue(torch.isfinite(yaw_direction.grad).all())

    def test_target_builder_adds_projected_center_offset_target(self):
        P2 = [
            [1000.0, 0.0, 0.0, 0.0],
            [0.0, 1000.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ]
        targets = build_targets_for_sample(
            objects=[
                {
                    "class_name": "Car",
                    "bbox_2d": [190.0, 280.0, 210.0, 320.0],
                    "location_3d": [2.0, 3.0, 10.0],
                    "dimensions_3d": [1.5, 1.6, 3.9],
                    "rotation_y": 0.0,
                }
            ],
            original_width=1280,
            original_height=384,
            input_width=1280,
            input_height=384,
            output_stride=16,
            classes=["Car", "Pedestrian", "Cyclist"],
            class_mean_dims={"Car": [1.5, 1.6, 3.9]},
            center_sampling_radius=0,
            P2=P2,
        )

        self.assertIn("projected_center_offset_target", targets)
        self.assertIn("projected_center_valid_mask", targets)
        self.assertIn("quality_target", targets)
        self.assertIn("yaw_axis_target", targets)
        self.assertIn("yaw_direction_target", targets)
        self.assertEqual(float(targets["valid_mask"][0, 18, 12]), 1.0)
        self.assertEqual(float(targets["projected_center_valid_mask"][0, 18, 12]), 1.0)
        self.assertGreater(float(targets["quality_target"][0, 18, 12]), 0.9)
        projected_offset = targets["projected_center_offset_target"][:, 18, 12]
        self.assertAlmostEqual(float(projected_offset[0]), 0.0, places=5)
        self.assertAlmostEqual(float(projected_offset[1]), 0.25, places=5)

    def test_decode_projected_center_backprojects_with_p2(self):
        outputs = {
            "cls_logits": torch.full((1, 3, 24, 80), -20.0),
            "box2d": torch.full((1, 4, 24, 80), 0.01),
            "log_depth": torch.zeros(1, 1, 24, 80),
            "dim": torch.zeros(1, 3, 24, 80),
            "yaw": torch.zeros(1, 2, 24, 80),
            "center_offset": torch.zeros(1, 2, 24, 80),
            "depth_uncertainty": torch.zeros(1, 1, 24, 80),
            "loc_xy": torch.zeros(1, 2, 24, 80),
            "projected_center_offset": torch.zeros(1, 2, 24, 80),
        }
        outputs["cls_logits"][0, 0, 18, 12] = 10.0
        outputs["log_depth"][0, 0, 18, 12] = torch.log(torch.tensor(10.0))
        outputs["yaw"][0, :, 18, 12] = torch.tensor([0.0, 1.0])
        outputs["projected_center_offset"][0, :, 18, 12] = torch.tensor([0.0, 0.25])

        P2 = torch.tensor(
            [
                [1000.0, 0.0, 0.0, 0.0],
                [0.0, 1000.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ]
        )
        predictions = decode_mobile_adas3d_outputs(
            outputs=outputs,
            classes=["Car", "Pedestrian", "Cyclist"],
            class_mean_dims={
                "Car": [1.5, 1.6, 3.9],
                "Pedestrian": [1.7, 0.6, 0.8],
                "Cyclist": [1.7, 0.6, 1.76],
            },
            input_height=384,
            input_width=1280,
            score_threshold=0.5,
            topk=1,
            nms_iou_threshold=0.5,
            P2=P2,
            location_source="projected_center",
        )[0]

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["location_decode_source"], "projected_center")
        self.assertAlmostEqual(predictions[0]["location_3d"][0], 2.0, places=4)
        self.assertAlmostEqual(predictions[0]["location_3d"][1], 3.0, places=4)
        self.assertAlmostEqual(predictions[0]["location_3d"][2], 10.0, places=4)

    def test_decode_quality_score_changes_ranking_when_enabled(self):
        outputs = {
            "cls_logits": torch.full((1, 3, 24, 80), -20.0),
            "box2d": torch.full((1, 4, 24, 80), 0.01),
            "log_depth": torch.zeros(1, 1, 24, 80),
            "dim": torch.zeros(1, 3, 24, 80),
            "yaw": torch.zeros(1, 2, 24, 80),
            "center_offset": torch.zeros(1, 2, 24, 80),
            "depth_uncertainty": torch.zeros(1, 1, 24, 80),
            "loc_xy": torch.zeros(1, 2, 24, 80),
            "quality": torch.full((1, 1, 24, 80), -6.0),
        }
        outputs["cls_logits"][0, 0, 10, 10] = 8.0
        outputs["cls_logits"][0, 0, 12, 12] = 8.0
        outputs["quality"][0, 0, 10, 10] = -2.0
        outputs["quality"][0, 0, 12, 12] = 2.0
        outputs["yaw"][0, :, 10, 10] = torch.tensor([0.0, 1.0])
        outputs["yaw"][0, :, 12, 12] = torch.tensor([0.0, 1.0])

        predictions = decode_mobile_adas3d_outputs(
            outputs=outputs,
            classes=["Car", "Pedestrian", "Cyclist"],
            class_mean_dims={
                "Car": [1.5, 1.6, 3.9],
                "Pedestrian": [1.7, 0.6, 0.8],
                "Cyclist": [1.7, 0.6, 1.76],
            },
            input_height=384,
            input_width=1280,
            score_threshold=0.0,
            topk=1,
            nms_iou_threshold=0.5,
            score_mode="class_quality",
        )[0]

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["cell_x"], 12)
        self.assertEqual(predictions[0]["cell_y"], 12)
        self.assertEqual(predictions[0]["score_mode"], "class_quality")
        self.assertGreater(predictions[0]["quality_score"], 0.8)

    def test_training_ready_cli_accepts_run_name(self):
        with unittest.mock.patch(
            "sys.argv",
            [
                "check_training_ready.py",
                "--config",
                str(AP_V1_CONFIG_PATH),
                "--profile",
                "colab_drive",
                "--run-name",
                "mnv4_v1_long80_no_earlystop",
            ],
        ):
            args = parse_training_ready_args()
        self.assertEqual(args.run_name, "mnv4_v1_long80_no_earlystop")

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
