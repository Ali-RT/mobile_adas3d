import tempfile
import unittest
from pathlib import Path

from data.kitti_prediction_parser import (
    load_kitti_prediction_directory,
    parse_kitti_prediction_file,
)
from data.kitti_r40 import evaluate_kitti_r40


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEACHER_NOTEBOOK = (
    PROJECT_ROOT / "notebooks" / "MonoDETR_Teacher_Feasibility_Colab.ipynb"
)


class KITTIPredictionParserTests(unittest.TestCase):
    def test_teacher_notebook_resolves_drive_aliases(self):
        source = TEACHER_NOTEBOOK.read_text(encoding="utf-8")

        self.assertIn("/content/drive/MyDrive/datasets/kitti", source)
        self.assertIn("training/image_02", source)
        self.assertIn("training/label_02", source)
        self.assertIn("KITTI_SOURCE_ROOT", source)
        self.assertIn("--dataset-root", source)
        self.assertIn("MONODETR_KITTI", source)
        self.assertIn("value.scalar_type()", source)
        self.assertIn("deprecated_count == 2", source)
        self.assertIn("MultiScaleDeformableAttention", source)
        self.assertIn("'ninja'", source)
        self.assertIn("from torch.nn import Linear as _LinearWithBias", source)
        self.assertIn("private_linear_count == 1", source)
        self.assertIn("from torch.overrides import has_torch_function", source)
        self.assertIn("private_overrides_count == 1", source)
        self.assertIn("from lib.models.monodetr import build_monodetr", source)

    def test_parses_scored_kitti_detection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "000001.txt"
            path.write_text(
                "Car 0.0 0 0.0 10 10 70 70 1.5 1.6 3.9 1.0 1.5 20.0 0.1 0.95\n",
                encoding="utf-8",
            )

            predictions = parse_kitti_prediction_file(path)

        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0]["class_name"], "Car")
        self.assertAlmostEqual(predictions[0]["score"], 0.95)
        self.assertEqual(predictions[0]["dimensions_3d_hwl"], [1.5, 1.6, 3.9])
        self.assertAlmostEqual(predictions[0]["yaw"], 0.1)

    def test_requires_score_column(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "000001.txt"
            path.write_text(
                "Car 0.0 0 0.0 10 10 70 70 1.5 1.6 3.9 1.0 1.5 20.0 0.1\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "including score"):
                parse_kitti_prediction_file(path)

    def test_strict_directory_load_rejects_partial_split(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "000001.txt"
            path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(FileNotFoundError, "Missing 1 prediction"):
                load_kitti_prediction_directory(
                    temp_dir,
                    sample_ids=["000001", "000002"],
                )

    def test_parsed_perfect_prediction_scores_full_ap(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "000001.txt"
            path.write_text(
                "Car 0.0 0 0.0 10 10 70 70 1.5 1.6 3.9 1.0 1.5 20.0 0.1 0.95\n",
                encoding="utf-8",
            )
            predictions = {
                "000001": parse_kitti_prediction_file(path),
            }

        ground_truth = {
            "000001": [
                {
                    "class_name": "Car",
                    "truncated": 0.0,
                    "occluded": 0,
                    "bbox_2d": [10.0, 10.0, 70.0, 70.0],
                    "dimensions_3d": [1.5, 1.6, 3.9],
                    "location_3d": [1.0, 1.5, 20.0],
                    "rotation_y": 0.1,
                }
            ]
        }
        results = evaluate_kitti_r40(
            ground_truth,
            predictions,
            classes=["Car"],
        )

        self.assertTrue(all(result.ap_r40 == 100.0 for result in results))


if __name__ == "__main__":
    unittest.main()
