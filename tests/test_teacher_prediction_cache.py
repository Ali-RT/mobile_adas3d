import json
import tempfile
import unittest
from pathlib import Path

from scripts.create_teacher_prediction_cache import create_cache


class TeacherPredictionCacheTests(unittest.TestCase):
    def test_creates_complete_cache_with_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions = root / "source"
            predictions.mkdir()
            (predictions / "000001.txt").write_text(
                "Car 0 0 0 1 2 3 4 1.5 1.6 3.9 1 2 20 0.1 0.9\n",
                encoding="utf-8",
            )
            (predictions / "000002.txt").write_text("", encoding="utf-8")
            split = root / "train.txt"
            split.write_text("000001\n000002\n", encoding="utf-8")
            config = root / "runtime.yaml"
            inference_root = root / "inference_view"
            (inference_root / "ImageSets").mkdir(parents=True)
            (inference_root / "ImageSets/val.txt").write_text(
                "000001\n000002\n", encoding="utf-8"
            )
            config.write_text(
                f"dataset:\n  root_dir: {inference_root}\n  test_split: val\n",
                encoding="utf-8",
            )

            manifest = create_cache(
                prediction_dir=predictions,
                split_file=split,
                output_dir=root / "cache",
                runtime_config=config,
                teacher_name="MonoDETR_official",
                teacher_source_commit="abc123",
                checkpoint_sha256="f" * 64,
                expected_count=2,
                allowed_classes=["Car"],
            )

            saved = json.loads(
                (root / "cache/teacher_cache_manifest.json").read_text()
            )
            self.assertTrue(manifest["complete"])
            self.assertEqual(saved, manifest)
            self.assertEqual(manifest["prediction_files"], 2)
            self.assertEqual(manifest["detection_count"], 1)
            self.assertEqual(manifest["empty_prediction_files"], 1)
            self.assertEqual(manifest["teacher_source_commit"], "abc123")
            self.assertEqual(manifest["inference_dataset_split"], "val")
            self.assertFalse(manifest["inference_data_augmentation"])
            self.assertTrue((root / "cache/predictions/000002.txt").is_file())

    def test_rejects_missing_or_extra_prediction_files(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions = root / "source"
            predictions.mkdir()
            (predictions / "000001.txt").write_text("", encoding="utf-8")
            (predictions / "999999.txt").write_text("", encoding="utf-8")
            split = root / "train.txt"
            split.write_text("000001\n000002\n", encoding="utf-8")
            config = root / "runtime.yaml"
            inference_root = root / "inference_view"
            (inference_root / "ImageSets").mkdir(parents=True)
            (inference_root / "ImageSets/val.txt").write_text(
                "000001\n000002\n", encoding="utf-8"
            )
            config.write_text(
                f"dataset:\n  root_dir: {inference_root}\n  test_split: val\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "missing=1, extra=1"):
                create_cache(
                    prediction_dir=predictions,
                    split_file=split,
                    output_dir=root / "cache",
                    runtime_config=config,
                    teacher_name="teacher",
                    teacher_source_commit="abc",
                    checkpoint_sha256="f" * 64,
                    expected_count=2,
                )

    def test_rejects_train_split_because_it_enables_augmentation(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            predictions = root / "source"
            predictions.mkdir()
            (predictions / "000001.txt").write_text("", encoding="utf-8")
            split = root / "train.txt"
            split.write_text("000001\n", encoding="utf-8")
            config = root / "runtime.yaml"
            config.write_text(
                f"dataset:\n  root_dir: {root}\n  test_split: train\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "random data augmentation"):
                create_cache(
                    prediction_dir=predictions,
                    split_file=split,
                    output_dir=root / "cache",
                    runtime_config=config,
                    teacher_name="teacher",
                    teacher_source_commit="abc",
                    checkpoint_sha256="f" * 64,
                    expected_count=1,
                )


if __name__ == "__main__":
    unittest.main()
