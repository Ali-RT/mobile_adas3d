import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from data.teacher_target_adapter import TeacherTargetAdapter, pad_teacher_targets


def prediction_tree_sha256(prediction_dir: Path, sample_ids: list[str]) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        path = prediction_dir / f"{sample_id}.txt"
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class TeacherTargetAdapterTest(unittest.TestCase):
    def make_adapter(self, root: Path) -> TeacherTargetAdapter:
        split = root / "train.txt"
        split.write_text("000001\n")
        prediction_dir = root / "cache/predictions"
        prediction_dir.mkdir(parents=True)
        prediction_dir.joinpath("000001.txt").write_text(
            "Car 0 0 0 0 0 10 10 1.5 1.6 4.0 1 2 30 0.2 0.90\n"
            "Car 0 0 0 20 20 30 30 1.4 1.5 3.9 2 2 20 -0.4 0.80\n"
            "Car 0 0 0 40 40 50 50 1.4 1.5 3.9 3 2 20 0.1 0.20\n"
        )
        manifest = {
            "schema_version": 1,
            "complete": True,
            "inference_data_augmentation": False,
            "split_images": 1,
            "split_file_sha256": hashlib.sha256(split.read_bytes()).hexdigest(),
            "allowed_classes": ["Car"],
            "checkpoint_sha256": "checkpoint",
            "prediction_tree_sha256": prediction_tree_sha256(
                prediction_dir, ["000001"]
            ),
        }
        root.joinpath("cache/teacher_cache_manifest.json").write_text(
            json.dumps(manifest)
        )
        return TeacherTargetAdapter(
            root / "cache",
            split,
            expected_checkpoint_sha256="checkpoint",
            expected_prediction_tree_sha256=manifest["prediction_tree_sha256"],
        )

    def test_aligns_matches_to_original_object_order(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            adapter = self.make_adapter(Path(temp_dir))
            objects = [
                {"class_name": "Pedestrian", "bbox_2d": [0, 0, 10, 10], "location_3d": [0, 0, 10]},
                {"class_name": "Car", "bbox_2d": [20, 20, 30, 30], "location_3d": [0, 0, 20]},
                {"class_name": "Car", "bbox_2d": [0, 0, 10, 10], "location_3d": [0, 0, 30]},
                {"class_name": "Car", "bbox_2d": [40, 40, 50, 50], "location_3d": [0, 0, 70]},
            ]
            targets = adapter.build_for_sample("000001", objects)
            self.assertEqual(targets["teacher_valid_mask"].tolist(), [False, True, True, False])
            self.assertTrue(torch.allclose(targets["teacher_score"], torch.tensor([0.0, 0.8, 0.9, 0.0])))
            self.assertTrue(torch.allclose(targets["teacher_location_3d"][1], torch.tensor([2.0, 2.0, 20.0])))

    def test_matches_all_cars_before_applying_distance_mask(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = self.make_adapter(root)
            root.joinpath("cache/predictions/000001.txt").write_text(
                "Car 0 0 0 0 0 10 10 1.5 1.6 4.0 1 2 70 0.2 0.90\n"
            )
            adapter = TeacherTargetAdapter(
                root / "cache", root / "train.txt", verify_prediction_tree=False
            )
            objects = [
                {"class_name": "Car", "bbox_2d": [0, 0, 10, 10], "location_3d": [0, 0, 70]},
                {"class_name": "Car", "bbox_2d": [0, 0, 9, 9], "location_3d": [0, 0, 30]},
            ]
            targets = adapter.build_for_sample("000001", objects)
            self.assertEqual(targets["teacher_valid_mask"].tolist(), [False, False])

    def test_rejects_tampered_prediction_tree(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = self.make_adapter(root)
            self.assertIsNotNone(adapter)
            root.joinpath("cache/predictions/000001.txt").write_text("")
            with self.assertRaisesRegex(RuntimeError, "do not match"):
                TeacherTargetAdapter(root / "cache", root / "train.txt")

    def test_pads_variable_object_counts(self):
        first = {
            "teacher_valid_mask": torch.tensor([True, False]),
            "teacher_score": torch.tensor([0.8, 0.0]),
        }
        second = {
            "teacher_valid_mask": torch.tensor([True]),
            "teacher_score": torch.tensor([0.9]),
        }
        batch = pad_teacher_targets([first, second])
        self.assertEqual(batch["teacher_score"].shape, (2, 2))
        self.assertEqual(batch["teacher_object_mask"].tolist(), [[True, True], [True, False]])


if __name__ == "__main__":
    unittest.main()
