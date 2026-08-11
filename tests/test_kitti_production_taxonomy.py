import json
import tempfile
import unittest
from pathlib import Path

from data.class_taxonomy import (
    KITTI_PRODUCTION_CLASS_MAPPING,
    map_objects,
    normalize_class_mapping,
    taxonomy_sha256,
    validate_taxonomy_manifest,
    split_label_tree_sha256,
)
from data.kitti_dataset import KITTIDataset


class KittiProductionTaxonomyTests(unittest.TestCase):
    def test_real_sample_dataset_maps_and_assigns_production_ids(self):
        root = Path(__file__).resolve().parents[1] / "datasets" / "kitti"
        dataset = KITTIDataset(
            root_dir=str(root),
            classes=["Vehicle", "Pedestrian"],
            sample_ids=["000000"],
            class_mapping=KITTI_PRODUCTION_CLASS_MAPPING,
        )
        objects = dataset[0]["objects"]
        self.assertEqual(
            [(obj["class_name"], obj["source_class_name"], obj["class_id"]) for obj in objects],
            [("Vehicle", "Car", 0), ("Pedestrian", "Pedestrian", 1)],
        )

    def test_mapping_merges_only_approved_sources_and_preserves_source(self):
        objects = [
            {"class_name": "Car", "dimensions_3d": [1.5, 1.6, 3.9]},
            {"class_name": "Van", "dimensions_3d": [2.0, 1.9, 4.8]},
            {"class_name": "Person_sitting", "dimensions_3d": [1.2, 0.6, 0.8]},
            {"class_name": "Cyclist", "dimensions_3d": [1.7, 0.6, 1.7]},
            {"class_name": "DontCare", "dimensions_3d": [0.0, 0.0, 0.0]},
        ]
        mapped = map_objects(objects, KITTI_PRODUCTION_CLASS_MAPPING)
        self.assertEqual(
            [obj["class_name"] for obj in mapped],
            ["Vehicle", "Vehicle", "Pedestrian"],
        )
        self.assertEqual(
            [obj["source_class_name"] for obj in mapped],
            ["Car", "Van", "Person_sitting"],
        )

    def test_mapping_must_cover_exact_production_targets(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            normalize_class_mapping({"Car": "Vehicle"}, ["Vehicle", "Pedestrian"])
        with self.assertRaisesRegex(ValueError, "unknown"):
            normalize_class_mapping(
                {"Car": "Vehicle", "Cyclist": "Cyclist"},
                ["Vehicle"],
            )

    def test_manifest_validation_is_fail_closed(self):
        classes = ["Vehicle", "Pedestrian"]
        mapping = KITTI_PRODUCTION_CLASS_MAPPING
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train = root / "train.txt"
            val = root / "val.txt"
            train.write_text("000000\n", encoding="utf-8")
            val.write_text("000001\n", encoding="utf-8")
            labels = root / "labels"
            labels.mkdir()
            (labels / "000000.txt").write_text("Car example\n", encoding="utf-8")
            (labels / "000001.txt").write_text("Pedestrian example\n", encoding="utf-8")
            from data.class_taxonomy import file_sha256

            manifest = {
                "complete": True,
                "taxonomy_sha256": taxonomy_sha256(classes, mapping),
                "splits": {
                    "train": {"split_file_sha256": file_sha256(train)},
                    "val": {"split_file_sha256": file_sha256(val)},
                },
            }
            manifest["splits"]["train"]["label_tree_sha256"] = (
                split_label_tree_sha256(labels, train)
            )
            manifest["splits"]["val"]["label_tree_sha256"] = (
                split_label_tree_sha256(labels, val)
            )
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            validated = validate_taxonomy_manifest(
                path, classes, mapping, {"train": train, "val": val}, labels
            )
            self.assertTrue(validated["complete"])
            val.write_text("000002\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "val split hash"):
                validate_taxonomy_manifest(
                    path, classes, mapping, {"train": train, "val": val}, labels
                )


if __name__ == "__main__":
    unittest.main()
