"""Synthetic signed-error aggregation tests: no models or data downloads."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts import diagnose_parking_v0_errors as audit
from scripts import parking_v0_v1_metrics as metrics
from scripts import prepare_parking_v0_v1_benchmark as benchmark
from scripts import parking_edge_common as common


def row(log="l1", track="t1", ann="a1", camera="c1", error=1., count=5):
    return {"cell": "Vehicle:2-4m", "recording_log": log, "instance_token": track,
        "annotation_token": ann, "camera_token": camera, "reference_points": count,
        "valid_points": count, "farther_points": count if error > 0 else 0,
        "closer_points": count if error < 0 else 0, "mae_m": abs(error),
        "signed_error_m": error, "ground_truth_depth_m": 3., "predicted_depth_m": 3.+error}


class DiagnosticTests(unittest.TestCase):
    def test_positive_means_too_far(self):
        result = audit.summarize([row(error=1.5)])
        self.assertEqual(result["log_track_observation_macro"]["signed_error_m"], 1.5)
        self.assertEqual(result["point_fraction_farther_than_truth"], 1.)

    def test_negative_means_too_close(self):
        result = audit.summarize([row(error=-1.5)])
        self.assertEqual(result["log_track_observation_macro"]["signed_error_m"], -1.5)
        self.assertEqual(result["point_fraction_closer_than_truth"], 1.)

    def test_bias_cancels_but_absolute_error_does_not(self):
        result = audit.summarize([row(error=1.), row(camera="c2", error=-1.)])
        self.assertEqual(result["log_track_observation_macro"]["signed_error_m"], 0.)
        self.assertEqual(result["log_track_observation_macro"]["mae_m"], 1.)

    def test_points_do_not_dominate_logs(self):
        result = audit.summarize([row(error=1., count=5000), row(log="l2", error=3., count=5)])
        self.assertEqual(result["log_track_observation_macro"]["mae_m"], 2.)
        self.assertEqual(result["reference_recording_logs"], 2)

    def test_equal_tracks_not_number_of_observations(self):
        rows = [row(error=1.), row(ann="a2", camera="c2", error=1.), row(track="t2", ann="a3", error=3.)]
        self.assertEqual(audit.summarize(rows)["log_track_observation_macro"]["mae_m"], 2.)

    def test_equal_cameras_inside_observation(self):
        rows = [row(error=1.), row(camera="c2", error=3.), row(ann="a2", camera="c3", error=4.)]
        self.assertEqual(audit.summarize(rows)["log_track_observation_macro"]["mae_m"], 3.)

    def test_empty_is_null_not_zero(self):
        result = audit.summarize([])
        self.assertIsNone(result["prediction_coverage"])
        self.assertIsNone(result["log_track_observation_macro"]["mae_m"])

    def test_invalid_predictions_keep_denominator(self):
        invalid = row(); invalid.update(valid_points=0, farther_points=0, closer_points=0)
        result = audit.summarize([invalid, row(camera="c2")])
        self.assertEqual(result["prediction_coverage"], .5)
        self.assertEqual(result["reference_object_camera_observations"], 2)

    def test_duplicate_observation_rejected(self):
        with self.assertRaises(ValueError):
            audit.summarize([row(), row()])

    def test_missing_primary_cell_never_becomes_zero(self):
        result = audit.cell_summary([row()], benchmark.load_protocol())
        self.assertIsNone(result["conditional_equal_four_cell_mae_m"])

    def test_cell_structure_matches_frozen_protocol(self):
        protocol = benchmark.load_protocol()
        self.assertEqual(set(audit.cell_summary([], protocol)["cells"]), set(metrics.cell_names(protocol)))

    def test_counts_are_distinct_not_camera_sums(self):
        result = audit.summarize([row(), row(camera="c2")])
        self.assertEqual(result["reference_annotation_observations"], 1)
        self.assertEqual(result["reference_instance_tracks"], 1)
        self.assertEqual(result["reference_object_camera_observations"], 2)

    def fixture(self):
        sample = {"role": "dev", "width": 100, "height": 10, "camera_token": "camera",
            "sample_token": "sample", "recording_log": "log", "scene": "scene",
            "intrinsics": [[100., 0., 50.], [0., 100., 5.], [0., 0., 1.]],
            "objects": [{"class_name": "Vehicle", "source_class": "vehicle.car", "semantic_index": 17,
                "instance_token": "track", "annotation_token": "ann"}]}
        points = {"xy": np.array([[x, 1] for x in range(5)], np.int32),
            "z": np.full(5, 3., np.float32), "semantic": np.full(5, 17, np.uint8),
            "object_index": np.zeros(5, np.int32)}
        protocol = benchmark.load_protocol()
        points["cell_masks"] = metrics.make_masks(points, sample["objects"], protocol)
        return sample, points, protocol

    def test_point_rows_match_frozen_metric(self):
        sample, points, protocol = self.fixture()
        pred = points["z"] + 1
        rows = audit.observation_rows(sample, points, pred, "CAM_FRONT", protocol)
        own = audit.cell_summary(rows, protocol)["cells"]["Vehicle:2-4m"]
        frozen = metrics.Metrics(protocol); frozen.add(sample, points, pred)
        self.assertEqual(own["log_track_observation_macro"]["mae_m"],
            frozen.result()["cells"]["Vehicle:2-4m"]["conditional_log_track_observation_macro_mae_m"])

    def test_invalid_point_rows_preserve_coverage(self):
        sample, points, protocol = self.fixture()
        pred = points["z"].copy(); pred[:2] = [np.nan, 0]
        rows = audit.observation_rows(sample, points, pred, "CAM_FRONT", protocol)
        self.assertEqual(audit.cell_summary(rows, protocol)["cells"]["Vehicle:2-4m"]["prediction_coverage"], .6)

    def test_heldout_observation_rejected(self):
        sample, points, protocol = self.fixture(); sample["role"] = "heldout_eval"
        with self.assertRaises(ValueError):
            audit.observation_rows(sample, points, points["z"], "CAM_FRONT", protocol)

    def test_unknown_camera_rejected(self):
        sample, points, protocol = self.fixture()
        with self.assertRaises(ValueError):
            audit.observation_rows(sample, points, points["z"], "UNKNOWN", protocol)

    def camera_fixture(self, folder):
        sample, _, _ = self.fixture()
        tables = {"sensor": [{"token": "sensor", "modality": "camera", "channel": "CAM_FRONT"}],
            "calibrated_sensor": [{"token": "cal", "sensor_token": "sensor", "camera_intrinsic": sample["intrinsics"]}],
            "sample_data": [{"token": "camera", "is_key_frame": True, "sample_token": "sample",
                "width": 100, "height": 10, "calibrated_sensor_token": "cal"}]}
        for name, table in tables.items():
            common.immutable(folder/(name+".json"), common.encoded(table))
        return sample, {name: common.file_sha(folder/(name+".json")) for name in tables}

    def camera_join(self, sample, folder, hashes):
        fake = SimpleNamespace(items=lambda stream, *a, **kw: iter(json.load(stream)))
        with mock.patch.dict(sys.modules, {"ijson": fake}), mock.patch.object(audit, "METADATA_SHA", hashes):
            return audit.camera_channels([sample], folder)

    def test_camera_join_is_token_and_calibration_checked(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); sample, hashes = self.camera_fixture(folder)
            joined, _ = self.camera_join(sample, folder, hashes)
            self.assertEqual(joined, {"camera": "CAM_FRONT"})
            sample["intrinsics"][0][0] += 1
            with self.assertRaises(ValueError): self.camera_join(sample, folder, hashes)

    def test_missing_camera_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); sample, hashes = self.camera_fixture(folder)
            sample["camera_token"] = "missing"
            with self.assertRaises(ValueError): self.camera_join(sample, folder, hashes)

    def test_modified_metadata_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary); sample, hashes = self.camera_fixture(folder)
            (folder/"sensor.json").write_bytes(b"[]")
            with self.assertRaises(ValueError): self.camera_join(sample, folder, hashes)

    def test_unreviewed_result_rejected_before_open(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/"not_reviewed.zip"
            common.immutable(path, b"not a checkpoint or executable")
            with self.assertRaises(ValueError): audit.verify_result(path, {})


if __name__ == "__main__":
    unittest.main()
