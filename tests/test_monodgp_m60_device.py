import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.prepare_monodgp_m60_device_bundle import write_tensor, read_tensor, parity_row
from scripts.evaluate_monodgp_m60_device import timing_stats
from scripts import evaluate_monodgp_m60_device as review


class M60DeviceTests(unittest.TestCase):
    def test_tensor_bytes_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            value = np.arange(24, dtype=np.float32).reshape(1, 2, 3, 4)
            row = write_tensor(root, "sample/inputs/image.bin", value)
            self.assertEqual((root / row["file"]).read_bytes(), value.astype("<f4").tobytes())
            np.testing.assert_array_equal(read_tensor(root, row), value)

    def test_tensor_checksum_shape_path_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = write_tensor(root, "image.bin", np.ones((2, 3), dtype=np.float32))
            for changes in ({"sha256": "bad"}, {"shape": [1]}, {"file": "../image.bin"}):
                with self.assertRaises(RuntimeError):
                    read_tensor(root, {**row, **changes})

    def test_fixture_rejects_nonfinite_or_wrong_precision(self):
        with tempfile.TemporaryDirectory() as directory:
            for value in (np.array([np.nan], dtype=np.float32), np.array([1.], dtype=np.float16)):
                with self.assertRaises(ValueError):
                    write_tensor(Path(directory), "image.bin", value)

    def test_timing_percentiles_and_invalid_values(self):
        stats = timing_stats(list(range(1, 101)))
        self.assertEqual(stats["count"], 100)
        self.assertAlmostEqual(stats["p95_ms"], 95.05)
        for values in ([], [np.nan], [np.inf], [0], [-1], [[1]]):
            with self.assertRaises(ValueError):
                timing_stats(values)

    def test_changed_rank_does_not_silently_align(self):
        candidates = np.zeros((1, 50, 37), dtype=np.float32)
        ids = np.arange(50)
        changed = ids[::-1]
        rows = np.zeros((50, 14))
        with patch("scripts.prepare_monodgp_m60_device_bundle.compare_tensor_outputs", return_value={}), \
             patch("scripts.prepare_monodgp_m60_device_bundle.geometry_rows", return_value=(rows, ids)), \
             patch("scripts.prepare_monodgp_m60_device_bundle.compare_geometry") as compare:
            outputs = iter(((candidates, ids), (candidates, changed)))
            row = parity_row({}, {}, {}, lambda _: next(outputs), lambda *_: rows)
        compare.assert_not_called()
        self.assertFalse(row["same_identity_order"])
        self.assertFalse(row["passed"])

    def test_protocol_rejects_simulator_changed_model_or_partial_set(self):
        ids = list(review.reviewed_samples())
        manifest = {"complete": True, "mlpackage_tree_sha256": review.PACKAGE_SHA256,
                    "m59i_report_sha256": review.M59I_REPORT_SHA256, "policy": review.load_policy(),
                    "sample_ids": ids, "samples": [{"sample_id": value} for value in ids],
                    "warmups": 5, "timed_predictions": 100, "sustain_seconds": 60}
        run = {"manifest_sha256": "fixture", "compute_units": "ALL", "physical_device": True,
               "samples": manifest["samples"]}
        review.verify_protocol(manifest, run, "fixture")
        for changes in ({"physical_device": False}, {"manifest_sha256": "changed"},
                        {"compute_units": "CPU_ONLY"}, {"samples": run["samples"][:-1]},
                        {"samples": run["samples"] + [run["samples"][0]]}):
            with self.assertRaises(RuntimeError):
                review.verify_protocol(manifest, {**run, **changes}, "fixture")
        for changes in ({"mlpackage_tree_sha256": "changed"}, {"timed_predictions": 1},
                        {"policy": {}}, {"samples": []}):
            with self.assertRaises(RuntimeError):
                review.verify_protocol({**manifest, **changes}, run, "fixture")

    def test_app_is_separate_and_handles_output_strides(self):
        root = Path(__file__).resolve().parents[1]
        swift = (root / "ios/M60Benchmark/M60Runner.swift").read_text()
        project = (root / "ios/M60Benchmark/M60Benchmark.xcodeproj/project.pbxproj").read_text()
        self.assertIn("array.strides", swift)
        self.assertIn("loadUnaligned", swift)
        self.assertIn('configuration.computeUnits = .all', swift)
        self.assertIn('com.ali.MonoDGPM60', project)
        self.assertNotIn("V7", swift)
        self.assertNotIn("NSCameraUsageDescription", project)


if __name__ == "__main__":
    unittest.main()
