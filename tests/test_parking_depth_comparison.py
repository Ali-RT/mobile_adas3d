"""Numerical and provenance guards for the inference-only two-depth screen."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import compare_parking_depth_models as compare


class ParkingDepthComparisonTests(unittest.TestCase):
    def test_official_checkpoint_is_metric_and_immutable(self):
        self.assertEqual(compare.HF_REVISION, "fd2c22027eaf20374204f14099b8341e1925ad39")
        self.assertEqual(compare.FILES["model.safetensors"][1], "sha256")
        self.assertNotIn(".pth", compare.FILES)
        raw = b'{"depth_estimation_type":"metric","max_depth":80}'
        git_hash = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        with mock.patch.dict(compare.FILES, {"config.json": (len(raw), "git-sha1", git_hash)}, clear=True):
            compare.verify_hf_file("config.json", raw)
            with self.assertRaises(ValueError): compare.verify_hf_file("config.json", raw + b" ")
            with tempfile.TemporaryDirectory() as d:
                path = Path(d) / "config.json"; path.write_bytes(raw)
                self.assertEqual(compare.verify_weights(d)["config.json"], hashlib.sha256(raw).hexdigest())
                path.unlink(); path.symlink_to(Path(d) / "missing")
                with self.assertRaises(ValueError): compare.verify_weights(d)

    def test_wrong_relative_head_rejected_even_with_matching_file_hash(self):
        raw = b'{"depth_estimation_type":"relative","max_depth":80}'
        spec = (len(raw), "sha256", hashlib.sha256(raw).hexdigest())
        with mock.patch.dict(compare.FILES, {"config.json": spec}, clear=True), tempfile.TemporaryDirectory() as d:
            (Path(d) / "config.json").write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "metric OUTDOOR"): compare.verify_weights(d)

    def test_shared_detector_and_identical_fixed_roi(self):
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = [200, 200, 400, 400, .9, 2]
        rows = compare.fixed_rows(raw, 640, 640)
        self.assertEqual(len(rows), 1); self.assertEqual(rows[0]["class_name"], "Vehicle")
        a = np.full((640, 640), 2, dtype=np.float32)
        b = np.full((640, 640), 3, dtype=np.float32)
        paired = compare.compare_rows(rows, a, b)
        self.assertEqual(paired[0]["yolo26n_depth"]["roi_xyxy"], [250, 250, 350, 350])
        self.assertEqual(paired[0]["yolo26n_depth"]["roi_xyxy"], paired[0]["depth_anything_v2_metric_outdoor_small"]["roi_xyxy"])
        self.assertEqual(paired[0]["anything_minus_yolo_m"], 1)
        self.assertTrue(paired[0]["difference_is_not_error"])
        self.assertNotIn("distance_m", rows[0])
        self.assertTrue((a == 2).all()); self.assertTrue((b == 3).all())

    def test_invalid_predictions_do_not_silently_change_roi(self):
        raw = np.zeros((1, 300, 6), dtype=np.float32)
        raw[0, 0] = [200, 200, 200, 400, .9, 2]
        with self.assertRaises(ValueError): compare.fixed_rows(raw, 640, 640)
        raw[0, 0, 0] = np.nan
        with self.assertRaises(ValueError): compare.fixed_rows(raw, 640, 640)
        with self.assertRaises(ValueError): compare.compare_rows([], np.ones((10, 10)), np.ones((20, 20)))

    def test_official_restoration_preserves_positive_constant(self):
        import torch
        a = compare.restore_anything(torch.ones((1, 14, 28)) * 3, 100, 200)
        self.assertEqual(a.shape, (100, 200)); self.assertEqual(a.dtype, np.float32)
        np.testing.assert_allclose(a, 3, atol=1e-5)
        with self.assertRaises(ValueError): compare.restore_anything(torch.full((1, 14, 28), float("nan")), 100, 200)

    def test_summary_never_calls_disagreement_accuracy(self):
        r = compare.summarize([{"detections": [{"class_name": "Vehicle", "anything_minus_yolo_m": -1}]}])
        self.assertEqual(r["Vehicle"]["median_absolute_model_disagreement_m"], 1)
        self.assertEqual(r["Pedestrian"]["detected_boxes"], 0)
        self.assertIsNone(r["Pedestrian"]["median_anything_minus_yolo_m"])
        self.assertIn("NOT distance error", r["Vehicle"]["interpretation"])

    def test_synthetic_phone_capture_is_not_field_data(self):
        with mock.patch("scripts.audit_parking_distance_samples.analyze", return_value={"synthetic": True}):
            args = type("Args", (), {"inputs": None, "clean_run": [Path("unused")], "limit": 3})()
            with self.assertRaisesRegex(ValueError, "Synthetic"):
                list(compare.input_items(args, None))

    def test_preview_is_labelled_and_does_not_change_inputs(self):
        from PIL import Image
        rgb = np.zeros((100, 200, 3), dtype=np.uint8)
        a = np.ones((100, 200), dtype=np.float32) * 2
        b = a * 2
        raw = compare.preview(rgb, a, b, [], "example")
        with Image.open(io.BytesIO(raw)) as im:
            self.assertEqual(im.size, (400, 300))
        self.assertFalse(rgb.any()); self.assertTrue((a == 2).all())


if __name__ == "__main__":
    unittest.main()
