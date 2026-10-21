"""Range-screen boundaries, missing-data safeguards, and CPU-only runbook."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
import zipfile

from scripts import evaluate_parking_range as audit

ROOT = Path(__file__).resolve().parents[1]


def row(cls="Vehicle", depth=3.5, score=1., index=1, box=(0, 0, 100, 100), valid=True):
    return dict(class_name=cls, source_class=cls, bbox_2d=list(box), depth_m=depth,
                score=score, row=index, valid_geometry=valid, truncated=0., occluded=0)


def settings():
    contract = audit.load_config()
    for key in ("minimum_gt_objects_per_cell", "minimum_gt_frames_per_cell",
                "minimum_matched_distances_per_cell", "minimum_predictions_per_precision_cell"):
        contract["preliminary_kitti_screen"][key] = 1
    return contract


class ParkingRangeTests(unittest.TestCase):
    def measure(self, predictions, targets, contract=None):
        contract = contract or settings()
        objects, assigned = audit.observations({"000001": predictions}, {"000001": targets},
                                             contract["preliminary_kitti_screen"])
        result = audit.summarize(objects, assigned, contract)
        return objects, assigned, result

    def test_frozen_thresholds_and_no_historical_gate_changes(self):
        c = audit.load_config()
        self.assertEqual(c["thresholds"]["median_absolute_distance_error_m"], .1)
        self.assertEqual(c["thresholds"]["p95_absolute_distance_error_m"], .2)
        self.assertEqual(c["thresholds"]["absolute_mean_signed_distance_error_m"], .05)
        self.assertEqual(c["thresholds"]["precision"], .95)
        self.assertEqual(c["thresholds"]["recall"], .95)
        self.assertFalse(c["existing_accuracy_and_runtime_gates_modified"])

    def test_range_edges_and_empty_quantile_not_zero(self):
        self.assertIsNone(audit.band(1.999))
        self.assertEqual(audit.band(2), "2_3m")
        self.assertEqual(audit.band(3), "3_4m")
        self.assertEqual(audit.band(6), "5_6m")
        self.assertIsNone(audit.band(6.00001))
        self.assertIsNone(audit.band(float("nan")))
        self.assertIsNone(audit.quantile([], .95))
        self.assertAlmostEqual(audit.quantile([0., .2], .95), .19)

    def test_far_prediction_for_near_gt_is_not_dropped(self):
        objects, _, r = self.measure([row(depth=20)], [row(depth=3.5)])
        self.assertEqual(len(objects), 1)
        self.assertTrue(objects[0]["matched"])
        self.assertEqual(objects[0]["absolute_error_m"], 16.5)
        cell = next(c for c in r["cells"] if c["class_name"] == "Vehicle" and c["band"] == "3_4m")
        self.assertEqual(cell["screen_status"], "fail")

    def test_matching_all_depths_prevents_stealing_an_outside_target(self):
        p = row(depth=3.5)
        gt = [row(depth=20), row(depth=3.5, index=2, box=(300, 0, 400, 100))]
        objects, assigned, _ = self.measure([p], gt)
        self.assertFalse(objects[0]["matched"])
        self.assertFalse(assigned[0]["correct_near_range_match"])

    def test_duplicate_prediction_false_positive_and_wrong_class_miss(self):
        objects, assigned, r = self.measure([row(), row(score=.9, index=2)], [row()])
        cell = next(c for c in r["cells"] if c["class_name"] == "Vehicle" and c["band"] == "3_4m")
        self.assertEqual(cell["metrics"]["precision"], .5)
        self.assertEqual(sum(p["matched"] for p in assigned), 1)
        objects, _, _ = self.measure([row(cls="Pedestrian")], [row()])
        self.assertFalse(objects[0]["matched"])

    def test_invalid_distance_cannot_disappear_from_coverage(self):
        objects, _, r = self.measure([row(depth=-1, valid=False)], [row()])
        self.assertTrue(objects[0]["matched"])
        self.assertIsNone(objects[0]["absolute_error_m"])
        c = next(c for c in r["cells"] if c["ground_truth"])
        self.assertEqual(c["metrics"]["valid_distance_coverage"], 0)
        self.assertEqual(c["screen_status"], "fail")

    def test_unknown_range_false_positive_blocks_precision_pass(self):
        _, _, r = self.measure([row(), row(depth=-1, valid=False, index=2, box=(200, 0, 300, 100))], [row()])
        c = next(c for c in r["cells"] if c["ground_truth"])
        self.assertEqual(c["screen_status"], "inconclusive")
        self.assertIn("unassigned_range_false_positives", c["evidence_gaps"])

    def test_small_or_empty_cells_never_pass(self):
        _, _, r = self.measure([row()], [row()], audit.load_config())
        self.assertEqual(r["screen_status"], "inconclusive")
        self.assertTrue(all(c["screen_status"] == "inconclusive" for c in r["cells"]))
        self.assertFalse(r["parking_validation_qualified"])
        self.assertFalse(r["ultrasonic_sensor_qualified"])

    def test_all_eight_numeric_cells_can_pass_without_sensor_qualification(self):
        c = settings()
        gt, pp = {}, {}
        for i, cls in enumerate(c["classes"]):
            for j, depth in enumerate((2.5, 3.5, 4.5, 5.5)):
                sample = f"{i*4+j:06d}"
                gt[sample] = [row(cls=cls, depth=depth)]
                pp[sample] = [row(cls=cls, depth=depth)]
        objects, assigned = audit.observations(pp, gt, c["preliminary_kitti_screen"])
        result = audit.summarize(objects, assigned, c)
        self.assertEqual(result["screen_status"], "pass")
        self.assertEqual(len(result["cells"]), 8)
        self.assertFalse(result["parking_validation_qualified"])
        self.assertFalse(result["ultrasonic_sensor_qualified"])

    def test_score_cutoff_is_fixed_and_empty_predictions_are_misses(self):
        objects, _, result = self.measure([row(score=.0009)], [row()])
        self.assertFalse(objects[0]["matched"])
        self.assertEqual(next(c for c in result["cells"] if c["ground_truth"])["screen_status"], "fail")
        objects, _, _ = self.measure([row(score=.001)], [row()])
        self.assertTrue(objects[0]["matched"])

    def test_occlusion_and_truncation_do_not_remove_close_objects(self):
        gt = row(); gt.update(truncated=1., occluded=3)
        objects, _, result = self.measure([], [gt])
        self.assertEqual(len(objects), 1)
        c = next(c for c in result["cells"] if c["ground_truth"])
        self.assertEqual((c["missed"], c["truncated_gt"], c["occluded_gt"]), (1, 1, 1))

    def test_exact_boundaries_and_signed_bias(self):
        objects, _, r = self.measure([row(depth=3.55)], [row(depth=3.5)])
        c = next(c for c in r["cells"] if c["ground_truth"])
        self.assertTrue(c["observed_threshold_checks"]["absolute_mean_signed_distance_error_m"])
        self.assertEqual(c["screen_status"], "pass")
        _, _, r = self.measure([row(depth=3.551)], [row(depth=3.5)])
        c = next(c for c in r["cells"] if c["ground_truth"])
        self.assertEqual(c["screen_status"], "fail")

    def test_parser_taxonomy_and_strict_nonfinite_rejection(self):
        raw = b"Car 0 0 0 0 0 100 100 1.5 2 4 0 1.5 3.5 0 0.9\n"
        p = audit.parse_rows(raw, True)[0]
        self.assertEqual(p["class_name"], "Vehicle")
        self.assertEqual(p["depth_m"], 3.5)
        self.assertEqual(p["score"], .9)
        self.assertTrue(p["valid_geometry"])
        with self.assertRaises(ValueError): audit.parse_rows(raw.replace(b"3.5", b"nan"), True)
        with self.assertRaises(ValueError): audit.parse_rows(raw, False)

    def test_prediction_band_precision_does_not_penalize_one_metre_boundary(self):
        _, _, r = self.measure([row(depth=2.99)], [row(depth=3.01)])
        c = next(c for c in r["cells"] if c["class_name"] == "Vehicle" and c["band"] == "2_3m")
        self.assertEqual(c["metrics"]["precision"], 1)

    def test_visible_long_vehicle_with_center_behind_camera_is_retained(self):
        raw = b"Truck 1 0 0 0 0 100 100 4 2.5 16 0 1.5 -0.16 0\n"
        gt = audit.parse_rows(raw)
        self.assertEqual(len(gt), 1)
        objects, predictions, _ = self.measure([row()], gt)
        self.assertEqual(objects, [])
        self.assertTrue(predictions[0]["matched"])
        self.assertFalse(predictions[0]["correct_near_range_match"])

    def test_named_paths_and_wrong_model_provenance(self):
        with self.assertRaises(ValueError): audit.named_paths(["A2=x", "A2=y"])
        with self.assertRaises(ValueError): audit.named_paths(["teacher=x"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(dict(checkpoint_sha256="wrong")))
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                audit.provenance(path, "A2", settings())

    def test_cli_end_to_end_and_immutable_resume_without_ml_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "gt").mkdir(); (root / "pred").mkdir()
            raw = b"Car 0 0 0 0 0 100 100 1.5 2 4 0 1.5 3.5 0"
            (root / "gt/000001.txt").write_bytes(raw + b"\n")
            (root / "pred/000001.txt").write_bytes(raw + b" 0.9\n")
            (root / "val.txt").write_bytes(b"000001\n")
            c = settings(); c["chen_val"] = dict(images=1, split_file_sha256=audit.digest(b"000001\n"))
            (root / "contract.json").write_text(json.dumps(c))
            command = [sys.executable, "-S", str(ROOT / "scripts/evaluate_parking_range.py"), "screen",
                       "--labels", str(root / "gt"), "--split", str(root / "val.txt"),
                       "--predictions", "A2_Lite=" + str(root / "pred"),
                       "--config", str(root / "contract.json"), "--output", str(root / "result")]
            for _ in range(2):
                p = subprocess.run(command, capture_output=True, text=True)
                self.assertEqual(p.returncode, 0, p.stderr)
            with zipfile.ZipFile(root / "result/parking_range_results.zip") as archive:
                self.assertIn("A2_Lite/objects.csv", archive.namelist())
                self.assertTrue(all(not n.endswith((".png", ".onnx", ".pth")) for n in archive.namelist()))
            before = (root / "result/parking_range_screen.json").read_bytes()
            (root / "pred/000001.txt").write_bytes(raw.replace(b"3.5", b"3.8") + b" 0.9\n")
            p = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(p.returncode, 0)
            self.assertEqual((root / "result/parking_range_screen.json").read_bytes(), before)
            (root / "pred/000001.txt").unlink()
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)

    def test_notebook_compiles_and_needs_no_gpu_or_dependency_install(self):
        path = ROOT / "notebooks/MonoDETR_Parking_2_to_6m_Screen_Colab.ipynb"
        notebook = json.loads(path.read_text())
        code = []
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                compile(source, f"{path}:{i}", "exec")
                self.assertEqual(cell["outputs"], [])
                self.assertIsNone(cell["execution_count"])
                code.append(source)
        text = "\n".join(code)
        self.assertIn("max_workers=4", text)
        self.assertIn("--ff-only", text)
        self.assertIn("evaluate_parking_range.py", text)
        self.assertNotIn("pip", text)
        self.assertNotIn("cuda", text.lower())
        self.assertNotIn("git reset", text)


if __name__ == "__main__":
    unittest.main()
