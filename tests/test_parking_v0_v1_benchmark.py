"""Synthetic CPU-only benchmark preparation/scoring; no downloads or models."""
import ast
import copy
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock
import zipfile

import numpy as np

from scripts import prepare_parking_v0_v1_benchmark as prepare
from scripts import parking_v0_v1_metrics as metrics
from scripts import parking_edge_common as common
from scripts import audit_nuscenes_coverage_0_6m as coverage

# File-based loading avoids a collision with third-party packages named tests.
spec = importlib.util.spec_from_file_location("coverage_fixture", prepare.ROOT / "tests/test_nuscenes_full_coverage.py")
fixture = importlib.util.module_from_spec(spec); spec.loader.exec_module(fixture)


def raw_fixture(root):
    c, roles, source_local, database, part, staged, archive, receipt, _ = fixture.tiny_dataset(root)
    roles["scenes"] = {"scene-fixture": {"role": "dev", "log_token": "logdev"},
                        "scene-final": {"role": "heldout_eval", "log_token": "logfinal"}}
    frames = copy.deepcopy(staged["keyframes"]); frames[1]["scene"] = "scene-final"
    protocol = prepare.load_protocol(); protocol.update(maximum_images_per_role=16, images_per_shard=2)
    rows = []
    for frame in frames:
        for row in coverage.audit_frame(frame, part, source_local / "lidarseg", database, c, roles)["camera_rows"]:
            rows.append({"camera_token": row["camera_token"], "sample_token": row["sample_token"],
                "scene": row["scene"], "role": row["role"], "recording_log": row["recording_log"],
                "source_part": 1, "coverage_row_sha256": common.identity(row), "coverage_row": row,
                "supported_counts": prepare.supported_counts(row, protocol)})
    chosen = sum((prepare.select_views([r for r in rows if r["role"] == role], protocol) for role in protocol["roles"]), [])
    selection = coverage.signed({"samples": chosen, "selected_pool": prepare.selection_inventory(chosen, protocol),
        "available_pool": prepare.selection_inventory(rows, protocol)})
    prepare.validate_selection(selection, protocol, roles)
    dataset = root / "drive"; (dataset / "archives").mkdir(parents=True)
    shutil.copy2(archive, dataset / "archives" / receipt["file"])
    local = root / "benchmark_local"; shutil.copytree(source_local / "lidarseg", local / "lidarseg")
    return c, roles, protocol, selection, dataset, local, database, receipt


def metric_fixture(camera="camera", log="log", prefix="a", points_per_cell=(5, 5, 5, 5)):
    protocol = prepare.load_protocol(); objects = []; zs = []; semantics = []; owners = []
    for index, (cls, z, sem, n) in enumerate(zip(["Vehicle", "Vehicle", "Pedestrian", "Pedestrian"], [3, 5, 3, 5], [17, 17, 2, 2], points_per_cell)):
        objects.append({"annotation_token": f"{prefix}_{index}", "instance_token": f"track{index}",
                        "class_name": cls, "semantic_index": sem})
        zs.extend([z] * n); semantics.extend([sem] * n); owners.extend([index] * n)
    n = len(zs)
    points = {"xy": np.column_stack((np.arange(n), np.zeros(n))).astype(np.int32),
        "z": np.asarray(zs, dtype=np.float32), "semantic": np.asarray(semantics, dtype=np.uint8),
        "object_index": np.asarray(owners, dtype=np.int32)}
    points["cell_masks"] = metrics.make_masks(points, objects, protocol)
    sample = {"camera_token": camera, "recording_log": log, "width": max(n + 1, 100), "height": 10, "objects": objects}
    return protocol, sample, points


class SelectionTests(unittest.TestCase):
    def test_protocol_has_no_training_or_accuracy_gate(self):
        p = prepare.load_protocol()
        self.assertIsNone(p["accuracy_threshold_m"])
        for key in ["accuracy_pass_fail_enabled", "training_authorized", "model_inference_authorized", "phone_app_changed"]:
            self.assertFalse(p[key])
        self.assertEqual(p["roles"], ["dev", "heldout_eval"])
        self.assertEqual(len(p["primary_cells"]), 4)

    def test_rare_views_retained_and_selection_deterministic(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, p, selection, *_ = raw_fixture(Path(folder))
            dev = [r for r in selection["samples"] if r["role"] == "dev"]
            self.assertEqual(len(dev), 6)  # All six mandatory sub-metre views survive.
            self.assertEqual(len({r["sample_token"] for r in dev}), 1)
            self.assertEqual(prepare.select_views(dev, p), prepare.select_views(list(reversed(dev)), p))
            p["maximum_images_per_role"] = 5
            with self.assertRaises(ValueError): prepare.select_views(dev, p)

    def test_train_and_previously_inspected_logs_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _, roles, p, selection, *_ = raw_fixture(Path(folder))
            bad = copy.deepcopy(selection); bad["samples"][0]["role"] = "train"
            with self.assertRaises(ValueError): prepare.validate_selection(bad, p, roles)
            roles["previously_inspected_evaluation_logs"] = ["logfinal"]
            with self.assertRaises(ValueError): prepare.validate_selection(selection, p, roles)

    def test_changed_counts_and_repeated_cameras_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            _, roles, p, selection, *_ = raw_fixture(Path(folder))
            bad = copy.deepcopy(selection); bad["samples"][0]["supported_counts"]["Vehicle:0-6m"] += 1
            with self.assertRaises(ValueError): prepare.validate_selection(bad, p, roles)
            bad = copy.deepcopy(selection); bad["samples"].append(bad["samples"][0])
            with self.assertRaises(ValueError): prepare.validate_selection(bad, p, roles)

    def test_fill_spreads_logs_and_does_not_add_extra_views(self):
        p = prepare.load_protocol(); p["maximum_images_per_role"] = 6
        rows = []
        for index in range(20):
            counts = {cell: 0 for cell in metrics.cell_names(p)}
            for cell in p["selection_cells"]:
                if cell.endswith(("2-4m", "4-6m")): counts[cell] = 5
            rows.append({"camera_token": f"c{index}", "sample_token": f"frame{index//2}",
                "role": "dev", "recording_log": f"log{index%2}", "source_part": 1, "supported_counts": counts})
        chosen = prepare.select_views(rows, p)
        self.assertEqual(len(chosen), 6)
        self.assertEqual(len({r["sample_token"] for r in chosen}), 6)
        self.assertEqual(sum(r["recording_log"] == "log0" for r in chosen), 3)

    def test_resume_plan_keeps_frozen_selection_without_reading_chunks(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, roles, p, selection, *_ = raw_fixture(root)
            identity = {"manifest_sha256": "fixture"}
            selection.pop("manifest_sha256")
            selection["benchmark_identity_sha256"] = "fixture"
            selection = coverage.signed(selection)
            output = root / "output"
            common.immutable(output / "selection_manifest.json", common.encoded(selection))
            with mock.patch.object(prepare, "candidates_from_chunks", side_effect=AssertionError("do not rescan")):
                self.assertEqual(prepare.plan(root, output, {}, roles, {}, p, identity), selection)


class MetricTests(unittest.TestCase):
    def test_cell_edges_and_minimum_support(self):
        p, sample, points = metric_fixture(points_per_cell=(4, 5, 5, 5))
        names = metrics.cell_names(p)
        self.assertFalse(points["cell_masks"][names.index("Vehicle:2-4m")].any())
        metrics.validate_gt(points, sample, p)
        points["cell_masks"][0, 0] = True
        with self.assertRaises(ValueError): metrics.validate_gt(points, sample, p)

    def test_equal_four_cells_not_large_vehicle_point_weight(self):
        p, sample, points = metric_fixture(points_per_cell=(100, 5, 5, 5))
        error = np.where(points["semantic"] == 17, 1., 3.)
        metric = metrics.Metrics(p); metric.add(sample, points, points["z"] + error)
        result = metric.result()
        self.assertEqual(result["conditional_equal_four_cell_mae_m"], 2.)
        self.assertEqual(result["cells"]["Vehicle:2-4m"]["prediction_coverage"], 1.)
        self.assertFalse(result["promotion_authorized"])

    def test_logs_and_tracks_are_equal_weight_not_frame_counts(self):
        p = prepare.load_protocol(); metric = metrics.Metrics(p)
        for i in range(10):
            _, sample, points = metric_fixture(f"long{i}", "logA", f"long{i}")
            metric.add(sample, points, points["z"].copy())
        _, sample, points = metric_fixture("short", "logA", "short")
        for obj in sample["objects"]: obj["instance_token"] += "_short"
        metric.add(sample, points, points["z"] + 4.)
        _, sample, points = metric_fixture("otherlog", "logB", "other")
        metric.add(sample, points, points["z"] + 2.)
        self.assertEqual(metric.result()["conditional_equal_four_cell_mae_m"], 2.)

    def test_invalid_predictions_keep_gt_denominator(self):
        p, sample, points = metric_fixture()
        prediction = points["z"].copy(); prediction[:5] = np.nan
        metric = metrics.Metrics(p); metric.add(sample, points, prediction)
        result = metric.result(); cell = result["cells"]["Vehicle:2-4m"]
        self.assertEqual(cell["reference_points"], 5); self.assertEqual(cell["prediction_coverage"], 0.)
        self.assertIsNone(cell["conditional_point_mae_m"])
        self.assertIsNone(result["conditional_equal_four_cell_mae_m"])
        self.assertEqual(result["images_with_invalid_predictions"], 1)

    def test_no_gt_band_clamping_and_no_silent_missing_images(self):
        p, sample, points = metric_fixture()
        prediction = points["z"] + 100.
        metric = metrics.Metrics(p); metric.add(sample, points, prediction)
        self.assertEqual(metric.result()["conditional_equal_four_cell_mae_m"], 100.)
        with self.assertRaises(ValueError): metrics.paired_metrics([sample], {"camera": points}, {}, {"camera": prediction}, p)
        with self.assertRaises(ValueError): metric.add(sample, points, prediction)

    def test_paired_common_mask_does_not_hide_missing_predictions(self):
        p, sample, points = metric_fixture(); p0 = points["z"] + 1.; p1 = points["z"] + .5
        p1[0] = np.nan
        result = metrics.paired_metrics([sample], {"camera": points}, {"camera": p0}, {"camera": p1}, p)
        self.assertEqual(result["v0_own_valid"]["cells"]["Vehicle:2-4m"]["prediction_coverage"], 1.)
        self.assertEqual(result["v1_own_valid"]["cells"]["Vehicle:2-4m"]["prediction_coverage"], .8)
        self.assertEqual(result["v0_common_valid"]["cells"]["Vehicle:2-4m"]["prediction_coverage"], .8)
        self.assertEqual(result["v0_common_valid"]["conditional_equal_four_cell_mae_m"], 1.)
        self.assertEqual(result["v1_common_valid"]["conditional_equal_four_cell_mae_m"], .5)

    def test_deterministic_npz_and_no_pickle(self):
        p, sample, points = metric_fixture()
        raw = prepare.encode_npz(points)
        self.assertEqual(raw, prepare.encode_npz(points))
        with np.load(io.BytesIO(raw), allow_pickle=False) as saved:
            actual = {k: saved[k] for k in saved.files}
        metrics.validate_gt(actual, sample, p)


class PreparationTests(unittest.TestCase):
    def test_synthetic_archives_to_frozen_shards_resume_and_bundle(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, roles, p, selection, dataset, local, db, receipt = raw_fixture(root)
            output = root / "output"; identity = {"manifest_sha256": "fixture"}
            receipts = {"v1.0-trainval01_keyframes.tgz": receipt}
            result = prepare.prepare_part(1, selection, dataset, output, local, db, receipts, c, p, identity, 2)
            self.assertEqual(len(result), 4)
            self.assertFalse((local / "parts/part01").exists())
            with mock.patch.object(prepare.data, "copy_archive", side_effect=AssertionError("completed part must not restage")):
                self.assertEqual(result, prepare.prepare_part(1, selection, dataset, output, local, db, receipts, c, p, identity, 2))
            report = prepare.finalize(selection, output, p, identity)
            self.assertEqual(report["images"], 7); self.assertFalse(report["model_artifacts_bound"])
            common.immutable(output / "benchmark_identity.json", common.encoded(identity))
            common.immutable(output / "selection_manifest.json", common.encoded(selection))
            prepare.bundle(output, p); first = common.file_sha(output / "parking_v0_v1_benchmark_results.zip")
            prepare.bundle(output, p)
            self.assertEqual(first, common.file_sha(output / "parking_v0_v1_benchmark_results.zip"))
            with zipfile.ZipFile(output / "parking_v0_v1_benchmark_results.zip") as archive:
                self.assertFalse(any(n.endswith((".jpg", ".npz")) for n in archive.namelist()))
            shard = output / result[0]["file"]; shard.write_bytes(b"changed")
            with self.assertRaises(ValueError): prepare.finalize(selection, output, p, identity)

    def test_interrupted_preparation_reuses_first_durable_shard(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, _, p, selection, dataset, local, db, receipt = raw_fixture(root)
            output = root / "output"; identity = {"manifest_sha256": "fixture"}
            receipts = {"v1.0-trainval01_keyframes.tgz": receipt}; original = prepare.reference_for_view; count = 0
            def interrupt(*args):
                nonlocal count
                count += 1
                if count == 3: raise OSError("simulated interruption")
                return original(*args)
            with mock.patch.object(prepare, "reference_for_view", side_effect=interrupt):
                with self.assertRaises(OSError): prepare.prepare_part(1, selection, dataset, output, local, db, receipts, c, p, identity, 1)
            first = next((output / "shards").rglob("*.zip")); before = first.read_bytes()
            with mock.patch.object(prepare, "reference_for_view", wraps=original) as compute:
                prepare.prepare_part(1, selection, dataset, output, local, db, receipts, c, p, identity, 1)
                self.assertEqual(compute.call_count, 5)
            self.assertEqual(first.read_bytes(), before)

    def test_changed_selected_reprojection_fails_closed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            c, _, p, selection, dataset, local, db, receipt = raw_fixture(root)
            selection["samples"][0]["coverage_row"]["point_counts"]["Vehicle:0-6m"] += 1
            with self.assertRaises(ValueError):
                prepare.prepare_part(1, selection, dataset, root / "output", local, db,
                    {"v1.0-trainval01_keyframes.tgz": receipt}, c, p, {"manifest_sha256": "fixture"}, 1)

    def test_missing_shards_cannot_finalize(self):
        with tempfile.TemporaryDirectory() as folder:
            _, _, p, selection, *_ = raw_fixture(Path(folder))
            with self.assertRaises(FileNotFoundError): prepare.finalize(selection, Path(folder) / "output", p, {"manifest_sha256": "fixture"})

    def test_notebook_cells_have_no_gpu_or_model_dependency(self):
        notebook = json.loads((prepare.ROOT / "notebooks/Parking_V0_V1_Benchmark_CPU_Colab.ipynb").read_text())
        sources = []
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                source = "".join(cell["source"]); ast.parse(source); sources.append(source)
                self.assertEqual(cell["outputs"], []); self.assertIsNone(cell["execution_count"])
        source = "\n".join(sources)
        self.assertIn("CUDA_VISIBLE_DEVICES", source); self.assertIn("--no-deps", source)
        for stage in ["plan", "prepare", "bundle"]: self.assertIn(f"stage('{stage}')", source)
        self.assertNotIn("git reset", source); self.assertNotIn("import torch", source)
        self.assertLess(source.index("def run("), source.index("run(['git'"))


if __name__ == "__main__":
    unittest.main()
