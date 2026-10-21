"""No model download or inference is performed by these unit tests."""
import copy
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
from types import ModuleType
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import parking_edge_common as common
import screen_parking_edge_candidates as screen


def label(cls="Car", z=3.5, box="0 0 100 100"):
    return f"{cls} 0 0 0 {box} 1.5 2 4 0 1.5 {z} 0\n".encode()


def detection(cls=2, box=(0., 0., 100., 100.), score=.9):
    return dict(class_name="Pedestrian" if cls == 0 else "Vehicle", coco_class_id=cls,
                bbox=list(box), score=score)


class ParkingEdgeTests(unittest.TestCase):
    def test_metric_flags_never_qualify_a_model(self):
        f = screen.flags()
        self.assertEqual(f["optimizer_steps"], 0)
        self.assertTrue(all(not v for v in f.values()))
        c = common.load_config()
        self.assertFalse(c["candidates"]["urban_nano"]["metric_range_available"])
        self.assertNotEqual(c["candidates"]["urban_nano"]["ultralytics"], c["candidates"]["yolo_depth"]["ultralytics"])

    def test_pinned_download_hash_and_conflict(self):
        with tempfile.TemporaryDirectory() as d:
            folder = Path(d); raw = b"trusted-checkpoint"; spec = dict(file="weights.pt", url="https://example.invalid", sha256=common.sha(raw))
            with patch.object(screen.urllib.request, "urlopen", return_value=io.BytesIO(raw)):
                path = screen.download(spec, folder)
            self.assertEqual(path.read_bytes(), raw)
            with patch.object(screen.urllib.request, "urlopen", side_effect=AssertionError("No second download")):
                screen.download(spec, folder)
            path.write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError, "identity"):
                screen.download(spec, folder)
            path.unlink()
            with patch.object(screen.urllib.request, "urlopen", return_value=io.BytesIO(b"bad")):
                with self.assertRaisesRegex(RuntimeError, "hash changed"):
                    screen.download(spec, folder)
            self.assertFalse(path.exists())

    def test_selection_is_fixed_deduplicated_and_label_only(self):
        c = common.load_config(); ids = [f"{i:06d}" for i in range(80)]
        labels = {s: label("Car" if i % 2 else "Pedestrian", 2.5 + (i // 2) % 4) for i, s in enumerate(ids)}
        a, counts = common.select(ids, labels, c)
        b, _ = common.select(list(reversed(ids)), labels, c)
        self.assertEqual(a, b); self.assertEqual(len(a), 64); self.assertEqual(len(set(a)), 64)
        self.assertTrue(all(count == 10 for count in counts.values()))

    def test_band_boundaries_and_nonpositive_center(self):
        self.assertEqual([common.band(z) for z in (1.99, 2., 3., 4., 5., 6., 6.01)], [None, 2, 3, 4, 5, 5, None])
        self.assertEqual(common.targets(label("Truck", -.1))[0]["camera_center_z_m"], -.1)
        with self.assertRaises(ValueError): common.targets(label(z=float("nan")))

    def test_depth_must_be_numeric_original_resolution_not_colored_image(self):
        class Tensor:
            def __init__(self, a): self.a = a
            def cpu(self): return self
            def numpy(self): return self.a
        result = lambda a: SimpleNamespace(depth=SimpleNamespace(data=Tensor(a)))
        np.testing.assert_array_equal(screen.depth_array(result(np.full((8, 10), 3.)), (8, 10)), np.full((8, 10), 3.))
        for a in (np.ones((8, 10, 3)), np.zeros((8, 10)), np.full((8, 10), np.nan)):
            with self.assertRaises(RuntimeError): screen.depth_array(result(a), (8, 10))
        with self.assertRaisesRegex(RuntimeError, "numeric"):
            screen.depth_array(SimpleNamespace(depth=None), (8, 10))

    def test_roi_fixed_median_no_global_alignment_or_minimum(self):
        a = np.full((20, 20), 100., dtype=np.float32); a[5:15, 5:15] = 3.7
        r = screen.roi_depth(a, [0, 0, 20, 20])
        self.assertEqual(r["roi_xyxy"], [5, 5, 15, 15]); self.assertAlmostEqual(r["distance_m"], 3.7, places=5)
        a[5:15, 5:15] = np.nan
        self.assertIsNone(screen.roi_depth(a, [0, 0, 20, 20])["distance_m"])
        self.assertIsNone(screen.roi_depth(a, [50, 50, 60, 60])["distance_m"])

    def test_detection_parity_cannot_hide_class_count_or_corner_change(self):
        limits = common.load_config()["parity"]; a = [detection(), detection(0, (100, 0, 200, 100))]
        self.assertTrue(screen.detection_parity(a, list(reversed(a)), limits)["passed"])
        b = copy.deepcopy(a); b[0]["coco_class_id"] = 7
        self.assertFalse(screen.detection_parity(a, b, limits)["passed"])
        self.assertFalse(screen.detection_parity(a, a[:1], limits)["passed"])
        b = copy.deepcopy(a); b[0]["bbox"][0] += 1
        self.assertFalse(screen.detection_parity(a, b, limits)["passed"])
        a = [detection()]; a[0]["projected_corners_xy_confidence"] = [[float(i), float(i), 1.] for i in range(8)]
        b = copy.deepcopy(a); b[0]["projected_corners_xy_confidence"][3][1] += 1
        self.assertFalse(screen.detection_parity(a, b, limits)["passed"])

    def test_depth_parity_finite_and_shape_checks(self):
        limits = common.load_config()["parity"]; a = np.full((10, 10), 3.)
        self.assertTrue(screen.depth_parity(a, a+.0001, limits)["passed"])
        for b in (a + .1, a[:1], a * np.nan):
            self.assertFalse(screen.depth_parity(a, b, limits)["passed"])

    def test_near_recall_uses_gt_depth_not_predicted_surface_depth(self):
        rows = [detection()]; rows[0]["distance_m"] = 90.
        values = screen.recall_cells(rows, label())
        self.assertTrue(values[0]["matched"]); self.assertEqual(values[0]["band"], 3)
        self.assertNotIn("distance_error_m", values[0])
        # All depths participate in matching before the 2–6 m filter.
        values = screen.recall_cells(rows, label(z=20.) + label(z=3.5))
        self.assertEqual(len(values), 1); self.assertFalse(values[0]["matched"])

    def test_immutable_conflicts_and_symlinks_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "record.json"; common.immutable(p, b"original")
            common.immutable(p, b"original")
            with self.assertRaises(RuntimeError): common.immutable(p, b"changed")
            link = Path(d) / "link"; link.symlink_to(p)
            with self.assertRaises(RuntimeError): common.immutable(link, b"original")
            self.assertEqual(p.read_bytes(), b"original")

    def test_archive_prepare_restore_reuse_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); c = common.load_config(); ids = [f"{i:06d}" for i in range(64)]
            raw = ("\n".join(ids) + "\n").encode(); (root / "val.txt").write_bytes(raw)
            c["split_images"] = 64; c["split_sha256"] = common.sha(raw)
            (root / "config.json").write_bytes(common.encoded(c))
            for i, s in enumerate(ids):
                for kind, suffix, value in (("label_2", ".txt", label(z=2.5 + i % 4)), ("image_2", ".png", ("image"+s).encode())):
                    p = root / "source/training" / kind / (s + suffix); p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(value)
            a = SimpleNamespace(config=root / "config.json", split=root / "val.txt", cache=root / "cache",
                                local_data=root / "local", dataset_root=root / "source", workers=4)
            first = common.prepare(a)
            historical = copy.deepcopy(first)
            historical["implementation_sha256"] = "older_io_helper"
            historical["manifest_sha256"] = common.identity({k:v for k,v in historical.items() if k != "manifest_sha256"})
            common.verify_manifest(historical, c)  # byte verification, not obsolete runtime restoration
            shutil_source = root / "source/training/label_2/000000.txt"
            shutil_source.unlink()  # Cached restore must not touch the 3769 source files again.
            a.local_data = root / "another_local"
            second = common.prepare(a)
            self.assertEqual(first, second)
            self.assertEqual((a.local_data / "training/image_2/000001.png").read_bytes(), b"image000001")
            (a.local_data / "training/image_2/000001.png").write_bytes(b"conflict")
            with self.assertRaises(RuntimeError): common.prepare(a)
            (a.cache / "inputs64.zip").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "archive changed"): common.prepare(a)

    def test_full_orchestration_with_fake_models_resume_and_bundle(self):
        """Exercise all 64 iterations and artifact plumbing, not model accuracy."""
        from PIL import Image
        class Tensor:
            def __init__(self, value): self.value = value
            def cpu(self): return self
            def numpy(self): return self.value
        boxes = SimpleNamespace(xyxy=Tensor(np.array([[0., 0., 4., 4.]], dtype=np.float32)),
                                conf=Tensor(np.array([.9], dtype=np.float32)), cls=Tensor(np.array([2.])))
        class Model:
            calls = 0
            def __init__(self, path, task):
                self.task = task
                self.predictor = SimpleNamespace(preprocess=lambda _: Tensor(np.zeros((1,3,4,4), dtype=np.float32)))
            def predict(self, **kwargs):
                Model.calls += 1
                return [SimpleNamespace(boxes=boxes, depth=SimpleNamespace(data=Tensor(np.full((4,4), 3.5, dtype=np.float32))))]
        class Session:
            def run(self, *args): return [np.zeros((1,1,4,4), dtype=np.float32)]
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); cfg = common.load_config(); data = root / "data"; hashes = {}; ids = []
            buffer = io.BytesIO(); Image.new("RGB", (4,4)).save(buffer, format="PNG")
            for i in range(64):
                sample = f"{i:06d}"; ids.append(sample)
                for kind, suffix, raw in (("image_2", ".png", buffer.getvalue()), ("label_2", ".txt", label(box="0 0 4 4"))):
                    name = f"training/{kind}/{sample}{suffix}"; common.immutable(data / name, raw); hashes[name] = common.sha(raw)
            for spec in cfg["candidates"]["yolo_depth"]["components"].values():
                spec["sha256"] = common.sha(b"fake_checkpoint")
                common.immutable(root / "weights" / spec["file"], b"fake_checkpoint")
            m = dict(ids=ids, files=hashes, config_sha256=common.identity(cfg),
                     implementation_sha256=common.file_sha(Path(common.__file__)))
            m["manifest_sha256"] = common.identity(m); common.immutable(data / "input_manifest.json", common.encoded(m))
            args = SimpleNamespace(config=root / "unused", candidate="yolo_depth", local_data=data,
                                   weights=root / "weights", source=None, output=root / "result")
            ultra = ModuleType("ultralytics"); ultra.YOLO = Model
            torch = ModuleType("torch"); torch.set_num_threads = lambda _: None
            cv2 = ModuleType("cv2"); cv2.imread = lambda _: np.zeros((4,4,3), dtype=np.uint8)
            def export(model, path, spec, c, run_id):
                common.immutable(path, b"fake_onnx")
                return Session(), dict(input_shape=[1,3,4,4], input_name="images", sha256=common.file_sha(path))
            patches = {"ultralytics": ultra, "torch": torch, "cv2": cv2}
            with patch.dict(sys.modules, patches), patch.dict(screen.os.environ, {}, clear=False), \
                 patch.object(common, "load_config", return_value=cfg), \
                 patch.object(screen.importlib.metadata, "version", return_value="8.4.183"), \
                 patch.object(screen, "runtime", return_value={"fake_test_runtime": True}), \
                 patch.object(screen, "print", create=True), \
                 patch.object(screen, "export_component", side_effect=export):
                screen.screen(args)
                report = json.loads((args.output / "candidate_screen.json").read_text())
                self.assertEqual(report["frames"], 64); self.assertTrue(report["technical_export_screen_passed"])
                self.assertFalse(report["independent_surface_distance_evaluated"])
                before = (args.output / "candidate_screen.json").read_bytes(); Model.calls = 0
                screen.screen(args)
                self.assertEqual(Model.calls, 8)  # warm-up only; all 64 records reused
                self.assertEqual((args.output / "candidate_screen.json").read_bytes(), before)
            screen.bundle(args)
            with zipfile.ZipFile(args.output / "yolo_depth_parking_edge_results.zip") as archive:
                self.assertIn("candidate_screen.json", archive.namelist())
                self.assertIn("fixtures/depth.npz", archive.namelist())
                self.assertEqual(len([n for n in archive.namelist() if n.startswith("overlays/")]), 16)
                self.assertFalse(any(n.endswith(".pt") or n.startswith("training/") for n in archive.namelist()))

    def test_distinct_notebooks_compile_stage_before_dependencies_and_never_restore_m67(self):
        for filename, candidate, version in (("Parking_YOLO26n_Metric_Depth_Screen_Colab.ipynb", "yolo_depth", "8.4.183"),
                                              ("Parking_UrbanOmniDetect2_Nano_Screen_Colab.ipynb", "urban_nano", "8.4.61")):
            p = ROOT / "notebooks" / filename
            nb = json.loads(p.read_text()); code = []
            for i, cell in enumerate(nb["cells"]):
                if cell["cell_type"] == "code":
                    source = "".join(cell["source"]); compile(source, f"{p}:{i}", "exec"); code.append(source)
                    self.assertIsNone(cell["execution_count"]); self.assertEqual(cell["outputs"], [])
            text = "\n".join(code)
            self.assertIn("CANDIDATE = '"+candidate+"'", text)
            self.assertIn("ultralytics=="+version, text)
            self.assertLess(text.index("parking_edge_common.py"), text.index("pip','install"))
            self.assertIn("--ff-only", text); self.assertIn("run_logged", text)
            self.assertNotIn("M67", text); self.assertNotIn("git reset", text)
            self.assertNotIn(".train(", text); self.assertNotIn(".calibrate(", text)


if __name__ == "__main__":
    unittest.main()
