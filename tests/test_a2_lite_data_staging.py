"""Parallel byte-verified staging, partial resume and failure preservation."""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage_a2_lite_data", ROOT / "scripts/stage_a2_lite_data.py")
staging = importlib.util.module_from_spec(spec); spec.loader.exec_module(staging)


class StagingTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"; dataset = root / "kitti_a2_lite_accuracy_fast"
        ids = [f"{n:06d}" for n in range(12)]; expected = {}
        for sample in ids:
            expected[sample] = {}
            for kind, suffix in staging.KINDS:
                alias = kind[:-1] + "02" if kind in ("image_2", "label_2") else kind
                path = source / "training" / alias / (sample + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                data = (sample + kind).encode(); path.write_bytes(data)
                expected[sample][kind] = hashlib.sha256(data).hexdigest()
        return source, dataset, ids, expected

    def test_parallel_and_serial_produce_identical_verified_bytes(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); source, dataset, ids, expected = self.fixture(root)
            report = staging.stage(dataset, [source], ids, expected, workers=4)
            self.assertEqual(report["copied_files"], 36)
            serial = root / "kitti_a2_lite_accuracy_serial"
            staging.stage(serial, [source], ids, expected, workers=1)
            for sample in ids:
                for kind, suffix in staging.KINDS:
                    p = Path("training") / kind / (sample + suffix)
                    self.assertEqual((dataset / p).read_bytes(), (serial / p).read_bytes())
            self.assertFalse(report["gpu_used"]); self.assertFalse(report["training_recipe_changed"])

    def test_partial_resume_reuses_verified_local_files(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); source, dataset, ids, expected = self.fixture(root)
            first = staging.stage(dataset, [source], ids[:2], expected)
            self.assertEqual(first["copied_files"], 6)
            # Already local files must not be read from Drive candidates again.
            for sample in ids[:2]:
                for kind, suffix in staging.KINDS:
                    alias = kind[:-1] + "02" if kind in ("image_2", "label_2") else kind
                    (source / "training" / alias / (sample + suffix)).unlink()
            resumed = staging.stage(dataset, [source], ids, expected)
            self.assertEqual(resumed["reused_files"], 6); self.assertEqual(resumed["copied_files"], 30)
            complete = staging.stage(dataset, [], ids, expected)
            self.assertEqual(complete["reused_files"], 36); self.assertEqual(complete["copied_files"], 0)

    def test_missing_inputs_stop_before_copy(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); source, dataset, ids, expected = self.fixture(root)
            (source / "training/calib/000001.txt").unlink()
            with self.assertRaisesRegex(RuntimeError, "none copied"):
                staging.stage(dataset, [source], ids, expected)
            self.assertFalse(dataset.exists())

    def test_wrong_source_hash_is_never_copied(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); source, dataset, ids, expected = self.fixture(root)
            (source / "training/image_02/000001.png").write_bytes(b"wrong")
            with self.assertRaisesRegex(RuntimeError, "No byte-identical"):
                staging.stage(dataset, [source], ids, expected)
            self.assertFalse((dataset / "training/image_2/000001.png").exists())

    def test_conflicting_destination_preserved(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); source, dataset, ids, expected = self.fixture(root)
            destination = dataset / "training/image_2/000000.png"
            destination.parent.mkdir(parents=True); destination.write_bytes(b"preserve")
            with self.assertRaisesRegex(RuntimeError, "no overwrite"):
                staging.stage(dataset, [source], ids, expected)
            self.assertEqual(destination.read_bytes(), b"preserve")
            self.assertFalse(list(dataset.rglob(".staging_*")))

    def test_bad_worker_count_and_shared_destination_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source, dataset, ids, expected = self.fixture(root)
            for workers in (0, 9):
                with self.assertRaises(ValueError): staging.stage(dataset, [source], ids, expected, workers)
            with self.assertRaises(RuntimeError): staging.stage(source, [source], ids, expected)
            destination = root / "kitti_a2_lite_accuracy_symlink"
            destination.symlink_to(source, target_is_directory=True)
            with self.assertRaises(RuntimeError): staging.stage(destination, [], ids, expected)

    def test_atomic_publication_never_clobbers_concurrent_file(self):
        with tempfile.TemporaryDirectory() as d:
            destination = Path(d) / "frame.png"
            def race(temporary, path):
                path.write_bytes(b"external edit")
                raise FileExistsError(path)
            with patch.object(staging.os, "link", side_effect=race):
                with self.assertRaisesRegex(RuntimeError, "conflicting"):
                    staging.atomic_copy(destination, b"reviewed bytes")
            self.assertEqual(destination.read_bytes(), b"external edit")
            self.assertFalse(list(Path(d).glob(".staging_*")))

    def test_transient_drive_read_retries_without_gpu(self):
        with patch.object(Path, "read_bytes", side_effect=[OSError("temporary I/O"), b"ok"]) as read, \
             patch.object(staging.time, "sleep"):
            self.assertEqual(staging.retry_read(Path("source")), b"ok")
            self.assertEqual(read.call_count, 2)

    def test_manifest_requires_original_pilot(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); (root / "recovery_manifest.json").write_text("{}")
            with self.assertRaisesRegex(RuntimeError, "original reviewed512"):
                staging.reviewed_data(root)

    def test_notebook_stages_before_unchanged_model_prepare(self):
        nb = json.loads((ROOT / "notebooks/MonoDETR_A2_Lite_512_Geometry_KD_Colab.ipynb").read_text())
        codes = ["".join(cell["source"]) for cell in nb["cells"] if cell["cell_type"] == "code"]
        for index, code in enumerate(codes): compile(code, f"staging-{index}", "exec")
        self.assertIn(staging.REVISION, "".join("".join(cell["source"]) for cell in nb["cells"]))
        self.assertIn("stage_a2_lite_data.py", codes[2])
        self.assertLess(codes[2].index("'parallel_staging'"), codes[2].index("stage('prepare'"))
        self.assertIn("'--workers','4'", codes[2])
        self.assertEqual(len(codes), 8)


if __name__ == "__main__": unittest.main()
