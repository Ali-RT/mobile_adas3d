"""Reusable archives: exact inputs/splits, resumable transfer and safe restore."""
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cache_a2_lite_inputs", ROOT / "scripts/cache_a2_lite_inputs.py")
cache = importlib.util.module_from_spec(spec); spec.loader.exec_module(cache)


class CacheTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "source"; dataset = root / "kitti_a2_lite_accuracy_prepare"
        ids = [f"{n:06d}" for n in range(12)]; expected = {}
        for sample in ids:
            expected[sample] = {}
            for kind, suffix in cache.staging.KINDS:
                path = source / "training" / kind / (sample + suffix)
                path.parent.mkdir(parents=True, exist_ok=True)
                data = (sample + kind).encode(); path.write_bytes(data)
                expected[sample][kind] = hashlib.sha256(data).hexdigest()
        splits = {"train": ids[:6], "val": ids[6:]}
        args = SimpleNamespace(pilot=root / "pilot", cache=root / "shared_cache", dataset=dataset,
                               local_archive=root / "build/inputs.zip", candidate=[source], workers=4)
        return args, ids, expected, splits

    def prepared(self, root):
        args, ids, expected, splits = self.fixture(root)
        with patch.object(cache, "oracle", return_value=(ids, expected, splits)):
            cache.prepare(args)
        return args, ids, expected, splits

    def test_prepare_and_restore_exact_data_and_disjoint_splits(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); args, ids, expected, splits = self.prepared(root)
            restore = SimpleNamespace(**vars(args)); restore.dataset = root / "kitti_a2_lite_accuracy_next"
            restore.local_archive = root / "gpu_local/inputs.zip"
            with patch.object(cache, "oracle", return_value=(ids, expected, splits)):
                cache.restore(restore)
            for sample in ids:
                for kind, suffix in cache.staging.KINDS:
                    p = Path("training") / kind / (sample + suffix)
                    self.assertEqual((args.dataset / p).read_bytes(), (restore.dataset / p).read_bytes())
            for name, values in splits.items():
                self.assertEqual((restore.dataset / "ImageSets" / (name + ".txt")).read_text().splitlines(), values)
            report = json.loads((restore.dataset / "a2_lite_archive_restore.json").read_text())
            self.assertTrue(report["every_input_hash_verified"])
            self.assertFalse(report["training_recipe_changed"]); self.assertFalse(report["gpu_used"])

    def test_existing_cache_and_local_restore_are_reused(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); args, ids, expected, splits = self.prepared(root)
            with patch.object(cache, "oracle", return_value=(ids, expected, splits)), \
                 patch.object(cache.staging, "stage", side_effect=AssertionError("no small-file reads")):
                cache.prepare(args)
            args.dataset = root / "kitti_a2_lite_accuracy_reuse"; args.local_archive = root / "local/cache.zip"
            with patch.object(cache, "oracle", return_value=(ids, expected, splits)):
                cache.restore(args); cache.restore(args)
            report = json.loads((args.dataset / "a2_lite_archive_restore.json").read_text())
            self.assertEqual(report["transfer"]["copied_bytes"], 0)
            self.assertEqual(report["extraction"]["restored_files"], 0)
            self.assertEqual(report["extraction"]["reused_files"], 36)

    def test_deterministic_archive_has_no_compression_or_model_assets(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); args, ids, expected, splits = self.prepared(root)
            info = cache.info_for(ids, expected, splits)
            second = root / "second.zip"
            a = cache.build_archive(args.dataset, second, ids, expected, splits, info)
            self.assertEqual(a, cache.archive_record(args.local_archive))
            with zipfile.ZipFile(second) as archive:
                self.assertEqual(len(archive.namelist()), 39)
                self.assertTrue(all(r.compress_type == zipfile.ZIP_STORED for r in archive.infolist()))
                self.assertTrue(all(not n.endswith((".onnx", ".pth", ".npz")) for n in archive.namelist()))

    def test_chunk_resume_keeps_complete_prefix_and_rewrites_owned_partial_tail(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), patch.object(cache, "CHUNK_BYTES", 16):
            root = Path(d); source = root / "source.zip"; data = bytes(range(100)); source.write_bytes(data)
            record = cache.archive_record(source); destination = root / "destination.zip"
            partial = root / "destination.zip.partial"; partial.write_bytes(data[:21])
            cache.write_once(root / "destination.zip.partial.json", cache.json_bytes(record))
            report = cache.copy_archive(source, destination, record)
            self.assertEqual(report["resumed_chunks"], 1); self.assertEqual(report["copied_bytes"], 84)
            self.assertEqual(destination.read_bytes(), data)
            self.assertFalse(partial.exists())

    def test_changed_prefix_is_preserved_not_overwritten(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()), patch.object(cache, "CHUNK_BYTES", 16):
            root = Path(d); source = root / "source.zip"; source.write_bytes(b"a" * 64)
            record = cache.archive_record(source); partial = root / "local.zip.partial"; partial.write_bytes(b"b" * 16)
            cache.write_once(root / "local.zip.partial.json", cache.json_bytes(record))
            with self.assertRaisesRegex(RuntimeError, "Changed complete partial"):
                cache.copy_archive(source, root / "local.zip", record)
            self.assertEqual(partial.read_bytes(), b"b" * 16)
            self.assertFalse((root / "local.zip").exists())

    def test_changed_source_chunk_never_published(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); source = root / "source.zip"; source.write_bytes(b"original")
            record = cache.archive_record(source); source.write_bytes(b"modified")
            with self.assertRaisesRegex(RuntimeError, "source chunk"):
                cache.copy_archive(source, root / "local.zip", record)
            self.assertFalse((root / "local.zip").exists())

    def test_unowned_partial_and_conflicting_final_preserved(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); source = root / "source.zip"; source.write_bytes(b"original")
            record = cache.archive_record(source)
            partial = root / "local.zip.partial"; partial.write_bytes(b"preserve")
            with self.assertRaisesRegex(RuntimeError, "Unowned partial"):
                cache.copy_archive(source, root / "local.zip", record)
            destination = root / "other.zip"; destination.write_bytes(b"preserve final")
            with self.assertRaisesRegex(RuntimeError, "conflicting archive"):
                cache.copy_archive(source, destination, record)
            self.assertEqual(partial.read_bytes(), b"preserve"); self.assertEqual(destination.read_bytes(), b"preserve final")

    def test_receipt_tampering_and_new_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); args, ids, expected, splits = self.prepared(root)
            info = cache.info_for(ids, expected, splits)
            changed = copy.deepcopy(info); changed["inventory_sha256"] = "different"
            with self.assertRaisesRegex(RuntimeError, "provenance"):
                cache.read_receipt(args.cache, changed)
            path = args.cache / cache.RECEIPT; value = json.loads(path.read_text()); value["archive"]["file"] = "../../escape"
            value.pop("receipt_sha256"); value["receipt_sha256"] = cache.signature(value); path.write_bytes(cache.json_bytes(value))
            with self.assertRaisesRegex(RuntimeError, "Malformed"):
                cache.read_receipt(args.cache, info)

    def test_unsafe_or_duplicate_members_rejected_before_extraction(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); args, ids, expected, splits = self.prepared(root)
            with zipfile.ZipFile(args.local_archive, "a") as archive:
                cache.zip_entry(archive, "../../escape.txt", b"bad")
            destination = root / "kitti_a2_lite_accuracy_unsafe"
            with self.assertRaisesRegex(RuntimeError, "unsafe archive"):
                cache.verify_archive(args.local_archive, destination, ids, expected, splits, cache.info_for(ids, expected, splits))
            self.assertFalse(destination.exists()); self.assertFalse((root / "escape.txt").exists())

    def test_changed_input_bytes_not_written_and_existing_conflict_preserved(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); args, ids, expected, splits = self.prepared(root); info = cache.info_for(ids, expected, splits)
            wrong = copy.deepcopy(expected); wrong[ids[0]]["image_2"] = "a" * 64
            # Keep metadata unchanged here to exercise the per-file oracle check.
            with self.assertRaisesRegex(RuntimeError, "input bytes differ"):
                cache.verify_archive(args.local_archive, root / "kitti_a2_lite_accuracy_wrong", ids, wrong, splits, info)
            destination = root / "kitti_a2_lite_accuracy_conflict"; bad = destination / "training/image_2/000000.png"
            bad.parent.mkdir(parents=True); bad.write_bytes(b"preserve")
            with self.assertRaisesRegex(RuntimeError, "conflicting restored"):
                cache.verify_archive(args.local_archive, destination, ids, expected, splits, info)
            self.assertEqual(bad.read_bytes(), b"preserve")

    def test_no_space_stops_before_build_and_split_overlap_rejected(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); args, ids, expected, splits = self.prepared(root)
            with patch.object(cache.shutil, "disk_usage", return_value=SimpleNamespace(free=0)):
                with self.assertRaisesRegex(RuntimeError, "Insufficient"):
                    cache.build_archive(args.dataset, root / "no_space/cache.zip", ids, expected, splits, cache.info_for(ids, expected, splits))
            with self.assertRaises(RuntimeError):
                cache.info_for(ids, expected, {"train": ids, "val": ids})

    def test_private_local_destination_required(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); args, ids, expected, splits = self.prepared(root)
            with self.assertRaises(RuntimeError):
                cache.verify_archive(args.local_archive, root / "shared", ids, expected, splits, cache.info_for(ids, expected, splits))

    def test_historical_pilot_and_shared_cache_paths_are_protected(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); args, _, _, _ = self.fixture(root)
            args.pilot.mkdir(); args.cache = args.pilot / "cache"
            with self.assertRaisesRegex(RuntimeError, "read-only"):
                cache.separate_paths(args)
            args.cache = args.dataset / "cache"
            with self.assertRaisesRegex(RuntimeError, "separate"):
                cache.separate_paths(args)

    def test_restore_checks_free_space_before_input_writes(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            root = Path(d); args, ids, expected, splits = self.prepared(root)
            destination = root / "kitti_a2_lite_accuracy_no_space"
            with patch.object(cache, "free_space", return_value=0):
                with self.assertRaisesRegex(RuntimeError, "remaining extracted"):
                    cache.verify_archive(args.local_archive, destination, ids, expected, splits, cache.info_for(ids, expected, splits))
            self.assertFalse(destination.exists())

    def test_transient_archive_read_retries_same_offset(self):
        class Stream:
            def __init__(self): self.offsets = []; self.calls = 0
            def seek(self, offset): self.offsets.append(offset)
            def read(self, count):
                self.calls += 1
                if self.calls == 1: raise OSError("transient Drive I/O")
                return b"x" * count
        stream = Stream()
        with patch.object(cache.time, "sleep"):
            self.assertEqual(cache.read_at(stream, 64, 16), b"x" * 16)
        self.assertEqual(stream.offsets, [64, 64])

    def test_notebooks_self_contained_cpu_prep_and_gpu_local_restore(self):
        names = ("KITTI_A2_Input_Cache_CPU_Colab.ipynb", "KITTI_A2_Input_Cache_Restore_Colab.ipynb")
        texts = []
        for name in names:
            nb = json.loads((ROOT / "notebooks" / name).read_text())
            codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
            for i, code in enumerate(codes): compile(code, f"{name}-{i}", "exec")
            text = "\n".join(codes); texts.append(text)
            self.assertIn(cache.REVISION, text); self.assertIn("def run_logged(", text)
            self.assertIn("'pull','--ff-only'", text); self.assertNotIn("pip", text)
            self.assertNotIn("torch", text)
            for c in nb["cells"]:
                if c["cell_type"] == "code": self.assertEqual(c["outputs"], [])
        self.assertIn("'prepare'", texts[0]); self.assertIn("ALLOW_GPU_PREP=False", texts[0])
        self.assertIn("'restore'", texts[1]); self.assertIn("--local-archive", texts[1])


if __name__ == "__main__": unittest.main()
