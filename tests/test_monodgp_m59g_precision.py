"""M59g selection, preprocessing, provenance, and numerical-audit regressions."""
import ast
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import collect_monodgp_m59g_inputs as collector
from scripts import audit_monodgp_m59g_precision as audit
from scripts.validate_monodgp_m58_macos_parity import OUTPUT_SHAPES


def outputs():
    return {name: np.zeros(shape, dtype=np.float32) for name, shape in OUTPUT_SHAPES.items()}


class M59gPrecisionTests(unittest.TestCase):
    def test_fixed_selection_and_hash_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "val.txt"
            path.write_text("".join(f"{n:06d}\n" for n in range(1, 3770)))
            with self.assertRaisesRegex(RuntimeError, "split hash"):
                collector.choose_samples(path)
            with patch.object(collector, "sha256_file", return_value=collector.VAL_SHA256):
                indices, ids = collector.choose_samples(path)
                self.assertEqual(indices, [0, 251, 502, 753, 1004, 1256, 1507, 1758, 2009, 2260, 2512, 2763, 3014, 3265, 3516, 3768])
                self.assertEqual(len(set(ids)), 16)
                self.assertEqual(ids[0], "000001")
                path.write_text("000001\n" * 3769)
                with self.assertRaisesRegex(RuntimeError, "unique"):
                    collector.choose_samples(path)

    def test_affine_preserves_aspect_ratio_and_center(self):
        for width, height in ((1242, 375), (1224, 370), (1280, 384), (640, 480)):
            matrix = collector.inverse_affine(width, height)
            expected = [[width / 1280, 0, 0], [0, width / 1280, height / 2 - width / 1280 * 192]]
            np.testing.assert_allclose(matrix, expected, atol=1e-12)

    def test_preprocess_and_strict_anchor(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image, calib = root / "000001.png", root / "000001.txt"
            Image.new("RGB", (1242, 375), (50, 100, 200)).save(image)
            calib.write_text("P2: 700 0 620 0 0 700 180 0 0 0 1 0\n")
            values = collector.preprocess(image, calib)
            collector.validate_inputs(values)
            self.assertEqual(values["image_size"].tolist(), [[1242, 375]])
            collector.require_anchor_match(values, values)
            different = dict(values, calibration=values["calibration"] + np.float32(.001))
            with self.assertRaisesRegex(RuntimeError, "Preprocessing differs"):
                collector.require_anchor_match(values, different)
            with self.assertRaisesRegex(RuntimeError, "dtype"):
                collector.validate_inputs(dict(values, image=values["image"].astype(np.float64)))

    def test_repeatability_and_nonfinite_rejection(self):
        first, second = outputs(), outputs()
        self.assertTrue(audit.repeat_summary(first, second)["bit_exact"])
        second["pred_depth"][0, 0, 0] = .0004
        self.assertFalse(audit.repeat_summary(first, second)["bit_exact"])
        second["pred_depth"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(RuntimeError, "Invalid output"):
            audit.repeat_summary(first, second)

    def test_collector_writes_complete_bundle_without_model_execution(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir, calib_dir, m58 = (root / name for name in ("images", "calib", "m58"))
            for path in (image_dir, calib_dir, m58):
                path.mkdir()
            split = root / "val.txt"
            split.write_text("".join(f"{n:06d}\n" for n in range(1, 3770)))
            real_hash = collector.sha256_file
            def fixture_hash(path):
                return collector.VAL_SHA256 if path == split else real_hash(path)
            with patch.object(collector, "sha256_file", side_effect=fixture_hash):
                _, ids = collector.choose_samples(split)
                for sample_id in ids:
                    Image.new("RGB", (1242, 375), (25, 50, 100)).save(image_dir / f"{sample_id}.png")
                    (calib_dir / f"{sample_id}.txt").write_text("P2: 700 0 620 0 0 700 180 0 0 0 1 0\n")
                anchor = collector.preprocess(image_dir / "000001.png", calib_dir / "000001.txt")
                np.savez_compressed(m58 / "m58_reference_io.npz", **anchor)
                source = {"artifacts": {"torchscript_sha256": collector.TRACE_SHA256,
                                        "reference_io_sha256": collector.ANCHOR_SHA256}}
                output = root / "bundle"
                argv = ["collect", "--image-dir", str(image_dir), "--calibration-dir", str(calib_dir),
                        "--split-file", str(split), "--m58-dir", str(m58), "--output-dir", str(output)]
                with patch.object(collector, "verify_full_source", return_value=source), \
                     patch("sys.argv", argv):
                    collector.main()
                    with self.assertRaises(FileExistsError):
                        collector.main()
                manifest = json.loads((output / "m59g_input_manifest.json").read_text())
                self.assertTrue(manifest["complete"])
                self.assertFalse(manifest["model_execution_performed"])
                self.assertEqual(len(manifest["samples"]), 16)
                with zipfile.ZipFile(output.with_suffix(".zip")) as archive:
                    self.assertIsNone(archive.testzip())
                    self.assertEqual(sum(name.endswith(".npz") for name in archive.namelist()), 16)

    def test_rank_change_distinct_from_identity_change(self):
        first, second = outputs(), outputs()
        logits = np.arange(150, dtype=np.float32).reshape(1, 50, 3) / 100
        first["pred_logits"] = logits.copy()
        second["pred_logits"] = logits.copy()
        second["pred_logits"].flat[149], second["pred_logits"].flat[148] = logits.flat[148], logits.flat[149]
        result = audit.selection_summary(first, second)
        self.assertEqual(result["rank_positions_changed"], 2)
        self.assertTrue(result["same_selected_identity_set"])
        second["pred_logits"].flat[0] = 3
        self.assertFalse(audit.selection_summary(first, second)["same_selected_identity_set"])

    def test_small_residual_still_fails_unchanged_decoded_gate(self):
        first, second = outputs(), outputs()
        second["pred_depth"][0, 0, 0] = .0004
        result = audit.compare_stage("full", first, second, audit.OUTPUT_NAMES)
        self.assertTrue(all(row["passed"] for row in result["raw_output_parity"].values()))
        self.assertFalse(result["all_parity_gates_passed"])
        self.assertEqual(result["decoded_candidate_parity"]["limit"], .0001)

    def test_bundle_checksum_and_path_guards(self):
        ids = [f"{n:06d}" for n in range(1, 17)]
        inputs = {"image_size": np.array([[1242, 375]], dtype=np.float32)}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = {"schema_version": 1, "complete": True, "sample_count": 16,
                        "sample_ids": ids, "sample_indices": list(range(16)),
                        "val_split_sha256": collector.VAL_SHA256,
                        "source_torchscript_sha256": collector.TRACE_SHA256,
                        "frozen_anchor_sha256": collector.ANCHOR_SHA256,
                        "preprocessing": collector.PREPROCESSING, "anchor_preprocessing_bit_exact": True,
                        "samples": []}
            for sample_id in ids:
                path = root / f"{sample_id}.npz"
                path.write_bytes(b"fixture; data loader mocked")
                manifest["samples"].append({"sample_id": sample_id, "file": path.name,
                                            "sha256": audit.sha256_file(path), "original_size_wh": [1242, 375]})
            path = root / "m59g_input_manifest.json"
            with patch.object(audit, "choose_samples", return_value=(list(range(16)), ids)), \
                 patch.object(audit, "load_inputs", return_value=inputs), \
                 patch.object(audit, "require_anchor_match"):
                path.write_text(json.dumps(manifest))
                self.assertEqual(len(audit.verify_bundle(root, inputs)), 16)
                manifest["samples"][1]["file"] = "../000002.npz"
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "file path or checksum"):
                    audit.verify_bundle(root, inputs)
                manifest["samples"][1]["file"] = "000002.npz"
                manifest["samples"][1]["sha256"] = "wrong"
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    audit.verify_bundle(root, inputs)

    def test_notebook_self_contained_cpu_workflow(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/MonoDGP_M59g_Precision_Inputs_Colab.ipynb"
        notebook = json.loads(path.read_text())
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        self.assertEqual(len(code), 3)
        for source in code:
            ast.parse(source)
        combined = "\n".join(code)
        self.assertIn("M59g-2026-09-18-r1", code[0])
        self.assertIn("def run_logged", code[0])
        self.assertNotIn("reset', '--hard", combined)
        self.assertNotIn("setup.py", combined)
        self.assertNotIn("nvidia-smi", combined)
        self.assertNotIn("PuFanqi23/MonoDETR", combined)


if __name__ == "__main__":
    unittest.main()
