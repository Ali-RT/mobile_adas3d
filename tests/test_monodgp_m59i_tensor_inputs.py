import copy
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import prepare_monodgp_m59i_tensor_inputs as tensors


class TensorInputTests(unittest.TestCase):
    def values(self):
        return {"image": np.zeros((1, 3, 384, 1280), dtype=np.float32),
                "calibration": np.zeros((1, 3, 4), dtype=np.float32),
                "image_size": np.array([[1242, 375]], dtype=np.float32)}

    def test_digest_preserves_every_bit_and_rejects_bad_fields(self):
        values = self.values()
        before = tensors.tensor_digest(values)
        values["image"][0, 0, 0, 0] = 1e-8
        self.assertNotEqual(before, tensors.tensor_digest(values))
        with self.assertRaisesRegex(RuntimeError, "fields"):
            tensors.tensor_digest({**values, "extra": np.zeros(1)})
        values["image"] = values["image"].astype(np.float16)
        with self.assertRaisesRegex(RuntimeError, "dtype"):
            tensors.tensor_digest(values)

    def test_reference_archive_checks_hash_and_ambiguity(self):
        values = self.values()
        content = io.BytesIO()
        np.savez(content, **values)
        data = content.getvalue()
        reviewed = {"000001": {"sha256": hashlib.sha256(data).hexdigest()}}
        with tempfile.TemporaryDirectory() as directory:
            archive = Path(directory) / "reference.zip"
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("inputs/000001.npz", data)
            with patch.object(tensors, "reviewed_samples", return_value=reviewed):
                found = tensors.load_reviewed_inputs(archive)
                self.assertTrue(np.array_equal(found["000001"]["image"], values["image"]))
                with zipfile.ZipFile(archive, "a") as handle:
                    handle.writestr("duplicate/000001.npz", data)
                with self.assertRaisesRegex(RuntimeError, "one reviewed"):
                    tensors.load_reviewed_inputs(archive)
            with zipfile.ZipFile(archive, "w") as handle:
                handle.writestr("inputs/000001.npz", b"corrupt")
            with patch.object(tensors, "reviewed_samples", return_value=reviewed):
                with self.assertRaisesRegex(RuntimeError, "checksum"):
                    tensors.load_reviewed_inputs(archive)

    def test_verified_load_rejects_mutation_and_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = self.values()
            path = root / "000001.npz"
            np.savez(path, **values)
            row = {"sample_id": "000001", "file": "000001.npz",
                   "sha256": tensors.sha256_file(path), "tensor_sha256": tensors.tensor_digest(values)}
            self.assertEqual(tensors.tensor_digest(tensors.load_tensor(root, row)), row["tensor_sha256"])
            for changes in ({"file": "../000001.npz"}, {"sample_id": "../000001"},
                            {"sha256": "0" * 64}, {"tensor_sha256": "0" * 64}):
                with self.assertRaises(RuntimeError):
                    tensors.load_tensor(root, {**row, **changes})

    def test_complete_bundle_source_software_and_exact_reference_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / "dataset"
            dataset.mkdir()
            (dataset / "m59i_dataset_manifest.json").write_text("original")
            target = root / "tensors"
            target.mkdir()
            values = self.values()
            anchor = root / "anchor.npz"
            np.savez(anchor, **values)
            np.savez(target / "000001.npz", **values)
            source = {"sample_id": "000001", "image_sha256": "image", "calibration_sha256": "calib"}
            dataset_manifest = {"sample_ids": ["000001"], "samples": [source]}
            software = {**tensors.SOFTWARE_VERSIONS, "system": "Linux", "machine": "x86_64"}
            row = {**source, "file": "000001.npz", "sha256": tensors.sha256_file(target / "000001.npz"),
                   "tensor_sha256": tensors.tensor_digest(values)}
            manifest = {"schema_version": 1, "complete": True, "sample_count": 3769,
                        "sample_ids": ["000001"], "samples": [row], "fixed16_bit_exact": True,
                        "anchor_bit_exact": True,
                        "binding": tensors.source_binding(dataset, "sha256:" + "a" * 64, software)}
            path = target / "m59i_tensor_manifest.json"
            with patch.object(tensors, "load_reviewed_inputs", return_value={"000001": values}), \
                 patch.object(tensors, "ANCHOR_SHA256", tensors.sha256_file(anchor)):
                # Anchor is intentionally fixture-bound; production hashes stay strict.
                manifest["binding"]["anchor_sha256"] = tensors.sha256_file(anchor)
                def verify(changed):
                    path.write_text(json.dumps(changed))
                    return tensors.verify_tensor_bundle(target, dataset, dataset_manifest, root / "ref.zip", anchor)
                self.assertTrue(verify(manifest)["complete"])
                for field, value in (("complete", False), ("fixed16_bit_exact", False),
                                     ("sample_ids", ["000002"]), ("sample_count", 1)):
                    with self.assertRaises(RuntimeError):
                        verify({**manifest, field: value})
                for key, value in (("machine", "arm64"), ("numpy", "other")):
                    changed = copy.deepcopy(manifest)
                    changed["binding"]["software"][key] = value
                    with self.assertRaisesRegex(RuntimeError, "binding"):
                        verify(changed)
                changed = copy.deepcopy(manifest)
                changed["samples"][0]["image_sha256"] = "different raw image"
                with self.assertRaisesRegex(RuntimeError, "raw source"):
                    verify(changed)
                values["image"][0, 0, 0, 0] = 1e-8
                with self.assertRaisesRegex(RuntimeError, "frozen M58 anchor"):
                    verify(manifest)


if __name__ == "__main__":
    unittest.main()
