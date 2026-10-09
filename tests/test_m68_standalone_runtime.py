"""CPU checks for historical provenance and the independent M68 environment."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m64_teacher_qualification as q
import m68_a2_runtime as runtime
import prepare_m68_a2_onnx as prepare
import setup_m68_onnx_runtime as setup


def signed(value):
    value = copy.deepcopy(value)
    value["signature_sha256"] = q.signature(value)
    return value


class M68RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def receipt(self, environment):
        return signed(dict(revision=setup.REVISION, complete=True, recipe=setup.RECIPE,
                           python=sys.executable, environment=environment,
                           onnx="1.20.1", onnxruntime="1.30.0",
                           pip_freeze=["torch==2.10.0+cu130", "onnxruntime==1.30.0"]))

    def test_historical_receipt_is_evidence_without_live_environment_comparison(self):
        checkpoint = self.root / "a2.pth"
        checkpoint.write_bytes(b"original weights")
        receipt_path = self.root / "old_receipt.json"
        receipt_path.write_text(json.dumps(signed(dict(environment={"gpu": "old GPU"}))))
        source = dict(runtime_receipt=str(receipt_path), runtime_receipt_sha256=q.sha(receipt_path),
                      checkpoint=str(checkpoint), checkpoint_epoch=130)
        with patch.object(runtime.historical, "load_manifest", return_value=source) as loader, \
                patch.object(q, "A2_SHA", q.sha(checkpoint)), \
                patch.object(q, "environment", side_effect=AssertionError("historical environment compared")):
            self.assertEqual(runtime.historical_inputs(self.root / "m67.json"), source)
            loader.assert_called_once_with(self.root / "m67.json", native=False)
        receipt_path.write_text("{}")
        with patch.object(runtime.historical, "load_manifest", return_value=source):
            with self.assertRaisesRegex(RuntimeError, "Historical M67 receipt changed"):
                runtime.historical_inputs(self.root / "m67.json")

    def test_current_runtime_checks_only_its_own_versions_and_environment(self):
        current = dict(gpu="current GPU", cuda="13.0")
        path = self.root / "m68_runtime_receipt.json"
        path.write_text(json.dumps(self.receipt(current)))
        versions = {"onnx": "1.20.1", "onnxruntime": "1.30.0"}
        with patch.object(q, "environment", return_value=current), \
                patch.object(runtime.importlib.metadata, "version", side_effect=versions.__getitem__):
            self.assertEqual(runtime.validate_current_runtime(path)["environment"], current)
        with patch.object(q, "environment", return_value=dict(gpu="different GPU", cuda="13.0")):
            with self.assertRaisesRegex(RuntimeError, "execution environment changed"):
                runtime.validate_current_runtime(path)

    def test_saved_package_pins_cannot_supply_pip_options(self):
        value = self.receipt({})
        value.pop("signature_sha256")
        value["pip_freeze"] = ["--extra-index-url=https://example.invalid"]
        with self.assertRaisesRegex(RuntimeError, "exact package/version pins"):
            setup.validate_receipt(signed(value))

    def test_new_execution_manifest_checks_checkpoint_source_and_binary(self):
        repo = self.root / "MonoDETR_M68_test"
        repo.mkdir()
        checkpoint, binary = self.root / "a2.pth", repo / "attention.so"
        checkpoint.write_bytes(b"weights")
        binary.write_bytes(b"new GPU build")
        source_path, receipt_path = self.root / "m67.json", self.root / "runtime.json"
        source_path.write_text("historical signed evidence")
        receipt_path.write_text("current runtime evidence")
        config = {"dataset": {"root_dir": str(self.root / "fixtures")}, "model": {"backbone": "A2"}}
        source = dict(signature_sha256="old-signature", checkpoint=str(checkpoint),
                      sample_ids=["000001"], export_source_sha256="source-hash",
                      input_files={"image": "input-hash"}, config=copy.deepcopy(config))
        value = signed(dict(revision=runtime.REVISION, implementation_sha256="implementation",
                            source_m67_manifest=str(source_path), source_m67_manifest_sha256=q.sha(source_path),
                            source_m67_manifest_signature="old-signature", repo=str(repo), config=config,
                            checkpoint=str(checkpoint), checkpoint_sha256=q.sha(checkpoint), sample_ids=["000001"],
                            export_source_sha256="source-hash", attention_binary=str(binary),
                            attention_binary_sha256=q.sha(binary), runtime_receipt=str(receipt_path),
                            runtime_receipt_sha256=q.sha(receipt_path), environment={"gpu": "new GPU"},
                            dataset_root=str(self.root / "fixtures")))
        path = self.root / "m68.json"
        path.write_text(json.dumps(value))
        with patch.object(runtime, "implementation_hash", return_value="implementation"), \
                patch.object(runtime, "historical_inputs", return_value=source), \
                patch.object(runtime, "validate_current_runtime", return_value={"environment": {"gpu": "new GPU"}}), \
                patch.object(runtime.historical, "fixture_inventory", return_value=source["input_files"]), \
                patch.object(q, "source_hash", return_value="source-hash"), \
                patch.object(q, "A2_SHA", q.sha(checkpoint)):
            self.assertEqual(runtime.load_manifest(path)["environment"], {"gpu": "new GPU"})
            binary.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "attention binary changed"):
                runtime.load_manifest(path)

    def test_clean_upstream_checkout_can_receive_reviewed_patches(self):
        repo = self.root / "MonoDETR_M68_clean"
        (repo / ".git").mkdir(parents=True)
        with patch.object(prepare.subprocess, "check_output", side_effect=[q.A2_COMMIT, prepare.A2_URL]), \
                patch.object(prepare.subprocess, "run") as command, \
                patch.object(q, "source_hash", side_effect=AssertionError("clean upstream rejected before patch")):
            command.return_value.returncode = 0
            prepare.ensure_repo(repo, {"native_source_sha256": "native", "export_source_sha256": "portable"})


if __name__ == "__main__":
    unittest.main()
