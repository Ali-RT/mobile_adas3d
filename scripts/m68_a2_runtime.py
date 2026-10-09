"""Validate M68 execution records independently of M67's converter environment."""
from __future__ import annotations

import copy
import importlib.metadata
from pathlib import Path
import sys

import audit_m67_a2_coreml as historical
import m64_teacher_qualification as q
from setup_m68_onnx_runtime import REVISION, validate_receipt

ROOT = Path(__file__).resolve().parents[1]


def implementation_hash():
    names = ("setup_m68_onnx_runtime.py", "prepare_m68_a2_onnx.py", "m68_a2_runtime.py",
             "m68_onnx_common.py", "export_monodetr_a2_onnx.py",
             "evaluate_monodetr_a2_onnx_cpu_fullval.py")
    return q.signature({name: q.sha(ROOT / "scripts" / name) for name in names})


def verify_signed(value):
    body = {key: item for key, item in value.items() if key != "signature_sha256"}
    if q.signature(body) != value.get("signature_sha256"):
        raise RuntimeError("M68 signed record changed")


def historical_inputs(path: Path, original_path: Path | None = None):
    # native=False validates the historical signed protocol/code record only.
    # M68 checks its current native environment and binary in its own record.
    source = historical.load_manifest(path, native=False)
    receipt = Path(source["runtime_receipt"])
    if q.sha(receipt) != source["runtime_receipt_sha256"]:
        raise RuntimeError("Historical M67 receipt changed")
    historical.verify_signed(q.read_json(receipt))
    if q.sha(source["checkpoint"]) != q.A2_SHA or source.get("checkpoint_epoch") != 130:
        raise RuntimeError("Original A2 checkpoint changed")
    if original_path is not None:
        if q.sha(original_path) != source["original_manifest_file_sha256"]:
            raise RuntimeError("M66 input manifest differs from the file frozen by M67")
        original = historical.validate_original(q.read_json(original_path))
        config = copy.deepcopy(original["config"])
        config["dataset"]["root_dir"] = source["dataset_root"]
        if original["checkpoint"] != source["checkpoint"] or config != source["config"]:
            raise RuntimeError("Historical A2 configuration/checkpoint records disagree")
    return source


def validate_current_runtime(path: Path):
    receipt = validate_receipt(q.read_json(path))
    if Path(receipt["python"]).absolute() != Path(sys.executable).absolute():
        raise RuntimeError("Use this M68 run's isolated interpreter")
    if receipt["environment"] != q.environment():
        raise RuntimeError("M68 execution environment changed; use a fresh RUN_ID")
    for name in ("onnx", "onnxruntime"):
        if importlib.metadata.version(name) != receipt[name]:
            raise RuntimeError(f"M68 {name} version changed")
    return receipt


def load_manifest(path: Path):
    value = q.read_json(path)
    verify_signed(value)
    if value.get("revision") != REVISION or value.get("implementation_sha256") != implementation_hash():
        raise RuntimeError("M68 runtime protocol/implementation changed")
    source_path = Path(value["source_m67_manifest"])
    if q.sha(source_path) != value["source_m67_manifest_sha256"]:
        raise RuntimeError("Source M67 manifest changed")
    source = historical_inputs(source_path)
    if source["signature_sha256"] != value["source_m67_manifest_signature"]:
        raise RuntimeError("M68 references a different historical A2 record")
    repo = Path(value["repo"])
    if (q.sha(value["checkpoint"]) != q.A2_SHA
            or value["checkpoint_sha256"] != q.A2_SHA
            or value["checkpoint"] != source["checkpoint"]
            or value["sample_ids"] != source["sample_ids"]
            or q.source_hash(repo) != source["export_source_sha256"]
            or value["export_source_sha256"] != source["export_source_sha256"]):
        raise RuntimeError("M68 A2 checkpoint/source/fixed16 identity changed")
    config = copy.deepcopy(source["config"])
    config["dataset"]["root_dir"] = value["dataset_root"]
    if value["config"] != config:
        raise RuntimeError("M68 original A2 configuration changed")
    binary = Path(value["attention_binary"])
    if not binary.resolve().is_relative_to(repo.resolve()) or q.sha(binary) != value["attention_binary_sha256"]:
        raise RuntimeError("M68 compiled attention binary changed")
    receipt_path = Path(value["runtime_receipt"])
    if q.sha(receipt_path) != value["runtime_receipt_sha256"]:
        raise RuntimeError("M68 runtime receipt changed")
    current = validate_current_runtime(receipt_path)
    if current["environment"] != value["environment"]:
        raise RuntimeError("M68 manifest/runtime identity differs")
    if historical.fixture_inventory(Path(value["dataset_root"]), value["sample_ids"]) != source["input_files"]:
        raise RuntimeError("M68 fixed16 KITTI inputs differ from historical inputs")
    return value


def set_export(model, enabled):
    return historical.set_export(model, enabled)
