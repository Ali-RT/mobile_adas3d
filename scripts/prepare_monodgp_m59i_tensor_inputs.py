"""Freeze Linux/x86 preprocessed inputs for unchanged paired Mac inference.

The raw dataset remains authoritative. All 16 reviewed tensors and the M58
anchor must match exactly before collection; no approximate-input fallback.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import platform
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.collect_monodgp_m59g_inputs import (
    ANCHOR_SHA256, INPUT_SHAPES, PREPROCESSING, preprocess, require_anchor_match,
    sha256_file, validate_inputs,
)
from scripts.collect_monodgp_m59i_validation import reviewed_samples, verify_bundle, write_json

PREPROCESSOR_SHA256 = "ef1d8aaee0e0f12fc5ddae7b533315f2005308d3210bafe9925dd93be8b3297c"
SOFTWARE_VERSIONS = {"python": "3.13.15", "numpy": "2.1.3", "pillow": "11.3.0", "opencv": "5.0.0"}


def tensor_digest(values):
    validate_inputs(values)
    if set(values) != set(INPUT_SHAPES):
        raise RuntimeError("Unexpected tensor input fields")
    digest = hashlib.sha256()
    for name in INPUT_SHAPES:
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode() + b"\0" + str(value.dtype).encode() + b"\0")
        digest.update(json.dumps(list(value.shape)).encode() + b"\0" + value.tobytes())
    return digest.hexdigest()


def load_reviewed_inputs(archive_path):
    """Read only exact hash-pinned NPZ members; never extract archive paths."""
    result = {}
    with zipfile.ZipFile(archive_path) as archive:
        for sample_id, row in reviewed_samples().items():
            matches = [name for name in archive.namelist() if name.endswith("/" + sample_id + ".npz")]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one reviewed input: {sample_id}")
            content = archive.read(matches[0])
            if hashlib.sha256(content).hexdigest() != row["sha256"]:
                raise RuntimeError(f"Reviewed tensor checksum changed: {sample_id}")
            with np.load(io.BytesIO(content), allow_pickle=False) as values:
                result[sample_id] = {name: values[name].copy() for name in INPUT_SHAPES}
            validate_inputs(result[sample_id])
    return result


def load_tensor(root, row):
    sample_id = row["sample_id"]
    if not re.fullmatch(r"\d{6}", sample_id) or row["file"] != sample_id + ".npz":
        raise RuntimeError("Invalid tensor sample filename")
    path = root / row["file"]
    if sha256_file(path) != row["sha256"]:
        raise RuntimeError(f"Tensor file checksum changed: {sample_id}")
    with np.load(path, allow_pickle=False) as data:
        values = {name: data[name].copy() for name in data.files}
    if tensor_digest(values) != row["tensor_sha256"]:
        raise RuntimeError(f"Tensor contents changed: {sample_id}")
    return values


def source_binding(dataset, image_id, software):
    if sha256_file(ROOT / "scripts/collect_monodgp_m59g_inputs.py") != PREPROCESSOR_SHA256:
        raise RuntimeError("Frozen preprocessing code changed")
    return {"dataset_manifest_sha256": sha256_file(dataset / "m59i_dataset_manifest.json"),
            "preprocessing": PREPROCESSING, "preprocessor_sha256": PREPROCESSOR_SHA256,
            "producer_sha256": sha256_file(Path(__file__)), "runtime_image_id": image_id,
            "software": software, "anchor_sha256": ANCHOR_SHA256}


def verify_tensor_bundle(root, dataset, dataset_manifest, archive_path, anchor_file):
    """Verify the complete source-bound tensor set before any model execution."""
    manifest = json.loads((root / "m59i_tensor_manifest.json").read_text())
    ids = dataset_manifest["sample_ids"]
    if (manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or manifest.get("sample_ids") != ids or manifest.get("sample_count") != 3769
            or manifest.get("fixed16_bit_exact") is not True
            or manifest.get("anchor_bit_exact") is not True):
        raise RuntimeError("Incomplete or changed tensor bundle")
    binding = manifest["binding"]
    software = binding["software"]
    if (binding != source_binding(dataset, binding["runtime_image_id"], software)
            or not re.fullmatch(r"sha256:[0-9a-f]{64}", binding["runtime_image_id"])
            or software.get("system") != "Linux" or software.get("machine") != "x86_64"
            or any(software.get(key) != value for key, value in SOFTWARE_VERSIONS.items())):
        raise RuntimeError("Tensor source/software binding changed")
    rows = manifest["samples"]
    if [row["sample_id"] for row in rows] != ids or {p.stem for p in root.glob("*.npz")} != set(ids):
        raise RuntimeError("Tensor sample IDs are missing or extra")
    original = {row["sample_id"]: row for row in dataset_manifest["samples"]}
    reviewed = load_reviewed_inputs(archive_path)
    if sha256_file(anchor_file) != ANCHOR_SHA256:
        raise RuntimeError("Frozen anchor changed")
    with np.load(anchor_file, allow_pickle=False) as anchor:
        for row in rows:
            sample_id = row["sample_id"]
            for kind in ("image", "calibration"):
                if row[kind + "_sha256"] != original[sample_id][kind + "_sha256"]:
                    raise RuntimeError(f"Tensor raw source changed: {sample_id}")
            # Validate hashes and dimensions for every delivered tensor, not only
            # the diagnostic set. load_tensor repeats this at actual consumption.
            values = load_tensor(root, row)
            if sample_id in reviewed:
                require_anchor_match(values, reviewed[sample_id])
            if sample_id == "000001":
                require_anchor_match(values, anchor)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset-bundle", "reviewed-input-archive", "anchor-npz", "output-dir"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--runtime-image-id", required=True)
    args = parser.parse_args()
    import cv2
    import PIL
    software = {"python": platform.python_version(), "numpy": np.__version__, "pillow": PIL.__version__,
                "opencv": cv2.__version__, "system": platform.system(), "machine": platform.machine()}
    if (software["system"] != "Linux" or software["machine"] != "x86_64"
            or any(software[key] != value for key, value in SOFTWARE_VERSIONS.items())):
        raise RuntimeError(f"Use the reviewed Linux/x86 preprocessing environment: {software}")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.runtime_image_id):
        raise RuntimeError("Record the Docker image content ID, not a mutable tag")
    dataset = args.dataset_bundle
    original = verify_bundle(dataset)
    print("Raw dataset verified: 3769/3769", flush=True)
    reviewed = load_reviewed_inputs(args.reviewed_input_archive)
    if sha256_file(args.anchor_npz) != ANCHOR_SHA256:
        raise RuntimeError("Frozen anchor changed")
    fixed = {}
    for sample_id, expected in reviewed.items():
        values = preprocess(dataset / "training/image_2" / (sample_id + ".png"),
                            dataset / "training/calib" / (sample_id + ".txt"))
        require_anchor_match(values, expected)
        fixed[sample_id] = values
    with np.load(args.anchor_npz, allow_pickle=False) as anchor:
        require_anchor_match(fixed["000001"], anchor)
    print("Frozen inputs verified: 16/16 bit-exact; M58 anchor bit-exact", flush=True)
    binding = source_binding(dataset, args.runtime_image_id, software)
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    report_path = root / "m59i_tensor_manifest.json"
    cached = {}
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old["binding"] != binding or old["sample_ids"] != original["sample_ids"]:
            raise RuntimeError("Existing tensor collection binding changed; preserve it")
        cached = {row["sample_id"]: row for row in old["samples"]}
    elif any(root.iterdir()):
        raise RuntimeError("Nonempty tensor directory without provenance")
    manifest = {"schema_version": 1, "complete": False, "binding": binding,
                "sample_ids": original["sample_ids"], "sample_count": 3769,
                "fixed16_bit_exact": True, "anchor_bit_exact": True,
                "model_execution_performed": False, "training_performed": False, "samples": []}
    write_json(report_path, manifest)
    try:
        for number, source in enumerate(original["samples"], 1):
            sample_id = source["sample_id"]
            if sample_id in cached:
                row = cached[sample_id]
                load_tensor(root, row)
            else:
                values = fixed.get(sample_id)
                if values is None:
                    values = preprocess(dataset / "training/image_2" / (sample_id + ".png"),
                                        dataset / "training/calib" / (sample_id + ".txt"))
                path = root / (sample_id + ".npz")
                # A tensor saved before a prior crash is reused only if exact.
                if path.exists():
                    with np.load(path, allow_pickle=False) as existing:
                        require_anchor_match(values, existing)
                else:
                    temporary = path.with_suffix(".npz.tmp")
                    with temporary.open("wb") as handle:
                        np.savez(handle, **values)
                    temporary.replace(path)
                row = {"sample_id": sample_id, "file": path.name,
                       "sha256": sha256_file(path), "tensor_sha256": tensor_digest(values),
                       "image_sha256": source["image_sha256"], "calibration_sha256": source["calibration_sha256"]}
            manifest["samples"].append(row)
            if number % 100 == 0 or number == 3769:
                write_json(report_path, manifest)
                print(f"Linux tensors frozen {number}/3769", flush=True)
        manifest["complete"] = True
    finally:
        write_json(report_path, manifest)
    print(f"Complete tensor manifest: {report_path}", flush=True)


if __name__ == "__main__":
    main()
