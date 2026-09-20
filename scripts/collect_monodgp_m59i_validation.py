"""Package original Chen-val PNGs/calibration/GT for paired macOS validation.

CPU only, no inference. Existing matching files are reused; changed files stop
collection. Original labels are required, never the remapped training labels.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.collect_monodgp_m59g_inputs import (
    ANCHOR_SHA256, VAL_SHA256, choose_samples, preprocess, require_anchor_match,
)
from scripts.validate_monodgp_m58_macos_parity import sha256_file

ROOT = Path(__file__).resolve().parents[1]
LABEL_TREE_SHA256 = "adf36ff179f3cc66a6b9432cd135a6afa4bf52588b7742584b1870e76f3f9dc7"
M59G_MANIFEST_SHA256 = "b98049bfd1a6df04d97ecb1ad0afc7702327b86a21b7e602f6f0f45a52015647"
FOLDERS = {"image": ("training/image_2", ".png"),
           "calibration": ("training/calib", ".txt"), "label": ("training/label_2", ".txt")}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def full_ids(split_file):
    choose_samples(split_file)  # Exact hash, uniqueness, count and format.
    return split_file.read_text().splitlines()


def label_hash(directory, ids):
    digest = hashlib.sha256()
    for sample_id in ids:
        digest.update(sample_id.encode("ascii") + b"\0")
        digest.update((directory / f"{sample_id}.txt").read_bytes() + b"\0")
    return digest.hexdigest()


def reviewed_samples():
    path = ROOT / "artifacts/m59g_input_manifest_20260918.json"
    if sha256_file(path) != M59G_MANIFEST_SHA256:
        raise RuntimeError("Reviewed M59g input manifest changed")
    return {row["sample_id"]: row for row in json.loads(path.read_text())["samples"]}


def copy_exact(source, destination):
    expected = sha256_file(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != expected:
            raise RuntimeError(f"Existing bundle file differs; preserve and investigate: {destination}")
    else:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        if sha256_file(temporary) != expected:
            raise RuntimeError(f"Copy verification failed: {source}")
        temporary.replace(destination)
    return expected


def verify_bundle(root):
    manifest = json.loads((root / "m59i_dataset_manifest.json").read_text())
    ids = full_ids(root / "ImageSets/val.txt")
    if (manifest.get("complete") is not True or manifest.get("schema_version") != 1
            or manifest.get("sample_ids") != ids or manifest.get("sample_count") != 3769
            or manifest.get("val_split_sha256") != VAL_SHA256
            or manifest.get("label_tree_sha256") != LABEL_TREE_SHA256
            or manifest.get("m59g_manifest_sha256") != M59G_MANIFEST_SHA256
            or manifest.get("anchor_preprocessing_bit_exact") is not True):
        raise RuntimeError("Incomplete or changed validation bundle")
    rows = manifest["samples"]
    if [row["sample_id"] for row in rows] != ids:
        raise RuntimeError("Validation bundle sample list changed")
    reviewed = reviewed_samples()
    for row in rows:
        sample_id = row["sample_id"]
        for kind, (folder, suffix) in FOLDERS.items():
            path = root / folder / (sample_id + suffix)
            if sha256_file(path) != row[kind + "_sha256"]:
                raise RuntimeError(f"Bundle checksum failed: {path}")
            if kind != "label" and sample_id in reviewed:
                if row[kind + "_sha256"] != reviewed[sample_id][kind + "_sha256"]:
                    raise RuntimeError(f"Reviewed diagnostic input changed: {path}")
    if label_hash(root / "training/label_2", ids) != LABEL_TREE_SHA256:
        raise RuntimeError("Original Chen validation labels changed")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image-dir", "calibration-dir", "label-dir", "split-file", "anchor-npz", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    ids = full_ids(args.split_file)
    if sha256_file(args.anchor_npz) != ANCHOR_SHA256:
        raise RuntimeError("Original M58 reference I/O missing or changed")
    sources = {"image": args.image_dir, "calibration": args.calibration_dir, "label": args.label_dir}
    for sample_id in ids:
        for kind, (_, suffix) in FOLDERS.items():
            path = sources[kind] / (sample_id + suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
    if label_hash(args.label_dir, ids) != LABEL_TREE_SHA256:
        raise RuntimeError("Expected original KITTI labels, not remapped MonoDGP training labels")
    import numpy as np
    with np.load(args.anchor_npz, allow_pickle=False) as anchor:
        require_anchor_match(preprocess(args.image_dir / "000001.png", args.calibration_dir / "000001.txt"), anchor)
    reviewed = reviewed_samples()
    for sample_id, row in reviewed.items():
        for kind in ("image", "calibration"):
            if sha256_file(sources[kind] / (sample_id + FOLDERS[kind][1])) != row[kind + "_sha256"]:
                raise RuntimeError(f"Reviewed {kind} changed: {sample_id}")
    root = args.output_dir
    root.mkdir(parents=True, exist_ok=True)
    copy_exact(args.split_file, root / "ImageSets/val.txt")
    manifest = {"schema_version": 1, "complete": False, "sample_count": len(ids), "sample_ids": ids,
                "val_split_sha256": VAL_SHA256, "label_tree_sha256": LABEL_TREE_SHA256,
                "m59g_manifest_sha256": M59G_MANIFEST_SHA256, "anchor_preprocessing_bit_exact": True,
                "collector_sha256": sha256_file(Path(__file__)), "training_performed": False,
                "model_execution_performed": False, "samples": []}
    report = root / "m59i_dataset_manifest.json"
    for number, sample_id in enumerate(ids, 1):
        row = {"sample_id": sample_id}
        for kind, (folder, suffix) in FOLDERS.items():
            row[kind + "_sha256"] = copy_exact(sources[kind] / (sample_id + suffix), root / folder / (sample_id + suffix))
        manifest["samples"].append(row)
        if number % 100 == 0 or number == len(ids):
            write_json(report, manifest)
            print(f"Copied and verified {number}/{len(ids)} validation samples", flush=True)
    manifest["complete"] = True
    write_json(report, manifest)
    verify_bundle(root)
    archive = root.parent / (root.name + ".zip")
    temporary = archive.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as output:
        paths = [root / "ImageSets/val.txt", report]
        paths.extend(root / folder / (sample_id + suffix) for sample_id in ids for folder, suffix in FOLDERS.values())
        for path in paths:
            output.write(path, Path(root.name) / path.relative_to(root))
    with zipfile.ZipFile(temporary) as output:
        if output.testzip() is not None:
            raise RuntimeError("ZIP integrity check failed")
    temporary.replace(archive)
    print(f"Return this archive: {archive}\nSize: {archive.stat().st_size / 1e9:.2f} GB\nSHA256: {sha256_file(archive)}", flush=True)


if __name__ == "__main__":
    main()
