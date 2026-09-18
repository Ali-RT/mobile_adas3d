"""Collect 16 fixed Chen-val inputs on CPU; no training or model execution."""
from __future__ import annotations

import argparse
import json
import platform
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_monodgp_m59f_position_interleave import verify_full_source
from scripts.validate_monodgp_m58_macos_parity import INPUT_SHAPES, sha256_file

VAL_SHA256 = "6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8"
TRACE_SHA256 = "5f4cf0992dd39fc521793a66f53d754b200ef97d0b243f218f9410e1635d3fca"
ANCHOR_SHA256 = "6d018439a909678546cdb6060ca7ee5ad6fe08296e22462155e9314b243406f2"
SAMPLE_COUNT = 16
PREPROCESSING = "monodgp-aa059a1-val-width-affine-imagenet-fp32-v1"


def choose_samples(split_file):
    if sha256_file(split_file) != VAL_SHA256:
        raise RuntimeError("Expected the frozen Chen validation split hash")
    ids = split_file.read_text().splitlines()
    if len(ids) != 3769 or len(set(ids)) != 3769 or any(not re.fullmatch(r"\d{6}", x) for x in ids):
        raise RuntimeError("Expected 3,769 unique six-digit validation IDs")
    indices = np.linspace(0, len(ids) - 1, SAMPLE_COUNT, dtype=int).tolist()
    selected = [ids[i] for i in indices]
    if selected[0] != "000001":
        raise RuntimeError("Frozen anchor must be the first selected input")
    return indices, selected


def validate_inputs(values):
    for name, shape in INPUT_SHAPES.items():
        value = np.asarray(values[name])
        if list(value.shape) != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise RuntimeError(f"Invalid input shape/dtype/finite values: {name}")
    if (values["image_size"] <= 0).any():
        raise RuntimeError("Invalid original image size")


def inverse_affine(width, height):
    """Exact zero-augmentation branch of upstream get_affine_transform.

    MonoDGP aa059a18214aebf644510e7f0793971b403f9d14 uses width-scaled,
    centered affine sampling, NOT independent width/height resizing.
    """
    import cv2
    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    center = np.array([width, height]) / 2
    src[0] = center
    src[1] = center + np.array([0, -width * 0.5])
    dst[0] = [640, 192]
    dst[1] = [640, 192 - 640]
    for points in (src, dst):
        direction = points[0] - points[1]
        points[2] = points[1] + np.array([-direction[1], direction[0]], dtype=np.float32)
    return cv2.getAffineTransform(dst, src)


def preprocess(image_path, calibration_path):
    from PIL import Image
    with Image.open(image_path) as image:
        if image.mode != "RGB":
            raise RuntimeError("Expected original RGB KITTI PNG")
        width, height = image.size
        transformed = image.transform(
            (1280, 384), Image.Transform.AFFINE,
            tuple(inverse_affine(width, height).reshape(-1).tolist()),
            resample=Image.Resampling.BILINEAR,
        )
        pixels = np.array(transformed).astype(np.float32) / 255.0
    pixels = (pixels - np.array([.485, .456, .406], dtype=np.float32)) / np.array([.229, .224, .225], dtype=np.float32)
    p2 = [line.split()[1:] for line in calibration_path.read_text().splitlines() if line.startswith("P2:")]
    if len(p2) != 1 or len(p2[0]) != 12:
        raise RuntimeError(f"Expected one 3x4 P2 calibration: {calibration_path}")
    values = {"image": np.ascontiguousarray(pixels.transpose(2, 0, 1)[None]),
              "calibration": np.array(p2[0], dtype=np.float32).reshape(1, 3, 4),
              "image_size": np.array([[width, height]], dtype=np.float32)}
    validate_inputs(values)
    return values


def require_anchor_match(inputs, anchor):
    if any(not np.array_equal(inputs[name], anchor[name]) for name in INPUT_SHAPES):
        raise RuntimeError("Preprocessing differs from frozen M58 anchor; stop, do not relax or bypass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-dir", required=True, type=Path)
    parser.add_argument("--calibration-dir", required=True, type=Path)
    parser.add_argument("--split-file", required=True, type=Path)
    parser.add_argument("--m58-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    indices, ids = choose_samples(args.split_file)
    source = verify_full_source(args.m58_dir)
    if (source["artifacts"]["torchscript_sha256"] != TRACE_SHA256
            or source["artifacts"]["reference_io_sha256"] != ANCHOR_SHA256):
        raise RuntimeError("Use the reviewed original M58 artifacts")
    for sample_id in ids:
        for path in (args.image_dir / f"{sample_id}.png", args.calibration_dir / f"{sample_id}.txt"):
            if not path.is_file():
                raise FileNotFoundError(path)
    # Check preprocessing before creating any output. Never substitute synthetic inputs.
    first = preprocess(args.image_dir / "000001.png", args.calibration_dir / "000001.txt")
    with np.load(args.m58_dir / "m58_reference_io.npz", allow_pickle=False) as anchor:
        require_anchor_match(first, anchor)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    import cv2
    import PIL
    manifest = {"schema_version": 1, "complete": False, "experiment": "M59g fixed-input collection",
                "sample_count": SAMPLE_COUNT, "sample_indices": indices, "sample_ids": ids,
                "selection_rule": "16 evenly spaced indices via linspace(0,3768,16,dtype=int)",
                "val_split_sha256": VAL_SHA256, "source_torchscript_sha256": TRACE_SHA256,
                "frozen_anchor_sha256": ANCHOR_SHA256, "preprocessing": PREPROCESSING,
                "anchor_preprocessing_bit_exact": True, "training_performed": False,
                "model_execution_performed": False, "deployment_authorized": False,
                "software": {"python": platform.python_version(), "numpy": np.__version__,
                             "pillow": PIL.__version__, "opencv": cv2.__version__},
                "collector_sha256": sha256_file(Path(__file__)), "samples": []}
    manifest_path = args.output_dir / "m59g_input_manifest.json"
    try:
        shutil.copy2(args.split_file, args.output_dir / "val.txt")
        for number, sample_id in enumerate(ids, 1):
            image = args.image_dir / f"{sample_id}.png"
            calib = args.calibration_dir / f"{sample_id}.txt"
            values = first if sample_id == "000001" else preprocess(image, calib)
            path = args.output_dir / f"{sample_id}.npz"
            np.savez_compressed(path, **values)
            manifest["samples"].append({"sample_id": sample_id, "file": path.name,
                                        "sha256": sha256_file(path), "image_sha256": sha256_file(image),
                                        "calibration_sha256": sha256_file(calib),
                                        "original_size_wh": values["image_size"][0].tolist()})
            print(f"Captured {number}/{SAMPLE_COUNT}: {sample_id}", flush=True)
        manifest["complete"] = True
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    archive = shutil.make_archive(str(args.output_dir), "zip", args.output_dir.parent, args.output_dir.name)
    print(f"Return this archive for the macOS audit: {archive}", flush=True)


if __name__ == "__main__":
    main()
