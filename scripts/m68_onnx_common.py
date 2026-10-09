"""Shared, fail-closed helpers for the original A2 ONNX Runtime CPU audit."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REVISION = "M68-A2-ONNX-ORT-CPU-IPHONE-FEASIBILITY-2026-10-08-r1"
OPSET_VERSION = 17
INPUT_NAMES = ("image", "calibration", "image_size")
OUTPUT_NAMES = ("logits", "boxes", "dimensions", "depth", "angle")
OUTPUT_KEYS = ("pred_logits", "pred_boxes", "pred_3d_dim", "pred_depth", "pred_angle")
OUTPUT_SHAPES = {
    "logits": (1, 50, 3), "boxes": (1, 50, 6), "dimensions": (1, 50, 3),
    "depth": (1, 50, 2), "angle": (1, 50, 24),
}
RAW_LIMITS = {"logits": 0.001, "boxes": 0.0001, "dimensions": 0.001,
              "depth": 0.01, "angle": 0.001}
INPUT_SHAPES = {"image": (1, 3, 384, 1280), "calibration": (1, 3, 4), "image_size": (1, 2)}
FULLVAL_AP_DRIFT_LIMIT = 0.5
NEAR_RECALL_DRIFT_LIMIT = 0.02


def sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
    temporary.replace(path)


def write_f32(path: Path, value) -> dict:
    array = np.asarray(value)
    if array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"Expected finite float32 tensor, got {array.dtype}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(array.astype("<f4", copy=False).tobytes(order="C"))
    return {"file": path.name, "shape": list(array.shape), "dtype": "float32",
            "sha256": sha256(path), "bytes": path.stat().st_size}


def read_f32(root: Path, record: dict) -> np.ndarray:
    path = root / record["file"]
    if not path.resolve().is_relative_to(root.resolve()) or sha256(path) != record["sha256"]:
        raise RuntimeError(f"Tensor file path/checksum mismatch: {path}")
    shape = tuple(int(value) for value in record["shape"])
    if record.get("dtype") != "float32" or path.stat().st_size != math.prod(shape) * 4:
        raise RuntimeError(f"Tensor file dtype/shape mismatch: {path}")
    array = np.fromfile(path, dtype="<f4").reshape(shape).copy()
    if not np.isfinite(array).all():
        raise RuntimeError(f"Non-finite tensor: {path}")
    return array


def compare_outputs(actual: dict, expected: dict, limits: dict = RAW_LIMITS) -> dict:
    if set(actual) != set(OUTPUT_NAMES) or set(expected) != set(OUTPUT_NAMES):
        raise RuntimeError("Unexpected ONNX output names")
    rows = {}
    for name in OUTPUT_NAMES:
        got, reference = np.asarray(actual[name]), np.asarray(expected[name])
        if got.shape != OUTPUT_SHAPES[name] or reference.shape != OUTPUT_SHAPES[name]:
            raise RuntimeError(f"Unexpected {name} shape: {got.shape}/{reference.shape}")
        if got.dtype != np.float32 or reference.dtype != np.float32:
            raise RuntimeError(f"Unexpected {name} precision: {got.dtype}/{reference.dtype}")
        if not np.isfinite(got).all() or not np.isfinite(reference).all():
            raise RuntimeError(f"Non-finite {name} values")
        delta = np.abs(got.astype(np.float64) - reference.astype(np.float64))
        rows[name] = {"max_abs": float(delta.max()), "mean_abs": float(delta.mean()),
                      "limit": float(limits[name]), "passed": bool(delta.max() <= limits[name])}
    return {"outputs": rows, "passed": all(row["passed"] for row in rows.values())}
