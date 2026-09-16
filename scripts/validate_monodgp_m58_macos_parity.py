"""Validate the fixed-shape M58 Core ML package on macOS.

The Colab M58 gate can build an ML Program but cannot execute Core ML.  This
validator runs on macOS, compares every exported tensor with the frozen
``m58_reference_io.npz`` sample, and repeats MonoDGP's top-k candidate decode
on both sides.  It deliberately does not run KITTI evaluation or claim device
readiness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np


INPUT_NAMES = ("image", "calibration", "image_size")
OUTPUT_NAMES = (
    "pred_logits",
    "pred_boxes",
    "pred_3d_dim",
    "pred_depth",
    "pred_angle",
    "pred_depth_map_logits",
    "pred_region_prob_0",
    "pred_region_prob_1",
    "pred_region_prob_2",
    "pred_region_prob_3",
)
INPUT_SHAPES = {
    "image": [1, 3, 384, 1280],
    "calibration": [1, 3, 4],
    "image_size": [1, 2],
}
OUTPUT_SHAPES = {
    "pred_logits": [1, 50, 3],
    "pred_boxes": [1, 50, 6],
    "pred_3d_dim": [1, 50, 3],
    "pred_depth": [1, 50, 2],
    "pred_angle": [1, 50, 24],
    "pred_depth_map_logits": [1, 81, 24, 80],
    "pred_region_prob_0": [1, 1, 48, 160],
    "pred_region_prob_1": [1, 1, 24, 80],
    "pred_region_prob_2": [1, 1, 12, 40],
    "pred_region_prob_3": [1, 1, 6, 20],
}

# These are the unchanged M56-family limits.  Region-pyramid levels share the
# semantic pred_region_prob limit.
PARITY_LIMITS = {
    "pred_logits": 0.10,
    "pred_boxes": 0.01,
    "pred_3d_dim": 0.10,
    "pred_depth": 0.50,
    "pred_angle": 0.10,
    "pred_depth_map_logits": 0.10,
    "pred_region_prob": 0.01,
}


def _family(name: str) -> str:
    return "pred_region_prob" if name.startswith("pred_region_prob_") else name


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(item.read_bytes())
    return digest.hexdigest()


def _require_shapes(values: dict[str, np.ndarray], expected: dict[str, list[int]]) -> None:
    for name, shape in expected.items():
        if name not in values:
            raise RuntimeError(f"Reference I/O is missing {name}")
        if list(values[name].shape) != shape:
            raise RuntimeError(
                f"{name} shape changed: expected {shape}, found {list(values[name].shape)}"
            )


def compare_tensor_outputs(
    reference: dict[str, np.ndarray], prediction: dict[str, np.ndarray]
) -> dict[str, dict[str, Any]]:
    report: dict[str, dict[str, Any]] = {}
    for name in OUTPUT_NAMES:
        if name not in prediction:
            report[name] = {"passed": False, "error": "missing Core ML output"}
            continue
        expected = np.asarray(reference[name], dtype=np.float32)
        actual = np.asarray(prediction[name], dtype=np.float32)
        limit = PARITY_LIMITS[_family(name)]
        if actual.shape != expected.shape:
            report[name] = {
                "passed": False,
                "expected_shape": list(expected.shape),
                "actual_shape": list(actual.shape),
                "max_abs_delta": None,
                "limit": limit,
            }
            continue
        delta = np.abs(actual - expected)
        max_abs = float(delta.max())
        mean_abs = float(delta.mean())
        report[name] = {
            "passed": bool(np.isfinite(actual).all() and max_abs <= limit),
            "shape": list(actual.shape),
            "max_abs_delta": max_abs,
            "mean_abs_delta": mean_abs,
            "limit": limit,
        }
    return report


def _sigmoid(values: np.ndarray) -> np.ndarray:
    # Clip only for numerical stability; this does not affect ordinary model
    # logits and mirrors the sigmoid used by MonoDGP's decode helper.
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))


def decode_candidates(values: dict[str, np.ndarray], topk: int = 50) -> np.ndarray:
    """Port MonoDGP ``extract_dets_from_outputs`` to NumPy.

    The returned candidate tensor has shape ``[B,K,39]`` and contains class,
    score, normalized 2D center/size, depth, 24-angle values, dimensions,
    normalized 3D center, and depth uncertainty.  Calibration-dependent KITTI
    conversion remains outside this gate.
    """

    logits = np.asarray(values["pred_logits"], dtype=np.float32)
    boxes = np.asarray(values["pred_boxes"], dtype=np.float32)
    heading = np.asarray(values["pred_angle"], dtype=np.float32)
    dimensions = np.asarray(values["pred_3d_dim"], dtype=np.float32)
    depth = np.asarray(values["pred_depth"], dtype=np.float32)[..., :1]
    sigma = np.exp(-np.asarray(values["pred_depth"], dtype=np.float32)[..., 1:2])
    batch, queries, classes = logits.shape
    k = min(int(topk), queries * classes)
    scores_all = _sigmoid(logits).reshape(batch, -1)
    # Stable descending ordering makes the candidate decode deterministic even
    # when two logits are equal.
    indexes = np.argsort(-scores_all, axis=1, kind="stable")[:, :k]
    query_ids = indexes // classes
    labels = indexes % classes
    row = np.arange(batch)[:, None]
    selected_boxes = boxes[row, query_ids]
    selected_heading = heading[row, query_ids]
    selected_depth = depth[row, query_ids]
    selected_sigma = sigma[row, query_ids]
    selected_dimensions = dimensions[row, query_ids]
    cx = selected_boxes[..., 0:1]
    cy = selected_boxes[..., 1:2]
    # MonoDGP stores c_x, c_y, left, right, top, bottom offsets.
    xyxy = np.concatenate(
        (
            cx - selected_boxes[..., 2:3],
            cy - selected_boxes[..., 4:5],
            cx + selected_boxes[..., 3:4],
            cy + selected_boxes[..., 5:6],
        ),
        axis=-1,
    )
    size_2d = xyxy[..., 2:4] - xyxy[..., 0:2]
    center_2d = (xyxy[..., 0:2] + xyxy[..., 2:4]) * 0.5
    score = scores_all[row, indexes][..., None]
    decoded = np.concatenate(
        (
            labels[..., None].astype(np.float32),
            score,
            center_2d,
            size_2d,
            selected_depth,
            selected_heading,
            selected_dimensions,
            selected_boxes[..., 0:2],
            selected_sigma,
        ),
        axis=-1,
    )
    return decoded.astype(np.float32, copy=False)


def compare_candidates(reference: np.ndarray, actual: np.ndarray) -> dict[str, Any]:
    if reference.shape != actual.shape:
        return {
            "passed": False,
            "expected_shape": list(reference.shape),
            "actual_shape": list(actual.shape),
        }
    delta = np.abs(actual - reference)
    max_abs = float(delta.max()) if delta.size else 0.0
    return {
        "passed": bool(np.isfinite(actual).all() and max_abs <= 1e-4),
        "shape": list(actual.shape),
        "max_abs_delta": max_abs,
        "mean_abs_delta": float(delta.mean()) if delta.size else 0.0,
        "limit": 1e-4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the M58 Core ML raw-output and decoded-candidate parity gate on macOS."
    )
    parser.add_argument("--mlpackage", type=Path, required=True)
    parser.add_argument("--reference-io", type=Path, required=True)
    parser.add_argument("--export-gate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument(
        "--compute-units",
        choices=("CPU_ONLY", "CPU_AND_GPU", "ALL"),
        default="CPU_ONLY",
    )
    args = parser.parse_args()

    if platform.system() != "Darwin":
        raise RuntimeError("M58 Core ML prediction parity requires macOS; Colab cannot execute Core ML")
    if not args.mlpackage.is_dir():
        raise FileNotFoundError(args.mlpackage)
    if not args.reference_io.is_file():
        raise FileNotFoundError(args.reference_io)
    if not args.export_gate.is_file():
        raise FileNotFoundError(args.export_gate)

    import coremltools as ct

    export_gate = json.loads(args.export_gate.read_text(encoding="utf-8"))
    if (
        export_gate.get("complete") is not True
        or export_gate.get("all_export_gates_passed") is not True
        or export_gate.get("macos_coreml_prediction_parity_authorized") is not True
    ):
        raise RuntimeError("M58 export gate did not authorize macOS prediction parity")
    expected_reference_sha = export_gate.get("artifacts", {}).get("reference_io_sha256")
    if expected_reference_sha and sha256_file(args.reference_io) != expected_reference_sha:
        raise RuntimeError("Reference I/O hash differs from the passed M58 export gate")
    expected_package_tree_sha = export_gate.get("artifacts", {}).get("mlpackage_tree_sha256")
    if expected_package_tree_sha and tree_sha256(args.mlpackage) != expected_package_tree_sha:
        raise RuntimeError("ML package tree hash differs from the passed M58 export gate")

    with np.load(args.reference_io, allow_pickle=False) as archive:
        reference_inputs = {name: np.asarray(archive[name]) for name in INPUT_NAMES}
        reference_outputs = {name: np.asarray(archive[name]) for name in OUTPUT_NAMES}
    _require_shapes(reference_inputs, INPUT_SHAPES)
    _require_shapes(reference_outputs, OUTPUT_SHAPES)
    if not all(value.dtype == np.float32 for value in reference_inputs.values()):
        raise RuntimeError("M58 reference inputs must be float32")
    if not all(np.isfinite(value).all() for value in reference_outputs.values()):
        raise RuntimeError("M58 reference outputs contain non-finite values")

    units = getattr(ct.ComputeUnit, args.compute_units)
    prediction_start = time.perf_counter()
    model = ct.models.MLModel(str(args.mlpackage), compute_units=units)
    prediction = model.predict(reference_inputs)
    prediction_seconds = time.perf_counter() - prediction_start
    prediction_arrays = {name: np.asarray(prediction[name]) for name in OUTPUT_NAMES if name in prediction}
    raw_parity = compare_tensor_outputs(reference_outputs, prediction_arrays)
    reference_candidates = decode_candidates(reference_outputs, args.topk)
    coreml_candidates = decode_candidates(prediction_arrays, args.topk)
    candidate_parity = compare_candidates(reference_candidates, coreml_candidates)
    all_raw_passed = all(item.get("passed", False) for item in raw_parity.values())
    complete = all_raw_passed and candidate_parity["passed"]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "complete": complete,
        "experiment": "M58 macOS Core ML prediction parity",
        "scope": "fixed-shape FP32 package; one frozen M58 validation sample",
        "platform": platform.platform(),
        "coremltools_version": ct.__version__,
        "compute_units": args.compute_units,
        "mlpackage": str(args.mlpackage.resolve()),
        "mlpackage_tree_sha256": tree_sha256(args.mlpackage),
        "mlpackage_size_bytes": tree_size(args.mlpackage),
        "reference_io": str(args.reference_io.resolve()),
        "reference_io_sha256": sha256_file(args.reference_io),
        "export_gate": str(args.export_gate.resolve()),
        "export_gate_sha256": sha256_file(args.export_gate),
        "prediction_seconds": prediction_seconds,
        "raw_output_names": list(prediction_arrays),
        "raw_output_parity": raw_parity,
        "all_raw_output_gates_passed": all_raw_passed,
        "decoded_candidate_parity": candidate_parity,
        "decoded_candidate_topk": args.topk,
        "physical_device_testing_authorized": False,
        "fp16_or_quantization_authorized": False,
        "product_safety_qualified": False,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not complete:
        raise RuntimeError(f"M58 macOS parity gate failed; see {args.output}")


if __name__ == "__main__":
    main()
