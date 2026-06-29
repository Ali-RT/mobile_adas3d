#!/usr/bin/env python3
"""
MobileADAS3D Python-vs-Swift decoder parity checker.

Example:
python tools/mobileadas3d_parity.py \
  --torchscript artifacts/mobileadas3d_v6.ts \
  --image road_sample.jpg \
  --swift-json mobileadas3d_road_sample_swift_topk_parity.json \
  --out-dir parity_out \
  --topk 50 \
  --score-threshold 0.55
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch


CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]

CLASS_MEAN_DIMENSIONS_HWL = np.array(
    [
        [1.50, 1.60, 3.90],
        [1.70, 0.60, 0.80],
        [1.70, 0.60, 1.76],
    ],
    dtype=np.float32,
)


def sigmoid_np(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def softplus_float(x: float) -> float:
    if x > 20:
        return float(x)
    return float(math.log1p(math.exp(x)))


def clamp_float(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(x)))


def preprocess_image(
    image_path: Path,
    input_w: int = 1280,
    input_h: int = 384,
) -> tuple[torch.Tensor, Image.Image, tuple[int, int]]:
    image = Image.open(image_path).convert("RGB")
    original_w, original_h = image.size

    resized = image.resize((input_w, input_h), resample=Image.BILINEAR)

    arr = np.asarray(resized).astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    arr = (arr - mean) / std
    arr = arr.transpose(2, 0, 1)  # HWC -> CHW
    arr = arr[None, ...]          # NCHW

    tensor = torch.from_numpy(arr).float()

    return tensor, image, (original_w, original_h)


def normalize_model_outputs(outputs: Any) -> dict[str, torch.Tensor]:
    expected_names = [
        "cls_logits",
        "box2d",
        "log_depth",
        "dim",
        "yaw",
        "center_offset",
        "depth_uncertainty",
    ]

    if isinstance(outputs, dict):
        return {name: outputs[name] for name in expected_names}

    if isinstance(outputs, (tuple, list)):
        if len(outputs) != 7:
            raise ValueError(f"Expected 7 outputs, got {len(outputs)}")

        return dict(zip(expected_names, outputs))

    raise TypeError(f"Unsupported model output type: {type(outputs)}")


def run_torchscript(
    model_path: Path,
    input_tensor: torch.Tensor,
    device: str,
) -> dict[str, np.ndarray]:
    model = torch.jit.load(str(model_path), map_location=device)
    model.eval()

    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        outputs = model(input_tensor)

    outputs = normalize_model_outputs(outputs)

    return {
        key: value.detach().cpu().numpy().astype(np.float32)
        for key, value in outputs.items()
    }


def decode_box_exp_ltrb_grid_units(
    raw_box: np.ndarray,
    center_x: float,
    center_y: float,
    stride: float,
    input_w: int,
    input_h: int,
) -> list[float]:
    l = math.exp(clamp_float(raw_box[0], -5.0, 5.0)) * stride
    t = math.exp(clamp_float(raw_box[1], -5.0, 5.0)) * stride
    r = math.exp(clamp_float(raw_box[2], -5.0, 5.0)) * stride
    b = math.exp(clamp_float(raw_box[3], -5.0, 5.0)) * stride

    x1 = clamp_float(center_x - l, 0.0, input_w - 1.0)
    y1 = clamp_float(center_y - t, 0.0, input_h - 1.0)
    x2 = clamp_float(center_x + r, 0.0, input_w - 1.0)
    y2 = clamp_float(center_y + b, 0.0, input_h - 1.0)

    return [x1, y1, x2, y2]


def decode_candidate(
    outputs: dict[str, np.ndarray],
    class_id: int,
    grid_x: int,
    grid_y: int,
    original_w: int,
    original_h: int,
    input_w: int,
    input_h: int,
    stride: float,
) -> dict[str, Any]:
    cls_logit = float(outputs["cls_logits"][0, class_id, grid_y, grid_x])
    score = float(sigmoid_np(cls_logit))

    raw_box = outputs["box2d"][0, :, grid_y, grid_x].astype(np.float32)
    raw_offset = outputs["center_offset"][0, :, grid_y, grid_x].astype(np.float32)
    raw_log_depth = float(outputs["log_depth"][0, 0, grid_y, grid_x])
    raw_dim = outputs["dim"][0, :, grid_y, grid_x].astype(np.float32)
    raw_yaw = outputs["yaw"][0, :, grid_y, grid_x].astype(np.float32)
    raw_uncertainty = float(outputs["depth_uncertainty"][0, 0, grid_y, grid_x])

    offset_sigmoid = sigmoid_np(raw_offset).astype(np.float32)

    center_x_model = (float(grid_x) + float(offset_sigmoid[0])) * stride
    center_y_model = (float(grid_y) + float(offset_sigmoid[1])) * stride

    bbox_model = decode_box_exp_ltrb_grid_units(
        raw_box=raw_box,
        center_x=center_x_model,
        center_y=center_y_model,
        stride=stride,
        input_w=input_w,
        input_h=input_h,
    )

    scale_x = original_w / input_w
    scale_y = original_h / input_h

    bbox_original = [
        bbox_model[0] * scale_x,
        bbox_model[1] * scale_y,
        bbox_model[2] * scale_x,
        bbox_model[3] * scale_y,
    ]

    depth_m = math.exp(clamp_float(raw_log_depth, -4.0, 6.5))

    class_mean = CLASS_MEAN_DIMENSIONS_HWL[class_id]
    dims_hwl = class_mean * np.exp(np.clip(raw_dim, -5.0, 5.0))

    yaw_rad = math.atan2(float(raw_yaw[0]), float(raw_yaw[1]))
    decoded_uncertainty = softplus_float(raw_uncertainty)

    return {
        "raw": {
            "cls_logit": cls_logit,
            "box2d_ltrb": raw_box.tolist(),
            "center_offset_xy": raw_offset.tolist(),
            "log_depth": raw_log_depth,
            "dim_hwl": raw_dim.tolist(),
            "yaw_2": raw_yaw.tolist(),
            "depth_uncertainty": raw_uncertainty,
        },
        "decoded": {
            "score": score,
            "center_offset_sigmoid_xy": offset_sigmoid.tolist(),
            "center_model_input_xy": [center_x_model, center_y_model],
            "bbox_2d_model_input": bbox_model,
            "bbox_2d_original_image": bbox_original,
            "depth_m": depth_m,
            "dimensions_hwl_m": dims_hwl.astype(float).tolist(),
            "yaw_rad": yaw_rad,
            "depth_uncertainty": decoded_uncertainty,
        },
    }


def make_topk_export(
    outputs: dict[str, np.ndarray],
    image_name: str,
    original_w: int,
    original_h: int,
    input_w: int,
    input_h: int,
    stride: float,
    topk: int,
    score_threshold: float,
    nms_iou_threshold: float,
) -> dict[str, Any]:
    cls_logits = outputs["cls_logits"][0]  # [C,H,W]
    scores = sigmoid_np(cls_logits)

    c, h, w = scores.shape
    flat = scores.reshape(-1)

    top_indices = np.argsort(-flat)[:topk]

    candidates = []

    for rank, flat_idx in enumerate(top_indices):
        class_id = int(flat_idx // (h * w))
        rem = int(flat_idx % (h * w))
        grid_y = int(rem // w)
        grid_x = int(rem % w)

        decoded = decode_candidate(
            outputs=outputs,
            class_id=class_id,
            grid_x=grid_x,
            grid_y=grid_y,
            original_w=original_w,
            original_h=original_h,
            input_w=input_w,
            input_h=input_h,
            stride=stride,
        )

        candidates.append(
            {
                "rank": rank,
                "class_id": class_id,
                "class_name": CLASS_NAMES[class_id],
                "grid_x": grid_x,
                "grid_y": grid_y,
                "raw": decoded["raw"],
                "decoded": decoded["decoded"],
                "passed_score_threshold": decoded["decoded"]["score"] >= score_threshold,
                "kept_after_nms": None,
            }
        )

    return {
        "schema_version": "mobileadas3d_python_parity_topk_v1",
        "source": "python_torchscript",
        "project": "MobileADAS3D",
        "model_name": "MobileADAS3D TorchScript",
        "model_precision": "TorchScript",
        "image_name": image_name,
        "original_image_width": original_w,
        "original_image_height": original_h,
        "model_input_shape": [1, 3, input_h, input_w],
        "score_threshold": score_threshold,
        "topk": topk,
        "nms_iou_threshold": nms_iou_threshold,
        "box_decode_mode": "expLTRBGridUnits",
        "dimension_decode_mode": "class_mean_times_exp_raw",
        "class_mean_dimensions_hwl": CLASS_MEAN_DIMENSIONS_HWL.astype(float).tolist(),
        "topk_candidates": candidates,
    }


def placeholder_intrinsics(image_w: int, image_h: int) -> dict[str, float]:
    width = float(image_w)
    height = float(image_h)

    return {
        "fx": 0.9 * width,
        "fy": 0.9 * width,
        "cx": width / 2.0,
        "cy": height / 2.0,
    }


def project_cuboid(
    detection: dict[str, Any],
    intrinsics: dict[str, float],
) -> list[list[float]] | None:
    center_u, center_v = detection["decoded"]["center_model_input_xy"]

    # Convert model-input center to original image center if needed.
    # This function expects detection bbox/original values, but center is model input.
    # Use bbox center in original image as a stable overlay center.
    box = detection["decoded"]["bbox_2d_original_image"]
    u = 0.5 * (box[0] + box[2])
    v = 0.5 * (box[1] + box[3])

    depth = float(detection["decoded"]["depth_m"])

    if depth <= 0.1:
        return None

    h, w, l = detection["decoded"]["dimensions_hwl_m"]
    yaw = float(detection["decoded"]["yaw_rad"])

    fx = intrinsics["fx"]
    fy = intrinsics["fy"]
    cx = intrinsics["cx"]
    cy = intrinsics["cy"]

    center_x = (u - cx) * depth / fx
    center_y = (v - cy) * depth / fy
    center_z = depth

    corners = [
        (-w / 2, -h / 2, -l / 2),
        ( w / 2, -h / 2, -l / 2),
        ( w / 2,  h / 2, -l / 2),
        (-w / 2,  h / 2, -l / 2),
        (-w / 2, -h / 2,  l / 2),
        ( w / 2, -h / 2,  l / 2),
        ( w / 2,  h / 2,  l / 2),
        (-w / 2,  h / 2,  l / 2),
    ]

    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)

    projected = []

    for local_x, local_y, local_z in corners:
        rotated_x = cos_yaw * local_x + sin_yaw * local_z
        rotated_z = -sin_yaw * local_x + cos_yaw * local_z

        camera_x = center_x + rotated_x
        camera_y = center_y + local_y
        camera_z = center_z + rotated_z

        if camera_z <= 0.1:
            return None

        px = fx * camera_x / camera_z + cx
        py = fy * camera_y / camera_z + cy

        projected.append([float(px), float(py)])

    return projected


def draw_overlay(
    image: Image.Image,
    candidates: list[dict[str, Any]],
    out_path: Path,
    score_threshold: float,
) -> None:
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)

    colors = {
        0: (0, 255, 0),
        1: (255, 255, 0),
        2: (0, 255, 255),
    }

    intrinsics = placeholder_intrinsics(*image.size)

    kept = [
        c for c in candidates
        if c["decoded"]["score"] >= score_threshold
    ]

    for c in kept:
        class_id = c["class_id"]
        color = colors.get(class_id, (255, 0, 0))

        box = c["decoded"]["bbox_2d_original_image"]
        x1, y1, x2, y2 = box

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)

        label = (
            f"{c['class_name']} "
            f"{c['decoded']['score']:.2f} "
            f"d={c['decoded']['depth_m']:.2f}m"
        )

        draw.rectangle(
            [x1, max(0, y1 - 20), x1 + 220, y1],
            fill=color,
        )
        draw.text((x1 + 4, max(0, y1 - 18)), label, fill=(0, 0, 0))

        cuboid = project_cuboid(c, intrinsics)

        if cuboid is not None and len(cuboid) == 8:
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]

            for a, b in edges:
                draw.line(
                    [tuple(cuboid[a]), tuple(cuboid[b])],
                    fill=color,
                    width=2,
                )

    overlay.save(out_path)


def max_abs_delta(a: Any, b: Any) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)

    if aa.shape != bb.shape:
        return float("inf")

    return float(np.max(np.abs(aa - bb)))


def compare_swift_python(
    swift_json: Path,
    python_export: dict[str, Any],
) -> dict[str, Any]:
    swift = json.loads(swift_json.read_text())

    swift_by_key = {
        (c["class_id"], c["grid_x"], c["grid_y"]): c
        for c in swift["topk_candidates"]
    }

    rows = []

    for py_candidate in python_export["topk_candidates"]:
        key = (
            py_candidate["class_id"],
            py_candidate["grid_x"],
            py_candidate["grid_y"],
        )

        sw_candidate = swift_by_key.get(key)

        if sw_candidate is None:
            rows.append(
                {
                    "key": key,
                    "status": "missing_in_swift_topk",
                    "python_rank": py_candidate["rank"],
                }
            )
            continue

        row = {
            "key": {
                "class_id": key[0],
                "grid_x": key[1],
                "grid_y": key[2],
            },
            "status": "matched",
            "python_rank": py_candidate["rank"],
            "swift_rank": sw_candidate["rank"],
            "raw_deltas": {
                "cls_logit": abs(
                    py_candidate["raw"]["cls_logit"]
                    - sw_candidate["raw"]["cls_logit"]
                ),
                "box2d_ltrb": max_abs_delta(
                    py_candidate["raw"]["box2d_ltrb"],
                    sw_candidate["raw"]["box2d_ltrb"],
                ),
                "center_offset_xy": max_abs_delta(
                    py_candidate["raw"]["center_offset_xy"],
                    sw_candidate["raw"]["center_offset_xy"],
                ),
                "log_depth": abs(
                    py_candidate["raw"]["log_depth"]
                    - sw_candidate["raw"]["log_depth"]
                ),
                "dim_hwl": max_abs_delta(
                    py_candidate["raw"]["dim_hwl"],
                    sw_candidate["raw"]["dim_hwl"],
                ),
                "yaw_2": max_abs_delta(
                    py_candidate["raw"]["yaw_2"],
                    sw_candidate["raw"]["yaw_2"],
                ),
                "depth_uncertainty": abs(
                    py_candidate["raw"]["depth_uncertainty"]
                    - sw_candidate["raw"]["depth_uncertainty"]
                ),
            },
            "decoded_deltas": {
                "score": abs(
                    py_candidate["decoded"]["score"]
                    - sw_candidate["decoded"]["score"]
                ),
                "bbox_2d_model_input": max_abs_delta(
                    py_candidate["decoded"]["bbox_2d_model_input"],
                    sw_candidate["decoded"]["bbox_2d_model_input"],
                ),
                "bbox_2d_original_image": max_abs_delta(
                    py_candidate["decoded"]["bbox_2d_original_image"],
                    sw_candidate["decoded"]["bbox_2d_original_image"],
                ),
                "depth_m": abs(
                    py_candidate["decoded"]["depth_m"]
                    - sw_candidate["decoded"]["depth_m"]
                ),
                "dimensions_hwl_m": max_abs_delta(
                    py_candidate["decoded"]["dimensions_hwl_m"],
                    sw_candidate["decoded"]["dimensions_hwl_m"],
                ),
                "yaw_rad": abs(
                    py_candidate["decoded"]["yaw_rad"]
                    - sw_candidate["decoded"]["yaw_rad"]
                ),
                "depth_uncertainty": abs(
                    py_candidate["decoded"]["depth_uncertainty"]
                    - sw_candidate["decoded"]["depth_uncertainty"]
                ),
            },
        }

        rows.append(row)

    matched = [r for r in rows if r["status"] == "matched"]

    summary = {
        "matched_count": len(matched),
        "python_topk_count": len(python_export["topk_candidates"]),
        "missing_in_swift_count": len(rows) - len(matched),
    }

    if matched:
        summary["max_raw_delta"] = {
            name: max(r["raw_deltas"][name] for r in matched)
            for name in matched[0]["raw_deltas"].keys()
        }

        summary["max_decoded_delta"] = {
            name: max(r["decoded_deltas"][name] for r in matched)
            for name in matched[0]["decoded_deltas"].keys()
        }

    return {
        "summary": summary,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--torchscript", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--swift-json", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path("parity_out"))

    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--input-w", type=int, default=1280)
    parser.add_argument("--input-h", type=int, default=384)
    parser.add_argument("--stride", type=float, default=16.0)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)

    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    input_tensor, original_image, (original_w, original_h) = preprocess_image(
        args.image,
        input_w=args.input_w,
        input_h=args.input_h,
    )

    outputs = run_torchscript(
        model_path=args.torchscript,
        input_tensor=input_tensor,
        device=args.device,
    )

    python_export = make_topk_export(
        outputs=outputs,
        image_name=args.image.name,
        original_w=original_w,
        original_h=original_h,
        input_w=args.input_w,
        input_h=args.input_h,
        stride=args.stride,
        topk=args.topk,
        score_threshold=args.score_threshold,
        nms_iou_threshold=args.nms_iou_threshold,
    )

    python_json_path = args.out_dir / "python_topk_parity.json"
    python_json_path.write_text(json.dumps(python_export, indent=2))

    overlay_path = args.out_dir / "python_overlay.png"
    draw_overlay(
        image=original_image,
        candidates=python_export["topk_candidates"],
        out_path=overlay_path,
        score_threshold=args.score_threshold,
    )

    print(f"Wrote Python topK parity JSON: {python_json_path}")
    print(f"Wrote Python overlay PNG:      {overlay_path}")

    if args.swift_json is not None:
        comparison = compare_swift_python(
            swift_json=args.swift_json,
            python_export=python_export,
        )

        comparison_path = args.out_dir / "swift_vs_python_comparison.json"
        comparison_path.write_text(json.dumps(comparison, indent=2))

        print(f"Wrote comparison JSON:         {comparison_path}")
        print("\nComparison summary:")
        print(json.dumps(comparison["summary"], indent=2))


if __name__ == "__main__":
    main()