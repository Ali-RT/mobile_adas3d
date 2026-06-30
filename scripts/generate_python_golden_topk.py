from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# If this script is copied into scripts/, parents[1] is repo root.
# If run from somewhere else, also add current working directory.
for p in [PROJECT_ROOT, Path.cwd()]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from data.kitti_dataset import KITTIDataset
from data.split_resolver import get_split_file
from data.geometry import scale_p2_for_resize
from data.visualization import draw_gt_and_predictions_2d, draw_projected_3d_boxes
from models.decode import decode_mobile_adas3d_outputs


TORCHSCRIPT_OUTPUT_NAMES = [
    "cls_logits",
    "box2d",
    "log_depth",
    "dim",
    "yaw",
    "center_offset",
    "depth_uncertainty",
    "loc_xy",
]

CUBOID_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def load_yaml_config(path: Path) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tuple_outputs_to_dict(outputs: Any) -> Dict[str, torch.Tensor]:
    if isinstance(outputs, dict):
        return outputs

    if not isinstance(outputs, (tuple, list)):
        raise TypeError(f"Unexpected TorchScript output type: {type(outputs)}")

    if len(outputs) != len(TORCHSCRIPT_OUTPUT_NAMES):
        raise ValueError(
            f"Expected {len(TORCHSCRIPT_OUTPUT_NAMES)} TorchScript outputs, got {len(outputs)}"
        )

    return {name: tensor for name, tensor in zip(TORCHSCRIPT_OUTPUT_NAMES, outputs)}


def resize_image_tensor_to_rgb_uint8(
    image_tensor: torch.Tensor,
    input_height: int,
    input_width: int,
) -> np.ndarray:
    resized = F.interpolate(
        image_tensor.unsqueeze(0),
        size=(input_height, input_width),
        mode="bilinear",
        align_corners=False,
    )

    image_rgb = resized.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (image_rgb * 255.0).clip(0, 255).astype("uint8")


def preprocessing_debug_from_resized_tensor(image_resized: torch.Tensor) -> Dict[str, Any]:
    """
    image_resized: [1, 3, H, W], RGB, float in [0, 1].
    This intentionally matches KITTIDataset + visualize_predictions.py.
    No ImageNet normalization is used.
    """
    x = image_resized.detach().cpu()[0]  # [3,H,W]
    debug: Dict[str, Any] = {
        "color_order": "RGB",
        "scale": "divide_by_255_only",
        "normalization": "none",
        "layout": "NCHW",
        "channel_stats": {},
        "fixed_pixel_checks": [],
    }

    for ci, name in enumerate(["R", "G", "B"]):
        ch = x[ci]
        debug["channel_stats"][name] = {
            "min": float(ch.min().item()),
            "max": float(ch.max().item()),
            "mean": float(ch.mean().item()),
            "first10": [float(v) for v in ch.flatten()[:10].tolist()],
        }

    _, _, h, w = image_resized.shape
    fixed_pixels = [
        [0, 0],
        [1, 0],
        [10, 10],
        [100, 100],
        [w // 2, h // 2],
        [w - 1, h - 1],
    ]

    for px, py in fixed_pixels:
        r = float(x[0, py, px].item())
        g = float(x[1, py, px].item())
        b = float(x[2, py, px].item())
        debug["fixed_pixel_checks"].append(
            {"x": int(px), "y": int(py), "R": r, "G": g, "B": b}
        )

    return debug


def compute_kitti_box_3d(dims_hwl: List[float], location_xyz: List[float], rotation_y: float) -> np.ndarray:
    h, w, l = [float(v) for v in dims_hwl]
    x, y, z = [float(v) for v in location_xyz]

    x_corners = np.array([
        l / 2, l / 2, -l / 2, -l / 2,
        l / 2, l / 2, -l / 2, -l / 2,
    ], dtype=np.float32)

    y_corners = np.array([0, 0, 0, 0, -h, -h, -h, -h], dtype=np.float32)

    z_corners = np.array([
        w / 2, -w / 2, -w / 2, w / 2,
        w / 2, -w / 2, -w / 2, w / 2,
    ], dtype=np.float32)

    corners = np.stack([x_corners, y_corners, z_corners], axis=0)

    c = math.cos(float(rotation_y))
    s = math.sin(float(rotation_y))
    rot_y = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float32)

    corners_3d = rot_y @ corners
    corners_3d = corners_3d + np.array([[x], [y], [z]], dtype=np.float32)
    return corners_3d.T.astype(np.float32)


def project_points_p2(points_3d: np.ndarray, p2: np.ndarray, min_z: float = 0.25) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_3d = np.asarray(points_3d, dtype=np.float32)
    p2 = np.asarray(p2, dtype=np.float32)

    points_h = np.concatenate(
        [points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float32)],
        axis=1,
    )
    uvw = points_h @ p2.T
    z = uvw[:, 2]
    valid = z > min_z

    points_2d = np.full((points_3d.shape[0], 2), np.nan, dtype=np.float32)
    points_2d[valid] = uvw[valid, :2] / z[valid, None]
    return points_2d, valid, z


def add_cuboids_to_prediction(
    pred: Dict[str, Any],
    p2_model: np.ndarray,
    p2_original: np.ndarray,
    original_width: int,
    original_height: int,
    input_width: int,
    input_height: int,
) -> Dict[str, Any]:
    out = dict(pred)

    dims = pred["dimensions_3d_hwl"]
    loc = pred["location_3d"]
    yaw = float(pred["yaw"])

    corners_3d = compute_kitti_box_3d(dims, loc, yaw)
    corners_2d_model, valid_model, z_model = project_points_p2(corners_3d, p2_model)
    corners_2d_orig, valid_orig, z_orig = project_points_p2(corners_3d, p2_original)

    sx_to_orig = float(original_width) / float(input_width)
    sy_to_orig = float(original_height) / float(input_height)
    x1, y1, x2, y2 = pred["bbox_2d"]

    out.update(
        {
            "bbox_2d_model_input": [float(x1), float(y1), float(x2), float(y2)],
            "bbox_2d_original_image": [
                float(x1 * sx_to_orig),
                float(y1 * sy_to_orig),
                float(x2 * sx_to_orig),
                float(y2 * sy_to_orig),
            ],
            "center_2d_model_input": pred["center_2d"],
            "center_2d_original_image": [
                float(pred["center_2d"][0] * sx_to_orig),
                float(pred["center_2d"][1] * sy_to_orig),
            ],
            "depth_m": float(pred["depth"]),
            "location_xyz_camera_m": [float(v) for v in pred["location_3d"]],
            "dimensions_hwl_m": [float(v) for v in pred["dimensions_3d_hwl"]],
            "rotation_y_rad": float(pred["yaw"]),
            "cuboid_3d_camera_m": corners_3d.tolist(),
            "cuboid_2d_model_input": corners_2d_model.tolist(),
            "cuboid_2d_original_image": corners_2d_orig.tolist(),
            "cuboid_valid_mask_model_input": valid_model.tolist(),
            "cuboid_valid_mask_original_image": valid_orig.tolist(),
            "cuboid_corner_z_camera_m": z_orig.tolist(),
        }
    )

    return out


def make_raw_topk_candidates(
    outputs: Dict[str, torch.Tensor],
    classes: List[str],
    class_mean_dims: Dict[str, List[float]],
    input_height: int,
    input_width: int,
    original_width: int,
    original_height: int,
    p2_model: np.ndarray,
    p2_original: np.ndarray,
    topk: int,
) -> List[Dict[str, Any]]:
    cls_logits = outputs["cls_logits"][0].detach().float().cpu()
    box2d = outputs["box2d"][0].detach().float().cpu()
    log_depth = outputs["log_depth"][0].detach().float().cpu()
    dim = outputs["dim"][0].detach().float().cpu()
    yaw = outputs["yaw"][0].detach().float().cpu()
    center_offset = outputs["center_offset"][0].detach().float().cpu()
    depth_uncertainty = outputs["depth_uncertainty"][0].detach().float().cpu()
    loc_xy = outputs["loc_xy"][0].detach().float().cpu()

    num_classes, feature_h, feature_w = cls_logits.shape
    num_cells = feature_h * feature_w
    stride_x = float(input_width) / float(feature_w)
    stride_y = float(input_height) / float(feature_h)

    class_mean_tensor = torch.tensor(
        [class_mean_dims[c] for c in classes], dtype=torch.float32
    )

    scores_flat = torch.sigmoid(cls_logits).reshape(-1)
    k = min(int(topk), int(scores_flat.numel()))
    top_scores, top_indices = torch.topk(scores_flat, k=k, largest=True, sorted=True)

    raw_topk: List[Dict[str, Any]] = []
    sx_to_orig = float(original_width) / float(input_width)
    sy_to_orig = float(original_height) / float(input_height)

    for rank, (score_t, flat_idx_t) in enumerate(zip(top_scores, top_indices)):
        flat_idx = int(flat_idx_t.item())
        score = float(score_t.item())
        class_id = flat_idx // num_cells
        spatial_index = flat_idx % num_cells
        cell_y = spatial_index // feature_w
        cell_x = spatial_index % feature_w

        raw_box = box2d[:, cell_y, cell_x]
        raw_offset = center_offset[:, cell_y, cell_x]
        raw_log_depth = log_depth[0, cell_y, cell_x]
        raw_dim = dim[:, cell_y, cell_x]
        raw_yaw = yaw[:, cell_y, cell_x]
        raw_unc = depth_uncertainty[0, cell_y, cell_x]
        raw_loc_xy = loc_xy[:, cell_y, cell_x]

        # OFFICIAL v7 decode from models/decode.py:
        # center = (cell + 0.5 + raw_offset) * stride
        # l/t/r/b = raw_ltrb * input_width/input_height
        cx = (float(cell_x) + 0.5 + float(raw_offset[0].item())) * stride_x
        cy = (float(cell_y) + 0.5 + float(raw_offset[1].item())) * stride_y

        left = float(raw_box[0].item()) * float(input_width)
        top = float(raw_box[1].item()) * float(input_height)
        right = float(raw_box[2].item()) * float(input_width)
        bottom = float(raw_box[3].item()) * float(input_height)

        x1 = max(0.0, min(float(input_width - 1), cx - left))
        y1 = max(0.0, min(float(input_height - 1), cy - top))
        x2 = max(0.0, min(float(input_width - 1), cx + right))
        y2 = max(0.0, min(float(input_height - 1), cy + bottom))

        depth = float(torch.exp(raw_log_depth).clamp(min=0.1, max=200.0).item())
        location = [
            float(raw_loc_xy[0].item()) * depth,
            float(raw_loc_xy[1].item()) * depth,
            depth,
        ]
        dims = (
            class_mean_tensor[class_id] * torch.exp(raw_dim)
        ).clamp(min=0.01).tolist()
        rot_y = float(torch.atan2(raw_yaw[0], raw_yaw[1]).item())
        corners_3d = compute_kitti_box_3d(dims, location, rot_y)
        cuboid_model, valid_model, z_vals = project_points_p2(corners_3d, p2_model)
        cuboid_orig, valid_orig, _ = project_points_p2(corners_3d, p2_original)

        raw_topk.append(
            {
                "rank": int(rank),
                "flat_index": int(flat_idx),
                "class_id": int(class_id),
                "class_name": classes[class_id],
                "grid_x": int(cell_x),
                "grid_y": int(cell_y),
                "raw": {
                    "cls_logit": float(cls_logits[class_id, cell_y, cell_x].item()),
                    "box2d_ltrb_normalized_by_input_wh": [float(v) for v in raw_box.tolist()],
                    "center_offset_xy": [float(v) for v in raw_offset.tolist()],
                    "log_depth": float(raw_log_depth.item()),
                    "dim_hwl": [float(v) for v in raw_dim.tolist()],
                    "yaw_sin_cos": [float(v) for v in raw_yaw.tolist()],
                    "depth_uncertainty": float(raw_unc.item()),
                    "loc_xy": [float(v) for v in raw_loc_xy.tolist()],
                },
                "decoded": {
                    "score": score,
                    "center_2d_model_input": [float(cx), float(cy)],
                    "center_2d_original_image": [float(cx * sx_to_orig), float(cy * sy_to_orig)],
                    "box_distance_ltrb_model_px": [float(left), float(top), float(right), float(bottom)],
                    "bbox_2d_model_input": [float(x1), float(y1), float(x2), float(y2)],
                    "bbox_2d_original_image": [
                        float(x1 * sx_to_orig),
                        float(y1 * sy_to_orig),
                        float(x2 * sx_to_orig),
                        float(y2 * sy_to_orig),
                    ],
                    "depth_m": depth,
                    "loc_xy": [float(v) for v in raw_loc_xy.tolist()],
                    "location_xyz_camera_m": [float(v) for v in location],
                    "dimensions_hwl_m": [float(v) for v in dims],
                    "rotation_y_rad": rot_y,
                    "yaw_rad": rot_y,
                    "depth_uncertainty_raw": float(raw_unc.item()),
                    "cuboid_3d_camera_m": corners_3d.tolist(),
                    "cuboid_2d_model_input": cuboid_model.tolist(),
                    "cuboid_2d_original_image": cuboid_orig.tolist(),
                    "cuboid_valid_mask_model_input": valid_model.tolist(),
                    "cuboid_valid_mask_original_image": valid_orig.tolist(),
                    "cuboid_corner_z_camera_m": z_vals.tolist(),
                },
            }
        )

    return raw_topk


def scale_gt_objects_to_model_input(sample: Dict[str, Any], input_height: int, input_width: int) -> List[Dict[str, Any]]:
    original_width = int(sample["original_size"]["width"])
    original_height = int(sample["original_size"]["height"])
    x_scale = input_width / float(original_width)
    y_scale = input_height / float(original_height)

    gt_scaled = []
    for obj in sample["objects"]:
        x1, y1, x2, y2 = obj["bbox_2d"]
        gt_scaled.append(
            {
                "class_name": obj["class_name"],
                "class_id": obj.get("class_id"),
                "bbox_2d_model_input": [
                    float(x1 * x_scale),
                    float(y1 * y_scale),
                    float(x2 * x_scale),
                    float(y2 * y_scale),
                ],
                "bbox_2d_original_image": [float(x1), float(y1), float(x2), float(y2)],
                "depth_m": float(obj["location_3d"][2]),
                "dimensions_hwl_m": [float(v) for v in obj["dimensions_3d"]],
                "location_xyz_camera_m": [float(v) for v in obj["location_3d"]],
                "rotation_y_rad": float(obj["rotation_y"]),
            }
        )
    return gt_scaled


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if torch.is_tensor(obj):
        return json_safe(obj.detach().cpu().tolist())
    if isinstance(obj, (np.float32, np.float64)):
        v = float(obj)
        return None if math.isnan(v) or math.isinf(v) else v
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) or math.isinf(obj) else obj
    return obj


def main() -> None:
    parser = argparse.ArgumentParser("Generate MobileADAS3D v7 golden JSON using official repo preprocessing/decode")
    parser.add_argument("--config", type=Path, default=Path("configs/kitti_mobileadas3d.yaml"))
    parser.add_argument("--torchscript-path", type=Path, required=True)
    parser.add_argument("--image-id", type=str, required=True)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--score-threshold", type=float, default=0.25)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=Path("/content/drive/MyDrive/mobile_adas3d_outputs/golden_reference_v7"))
    parser.add_argument("--profile", type=str, default=None, help="Optional dataset active_profile override, e.g. colab_drive")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.profile is not None:
        config["dataset"]["active_profile"] = args.profile

    dataset_cfg = config["dataset"]
    model_cfg = config["model"]
    target_cfg = config["targets"]

    active_profile = dataset_cfg["active_profile"]
    root_dir = dataset_cfg["profiles"][active_profile]["root_dir"]
    split_file = get_split_file(config, args.split)

    input_height = int(model_cfg["input_height"])
    input_width = int(model_cfg["input_width"])

    dataset = KITTIDataset(
        root_dir=root_dir,
        classes=dataset_cfg["classes"],
        image_dir=dataset_cfg["image_dir"],
        label_dir=dataset_cfg["label_dir"],
        calib_dir=dataset_cfg["calib_dir"],
        split_file=split_file,
    )

    if args.image_id not in dataset.sample_ids:
        raise ValueError(
            f"Image id {args.image_id} not found in split={args.split}. "
            f"Use the correct --split, or first ids are: {dataset.sample_ids[:10]}"
        )

    dataset_idx = dataset.sample_ids.index(args.image_id)
    sample = dataset[dataset_idx]

    image = sample["image"].unsqueeze(0)
    image_resized = F.interpolate(
        image,
        size=(input_height, input_width),
        mode="bilinear",
        align_corners=False,
    )

    model = torch.jit.load(str(args.torchscript_path), map_location="cpu")
    model.eval()

    with torch.no_grad():
        outputs_raw = model(image_resized)

    outputs = tuple_outputs_to_dict(outputs_raw)

    predictions = decode_mobile_adas3d_outputs(
        outputs=outputs,
        classes=dataset_cfg["classes"],
        class_mean_dims=target_cfg["class_mean_dims"],
        input_height=input_height,
        input_width=input_width,
        score_threshold=args.score_threshold,
        topk=args.topk,
        nms_iou_threshold=args.nms_iou_threshold,
    )[0]

    original_width = int(sample["original_size"]["width"])
    original_height = int(sample["original_size"]["height"])
    p2_original = np.asarray(sample["P2"], dtype=np.float32)
    p2_model = scale_p2_for_resize(
        P2=p2_original,
        orig_w=original_width,
        orig_h=original_height,
        input_w=input_width,
        input_h=input_height,
    )

    predictions_augmented = [
        add_cuboids_to_prediction(
            pred=p,
            p2_model=p2_model,
            p2_original=p2_original,
            original_width=original_width,
            original_height=original_height,
            input_width=input_width,
            input_height=input_height,
        )
        for p in predictions
    ]

    raw_topk = make_raw_topk_candidates(
        outputs=outputs,
        classes=dataset_cfg["classes"],
        class_mean_dims=target_cfg["class_mean_dims"],
        input_height=input_height,
        input_width=input_width,
        original_width=original_width,
        original_height=original_height,
        p2_model=p2_model,
        p2_original=p2_original,
        topk=args.topk,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Write overlays using exactly the same drawing helpers as visualize_predictions.py.
    resized_rgb = resize_image_tensor_to_rgb_uint8(sample["image"], input_height, input_width)
    gt_scaled_for_overlay = []
    for obj in sample["objects"]:
        x1, y1, x2, y2 = obj["bbox_2d"]
        gt_scaled_for_overlay.append(
            {
                "class_name": obj["class_name"],
                "class_id": obj.get("class_id"),
                "bbox_2d": [
                    x1 * input_width / original_width,
                    y1 * input_height / original_height,
                    x2 * input_width / original_width,
                    y2 * input_height / original_height,
                ],
                "depth": float(obj["location_3d"][2]),
                "dimensions_3d": obj["dimensions_3d"],
                "rotation_y": obj["rotation_y"],
            }
        )

    overlay_2d_path = args.output_dir / f"python_golden_topk_v7_{args.image_id}_overlay_2d.png"
    draw_gt_and_predictions_2d(
        image_rgb=resized_rgb,
        gt_objects=gt_scaled_for_overlay,
        predictions=predictions,
        output_path=overlay_2d_path,
    )

    pred_cuboid_objects = [
        {
            "class_name": pred["class_name"],
            "bbox_2d": pred["bbox_2d"],
            "dimensions_3d": pred["dimensions_3d_hwl"],
            "location_3d": pred["location_3d"],
            "rotation_y": pred["yaw"],
        }
        for pred in predictions
    ]
    overlay_cuboid_path = args.output_dir / f"python_golden_topk_v7_{args.image_id}_cuboid_pred.png"
    draw_projected_3d_boxes(
        image_rgb=resized_rgb,
        objects=pred_cuboid_objects,
        P2=p2_model,
        output_path=overlay_cuboid_path,
    )

    golden = {
        "schema_version": "mobileadas3d_python_golden_topk_v7_repo_decode",
        "project": "MobileADAS3D",
        "version": "v7_cuboid_location",
        "image_id": args.image_id,
        "split": args.split,
        "paths": {
            "config": str(args.config),
            "torchscript_path": str(args.torchscript_path),
            "image_path": str(sample["image_path"]),
            "overlay_2d_path": str(overlay_2d_path),
            "overlay_cuboid_path": str(overlay_cuboid_path),
        },
        "model": {
            "input_shape": [1, 3, input_height, input_width],
            "output_names": TORCHSCRIPT_OUTPUT_NAMES,
            "score_threshold": args.score_threshold,
            "topk": args.topk,
            "nms_iou_threshold": args.nms_iou_threshold,
            "preprocessing": {
                "source": "data.kitti_dataset.KITTIDataset + F.interpolate",
                "color_order": "RGB",
                "scale": "divide_by_255_only",
                "normalization": "none",
                "layout": "NCHW",
                "resize": [input_width, input_height],
                "interpolation": "torch.nn.functional.interpolate bilinear align_corners=False",
            },
            "decode": {
                "source": "models.decode.decode_mobile_adas3d_outputs",
                "center_decode_mode": "(cell + 0.5 + raw_center_offset) * stride",
                "box_decode_mode": "raw_ltrb_normalized_by_input_width_height",
                "depth_decode_mode": "exp(log_depth)",
                "location_decode_mode": "[loc_xy[0] * depth, loc_xy[1] * depth, depth]",
                "dimension_decode_mode": "class_mean_dims_hwl * exp(raw_dim)",
                "yaw_decode_mode": "atan2(raw_yaw_sin, raw_yaw_cos)",
            },
            "class_names": dataset_cfg["classes"],
            "class_mean_dimensions_hwl": target_cfg["class_mean_dims"],
        },
        "image": {
            "original_width": original_width,
            "original_height": original_height,
            "input_width": input_width,
            "input_height": input_height,
            "scale_x_model_to_original": float(original_width) / float(input_width),
            "scale_y_model_to_original": float(original_height) / float(input_height),
        },
        "preprocessing_debug": preprocessing_debug_from_resized_tensor(image_resized),
        "calibration": {
            "P2_original": p2_original.tolist(),
            "P2_model_input": np.asarray(p2_model, dtype=np.float32).tolist(),
        },
        "kitti_ground_truth": scale_gt_objects_to_model_input(sample, input_height, input_width),
        "raw_topk_candidates": raw_topk,
        "detections_after_nms": predictions_augmented,
        "acceptance_tolerance": {
            "score_abs": 0.002,
            "bbox_model_px": 2.0,
            "bbox_original_px": 2.0,
            "depth_m": 0.05,
            "location_xyz_m": 0.05,
            "dimensions_m": 0.05,
            "yaw_rad": 0.01,
            "cuboid_2d_px_valid_points": 5.0,
        },
    }

    out_specific = args.output_dir / f"python_golden_topk_v7_{args.image_id}.json"
    out_v7 = args.output_dir / "python_golden_topk_v7.json"
    out_generic = args.output_dir / "python_golden_topk.json"

    for out_path in [out_specific, out_v7, out_generic]:
        with open(out_path, "w") as f:
            json.dump(json_safe(golden), f, indent=2)
        print("Wrote:", out_path)

    print("2D overlay:", overlay_2d_path)
    print("Cuboid overlay:", overlay_cuboid_path)
    print("Detections after NMS:", len(predictions_augmented))

    for i, pred in enumerate(predictions_augmented):
        print()
        print(f"[{i}] {pred['class_name']} score={pred['score']:.4f}")
        print("bbox_model:", [round(float(x), 2) for x in pred["bbox_2d_model_input"]])
        print("bbox_orig:", [round(float(x), 2) for x in pred["bbox_2d_original_image"]])
        print("depth:", round(float(pred["depth_m"]), 3))
        print("loc:", [round(float(x), 3) for x in pred["location_xyz_camera_m"]])
        print("dims:", [round(float(x), 3) for x in pred["dimensions_hwl_m"]])
        print("yaw:", round(float(pred["rotation_y_rad"]), 4))


if __name__ == "__main__":
    main()
