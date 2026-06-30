import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import torch


OUTPUT_NAMES = [
    "cls_logits",
    "box2d",
    "log_depth",
    "dim",
    "yaw",
    "center_offset",
    "depth_uncertainty",
    "loc_xy",
]

CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]

CLASS_MEAN_DIMS_HWL = {
    0: np.array([1.50, 1.60, 3.90], dtype=np.float32),
    1: np.array([1.70, 0.60, 0.80], dtype=np.float32),
    2: np.array([1.70, 0.60, 1.76], dtype=np.float32),
}

INPUT_W = 1280
INPUT_H = 384
STRIDE = 16

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CUBOID_EDGES = [
    [0, 1], [1, 2], [2, 3], [3, 0],
    [4, 5], [5, 6], [6, 7], [7, 4],
    [0, 4], [1, 5], [2, 6], [3, 7],
]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-abs(x))) + max(x, 0.0)


def read_kitti_calib_p2(calib_path: Path):
    data = {}

    with open(calib_path, "r") as f:
        for line in f:
            if ":" not in line:
                continue

            key, value = line.strip().split(":", 1)
            data[key] = np.array([float(x) for x in value.split()], dtype=np.float32)

    if "P2" not in data:
        raise KeyError(f"P2 not found in {calib_path}")

    return data["P2"].reshape(3, 4)


def read_kitti_labels(label_path: Path):
    objects = []

    if not label_path.exists():
        return objects

    with open(label_path, "r") as f:
        for line_idx, line in enumerate(f):
            parts = line.strip().split()

            if len(parts) < 15:
                continue

            class_name = parts[0]
            if class_name not in CLASS_NAMES:
                continue

            bbox = [float(x) for x in parts[4:8]]
            h, w, l = [float(x) for x in parts[8:11]]
            x, y, z = [float(x) for x in parts[11:14]]
            ry = float(parts[14])

            objects.append({
                "gt_index": line_idx,
                "class_name": class_name,
                "bbox_2d_original_image": bbox,
                "dimensions_hwl_m": [h, w, l],
                "location_xyz_camera_m": [x, y, z],
                "rotation_y_rad": ry,
                "depth_m": z,
            })

    return objects


def preprocess_image(image_bgr):
    orig_h, orig_w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(
        image_rgb,
        (INPUT_W, INPUT_H),
        interpolation=cv2.INTER_LINEAR,
    )

    x = resized_rgb.astype(np.float32) / 255.0
    x = (x - IMAGE_MEAN) / IMAGE_STD

    chw = np.transpose(x, (2, 0, 1)).astype(np.float32)
    nchw = np.expand_dims(chw, axis=0)

    tensor = torch.from_numpy(nchw).float()

    preprocess_debug = {}
    for ci, name in enumerate(["R", "G", "B"]):
        ch = x[:, :, ci]
        preprocess_debug[name] = {
            "min": float(ch.min()),
            "max": float(ch.max()),
            "mean": float(ch.mean()),
            "first10": [float(v) for v in ch.reshape(-1)[:10]],
        }

    fixed_pixels = []
    for px, py in [
        [0, 0],
        [1, 0],
        [10, 10],
        [100, 100],
        [640, 192],
        [1279, 383],
    ]:
        r, g, b = x[py, px, :].tolist()
        fixed_pixels.append({
            "x": int(px),
            "y": int(py),
            "R": float(r),
            "G": float(g),
            "B": float(b),
        })

    meta = {
        "original_width": int(orig_w),
        "original_height": int(orig_h),
        "input_width": INPUT_W,
        "input_height": INPUT_H,
        "scale_x_to_original": float(orig_w / INPUT_W),
        "scale_y_to_original": float(orig_h / INPUT_H),
    }

    return tensor, meta, {
        "channel_stats": preprocess_debug,
        "fixed_pixel_checks": fixed_pixels,
    }


def tuple_outputs_to_dict(outputs):
    if isinstance(outputs, dict):
        return outputs

    if not isinstance(outputs, (tuple, list)):
        raise TypeError(f"Unexpected TorchScript output type: {type(outputs)}")

    if len(outputs) != len(OUTPUT_NAMES):
        raise ValueError(f"Expected {len(OUTPUT_NAMES)} outputs, got {len(outputs)}")

    return {name: tensor for name, tensor in zip(OUTPUT_NAMES, outputs)}


def compute_kitti_box_3d(dims_hwl, location_xyz, rotation_y):
    """
    KITTI convention:
      dims = h,w,l
      location = bottom-center in camera coordinates
      rotation_y = yaw around camera Y axis
    """
    h, w, l = [float(v) for v in dims_hwl]
    x, y, z = [float(v) for v in location_xyz]

    x_corners = np.array([
        l / 2,  l / 2, -l / 2, -l / 2,
        l / 2,  l / 2, -l / 2, -l / 2,
    ], dtype=np.float32)

    y_corners = np.array([
        0, 0, 0, 0,
        -h, -h, -h, -h,
    ], dtype=np.float32)

    z_corners = np.array([
        w / 2, -w / 2, -w / 2,  w / 2,
        w / 2, -w / 2, -w / 2,  w / 2,
    ], dtype=np.float32)

    corners = np.stack([x_corners, y_corners, z_corners], axis=0)

    c = math.cos(float(rotation_y))
    s = math.sin(float(rotation_y))

    rot_y = np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float32)

    corners_3d = rot_y @ corners
    corners_3d = corners_3d + np.array([[x], [y], [z]], dtype=np.float32)

    return corners_3d.T.astype(np.float32)


def project_points_p2(points_3d, p2, min_z=0.25):
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


def bbox_iou_xyxy(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)

    box = np.asarray(box, dtype=np.float32)
    boxes = np.asarray(boxes, dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area_b = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    union = area_a + area_b - inter + 1e-9
    return inter / union


def class_aware_nms(detections, iou_threshold):
    final = []

    for class_id in sorted(set(d["class_id"] for d in detections)):
        dets = [d for d in detections if d["class_id"] == class_id]
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)

        while dets:
            best = dets.pop(0)
            final.append(best)

            if not dets:
                break

            boxes = np.array([d["bbox_2d_model_input"] for d in dets], dtype=np.float32)
            ious = bbox_iou_xyxy(best["bbox_2d_model_input"], boxes)

            dets = [
                d for d, iou in zip(dets, ious)
                if float(iou) <= iou_threshold
            ]

    return sorted(final, key=lambda d: d["score"], reverse=True)


def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [json_safe(v) for v in obj]

    if isinstance(obj, tuple):
        return [json_safe(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())

    if isinstance(obj, (np.float32, np.float64)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj

    return obj


def decode_topk_v7(outputs, meta, p2, score_threshold, topk, nms_iou_threshold):
    cls_logits = outputs["cls_logits"][0].detach().float().cpu()            # [3,24,80]
    box2d = outputs["box2d"][0].detach().float().cpu()                      # [4,24,80]
    log_depth = outputs["log_depth"][0, 0].detach().float().cpu()            # [24,80]
    dim = outputs["dim"][0].detach().float().cpu()                          # [3,24,80]
    yaw = outputs["yaw"][0].detach().float().cpu()                          # [2,24,80]
    center_offset = outputs["center_offset"][0].detach().float().cpu()      # [2,24,80]
    depth_unc = outputs["depth_uncertainty"][0, 0].detach().float().cpu()    # [24,80]
    loc_xy = outputs["loc_xy"][0].detach().float().cpu()                    # [2,24,80]

    c, h, w = cls_logits.shape

    scores_flat = torch.sigmoid(cls_logits).reshape(-1)
    topk_scores, topk_indices = torch.topk(
        scores_flat,
        k=min(topk, scores_flat.numel()),
    )

    raw_topk = []
    detections_before_nms = []

    sx = meta["scale_x_to_original"]
    sy = meta["scale_y_to_original"]

    for rank, (score_t, flat_idx_t) in enumerate(zip(topk_scores, topk_indices)):
        flat_idx = int(flat_idx_t.item())
        score = float(score_t.item())

        class_id = flat_idx // (h * w)
        rem = flat_idx % (h * w)
        grid_y = rem // w
        grid_x = rem % w

        raw_cls = float(cls_logits[class_id, grid_y, grid_x].item())
        raw_box = box2d[:, grid_y, grid_x].numpy().astype(np.float32)
        raw_center = center_offset[:, grid_y, grid_x].numpy().astype(np.float32)
        raw_log_depth = float(log_depth[grid_y, grid_x].item())
        raw_dim = dim[:, grid_y, grid_x].numpy().astype(np.float32)
        raw_yaw = yaw[:, grid_y, grid_x].numpy().astype(np.float32)
        raw_unc = float(depth_unc[grid_y, grid_x].item())
        raw_loc_xy = loc_xy[:, grid_y, grid_x].numpy().astype(np.float32)

        center_sig = sigmoid(raw_center)

        center_model_x = (float(grid_x) + float(center_sig[0])) * STRIDE
        center_model_y = (float(grid_y) + float(center_sig[1])) * STRIDE

        ltrb_model = np.exp(raw_box) * STRIDE

        bbox_model = np.array([
            center_model_x - float(ltrb_model[0]),
            center_model_y - float(ltrb_model[1]),
            center_model_x + float(ltrb_model[2]),
            center_model_y + float(ltrb_model[3]),
        ], dtype=np.float32)

        bbox_original = bbox_model.copy()
        bbox_original[[0, 2]] *= sx
        bbox_original[[1, 3]] *= sy

        center_original = np.array([
            center_model_x * sx,
            center_model_y * sy,
        ], dtype=np.float32)

        depth_m = float(np.exp(raw_log_depth))

        location_xyz = np.array([
            float(raw_loc_xy[0]) * depth_m,
            float(raw_loc_xy[1]) * depth_m,
            depth_m,
        ], dtype=np.float32)

        class_mean = CLASS_MEAN_DIMS_HWL[class_id]
        dims_hwl = class_mean * np.exp(raw_dim)

        rotation_y = float(np.arctan2(raw_yaw[0], raw_yaw[1]))

        corners_3d = compute_kitti_box_3d(
            dims_hwl=dims_hwl,
            location_xyz=location_xyz,
            rotation_y=rotation_y,
        )

        corners_2d, corners_valid, corner_z = project_points_p2(
            corners_3d,
            p2,
            min_z=0.25,
        )

        decoded = {
            "score": score,
            "center_offset_sigmoid_xy": center_sig.tolist(),
            "center_2d_model_input": [float(center_model_x), float(center_model_y)],
            "center_2d_original_image": center_original.tolist(),
            "box_distance_ltrb_model_px": ltrb_model.tolist(),
            "bbox_2d_model_input": bbox_model.tolist(),
            "bbox_2d_original_image": bbox_original.tolist(),
            "depth_m": depth_m,
            "loc_xy": raw_loc_xy.tolist(),
            "location_xyz_camera_m": location_xyz.tolist(),
            "class_mean_hwl": class_mean.tolist(),
            "dimensions_hwl_m": dims_hwl.tolist(),
            "rotation_y_rad": rotation_y,
            "yaw_rad": rotation_y,
            "depth_uncertainty": float(softplus(raw_unc)),
            "cuboid_3d_camera_m": corners_3d.tolist(),
            "cuboid_2d_original_image": corners_2d.tolist(),
            "cuboid_valid_mask": corners_valid.tolist(),
            "cuboid_corner_z_camera_m": corner_z.tolist(),
        }

        candidate = {
            "rank": int(rank),
            "flat_index": int(flat_idx),
            "class_id": int(class_id),
            "class_name": CLASS_NAMES[class_id],
            "grid_x": int(grid_x),
            "grid_y": int(grid_y),
            "raw": {
                "cls_logit": raw_cls,
                "box2d_ltrb": raw_box.tolist(),
                "center_offset_xy": raw_center.tolist(),
                "log_depth": raw_log_depth,
                "dim_hwl": raw_dim.tolist(),
                "yaw_sin_cos": raw_yaw.tolist(),
                "depth_uncertainty": raw_unc,
                "loc_xy": raw_loc_xy.tolist(),
            },
            "decoded": decoded,
        }

        raw_topk.append(candidate)

        if score >= score_threshold:
            det = {
                "rank": int(rank),
                "flat_index": int(flat_idx),
                "class_id": int(class_id),
                "class_name": CLASS_NAMES[class_id],
                "grid_x": int(grid_x),
                "grid_y": int(grid_y),
                "score": score,
                **decoded,
            }
            detections_before_nms.append(det)

    detections_after_nms = class_aware_nms(
        detections_before_nms,
        iou_threshold=nms_iou_threshold,
    )

    return raw_topk, detections_before_nms, detections_after_nms


def draw_overlay(image_bgr, detections, out_path):
    canvas = image_bgr.copy()

    for det in detections:
        x1, y1, x2, y2 = [int(round(v)) for v in det["bbox_2d_original_image"]]

        color = (0, 255, 0)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = (
            f"{det['class_name']} {det['score']:.2f} "
            f"z={det['depth_m']:.1f}m"
        )

        cv2.putText(
            canvas,
            label,
            (max(0, x1), max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

        corners = np.asarray(det["cuboid_2d_original_image"], dtype=np.float32)
        valid = det["cuboid_valid_mask"]

        for a, b in CUBOID_EDGES:
            if not valid[a] or not valid[b]:
                continue

            p1 = tuple(np.round(corners[a]).astype(int).tolist())
            p2 = tuple(np.round(corners[b]).astype(int).tolist())

            cv2.line(canvas, p1, p2, color, 2, cv2.LINE_AA)

    cv2.imwrite(str(out_path), canvas)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--torchscript-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--kitti-root",
        type=Path,
        default=Path("/content/drive/MyDrive/datasets/kitti"),
    )
    parser.add_argument(
        "--image-id",
        type=str,
        default="007479",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/content/drive/MyDrive/mobile_adas3d_outputs/golden_reference_v7"),
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.55,
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--nms-iou-threshold",
        type=float,
        default=0.5,
    )

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_path = args.kitti_root / "training" / "image_2" / f"{args.image_id}.png"
    label_path = args.kitti_root / "training" / "label_2" / f"{args.image_id}.txt"
    calib_path = args.kitti_root / "training" / "calib" / f"{args.image_id}.txt"

    print("TorchScript:", args.torchscript_path)
    print("Image:", image_path)
    print("Label:", label_path)
    print("Calib:", calib_path)

    assert args.torchscript_path.exists(), args.torchscript_path
    assert image_path.exists(), image_path
    assert calib_path.exists(), calib_path

    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(image_path)

    p2 = read_kitti_calib_p2(calib_path)
    k = p2[:, :3]
    gt_objects = read_kitti_labels(label_path)

    input_tensor, meta, preprocess_debug = preprocess_image(image_bgr)

    model = torch.jit.load(str(args.torchscript_path), map_location="cpu")
    model.eval()

    with torch.no_grad():
        outputs_raw = model(input_tensor)

    outputs = tuple_outputs_to_dict(outputs_raw)

    print("Output shapes:")
    for name in OUTPUT_NAMES:
        print(f"  {name}: {list(outputs[name].shape)}")

    raw_topk, dets_before_nms, dets_after_nms = decode_topk_v7(
        outputs=outputs,
        meta=meta,
        p2=p2,
        score_threshold=args.score_threshold,
        topk=args.topk,
        nms_iou_threshold=args.nms_iou_threshold,
    )

    overlay_path = args.output_dir / f"python_golden_topk_v7_{args.image_id}_overlay.png"
    draw_overlay(image_bgr, dets_after_nms, overlay_path)

    golden = {
        "schema_version": "mobileadas3d_python_golden_topk_v7",
        "project": "MobileADAS3D",
        "version": "v7_cuboid_location",
        "image_id": args.image_id,
        "image_name": image_path.name,
        "paths": {
            "torchscript_path": str(args.torchscript_path),
            "image_path": str(image_path),
            "label_path": str(label_path),
            "calib_path": str(calib_path),
            "overlay_path": str(overlay_path),
        },
        "model": {
            "input_shape": [1, 3, INPUT_H, INPUT_W],
            "output_names": OUTPUT_NAMES,
            "stride": STRIDE,
            "score_threshold": args.score_threshold,
            "topk": args.topk,
            "nms_iou_threshold": args.nms_iou_threshold,
            "box_decode_mode": "exp_log_ltrb_stride",
            "depth_decode_mode": "exp_log_z",
            "location_decode_mode": "loc_xy_times_depth",
            "dimension_decode_mode": "class_mean_times_exp_raw_dim",
            "yaw_decode_mode": "atan2_sin_cos",
            "class_names": CLASS_NAMES,
            "class_mean_dimensions_hwl": {
                str(k): v.tolist()
                for k, v in CLASS_MEAN_DIMS_HWL.items()
            },
            "preprocessing": {
                "color_order": "RGB",
                "resize": [INPUT_W, INPUT_H],
                "scale": "divide_by_255",
                "mean_rgb": IMAGE_MEAN.tolist(),
                "std_rgb": IMAGE_STD.tolist(),
                "layout": "NCHW",
            },
        },
        "image": meta,
        "preprocessing_debug": preprocess_debug,
        "calibration": {
            "P2": p2.tolist(),
            "K": k.tolist(),
            "fx": float(k[0, 0]),
            "fy": float(k[1, 1]),
            "cx": float(k[0, 2]),
            "cy": float(k[1, 2]),
        },
        "kitti_ground_truth": gt_objects,
        "raw_topk_candidates": raw_topk,
        "detections_before_nms": dets_before_nms,
        "detections_after_nms": dets_after_nms,
        "acceptance_tolerance": {
            "score_abs": 0.002,
            "bbox_px": 2.0,
            "depth_m": 0.05,
            "location_xyz_m": 0.05,
            "dimensions_m": 0.05,
            "yaw_rad": 0.01,
            "cuboid_2d_px": 5.0,
        },
    }

    out_specific = args.output_dir / f"python_golden_topk_v7_{args.image_id}.json"
    out_v7 = args.output_dir / "python_golden_topk_v7.json"
    out_generic = args.output_dir / "python_golden_topk.json"

    for out_path in [out_specific, out_v7, out_generic]:
        with open(out_path, "w") as f:
            json.dump(json_safe(golden), f, indent=2)

        print("Wrote:", out_path)

    print("Overlay:", overlay_path)
    print("Detections after NMS:", len(dets_after_nms))

    for i, det in enumerate(dets_after_nms):
        print()
        print(f"[{i}] {det['class_name']} score={det['score']:.4f}")
        print("bbox:", [round(x, 2) for x in det["bbox_2d_original_image"]])
        print("depth:", round(det["depth_m"], 3))
        print("loc:", [round(x, 3) for x in det["location_xyz_camera_m"]])
        print("dims:", [round(x, 3) for x in det["dimensions_hwl_m"]])
        print("yaw:", round(det["rotation_y_rad"], 4))


if __name__ == "__main__":
    main()