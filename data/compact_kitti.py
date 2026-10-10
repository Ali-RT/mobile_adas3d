"""KITTI targets for the compact CNN. No teacher or upstream CUDA extensions.

Corner ordering and two-bin rotation follow RTM3D/KM3D (Banconxuan/RTM3D,
commit 888c379e79d8a6d134f06a9b7d669118679e06dc). Geometry is implemented
here from the full P2 projection equations, including translation and fy.
The trained model uses RGB/ImageNet normalization, unlike the earlier BGR
random-weight timing fixture. A trained phone bundle must use this contract.
"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from data.class_taxonomy import KITTI_PRODUCTION_CLASS_MAPPING
from data.kitti_parser import parse_kitti_calib_file, parse_kitti_label_file

CLASSES = ("Vehicle", "Pedestrian")
PREPROCESSING = "COMPACT-KITTI-RGB-IMAGENET-AFFINE-384x1280-r1"
INPUT_HW = (384, 1280)
MEAN = np.asarray([0.485, 0.456, 0.406], np.float32)
STD = np.asarray([0.229, 0.224, 0.225], np.float32)


def read_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids or len(ids) != len(set(ids)) or any(len(i) != 6 or not i.isdigit() for i in ids):
        raise ValueError(f"Expected unique six-digit KITTI IDs: {path}")
    return ids


def image_affine(width: int, height: int, output_hw=INPUT_HW, scale=1.0, shift=(0.0, 0.0)):
    """Aspect-preserving affine; same zero-rotation geometry as RTM3D."""
    out_h, out_w = output_hw
    factor = out_w / (max(width, height) * scale)
    center = np.asarray([width / 2 + shift[0], height / 2 + shift[1]])
    return np.asarray([[factor, 0, out_w / 2 - center[0] * factor],
                       [0, factor, out_h / 2 - center[1] * factor]], np.float32)


def transform_points(points, affine):
    points = np.asarray(points)
    return points @ affine[:, :2].T + affine[:, 2]


def corner_offsets(dimensions, yaw):
    """Eight corners about geometric center, then geometric center itself."""
    h, w, length = map(float, dimensions)
    x = np.asarray([1, 1, -1, -1, 1, 1, -1, -1, 0]) * length / 2
    y = np.asarray([1, 1, 1, 1, -1, -1, -1, -1, 0]) * h / 2
    z = np.asarray([1, -1, -1, 1, 1, -1, -1, 1, 0]) * w / 2
    c, s = math.cos(yaw), math.sin(yaw)
    return np.stack([c * x + s * z, y, -s * x + c * z], axis=-1)


def project_keypoints(obj, projection):
    center = np.asarray(obj.location_3d, np.float64).copy()
    center[1] -= obj.dimensions_3d[0] / 2
    xyz = corner_offsets(obj.dimensions_3d, obj.rotation_y) + center
    pixels = np.column_stack([xyz, np.ones(9)]) @ projection.T
    if (pixels[:, 2] <= 0.1).any() or not np.isfinite(pixels).all():
        return None
    return (pixels[:, :2] / pixels[:, 2:3]).astype(np.float32)


def valid_geometry(obj):
    values = [*obj.dimensions_3d, *obj.location_3d, obj.rotation_y, obj.alpha]
    return bool(np.isfinite(values).all() and min(obj.dimensions_3d) > 0
                and obj.location_3d[2] > 0.1 and abs(obj.alpha) <= math.pi + 0.01)


def rotation_target(alpha):
    alpha = (float(alpha) + math.pi) % (2 * math.pi) - math.pi
    bins = np.asarray([alpha < math.pi / 6 or alpha > 5 * math.pi / 6,
                       alpha > -math.pi / 6 or alpha < -5 * math.pi / 6], np.int64)
    return bins, np.asarray([alpha + math.pi / 2, alpha - math.pi / 2], np.float32)


def draw_gaussian(heatmap, xy, radius):
    radius = max(0, int(radius))
    x, y = map(int, xy)
    diameter = 2 * radius + 1
    grid = np.arange(-radius, radius + 1)
    gaussian = np.exp(-(grid[:, None] ** 2 + grid[None, :] ** 2) / (2 * (diameter / 6) ** 2))
    h, w = heatmap.shape
    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)
    window = heatmap[y-top:y+bottom, x-left:x+right]
    np.maximum(window, gaussian[radius-top:radius+bottom, radius-left:radius+right], out=window)


def gaussian_radius(height, width, overlap=0.7):
    # Radius constraints from overlap of two rectangles (CenterNet convention).
    a = (1., 4., 4. * overlap)
    b = (height + width, 2 * (height + width), -2 * overlap * (height + width))
    c = (height * width * (1-overlap)/(1+overlap), (1-overlap)*height*width,
         (overlap-1)*height*width)
    return max(0, int(min((bb + math.sqrt(max(0., bb*bb - 4*aa*cc))) / 2
                         for aa, bb, cc in zip(a, b, c))))


def build_targets(objects, projection, output_affine, output_hw=(96, 320), max_objects=128):
    h, w = output_hw
    n = max_objects
    target = {
        "hm": np.zeros((2, h, w), np.float32),
        "hm_valid": np.ones((2, h, w), np.float32),
        "hm_hp": np.zeros((9, h, w), np.float32),
        "kp_valid": np.ones((9, h, w), np.float32),
        "indices": np.zeros(n, np.int64), "mask": np.zeros(n, np.float32),
        "geom_mask": np.zeros(n, np.float32), "wh": np.zeros((n, 2), np.float32),
        "reg": np.zeros((n, 2), np.float32), "hps": np.zeros((n, 18), np.float32),
        "dim": np.zeros((n, 3), np.float32), "location": np.zeros((n, 3), np.float32),
        "yaw": np.zeros(n, np.float32), "rotbin": np.zeros((n, 2), np.int64),
        "rotres": np.zeros((n, 2), np.float32),
        "hp_indices": np.zeros(n * 9, np.int64), "hp_mask": np.zeros(n * 9, np.float32),
        "hp_offset": np.zeros((n * 9, 2), np.float32),
    }
    positives = []
    for obj in objects:
        xy = transform_points(np.asarray(obj.bbox_2d).reshape(2, 2), output_affine)
        xy[:, 0] = np.clip(xy[:, 0], 0, w - 1)
        xy[:, 1] = np.clip(xy[:, 1], 0, h - 1)
        if not np.isfinite(xy).all() or (xy[1] <= xy[0]).any():
            continue
        mapped = KITTI_PRODUCTION_CLASS_MAPPING.get(obj.class_name)
        if mapped is None:
            # Do not train DontCare/Cyclist/Misc as confident negative objects.
            x0, y0 = np.floor(xy[0]).astype(int)
            x1, y1 = np.ceil(xy[1]).astype(int)
            target["hm_valid"][:, y0:y1+1, x0:x1+1] = 0
            target["kp_valid"][:, y0:y1+1, x0:x1+1] = 0
        else:
            positives.append((obj, xy, CLASSES.index(mapped)))
    if len(positives) > n:
        raise ValueError(f"{len(positives)} objects exceed max_objects={n}; do not silently truncate GT")
    used_centers = set()
    # Dense regressions cannot represent two objects at the same center cell.
    # Retain both heatmap positives, but regress the larger box at that cell.
    positives.sort(key=lambda item: -float(np.prod(item[1][1] - item[1][0])))
    for slot, (obj, xy, cls) in enumerate(positives):
        ct = (xy[0] + xy[1]) / 2
        ct_int = ct.astype(int)
        size = xy[1] - xy[0]
        radius = gaussian_radius(*size[::-1])
        draw_gaussian(target["hm"][cls], ct_int, radius)
        index = int(ct_int[1] * w + ct_int[0])
        if index in used_centers:
            continue
        used_centers.add(index)
        target["indices"][slot] = index
        target["mask"][slot] = 1
        target["wh"][slot] = size
        target["reg"][slot] = ct - ct_int
        if not valid_geometry(obj):
            continue
        keypoints = project_keypoints(obj, projection)
        if keypoints is None:
            continue
        kp = transform_points(keypoints, output_affine)
        target["geom_mask"][slot] = 1
        target["hps"][slot] = (kp - ct_int).reshape(18)
        target["dim"][slot] = obj.dimensions_3d
        target["location"][slot] = obj.location_3d  # KITTI bottom center.
        target["yaw"][slot] = obj.rotation_y
        target["rotbin"][slot], target["rotres"][slot] = rotation_target(obj.alpha)
        for joint, point in enumerate(kp):
            if 0 <= point[0] < w and 0 <= point[1] < h:
                integer = point.astype(int)
                j = slot * 9 + joint
                target["hp_indices"][j] = integer[1] * w + integer[0]
                target["hp_mask"][j] = 1
                target["hp_offset"][j] = point - integer
                draw_gaussian(target["hm_hp"][joint], integer, radius)
    return {key: torch.from_numpy(value) for key, value in target.items()}


class CompactKittiDataset(Dataset):
    def __init__(self, root, ids, training=False, max_objects=128):
        self.root, self.ids = Path(root), list(ids)
        self.training, self.max_objects = training, max_objects

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        sample = self.ids[index]
        base = self.root / "training"
        image = cv2.imread(str(base / "image_2" / f"{sample}.png"))
        if image is None:
            raise OSError(f"Cannot read KITTI image {sample}; preserve checkpoints and retry")
        p2 = parse_kitti_calib_file(base / "calib" / f"{sample}.txt")["P2"]
        if not np.isfinite(p2).all() or min(p2[0, 0], p2[1, 1]) <= 0:
            raise ValueError(f"Invalid P2 for {sample}")
        objects = parse_kitti_label_file(base / "label_2" / f"{sample}.txt")
        height, width = image.shape[:2]
        scale, shift = 1., (0., 0.)
        if self.training:
            scale = float(np.random.uniform(0.85, 1.15))
            shift = (np.random.uniform(-0.05, 0.05) * width,
                     np.random.uniform(-0.05, 0.05) * height)
        affine = image_affine(width, height, scale=scale, shift=shift)
        output_affine = affine / 4
        image = cv2.warpAffine(image, affine, INPUT_HW[::-1], flags=cv2.INTER_LINEAR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
        if self.training:
            image = np.clip(image * np.random.uniform(0.85, 1.15) + np.random.uniform(-0.05, 0.05), 0, 1)
        image = np.ascontiguousarray(((image - MEAN) / STD).transpose(2, 0, 1), dtype=np.float32)
        return {"sample_id": sample, "image": torch.from_numpy(image),
                "projection": torch.from_numpy(p2),
                "inverse_affine": torch.from_numpy(cv2.invertAffineTransform(output_affine)),
                "targets": build_targets(objects, p2, output_affine, max_objects=self.max_objects)}
