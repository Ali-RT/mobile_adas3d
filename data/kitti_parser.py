from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class KittiObject:
    class_name: str
    truncated: float
    occluded: int
    alpha: float
    bbox_2d: List[float]          # [x1, y1, x2, y2]
    dimensions_3d: List[float]    # [h, w, l]
    location_3d: List[float]      # [x, y, z] in camera coordinates
    rotation_y: float


@dataclass
class KittiSample:
    sample_id: str
    image_path: str
    label_path: str
    calib_path: str
    objects: List[KittiObject]
    P2: List[List[float]]
    K: List[List[float]]


def parse_kitti_label_file(
    label_path: str | Path,
    allowed_classes: Optional[List[str]] = None,
) -> List[KittiObject]:
    """
    Parse one KITTI label_2/*.txt file.

    KITTI label format:
    type truncated occluded alpha bbox_left bbox_top bbox_right bbox_bottom
    height width length x y z rotation_y
    """
    label_path = Path(label_path)

    if not label_path.exists():
        raise FileNotFoundError(f"Label file not found: {label_path}")

    objects: List[KittiObject] = []

    with label_path.open("r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) < 15:
            raise ValueError(
                f"Invalid KITTI label line in {label_path}: expected >=15 fields, got {len(parts)}"
            )

        class_name = parts[0]

        if allowed_classes is not None and class_name not in allowed_classes:
            continue

        truncated = float(parts[1])
        occluded = int(parts[2])
        alpha = float(parts[3])

        bbox_2d = [
            float(parts[4]),
            float(parts[5]),
            float(parts[6]),
            float(parts[7]),
        ]

        dimensions_3d = [
            float(parts[8]),   # height
            float(parts[9]),   # width
            float(parts[10]),  # length
        ]

        location_3d = [
            float(parts[11]),  # x
            float(parts[12]),  # y
            float(parts[13]),  # z
        ]

        rotation_y = float(parts[14])

        obj = KittiObject(
            class_name=class_name,
            truncated=truncated,
            occluded=occluded,
            alpha=alpha,
            bbox_2d=bbox_2d,
            dimensions_3d=dimensions_3d,
            location_3d=location_3d,
            rotation_y=rotation_y,
        )

        objects.append(obj)

    return objects


def parse_kitti_calib_file(calib_path: str | Path) -> Dict[str, np.ndarray]:
    """
    Parse KITTI calibration file.

    We mainly need P2, the projection matrix for camera image_2.
    P2 has shape [3, 4].

    Example line:
    P2: 7.215377e+02 0.000000e+00 ...
    """
    calib_path = Path(calib_path)

    if not calib_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {calib_path}")

    calib: Dict[str, np.ndarray] = {}

    with calib_path.open("r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if not line:
            continue

        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        values = np.array([float(x) for x in value.strip().split()], dtype=np.float32)

        if key.startswith("P"):
            calib[key] = values.reshape(3, 4)
        elif key in {"R0_rect", "R_rect"}:
            calib[key] = values.reshape(3, 3)
        elif key.startswith("Tr"):
            calib[key] = values.reshape(3, 4)
        else:
            calib[key] = values

    if "P2" not in calib:
        raise KeyError(f"P2 not found in calibration file: {calib_path}")

    return calib


def get_camera_intrinsics_from_P2(P2: np.ndarray) -> np.ndarray:
    """
    Extract approximate camera intrinsic matrix K from KITTI P2.

    P2:
      [fx  0 cx tx]
      [0  fy cy ty]
      [0   0  1  0]

    K:
      [fx  0 cx]
      [0  fy cy]
      [0   0  1]
    """
    if P2.shape != (3, 4):
        raise ValueError(f"Expected P2 shape [3, 4], got {P2.shape}")

    return P2[:, :3].copy()


def load_kitti_sample(
    root_dir: str | Path,
    sample_id: str,
    allowed_classes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Load one KITTI sample and convert it into our internal dictionary format.
    """
    root_dir = Path(root_dir)

    image_path = root_dir / "training" / "image_2" / f"{sample_id}.png"
    label_path = root_dir / "training" / "label_2" / f"{sample_id}.txt"
    calib_path = root_dir / "training" / "calib" / f"{sample_id}.txt"

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    objects = parse_kitti_label_file(
        label_path=label_path,
        allowed_classes=allowed_classes,
    )

    calib = parse_kitti_calib_file(calib_path)
    P2 = calib["P2"]
    K = get_camera_intrinsics_from_P2(P2)

    sample = KittiSample(
        sample_id=sample_id,
        image_path=str(image_path),
        label_path=str(label_path),
        calib_path=str(calib_path),
        objects=objects,
        P2=P2.tolist(),
        K=K.tolist(),
    )

    return asdict(sample)