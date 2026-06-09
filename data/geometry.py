from typing import List, Tuple

import numpy as np


def compute_kitti_3d_box_corners(
    dimensions_3d: List[float],
    location_3d: List[float],
    rotation_y: float,
) -> np.ndarray:
    """
    Compute 8 corners of a KITTI 3D bounding box in camera coordinates.

    KITTI dimensions are:
      height, width, length

    KITTI location is the bottom center of the object in camera coordinates.

    Returns:
      corners_3d: [8, 3]
    """
    h, w, l = dimensions_3d
    x, y, z = location_3d

    # Corners relative to object bottom center.
    # x: left/right along object length
    # y: vertical, bottom=0, top=-h in KITTI camera coordinates
    # z: object width/depth direction
    x_corners = np.array(
        [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
        dtype=np.float32,
    )
    y_corners = np.array(
        [0, 0, 0, 0, -h, -h, -h, -h],
        dtype=np.float32,
    )
    z_corners = np.array(
        [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
        dtype=np.float32,
    )

    corners = np.stack([x_corners, y_corners, z_corners], axis=0)

    cos_ry = np.cos(rotation_y)
    sin_ry = np.sin(rotation_y)

    rotation_matrix = np.array(
        [
            [cos_ry, 0, sin_ry],
            [0, 1, 0],
            [-sin_ry, 0, cos_ry],
        ],
        dtype=np.float32,
    )

    corners_3d = rotation_matrix @ corners

    corners_3d[0, :] += x
    corners_3d[1, :] += y
    corners_3d[2, :] += z

    return corners_3d.T


def project_points_to_image(
    points_3d: np.ndarray,
    P2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D camera-coordinate points to image using KITTI P2.

    Args:
      points_3d: [N, 3]
      P2: [3, 4]

    Returns:
      points_2d: [N, 2]
      valid_mask: [N], True where depth > 0
    """
    if points_3d.ndim != 2 or points_3d.shape[1] != 3:
        raise ValueError(f"Expected points_3d shape [N, 3], got {points_3d.shape}")

    if P2.shape != (3, 4):
        raise ValueError(f"Expected P2 shape [3, 4], got {P2.shape}")

    num_points = points_3d.shape[0]

    points_hom = np.concatenate(
        [points_3d, np.ones((num_points, 1), dtype=np.float32)],
        axis=1,
    )

    projected = points_hom @ P2.T

    depths = projected[:, 2]
    valid_mask = depths > 1e-6

    points_2d = np.zeros((num_points, 2), dtype=np.float32)
    points_2d[valid_mask, 0] = projected[valid_mask, 0] / depths[valid_mask]
    points_2d[valid_mask, 1] = projected[valid_mask, 1] / depths[valid_mask]

    return points_2d, valid_mask


def compute_projected_3d_box(
    dimensions_3d: List[float],
    location_3d: List[float],
    rotation_y: float,
    P2: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Full KITTI 3D box projection.

    Returns:
      corners_3d: [8, 3]
      corners_2d: [8, 2]
      valid_mask: [8]
    """
    corners_3d = compute_kitti_3d_box_corners(
        dimensions_3d=dimensions_3d,
        location_3d=location_3d,
        rotation_y=rotation_y,
    )

    corners_2d, valid_mask = project_points_to_image(
        points_3d=corners_3d,
        P2=P2,
    )

    return corners_3d, corners_2d, valid_mask