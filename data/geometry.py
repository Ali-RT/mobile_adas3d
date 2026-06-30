from typing import List, Tuple

import numpy as np
import math
import numpy as np
import torch


def compute_kitti_corners_3d_torch(dims_hwl, location_xyz, rotation_y):
    """
    Torch version of the KITTI cuboid corner construction.

    Uses the same convention as the NumPy versions in this file:
      dims_hwl:     [..., 3] = h, w, l
      location_xyz: [..., 3], KITTI bottom-center in camera coordinates
      rotation_y:   [...]     yaw around camera Y axis

    Returns:
      corners: [..., 8, 3]
    """
    h = dims_hwl[..., 0]
    w = dims_hwl[..., 1]
    l = dims_hwl[..., 2]

    zeros = torch.zeros_like(h)

    x_corners = torch.stack(
        [
            l / 2, l / 2, -l / 2, -l / 2,
            l / 2, l / 2, -l / 2, -l / 2,
        ],
        dim=-1,
    )

    y_corners = torch.stack(
        [
            zeros, zeros, zeros, zeros,
            -h, -h, -h, -h,
        ],
        dim=-1,
    )

    z_corners = torch.stack(
        [
            w / 2, -w / 2, -w / 2, w / 2,
            w / 2, -w / 2, -w / 2, w / 2,
        ],
        dim=-1,
    )

    # [..., 3, 8]
    corners = torch.stack([x_corners, y_corners, z_corners], dim=-2)

    c = torch.cos(rotation_y)
    s = torch.sin(rotation_y)
    ones = torch.ones_like(c)

    row0 = torch.stack([c, zeros, s], dim=-1)
    row1 = torch.stack([zeros, ones, zeros], dim=-1)
    row2 = torch.stack([-s, zeros, c], dim=-1)

    # [..., 3, 3]
    rot = torch.stack([row0, row1, row2], dim=-2)

    # [..., 3, 8] -> [..., 8, 3]
    corners_rot = torch.matmul(rot, corners).transpose(-1, -2)

    return corners_rot + location_xyz.unsqueeze(-2)


def project_points_p2_torch(points_3d, P2, min_z=0.1):
    """
    Torch projection of 3D camera-coordinate points using KITTI P2.

      points_3d: [..., N, 3]
      P2:        [3, 4] or [..., 3, 4]

    Returns:
      uv: [..., N, 2], NaN where projected depth is invalid
      valid: [..., N]
    """
    ones = torch.ones_like(points_3d[..., :1])
    points_h = torch.cat([points_3d, ones], dim=-1)

    if P2.dim() == 2:
        uvw = torch.matmul(points_h, P2.t())
    else:
        uvw = torch.matmul(points_h, P2.transpose(-1, -2))

    z = uvw[..., 2:3]
    valid = z > float(min_z)

    uv = torch.full_like(uvw[..., :2], float("nan"))
    uv = torch.where(
        valid.expand_as(uv),
        uvw[..., :2] / z.clamp(min=float(min_z)),
        uv,
    )

    return uv, valid.squeeze(-1)


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
    min_z: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Project 3D camera-coordinate points to image using KITTI P2.

    Args:
      points_3d: [N, 3]
      P2: [3, 4]

    Returns:
      points_2d: [N, 2]
      valid_mask: [N], True where projected depth is greater than min_z
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

    uvw = points_hom @ P2.T

    depths = uvw[:, 2]
    valid_mask = depths > float(min_z)

    points_2d = np.full((num_points, 2), np.nan, dtype=np.float32)
    points_2d[valid_mask] = uvw[valid_mask, :2] / depths[valid_mask, None]

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




def scale_p2_for_resize(P2, orig_w, orig_h, input_w, input_h):
    """
    Scale KITTI P2 from original image coordinates to resized model-input coordinates.
    """
    sx = float(input_w) / float(orig_w)
    sy = float(input_h) / float(orig_h)

    P2_scaled = P2.copy().astype(np.float32)
    P2_scaled[0, :] *= sx
    P2_scaled[1, :] *= sy

    return P2_scaled


def project_points_p2_numpy(points_3d, P2, min_z=0.1):
    """
    Project Nx3 camera-coordinate points to image coordinates using KITTI P2.

    Returns NaN for invalid points instead of clamping near-zero depth.
    """
    points_3d = np.asarray(points_3d, dtype=np.float32)

    points_h = np.concatenate(
        [points_3d, np.ones((points_3d.shape[0], 1), dtype=np.float32)],
        axis=1,
    )

    uvw = points_h @ P2.T
    z = uvw[:, 2]
    valid = z > float(min_z)

    uv = np.full((points_3d.shape[0], 2), np.nan, dtype=np.float32)
    uv[valid] = uvw[valid, :2] / z[valid, None]

    return uv.astype(np.float32), valid


def project_points_p2(points_3d, P2, min_z=0.1):
    """
    Compatibility wrapper returning only projected points.

    Invalid points are NaN; callers that need validity should use
    project_points_p2_numpy.
    """
    uv, _ = project_points_p2_numpy(points_3d, P2, min_z=min_z)

    return uv


def project_kitti_location(location_xyz, P2):
    """
    Project KITTI object bottom-center location [x, y, z] to image.
    """
    loc = np.asarray(location_xyz, dtype=np.float32).reshape(1, 3)
    uv = project_points_p2(loc, P2)[0]
    return uv.astype(np.float32)


def backproject_kitti_location_from_uv_depth(u, v, z, P2):
    """
    Recover camera-coordinate [x, y, z] from projected location pixel and depth.

    Uses:
      u = (fx*x + cx*z + tx) / z
      v = (fy*y + cy*z + ty) / z
    """
    fx = float(P2[0, 0])
    fy = float(P2[1, 1])
    cx = float(P2[0, 2])
    cy = float(P2[1, 2])
    tx = float(P2[0, 3])
    ty = float(P2[1, 3])

    z = float(z)

    x = ((float(u) - cx) * z - tx) / fx
    y = ((float(v) - cy) * z - ty) / fy

    return np.array([x, y, z], dtype=np.float32)


def compute_kitti_cuboid_corners_3d(dims_hwl, location_xyz, rotation_y):
    """
    KITTI convention:
      dims = [h, w, l]
      location_xyz = bottom center in camera coordinates
      rotation_y = yaw around camera Y axis
    """
    h, w, l = [float(x) for x in dims_hwl]
    x, y, z = [float(x) for x in location_xyz]

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

    R = np.array([
        [ c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=np.float32)

    corners_3d = R @ corners
    corners_3d = corners_3d + np.array([[x], [y], [z]], dtype=np.float32)

    return corners_3d.T.astype(np.float32)


def project_kitti_cuboid(dims_hwl, location_xyz, rotation_y, P2):
    corners_3d = compute_kitti_cuboid_corners_3d(
        dims_hwl=dims_hwl,
        location_xyz=location_xyz,
        rotation_y=rotation_y,
    )

    corners_2d = project_points_p2(corners_3d, P2)

    return corners_3d, corners_2d
