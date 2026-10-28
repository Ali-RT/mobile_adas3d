"""Reviewed positive-Z near-field geometry, independent of legacy staging.

No model dependencies. Frozen 2-80m preparation remains unchanged.
"""
from scripts import prepare_parking_lidar_depth as original_gt


def band_name(lo, hi):
    return f"{lo}-{hi}m"


def band_mask(z, lo, hi):
    return (z > 0) & (z >= lo) & ((z <= hi) if hi == 6 else (z < hi))


def project_positive(xyz, ls, lp, cp, cs, width, height):
    """Same Float32 transform sequence as reviewed GT, but no 1m cutoff."""
    import numpy as np
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError("Malformed point cloud")
    points = xyz.T.astype(np.float32, copy=True)
    for record in [ls, lp]:
        points[:] = original_gt.rotation(record["rotation"]) @ points
        points[:] += np.asarray(record["translation"])[:, None]
    world = points.copy()
    for record in [cp, cs]:
        points[:] -= np.asarray(record["translation"])[:, None]
        points[:] = original_gt.rotation(record["rotation"]).T @ points
    k = np.asarray(cs["camera_intrinsic"], dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError("Invalid intrinsic matrix")
    projected = k @ points
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = projected[:2] / projected[2:]
    z = points[2].copy()
    keep = (z > 0) & np.isfinite(uv).all(axis=0)
    keep &= (uv[0] > 1) & (uv[0] < width - 1) & (uv[1] > 1) & (uv[1] < height - 1)
    return uv[:, keep].T, z[keep], np.flatnonzero(keep), world


def sparse_near_raster(uv, z, semantic, source_indices, width, height, guard):
    """Exact existing square guard at near points, without dense image arrays.

    All projected returns, including invalid semantics and >6m returns, enter
    the z-buffer/guard before filtering. This prevents revealing hidden points.
    """
    import numpy as np
    from scipy.spatial import cKDTree
    if (uv.shape != (len(z), 2) or semantic.shape != z.shape or source_indices.shape != z.shape
            or not np.isfinite(uv).all() or not np.isfinite(z).all() or not (z > 0).all()):
        raise ValueError("Invalid projected point arrays")
    xy = np.rint(uv).astype(np.int32)
    keep = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    xy, z, semantic, source_indices = xy[keep], z[keep], semantic[keep], source_indices[keep]
    pixel = xy[:, 1] * width + xy[:, 0]
    order = np.lexsort((np.arange(len(z)), z, pixel))
    first = np.r_[True, pixel[order][1:] != pixel[order][:-1]] if len(order) else np.zeros(0, dtype=bool)
    chosen = order[first]
    xy, z, semantic, source_indices = xy[chosen], z[chosen], semantic[chosen], source_indices[chosen]
    near = np.flatnonzero((z > 0) & (z <= 6) & ~np.isin(semantic, [0, 31]))
    conservative = np.zeros(len(near), dtype=bool)
    window = guard["window_pixels"]
    if window < 1 or window % 2 != 1:
        raise ValueError("Boundary guard requires a positive odd window")
    if len(near):
        neighbours = cKDTree(xy).query_ball_point(xy[near], r=window // 2, p=np.inf)
        conservative = np.array([float(z[n].max() - z[n].min()) <= guard["maximum_local_depth_span_m"]
                                 for n in neighbours], dtype=bool)
    return dict(xy=xy[near], z=z[near], semantic=semantic[near],
                source_indices=source_indices[near], conservative=conservative)


def associate_objects(world, semantics, annotations):
    """Exact oriented boxes and exact semantic category; ambiguous -> -2."""
    import numpy as np
    owner = np.full(world.shape[1], -1, dtype=np.int32)
    for index, ann in enumerate(annotations):
        candidate = np.flatnonzero(semantics == ann["semantic_index"])
        if not len(candidate):
            continue
        local = (world[:, candidate].T - ann["translation"]) @ original_gt.rotation(ann["rotation"])
        # nuScenes box axes are length, width, height; metadata size is W/L/H.
        half_size = np.asarray(ann["size"])[[1, 0, 2]] / 2
        hit = candidate[(np.abs(local) <= half_size + 1e-6).all(axis=1)]
        previous = owner[hit].copy()
        owner[hit] = np.where(previous == -1, index, -2)
    return owner


def centre_in_camera(ann, cp, cs, width, height):
    import numpy as np
    point = np.asarray(ann["translation"], dtype=np.float64)
    for record in [cp, cs]:
        point = original_gt.rotation(record["rotation"]).T @ (point - record["translation"])
    z = float(point[2])
    if z <= 0:
        return z, False
    projected = np.asarray(cs["camera_intrinsic"]) @ point
    x, y = projected[:2] / projected[2]
    return z, bool(1 < x < width - 1 and 1 < y < height - 1)
