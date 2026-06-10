from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np


def draw_2d_boxes(
    image_rgb: np.ndarray,
    objects: List[Dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Draw 2D KITTI boxes on an RGB image and save output.

    image_rgb: RGB image, shape [H, W, 3]
    objects: parsed object dictionaries
    output_path: where to save visualization
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_vis = image_rgb.copy()

    for obj in objects:
        x1, y1, x2, y2 = obj["bbox_2d"]

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        class_name = obj["class_name"]
        depth = obj["location_3d"][2]

        label = f"{class_name} z={depth:.1f}m"

        cv2.rectangle(
            image_vis,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),
            thickness=2,
        )

        cv2.putText(
            image_vis,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    # Convert RGB back to BGR for OpenCV save.
    image_bgr = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)

def draw_projected_3d_boxes(
    image_rgb: np.ndarray,
    objects: List[Dict[str, Any]],
    P2: np.ndarray,
    output_path: str | Path,
) -> None:
    """
    Draw projected KITTI 3D boxes on an RGB image and save output.
    """
    from data.geometry import compute_projected_3d_box

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_vis = image_rgb.copy()

    # KITTI 3D box edge connections.
    # First 4 corners are bottom face.
    # Last 4 corners are top face.
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # top
        (0, 4), (1, 5), (2, 6), (3, 7),  # vertical
    ]

    for obj in objects:
        corners_3d, corners_2d, valid_mask = compute_projected_3d_box(
            dimensions_3d=obj["dimensions_3d"],
            location_3d=obj["location_3d"],
            rotation_y=obj["rotation_y"],
            P2=P2,
        )

        if not valid_mask.all():
            continue

        pts = corners_2d.astype(int)

        for start_idx, end_idx in edges:
            p1 = tuple(pts[start_idx])
            p2 = tuple(pts[end_idx])

            cv2.line(
                image_vis,
                p1,
                p2,
                color=(255, 0, 0),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        # Draw front face with thicker lines to show orientation.
        front_edges = [(0, 1), (1, 5), (5, 4), (4, 0)]
        for start_idx, end_idx in front_edges:
            p1 = tuple(pts[start_idx])
            p2 = tuple(pts[end_idx])

            cv2.line(
                image_vis,
                p1,
                p2,
                color=(0, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA,
            )

        x1, y1, _, _ = obj["bbox_2d"]
        label = f"{obj['class_name']} 3D"

        cv2.putText(
            image_vis,
            label,
            (int(x1), max(int(y1) - 25, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color=(255, 0, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    image_bgr = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)

def draw_predictions_2d(
    image_rgb: np.ndarray,
    predictions: List[Dict[str, Any]],
    output_path: str | Path,
) -> None:
    """
    Draw predicted 2D boxes, class, score, and depth.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image_vis = image_rgb.copy()

    for pred in predictions:
        x1, y1, x2, y2 = pred["bbox_2d"]

        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        label = (
            f"{pred['class_name']} "
            f"{pred['score']:.2f} "
            f"z={pred['depth']:.1f}m"
        )

        cv2.rectangle(
            image_vis,
            (x1, y1),
            (x2, y2),
            color=(0, 255, 0),
            thickness=2,
        )

        cv2.putText(
            image_vis,
            label,
            (x1, max(y1 - 8, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color=(0, 255, 0),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    image_bgr = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), image_bgr)