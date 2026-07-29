from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_kitti_prediction_file(
    prediction_path: str | Path,
    allowed_classes: Iterable[str] | None = None,
) -> List[Dict[str, Any]]:
    """Parse a KITTI detection-result file with a required score column."""
    prediction_path = Path(prediction_path)
    if not prediction_path.is_file():
        raise FileNotFoundError(f"Prediction file not found: {prediction_path}")

    allowed = set(allowed_classes) if allowed_classes is not None else None
    predictions: List[Dict[str, Any]] = []

    for line_number, raw_line in enumerate(
        prediction_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 16:
            raise ValueError(
                f"Invalid KITTI prediction in {prediction_path}:{line_number}: "
                f"expected at least 16 fields including score, got {len(parts)}"
            )

        class_name = parts[0]
        if allowed is not None and class_name not in allowed:
            continue

        predictions.append(
            {
                "class_name": class_name,
                "truncated": float(parts[1]),
                "occluded": int(float(parts[2])),
                "alpha": float(parts[3]),
                "bbox_2d": [float(value) for value in parts[4:8]],
                "dimensions_3d_hwl": [float(value) for value in parts[8:11]],
                "location_3d": [float(value) for value in parts[11:14]],
                "rotation_y": float(parts[14]),
                "yaw": float(parts[14]),
                "score": float(parts[15]),
            }
        )

    return predictions


def load_kitti_prediction_directory(
    prediction_dir: str | Path,
    sample_ids: Iterable[str],
    allowed_classes: Iterable[str] | None = None,
    require_all_files: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load per-frame KITTI predictions and optionally require every split file."""
    prediction_dir = Path(prediction_dir)
    if not prediction_dir.is_dir():
        raise FileNotFoundError(f"Prediction directory not found: {prediction_dir}")

    predictions: Dict[str, List[Dict[str, Any]]] = {}
    missing: List[str] = []
    for sample_id in sample_ids:
        normalized_id = Path(str(sample_id).strip()).stem
        prediction_path = prediction_dir / f"{normalized_id}.txt"
        if not prediction_path.is_file():
            missing.append(normalized_id)
            predictions[normalized_id] = []
            continue
        predictions[normalized_id] = parse_kitti_prediction_file(
            prediction_path,
            allowed_classes=allowed_classes,
        )

    if require_all_files and missing:
        preview = ", ".join(missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} prediction files under {prediction_dir}; "
            f"first missing IDs: {preview}"
        )

    return predictions
