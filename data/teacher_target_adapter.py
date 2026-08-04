from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import torch

from data.kitti_prediction_parser import parse_kitti_prediction_file
from data.matching import bbox_iou


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prediction_tree_sha256(
    prediction_dir: Path,
    sample_ids: Iterable[str],
) -> str:
    digest = hashlib.sha256()
    for sample_id in sample_ids:
        path = prediction_dir / f"{sample_id}.txt"
        if not path.is_file():
            raise FileNotFoundError(f"Teacher prediction missing: {path}")
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


class TeacherTargetAdapter:
    """Load a verified KITTI teacher cache and align geometry to GT objects."""

    def __init__(
        self,
        cache_dir: str | Path,
        split_file: str | Path,
        *,
        score_threshold: float = 0.30,
        match_iou_threshold: float = 0.50,
        max_gt_depth_m: float = 60.0,
        class_name: str = "Car",
        expected_checkpoint_sha256: str | None = None,
        expected_prediction_tree_sha256: str | None = None,
        verify_prediction_tree: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.prediction_dir = self.cache_dir / "predictions"
        self.split_file = Path(split_file)
        self.score_threshold = float(score_threshold)
        self.match_iou_threshold = float(match_iou_threshold)
        self.max_gt_depth_m = float(max_gt_depth_m)
        self.class_name = class_name

        self.sample_ids = [
            Path(line.strip()).stem
            for line in self.split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.sample_ids or len(self.sample_ids) != len(set(self.sample_ids)):
            raise ValueError(f"Split must contain unique sample IDs: {self.split_file}")
        self.sample_id_set = set(self.sample_ids)
        self.manifest = self._validate_manifest(
            expected_checkpoint_sha256=expected_checkpoint_sha256,
            expected_prediction_tree_sha256=expected_prediction_tree_sha256,
            verify_prediction_tree=verify_prediction_tree,
        )

    def _validate_manifest(
        self,
        *,
        expected_checkpoint_sha256: str | None,
        expected_prediction_tree_sha256: str | None,
        verify_prediction_tree: bool,
    ) -> Dict[str, Any]:
        manifest_path = self.cache_dir / "teacher_cache_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Teacher cache manifest missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
            raise RuntimeError("Teacher cache manifest is incomplete or unsupported")
        if manifest.get("inference_data_augmentation") is not False:
            raise RuntimeError("Teacher cache must come from deterministic clean inference")
        if manifest.get("split_images") != len(self.sample_ids):
            raise RuntimeError("Teacher cache and requested split have different sizes")
        if manifest.get("split_file_sha256") != _sha256_file(self.split_file):
            raise RuntimeError("Teacher cache split digest does not match requested split")
        if self.class_name not in manifest.get("allowed_classes", []):
            raise RuntimeError(f"Teacher cache does not contain class {self.class_name!r}")

        checkpoint_digest = manifest.get("checkpoint_sha256")
        if (
            expected_checkpoint_sha256 is not None
            and checkpoint_digest != expected_checkpoint_sha256
        ):
            raise RuntimeError("Teacher checkpoint digest does not match expectation")

        manifest_tree_digest = manifest.get("prediction_tree_sha256")
        if (
            expected_prediction_tree_sha256 is not None
            and manifest_tree_digest != expected_prediction_tree_sha256
        ):
            raise RuntimeError("Teacher prediction-tree digest does not match expectation")
        if verify_prediction_tree:
            actual_tree_digest = _prediction_tree_sha256(
                self.prediction_dir,
                self.sample_ids,
            )
            if actual_tree_digest != manifest_tree_digest:
                raise RuntimeError("Teacher prediction files do not match the manifest")
        return manifest

    def build_for_sample(
        self,
        sample_id: str,
        objects: Sequence[Mapping[str, Any]],
    ) -> Dict[str, torch.Tensor]:
        sample_id = Path(str(sample_id).strip()).stem
        if sample_id not in self.sample_id_set:
            raise ValueError(f"Sample {sample_id} is not in the verified teacher split")

        predictions = parse_kitti_prediction_file(
            self.prediction_dir / f"{sample_id}.txt",
            allowed_classes=[self.class_name],
        )
        candidates = sorted(
            (
                prediction
                for prediction in predictions
                if float(prediction["score"]) >= self.score_threshold
            ),
            key=lambda prediction: float(prediction["score"]),
            reverse=True,
        )
        unmatched_gt = {
            index
            for index, obj in enumerate(objects)
            if obj["class_name"] == self.class_name
        }

        count = len(objects)
        targets = {
            "teacher_association_mask": torch.zeros(count, dtype=torch.bool),
            "teacher_valid_mask": torch.zeros(count, dtype=torch.bool),
            "teacher_score": torch.zeros(count, dtype=torch.float32),
            "teacher_match_iou_2d": torch.zeros(count, dtype=torch.float32),
            "teacher_bbox_2d": torch.zeros((count, 4), dtype=torch.float32),
            "teacher_dimensions_3d": torch.zeros((count, 3), dtype=torch.float32),
            "teacher_location_3d": torch.zeros((count, 3), dtype=torch.float32),
            "teacher_yaw": torch.zeros(count, dtype=torch.float32),
        }

        for prediction in candidates:
            overlaps = [
                (bbox_iou(prediction["bbox_2d"], objects[index]["bbox_2d"]), index)
                for index in unmatched_gt
            ]
            if not overlaps:
                break
            overlap, gt_index = max(overlaps, key=lambda item: (item[0], -item[1]))
            if overlap < self.match_iou_threshold:
                continue
            unmatched_gt.remove(gt_index)
            targets["teacher_association_mask"][gt_index] = True
            if float(objects[gt_index]["location_3d"][2]) >= self.max_gt_depth_m:
                continue
            targets["teacher_valid_mask"][gt_index] = True
            targets["teacher_score"][gt_index] = float(prediction["score"])
            targets["teacher_match_iou_2d"][gt_index] = overlap
            targets["teacher_bbox_2d"][gt_index] = torch.tensor(
                prediction["bbox_2d"], dtype=torch.float32
            )
            targets["teacher_dimensions_3d"][gt_index] = torch.tensor(
                prediction["dimensions_3d_hwl"], dtype=torch.float32
            )
            targets["teacher_location_3d"][gt_index] = torch.tensor(
                prediction["location_3d"], dtype=torch.float32
            )
            targets["teacher_yaw"][gt_index] = float(prediction["yaw"])

        return targets


def pad_teacher_targets(
    targets: Sequence[Mapping[str, torch.Tensor]],
) -> Dict[str, torch.Tensor]:
    """Pad object-aligned teacher tensors to the largest sample in a batch."""
    if not targets:
        raise ValueError("Cannot pad an empty teacher-target batch")
    max_objects = max(int(target["teacher_valid_mask"].shape[0]) for target in targets)
    batched: Dict[str, List[torch.Tensor]] = {key: [] for key in targets[0]}
    for target in targets:
        count = int(target["teacher_valid_mask"].shape[0])
        for key, value in target.items():
            shape = (max_objects, *value.shape[1:])
            padded = torch.zeros(shape, dtype=value.dtype, device=value.device)
            padded[:count] = value
            batched[key].append(padded)
    batched["teacher_object_mask"] = []
    for target in targets:
        count = int(target["teacher_valid_mask"].shape[0])
        mask = torch.zeros(max_objects, dtype=torch.bool)
        mask[:count] = True
        batched["teacher_object_mask"].append(mask)
    return {key: torch.stack(values, dim=0) for key, values in batched.items()}
