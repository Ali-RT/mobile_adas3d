from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import torch
from torch.utils.data import Dataset

from data.kitti_parser import load_kitti_sample


class KITTIDataset(Dataset):
    """
    PyTorch Dataset for KITTI object detection data.

    For now this returns:
      - image tensor
      - original image as numpy array
      - parsed objects
      - camera intrinsics K
      - projection matrix P2
      - sample_id

    Later we will add target maps for training.
    """

    def __init__(
        self,
        root_dir: str,
        classes: List[str],
        image_dir: str = "training/image_2",
        label_dir: str = "training/label_2",
        calib_dir: str = "training/calib",
        sample_ids: Optional[List[str]] = None,
        split_file: Optional[str] = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.classes = classes
        self.class_to_id = {name: idx for idx, name in enumerate(classes)}

        self.image_dir = self.root_dir / image_dir
        self.label_dir = self.root_dir / label_dir
        self.calib_dir = self.root_dir / calib_dir

        if sample_ids is not None and split_file is not None:
            raise ValueError("Use either sample_ids or split_file, not both.")

        if split_file is not None:
            from data.splits import read_split_file
            self.sample_ids = read_split_file(split_file)
        elif sample_ids is not None:
            self.sample_ids = sample_ids
        else:
            self.sample_ids = self._discover_sample_ids()

        self._validate_sample_files()
        
        if len(self.sample_ids) == 0:
            raise ValueError(
                f"No KITTI samples found in {self.image_dir}. "
                "Expected files like 000000.png"
            )

    def _validate_sample_files(self) -> None:
        missing = []

        for sample_id in self.sample_ids[:20]:
            image_path = self.image_dir / f"{sample_id}.png"
            label_path = self.label_dir / f"{sample_id}.txt"
            calib_path = self.calib_dir / f"{sample_id}.txt"

            if not image_path.exists():
                missing.append(str(image_path))

            if not label_path.exists():
                missing.append(str(label_path))

            if not calib_path.exists():
                missing.append(str(calib_path))

        if missing:
            raise FileNotFoundError(
                "Some sample files are missing. First missing files:\n"
                + "\n".join(missing[:20])
            )
    def _discover_sample_ids(self) -> List[str]:
        image_paths = sorted(self.image_dir.glob("*.png"))
        sample_ids = [p.stem for p in image_paths]
        return sample_ids

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample_id = self.sample_ids[index]

        sample = load_kitti_sample(
            root_dir=self.root_dir,
            sample_id=sample_id,
            allowed_classes=self.classes,
        )

        image_bgr = cv2.imread(sample["image_path"], cv2.IMREAD_COLOR)

        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {sample['image_path']}")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        # Convert image to torch tensor: [H, W, C] -> [C, H, W]
        image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float() / 255.0

        objects = sample["objects"]

        # Add class_id for each object.
        for obj in objects:
            obj["class_id"] = self.class_to_id[obj["class_name"]]

        height, width = image_rgb.shape[:2]

        return {
            "sample_id": sample_id,
            "image": image_tensor,
            "image_rgb": image_rgb,
            "image_path": sample["image_path"],
            "objects": objects,
            "K": torch.tensor(sample["K"], dtype=torch.float32),
            "P2": torch.tensor(sample["P2"], dtype=torch.float32),
            "original_size": {
                "height": height,
                "width": width,
            },
        }