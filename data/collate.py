from typing import Any, Dict, List

import torch
import torch.nn.functional as F

from data.target_builder import build_targets_for_sample


def mobile_adas3d_collate_fn(
    batch: List[Dict[str, Any]],
    classes: List[str],
    input_height: int,
    input_width: int,
    output_stride: int,
    class_mean_dims: Dict[str, List[float]],
) -> Dict[str, Any]:
    """
    Collate function for MobileADAS3D.

    Converts a list of dataset samples into:
      images:  [B, 3, H, W]
      targets: dict of [B, C, Hf, Wf] tensors
      metadata: sample-level info
    """
    images = []
    targets_list = []
    metadata = []

    for sample in batch:
        image = sample["image"].unsqueeze(0)

        image = F.interpolate(
            image,
            size=(input_height, input_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        targets = build_targets_for_sample(
            sample=sample,
            classes=classes,
            input_height=input_height,
            input_width=input_width,
            output_stride=output_stride,
            class_mean_dims=class_mean_dims,
        )

        images.append(image)
        targets_list.append(targets)

        metadata.append(
            {
                "sample_id": sample["sample_id"],
                "image_path": sample["image_path"],
                "original_size": sample["original_size"],
                "objects": sample["objects"],
                "K": sample["K"],
                "P2": sample["P2"],
            }
        )

    images_tensor = torch.stack(images, dim=0)

    batched_targets = {}

    for key in targets_list[0].keys():
        batched_targets[key] = torch.stack(
            [targets[key] for targets in targets_list],
            dim=0,
        )

    return {
        "images": images_tensor,
        "targets": batched_targets,
        "metadata": metadata,
    }