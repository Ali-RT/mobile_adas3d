from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(new) == 1:
        print(f"already patched {label}")
        return
    if text.count(old) == 1:
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"patched {label}")
        return
    raise RuntimeError(
        f"Unexpected {label} source in {path}: "
        f"old={text.count(old)}, new={text.count(new)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodetr_repo.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDETR {PINNED_COMMIT}, found {commit}")

    model = repo / "lib/models/monodetr/monodetr.py"
    replace_once(
        model,
        "    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, group_num=11):",
        "    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, group_num=11, small_pedestrian_box_loss_weight=1.0, small_pedestrian_box_height_norm=0.16666666666666666):",
        "criterion small-Pedestrian box-weight argument",
    )
    replace_once(
        model,
        "        self.focal_alpha = focal_alpha\n        self.ddn_loss = DDNLoss()",
        "        self.focal_alpha = focal_alpha\n"
        "        self.small_pedestrian_box_loss_weight = float(small_pedestrian_box_loss_weight)\n"
        "        if self.small_pedestrian_box_loss_weight < 1.0:\n"
        "            raise ValueError('small_pedestrian_box_loss_weight must be >= 1.0')\n"
        "        self.small_pedestrian_box_height_norm = float(small_pedestrian_box_height_norm)\n"
        "        if not 0.0 < self.small_pedestrian_box_height_norm <= 1.0:\n"
        "            raise ValueError('small_pedestrian_box_height_norm must be in (0, 1]')\n"
        "        self.ddn_loss = DDNLoss()",
        "criterion small-Pedestrian box-weight state",
    )
    replace_once(
        model,
        "        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)\n\n        # l1",
        "        target_2dboxes = torch.cat([t['boxes_3d'][:, 2: 6][i] for t, (_, i) in zip(targets, indices)], dim=0)\n"
        "        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0).reshape(-1).long()\n"
        "        target_box_heights = target_2dboxes[:, 2] + target_2dboxes[:, 3]\n"
        "        small_pedestrian_mask = (target_labels == 0) & (target_box_heights <= self.small_pedestrian_box_height_norm)\n"
        "        pair_weights = torch.ones_like(target_labels, dtype=src_2dboxes.dtype)\n"
        "        pair_weights = torch.where(\n"
        "            small_pedestrian_mask,\n"
        "            pair_weights * self.small_pedestrian_box_loss_weight,\n"
        "            pair_weights)\n\n"
        "        # l1",
        "matched small-Pedestrian pair weights",
    )
    replace_once(
        model,
        "        losses['loss_bbox'] = loss_bbox.sum() / num_boxes",
        "        losses['loss_bbox'] = (loss_bbox * pair_weights[:, None]).sum() / num_boxes",
        "weighted box L1",
    )
    replace_once(
        model,
        "        losses['loss_giou'] = loss_giou.sum() / num_boxes",
        "        losses['loss_giou'] = (loss_giou * pair_weights).sum() / num_boxes",
        "weighted box GIoU",
    )
    replace_once(
        model,
        "        focal_alpha=cfg['focal_alpha'],\n        losses=losses)",
        "        focal_alpha=cfg['focal_alpha'],\n"
        "        losses=losses,\n"
        "        small_pedestrian_box_loss_weight=cfg.get('small_pedestrian_box_loss_weight', 1.0),\n"
        "        small_pedestrian_box_height_norm=cfg.get('small_pedestrian_box_height_norm', 64.0 / 384.0))",
        "criterion small-Pedestrian box-weight config",
    )
    print("MonoDETR matched small-Pedestrian 2D box weighting patch ready")


if __name__ == "__main__":
    main()
