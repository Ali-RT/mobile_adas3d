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
        f"Unexpected {label} source in {path}: old={text.count(old)}, new={text.count(new)}"
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

    matcher = repo / "lib/models/monodetr/matcher.py"
    replace_once(
        matcher,
        "    def __init__(self, cost_class: float = 1, cost_3dcenter: float = 1, cost_bbox: float = 1, cost_giou: float = 1):",
        "    def __init__(self, cost_class: float = 1, cost_3dcenter: float = 1, cost_bbox: float = 1, cost_giou: float = 1, pedestrian_localization_cost_weight: float = 1.0):",
        "matcher Pedestrian localization-cost argument",
    )
    replace_once(
        matcher,
        "        self.cost_3dcenter = cost_3dcenter\n",
        "        self.cost_3dcenter = cost_3dcenter\n"
        "        self.pedestrian_localization_cost_weight = float(pedestrian_localization_cost_weight)\n"
        "        if self.pedestrian_localization_cost_weight < 1.0:\n"
        "            raise ValueError('pedestrian_localization_cost_weight must be >= 1.0')\n",
        "matcher Pedestrian localization-cost state",
    )
    replace_once(
        matcher,
        "        cost_giou = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox), box_cxcylrtb_to_xyxy(tgt_bbox))\n\n        # Final cost matrix",
        "        cost_giou = -generalized_box_iou(box_cxcylrtb_to_xyxy(out_bbox), box_cxcylrtb_to_xyxy(tgt_bbox))\n"
        "        target_localization_weights = torch.ones_like(tgt_ids, dtype=cost_bbox.dtype)\n"
        "        target_localization_weights = torch.where(\n"
        "            tgt_ids == 0,\n"
        "            target_localization_weights * self.pedestrian_localization_cost_weight,\n"
        "            target_localization_weights)\n\n"
        "        # Final cost matrix",
        "matcher target localization weights",
    )
    replace_once(
        matcher,
        "        C = self.cost_bbox * cost_bbox + self.cost_3dcenter * cost_3dcenter + self.cost_class * cost_class + self.cost_giou * cost_giou",
        "        C = (self.cost_bbox * cost_bbox * target_localization_weights[None, :] + self.cost_class * cost_class + self.cost_3dcenter * cost_3dcenter + self.cost_giou * cost_giou * target_localization_weights[None, :])",
        "weighted matcher box and GIoU costs",
    )
    replace_once(
        matcher,
        "        cost_giou=cfg['set_cost_giou'])",
        "        cost_giou=cfg['set_cost_giou'],\n"
        "        pedestrian_localization_cost_weight=cfg.get('pedestrian_localization_cost_weight', 1.0))",
        "matcher Pedestrian localization-cost config",
    )
    print("MonoDETR Pedestrian Hungarian localization-cost patch ready")


if __name__ == "__main__":
    main()
