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

    focal = repo / "lib/losses/focal_loss.py"
    replace_once(
        focal,
        "def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):",
        "def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2, positive_class_weights=None):",
        "focal positive-class argument",
    )
    replace_once(
        focal,
        "    return loss.mean(1).sum() / num_boxes\n",
        "    if positive_class_weights is not None:\n"
        "        weights = torch.as_tensor(positive_class_weights, device=inputs.device, dtype=inputs.dtype)\n"
        "        if weights.numel() != inputs.shape[-1]:\n"
        "            raise ValueError(f'Expected {inputs.shape[-1]} class weights, found {weights.numel()}')\n"
        "        loss = loss * (1.0 + targets * (weights.view(1, 1, -1) - 1.0))\n"
        "    return loss.mean(1).sum() / num_boxes\n",
        "focal positive-class weighting",
    )

    model = repo / "lib/models/monodetr/monodetr.py"
    replace_once(
        model,
        "    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, group_num=11):",
        "    def __init__(self, num_classes, matcher, weight_dict, focal_alpha, losses, group_num=11, positive_class_weights=None):",
        "criterion positive-class argument",
    )
    replace_once(
        model,
        "        self.focal_alpha = focal_alpha\n        self.ddn_loss = DDNLoss()",
        "        self.focal_alpha = focal_alpha\n        self.positive_class_weights = positive_class_weights\n        self.ddn_loss = DDNLoss()",
        "criterion positive-class state",
    )
    replace_once(
        model,
        "        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]",
        "        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2, positive_class_weights=self.positive_class_weights) * src_logits.shape[1]",
        "criterion weighted focal call",
    )
    replace_once(
        model,
        "        focal_alpha=cfg['focal_alpha'],\n        losses=losses)",
        "        focal_alpha=cfg['focal_alpha'],\n        losses=losses,\n        positive_class_weights=cfg.get('positive_class_weights'))",
        "criterion weighted focal config",
    )
    print("MonoDETR class-positive focal weighting patch ready")


if __name__ == "__main__":
    main()
