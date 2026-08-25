from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]


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

    loss_source = PROJECT_ROOT / "third_party/monodetr/a1_distillation_loss.py"
    loss_target = repo / "lib/helpers/a1_distillation_loss.py"
    if loss_target.is_file() and loss_target.read_bytes() != loss_source.read_bytes():
        raise RuntimeError(f"Refusing to overwrite unexpected {loss_target}")
    shutil.copy2(loss_source, loss_target)

    trainer = repo / "lib/helpers/trainer_helper.py"
    replace_once(
        trainer,
        "from utils import misc\n",
        "from utils import misc\nfrom lib.helpers.a1_distillation_loss import compute_a1_distillation_losses\n",
        "trainer distillation import",
    )
    replace_once(
        trainer,
        "                 loss,\n                 model_name):",
        "                 loss,\n                 model_name,\n                 teacher_model=None,\n                 distillation_cfg=None):",
        "trainer distillation constructor",
    )
    replace_once(
        trainer,
        "        self.model = model\n        self.optimizer = optimizer",
        "        self.model = model\n        self.teacher_model = teacher_model\n        self.distillation_cfg = distillation_cfg or {}\n        self.optimizer = optimizer",
        "trainer teacher state",
    )
    replace_once(
        trainer,
        "            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=dn_args)\n            mask_dict=None",
        "            teacher_outputs = None\n            if self.teacher_model is not None:\n                self.teacher_model.eval()\n                with torch.no_grad():\n                    teacher_outputs = self.teacher_model(\n                        inputs, calibs, targets, img_sizes, dn_args=None)\n            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args=dn_args)\n            mask_dict=None",
        "trainer teacher forward",
    )
    replace_once(
        trainer,
        "            detr_losses = sum(detr_losses_dict_weighted)\n\n            detr_losses_dict = misc.reduce_dict(detr_losses_dict)",
        "            detr_losses = sum(detr_losses_dict_weighted)\n            teacher_losses = {}\n            if teacher_outputs is not None:\n                teacher_losses = compute_a1_distillation_losses(\n                    outputs, teacher_outputs, targets, self.detr_loss.matcher,\n                    self.distillation_cfg, student_group_num=self.detr_loss.group_num)\n                detr_losses = detr_losses + teacher_losses['distill_total']\n\n            detr_losses_dict = misc.reduce_dict(detr_losses_dict)",
        "trainer teacher loss",
    )
    replace_once(
        trainer,
        "            detr_losses_dict_log[\"loss_detr\"] = detr_losses_log\n            epoch_loss_sum += detr_losses_log",
        "            for key, value in teacher_losses.items():\n                detr_losses_dict_log[key] = value.detach().item()\n            detr_losses_log += detr_losses_dict_log.get('distill_total', 0.0)\n            detr_losses_dict_log[\"loss_detr\"] = detr_losses_log\n            epoch_loss_sum += detr_losses_log",
        "trainer distillation logging",
    )

    train_val = repo / "tools/train_val.py"
    replace_once(
        train_val,
        "    #  build optimizer\n    optimizer = build_optimizer(cfg['optimizer'], model)",
        "    teacher_model = None\n    distillation_cfg = cfg.get('distillation', {})\n    if distillation_cfg.get('enabled', False):\n        teacher_model, _ = build_model(distillation_cfg['teacher_model'])\n        teacher_model = teacher_model.to(device)\n        load_checkpoint(\n            model=teacher_model, optimizer=None,\n            filename=distillation_cfg['teacher_checkpoint'],\n            map_location=device, logger=logger)\n        teacher_model.eval()\n        for parameter in teacher_model.parameters():\n            parameter.requires_grad_(False)\n        logger.info('A1 distillation teacher loaded and frozen')\n\n    #  build optimizer\n    optimizer = build_optimizer(cfg['optimizer'], model)",
        "teacher construction",
    )
    replace_once(
        train_val,
        "from lib.helpers.scheduler_helper import build_lr_scheduler\n",
        "from lib.helpers.scheduler_helper import build_lr_scheduler\nfrom lib.helpers.save_helper import load_checkpoint\n",
        "teacher checkpoint import",
    )
    replace_once(
        train_val,
        "                      loss=loss,\n                      model_name=model_name)",
        "                      loss=loss,\n                      model_name=model_name,\n                      teacher_model=teacher_model,\n                      distillation_cfg=distillation_cfg)",
        "trainer teacher arguments",
    )
    print("MonoDETR A1 query distillation patch ready")


if __name__ == "__main__":
    main()
