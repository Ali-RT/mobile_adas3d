from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PINNED_COMMIT = "aa059a18214aebf644510e7f0793971b403f9d14"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    old_count = text.count(old)
    new_count = text.count(new)
    if old_count == 1 and new_count == 0:
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"patched {label}")
        return
    if old_count == 0 and new_count == 1:
        print(f"already patched {label}")
        return
    raise RuntimeError(
        f"Unexpected {label} source in {path}: old={old_count}, new={new_count}"
    )


def replace_count(path: Path, old: str, new: str, expected: int, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    old_count = text.count(old)
    new_count = text.count(new)
    if old_count == expected and new_count == 0:
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"patched {label} ({expected} occurrences)")
        return
    if old_count == 0 and new_count == expected:
        print(f"already patched {label}")
        return
    raise RuntimeError(
        f"Unexpected {label} source in {path}: old={old_count}, new={new_count}"
    )


def patch_dataset(repo: Path) -> None:
    path = repo / "lib/datasets/kitti/kitti_dataset.py"
    replace_once(
        path,
        "        self.writelist = cfg.get('writelist', ['Car'])\n"
        "        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT",
        "        self.writelist = cfg.get('writelist', ['Car'])\n"
        "        self.class_mapping = cfg.get('class_mapping', {})\n"
        "        invalid_targets = set(self.class_mapping.values()) - set(self.cls2id)\n"
        "        if invalid_targets:\n"
        "            raise ValueError(f'Unknown class_mapping targets: {sorted(invalid_targets)}')\n"
        "        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT",
        "dataset class-mapping configuration",
    )
    replace_once(
        path,
        "        for i in range(object_num):\n"
        "            # filter objects by writelist\n"
        "            if objects[i].cls_type not in self.writelist:\n"
        "                continue",
        "        for i in range(object_num):\n"
        "            # Map source taxonomy before native target encoding.\n"
        "            source_class = objects[i].cls_type\n"
        "            mapped_class = self.class_mapping.get(source_class, source_class)\n"
        "            if mapped_class not in self.writelist:\n"
        "                continue",
        "primary target mapping",
    )
    replace_once(
        path,
        "                for i in range(object_num_temp):\n"
        "                    if objects[i].cls_type not in self.writelist:\n"
        "                        continue",
        "                for i in range(object_num_temp):\n"
        "                    source_class = objects[i].cls_type\n"
        "                    mapped_class = self.class_mapping.get(source_class, source_class)\n"
        "                    if mapped_class not in self.writelist:\n"
        "                        continue",
        "mixup target mapping",
    )
    replace_count(
        path,
        "self.cls2id[objects[i].cls_type]",
        "self.cls2id[mapped_class]",
        4,
        "mapped class IDs and mean-size lookups",
    )


def patch_trainer(repo: Path) -> None:
    path = repo / "lib/helpers/trainer_helper.py"
    replace_once(
        path,
        "        if cfg.get('resume_model', None):\n"
        "            resume_model_path = os.path.join(self.output_dir, \"checkpoint.pth\")",
        "        resume_model = cfg.get('resume_model', None)\n"
        "        if resume_model:\n"
        "            resume_model_path = (resume_model if isinstance(resume_model, str)\n"
        "                                 else os.path.join(self.output_dir, \"checkpoint.pth\"))",
        "explicit resume-checkpoint path",
    )
    replace_once(
        path,
        "            if batch_idx % 30 == 0:",
        "            if batch_idx % int(self.cfg.get('log_frequency', 20)) == 0:",
        "configurable batch logging",
    )
    replace_once(
        path,
        "            detr_losses.backward()\n"
        "            self.optimizer.step()",
        "            if not torch.isfinite(detr_losses):\n"
        "                raise RuntimeError(\n"
        "                    f'Non-finite loss at epoch={epoch + 1} batch={batch_idx}: '\n"
        "                    f'{float(detr_losses.detach().cpu())}'\n"
        "                )\n"
        "            detr_losses.backward()\n"
        "            finite_gradients = all(\n"
        "                parameter.grad is None or torch.isfinite(parameter.grad).all()\n"
        "                for parameter in self.model.parameters()\n"
        "            )\n"
        "            if not finite_gradients:\n"
        "                raise RuntimeError(\n"
        "                    f'Non-finite gradients at epoch={epoch + 1} batch={batch_idx}'\n"
        "                )\n"
        "            self.optimizer.step()\n"
        "            progress_bar.set_postfix(loss=f'{float(detr_losses.detach().cpu()):.4f}')",
        "finite-loss and gradient guard",
    )


def patch_training_entrypoint(repo: Path) -> None:
    path = repo / "tools/train_val.py"
    replace_once(
        path,
        "    if cfg['dataset']['test_split'] != 'test':\n"
        "        trainer.tester = tester",
        "    if (cfg['dataset']['test_split'] != 'test' and\n"
        "            cfg['trainer'].get('evaluate_during_training', True)):\n"
        "        trainer.tester = tester",
        "optional in-training evaluation",
    )
    replace_once(
        path,
        "    if cfg['dataset']['test_split'] == 'test':\n"
        "        return",
        "    if (cfg['dataset']['test_split'] == 'test' or\n"
        "            not cfg['trainer'].get('evaluate_after_training', True)):\n"
        "        return",
        "optional final evaluation",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply fail-closed M54 taxonomy, resume, and training guards."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodgp_repo.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP commit {PINNED_COMMIT}, found {commit}")
    patch_dataset(repo)
    patch_trainer(repo)
    patch_training_entrypoint(repo)


if __name__ == "__main__":
    main()
