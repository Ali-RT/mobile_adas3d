from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"


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


def patch_dataset(repo: Path) -> None:
    target = repo / "lib/datasets/kitti/kitti_dataset.py"
    replace_once(
        target,
        "        self.writelist = cfg.get('writelist', ['Car'])\n"
        "        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT",
        "        self.writelist = cfg.get('writelist', ['Car'])\n"
        "        self.class_mapping = cfg.get('class_mapping', {})\n"
        "        invalid_targets = set(self.class_mapping.values()) - set(self.cls2id)\n"
        "        if invalid_targets:\n"
        "            raise ValueError(f'Unknown class_mapping targets: {sorted(invalid_targets)}')\n"
        "        # anno: use src annotations as GT, proj: use projected 2d bboxes as GT",
        "class mapping configuration",
    )
    replace_once(
        target,
        "        for i in range(object_num):\n"
        "            # filter objects by writelist\n"
        "            if objects[i].cls_type not in self.writelist:\n"
        "                continue",
        "        for i in range(object_num):\n"
        "            # Map source labels before filtering and native target encoding.\n"
        "            source_class = objects[i].cls_type\n"
        "            mapped_class = self.class_mapping.get(source_class, source_class)\n"
        "            if mapped_class not in self.writelist:\n"
        "                continue",
        "pre-filter source mapping",
    )
    replace_once(
        target,
        "            cls_id = self.cls2id[objects[i].cls_type]",
        "            cls_id = self.cls2id[mapped_class]",
        "mapped target encoding",
    )
    replace_once(
        target,
        "            mean_size = self.cls_mean_size[self.cls2id[objects[i].cls_type]]",
        "            mean_size = self.cls_mean_size[self.cls2id[mapped_class]]",
        "mapped mean-size lookup",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the frozen MobileADAS3D product taxonomy to pinned MonoDETR."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodetr_repo.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDETR {PINNED_COMMIT}, found {commit}")
    patch_dataset(repo)


if __name__ == "__main__":
    main()
