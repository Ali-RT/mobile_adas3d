from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
MOBILE_BACKBONE_SHA256 = "89da029e45636a6d4d258ef75fc42aff5e5d87eb3d33fc4840c8b25dace64cb6"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    parser = argparse.ArgumentParser(
        description="Enable an explicit stride-4 MobileNetV4 feature in MonoDETR."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodetr_repo.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDETR {PINNED_COMMIT}, found {commit}")

    backbone = repo / "lib/models/monodetr/backbone.py"
    current = sha256_file(backbone)
    already_patched = "backbone_expected_strides" in backbone.read_text(encoding="utf-8")
    if current != MOBILE_BACKBONE_SHA256 and not already_patched:
        raise RuntimeError(
            "Apply patch_monodetr_mobilenetv4.py first; "
            f"unexpected backbone SHA-256 {current}"
        )
    replace_once(
        backbone,
        "        expected_strides = [8, 16, 32] if return_interm_layers else [32]\n",
        "        expected_strides = list(cfg.get(\n"
        "            'backbone_expected_strides', [8, 16, 32]))\n"
        "        if not return_interm_layers:\n"
        "            expected_strides = [expected_strides[-1]]\n",
        "configurable MobileNetV4 feature strides",
    )
    print("MonoDETR A2f stride-4 feature patch ready")


if __name__ == "__main__":
    main()
