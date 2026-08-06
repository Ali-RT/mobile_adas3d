from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
ORIGINAL_BACKBONE_SHA256 = "cea9a177c1c7c3566a7d00cff0b5edd7cdc669512ba07db5caeeb47ca5b7bd8a"
PATCHED_BACKBONE_SHA256 = "89da029e45636a6d4d258ef75fc42aff5e5d87eb3d33fc4840c8b25dace64cb6"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = PROJECT_ROOT / "third_party/monodetr/mobilenetv4_backbone.patch"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def patch_monodetr(repo: Path) -> str:
    repo = repo.resolve()
    target = repo / "lib/models/monodetr/backbone.py"
    if not target.is_file():
        raise FileNotFoundError(f"MonoDETR backbone source missing: {target}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDETR {PINNED_COMMIT}, found {commit}")

    current_digest = sha256_file(target)
    if current_digest == PATCHED_BACKBONE_SHA256:
        return "already_patched"
    if current_digest != ORIGINAL_BACKBONE_SHA256:
        raise RuntimeError(
            "Refusing to patch unexpected MonoDETR backbone source: "
            f"sha256={current_digest}"
        )
    subprocess.run(
        ["git", "apply", "--check", str(PATCH_PATH)], cwd=repo, check=True
    )
    subprocess.run(["git", "apply", str(PATCH_PATH)], cwd=repo, check=True)
    patched_digest = sha256_file(target)
    if patched_digest != PATCHED_BACKBONE_SHA256:
        raise RuntimeError(f"Patched backbone digest mismatch: {patched_digest}")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add the MobileNetV4 backbone option to pinned MonoDETR."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    args = parser.parse_args()
    result = patch_monodetr(args.monodetr_repo)
    print(f"MonoDETR MobileNetV4 backbone: {result}")
    print(f"Pinned commit: {PINNED_COMMIT}")
    print(f"Patched SHA-256: {PATCHED_BACKBONE_SHA256}")


if __name__ == "__main__":
    main()
