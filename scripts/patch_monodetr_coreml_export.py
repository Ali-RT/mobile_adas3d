from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH_PATH = PROJECT_ROOT / "third_party/monodetr/coreml_export.patch"
MARKER = "def ms_deform_attn_core_coreml("


def patch_monodetr(repo: Path) -> str:
    repo = repo.resolve()
    target = repo / "lib/models/monodetr/ops/modules/ms_deform_attn.py"
    if not target.is_file():
        raise FileNotFoundError(target)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDETR {PINNED_COMMIT}, found {commit}")
    target_text = target.read_text()
    if MARKER in target_text:
        return "already_patched"
    prerequisites = (
        "from torch.nn import Linear as _LinearWithBias",
        "from torch.overrides import has_torch_function, handle_torch_function",
    )
    if any(marker not in target_text for marker in prerequisites):
        raise RuntimeError("Apply patch_monodetr_colab_compat.py before Core ML patch")
    backbone = repo / "lib/models/monodetr/backbone.py"
    if "class TimmMobileNetV4Backbone" not in backbone.read_text():
        raise RuntimeError("Apply patch_monodetr_mobilenetv4.py before Core ML patch")
    apply_args = ["git", "apply", "--recount", "--unidiff-zero"]
    subprocess.run([*apply_args, "--check", str(PATCH_PATH)], cwd=repo, check=True)
    subprocess.run([*apply_args, str(PATCH_PATH)], cwd=repo, check=True)
    patched_text = target.read_text()
    if MARKER not in patched_text or "MSDeformAttnFunction.apply(" not in patched_text:
        raise RuntimeError("Core ML export patch marker missing after apply")
    return "patched"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add fixed-shape native Core ML export paths to pinned MonoDETR."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    args = parser.parse_args()
    print(f"MonoDETR Core ML export: {patch_monodetr(args.monodetr_repo)}")
    print(f"Pinned commit: {PINNED_COMMIT}")


if __name__ == "__main__":
    main()
