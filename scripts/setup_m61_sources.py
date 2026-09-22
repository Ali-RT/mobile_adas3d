"""Prepare separate pinned repos and repo-local CUDA extensions; never reset a checkout."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from m61_common import ROOT, STUDENT_COMMIT, TEACHER_COMMIT


def run(command, cwd=None):
    print("+", " ".join(map(str, command)), flush=True)
    env = os.environ.copy()
    env.setdefault("MAX_JOBS", "2")
    subprocess.run(list(map(str, command)), cwd=cwd, env=env, check=True)


def patch_active_architecture(path):
    text = path.read_text()
    flags = ["-arch=sm_60", "-gencode=arch=compute_60,code=sm_60",
             "-gencode=arch=compute_61,code=sm_61", "-gencode=arch=compute_70,code=sm_70",
             "-gencode=arch=compute_75,code=sm_75"]
    present = [f in text for f in flags]
    if not any(present):
        return
    if not all(present):
        raise RuntimeError("Unexpected mixed CUDA architecture patch state")
    for flag in flags:
        old = f'            "{flag}",\n'
        if text.count(old) != 1:
            raise RuntimeError(f"Unexpected CUDA flag: {flag}")
        text = text.replace(old, "")
    path.write_text(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--student-repo", type=Path, required=True)
    p.add_argument("--teacher-repo", type=Path, required=True)
    p.add_argument("--patch-only", action="store_true", help="Local source tests only; no CUDA claim")
    args = p.parse_args()
    repos = (
        (args.student_repo.resolve(), "ZrrSkywalker", "MonoDETR", STUDENT_COMMIT,
         "monodetr", ["patch_monodetr_colab_compat.py", "patch_monodetr_product_taxonomy.py", "patch_monodetr_mobilenetv4.py"]),
        (args.teacher_repo.resolve(), "PuFanqi23", "MonoDGP", TEACHER_COMMIT,
         "monodgp", ["patch_monodgp_colab_compat.py", "patch_monodgp_m54_training.py"]),
    )
    for repo, owner, name, commit, model, patches in repos:
        url = "/".join(["https:", "", "github.com", owner, name + ".git"])
        if not repo.exists():
            run(["git", "clone", url, repo])
        if not (repo / ".git").exists():
            raise RuntimeError(f"Not a git checkout; refusing to replace {repo}")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        if head != commit:
            if subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip():
                raise RuntimeError(f"Wrong commit with local changes: {repo}; use a new dedicated directory")
            run(["git", "fetch", "origin", commit], repo)
            run(["git", "checkout", "--detach", commit], repo)
        for patch in patches:
            run([sys.executable, ROOT / "scripts" / patch, "--" + model + "-repo", repo], ROOT)
        ops = repo / f"lib/models/{model}/ops"
        patch_active_architecture(ops / "setup.py")
        if not args.patch_only:
            run([sys.executable, "setup.py", "build_ext", "--inplace"], ops)
            probe = ("import sys; sys.path[:0]=" + repr([str(repo), str(ops)]) + "; "
                     "import torch, MultiScaleDeformableAttention; "
                     "from lib.helpers.model_helper import build_model; "
                     "assert torch.cuda.is_available(); print('Full model import passed:', torch.__version__)" )
            run([sys.executable, "-c", probe], repo)


if __name__ == "__main__":
    main()
