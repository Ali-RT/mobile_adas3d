from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


ORIGINAL_SHA256 = "9d01c57cd65aa6080b94dc3a8d1ab37fb1e45ed96993ddb03b25798f38682ff5"
PATCHED_SHA256 = "6bfde1bccdd883110edff09dc018c4f1714d3d57dbb64736a08e2fb2693eda5f"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH = PROJECT_ROOT / "third_party/monodetr/verbose_resume.patch"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodetr_repo.resolve()
    target = repo / "lib/helpers/trainer_helper.py"
    current = digest(target)
    if current == PATCHED_SHA256:
        print("MonoDETR verbose/resume patch: already_patched")
        return
    if current != ORIGINAL_SHA256:
        raise RuntimeError(f"Unexpected trainer_helper.py SHA-256: {current}")
    subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=repo, check=True)
    subprocess.run(["git", "apply", str(PATCH)], cwd=repo, check=True)
    if digest(target) != PATCHED_SHA256:
        raise RuntimeError("Patched trainer_helper.py digest mismatch")
    print("MonoDETR verbose/resume patch: patched")


if __name__ == "__main__":
    main()
