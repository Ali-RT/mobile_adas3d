from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


VERBOSE_RESUME_SHA256 = "6bfde1bccdd883110edff09dc018c4f1714d3d57dbb64736a08e2fb2693eda5f"
PATCHED_SHA256 = "c5e94a46ca2b3026ef6b14ddb830750145484a634d04eb8efaeb68e154dd5b7d"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PATCH = PROJECT_ROOT / "third_party/monodetr/checkpoint_metadata_order.patch"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize MonoDETR resumable checkpoint metadata after validation."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodetr_repo.resolve()
    target = repo / "lib/helpers/trainer_helper.py"
    current = digest(target)
    if current == PATCHED_SHA256:
        print("MonoDETR checkpoint metadata patch: already_patched")
        return
    if current != VERBOSE_RESUME_SHA256:
        raise RuntimeError(f"Unexpected trainer_helper.py SHA-256: {current}")
    subprocess.run(["git", "apply", "--check", str(PATCH)], cwd=repo, check=True)
    subprocess.run(["git", "apply", str(PATCH)], cwd=repo, check=True)
    patched = digest(target)
    if patched != PATCHED_SHA256:
        raise RuntimeError(f"Patched trainer_helper.py digest mismatch: {patched}")
    print("MonoDETR checkpoint metadata patch: patched")


if __name__ == "__main__":
    main()
