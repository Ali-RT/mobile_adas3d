"""Reuse the working GPU recipe and add the pinned converter in isolation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from setup_m64_runtime import runtime_env
from m64_teacher_qualification import signature, write_json

ROOT = Path(__file__).resolve().parents[1]
REVISION = "M67-A2-EXPORT-RUNTIME-AUDIT-2026-10-06-r1"


def save_receipt(path, receipt):
    """Never replace a completed run's dependency identity."""
    if path.exists() and json.loads(path.read_text()) != receipt:
        raise RuntimeError("M67 runtime differs; preserve the run and choose a new RUN_ID")
    write_json(path, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    python = args.venv.resolve() / "bin/python"
    env = runtime_env(args.cuda, python.parent)
    # The reusable base installer inventories all installed packages. Its
    # temporary receipt must not collide with the later converter inventory.
    with tempfile.TemporaryDirectory(prefix="m67_base_runtime_") as temporary:
        base = Path(temporary) / "base.json"
        subprocess.run([sys.executable, str(ROOT / "scripts/setup_m64_runtime.py"),
                        "--venv", str(args.venv), "--cuda", str(args.cuda),
                        "--receipt", str(base)], env=env, check=True)
        base_receipt = json.loads(base.read_text())
    subprocess.run([str(python), "-m", "pip", "install", "coremltools==9.0"], env=env, check=True)
    subprocess.run([str(python), "-m", "pip", "check"], env=env, check=True)
    probe = "import coremltools as ct; assert ct.__version__=='9.0'; print(ct.__version__)"
    subprocess.run([str(python), "-c", probe], env=env, check=True)
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True).splitlines()
    receipt = dict(schema_version=1, complete=True, revision=REVISION, python=str(python),
                   base_recipe=base_receipt["recipe"], cuda_home=base_receipt["cuda_home"],
                   nvcc=base_receipt["nvcc"], coremltools="9.0", pip_freeze=freeze,
                   historical_environment_equivalence_claimed=False)
    receipt["signature_sha256"] = signature(receipt)
    save_receipt(args.receipt, receipt)
    print("M67 isolated CUDA13 + converter ready; host packages and historical receipts untouched.", flush=True)


if __name__ == "__main__":
    main()
