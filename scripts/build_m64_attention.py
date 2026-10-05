"""Build each M64 attention extension in its own checkout and isolated runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--role", choices=("a2", "teacher"), required=True)
    args = parser.parse_args()
    import torch
    torch.cuda.init()
    if torch.version.cuda != "13.0":
        raise RuntimeError("Build with the M64 CUDA 13.0 isolated interpreter")
    repo = args.repo.resolve()
    name = "monodetr" if args.role == "a2" else "monoprio"
    ops = repo / f"lib/models/{name}/ops"
    if not (ops / "setup.py").is_file():
        raise FileNotFoundError(ops / "setup.py")
    source = {str(p.relative_to(ops)): sha(p) for p in sorted(ops.rglob("*"))
              if p.is_file() and p.suffix in {".py", ".cu", ".cpp", ".h", ".cuh"}
              and "build" not in p.relative_to(ops).parts}
    capability = torch.cuda.get_device_capability(0)
    arch = f"{capability[0]}.{capability[1]}"
    identity = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
                    cuda_home=os.environ.get("CUDA_HOME"), arch=arch, sources=source)
    receipt_path = ops / "m64_build_receipt.json"
    previous = json.loads(receipt_path.read_text()) if receipt_path.exists() else None
    binaries = sorted(ops.glob("MultiScaleDeformableAttention*.so"))
    valid = (previous and previous["identity"] == identity and len(binaries) == 1
             and previous.get("binary_sha256") == sha(binaries[0]))
    if not valid:
        # Recompile only generated outputs in this prospective model checkout.
        # Do not use a shared pip-installed extension or change model source.
        if len(binaries) > 1:
            raise RuntimeError("Multiple ABI builds in checkout; use a new M64 checkout path")
        env = dict(os.environ, TORCH_CUDA_ARCH_LIST=arch, MAX_JOBS="2")
        command = [sys.executable, "setup.py", "build_ext", "--inplace", "--force"]
        print("+", " ".join(command), flush=True)
        subprocess.run(command, cwd=ops, env=env, check=True)
        binaries = sorted(ops.glob("MultiScaleDeformableAttention*.so"))
        if len(binaries) != 1:
            raise RuntimeError("Expected exactly one locally built attention binary")
    # Import-only verification here. The next stage tests actual forward/backward.
    probe = "import sys,torch;sys.path.insert(0,sys.argv[1]);import MultiScaleDeformableAttention as m;print(m.__file__)"
    resolved = subprocess.check_output([sys.executable, "-X", "faulthandler", "-c", probe, str(ops)],
                                       text=True).strip()
    if Path(resolved).resolve() != binaries[0].resolve():
        raise RuntimeError("Attention import resolved to another checkout")
    receipt_path.write_text(json.dumps(dict(identity=identity, binary=str(binaries[0]),
                           binary_sha256=sha(binaries[0]), import_verified=True), indent=2) + "\n")
    print(f"{args.role}: attention binary ready at {resolved}", flush=True)


if __name__ == "__main__":
    main()
