"""Create a prospective isolated runtime, not a reconstruction of M62's runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

PACKAGES = [
    "numpy==2.2.6", "scipy==1.16.3", "pillow==11.3.0",
    "opencv-python-headless==4.12.0.88", "scikit-image==0.25.2",
    "scikit-learn==1.7.2", "numba==0.63.1", "timm==1.0.20",
    "PyYAML==6.0.2", "tqdm==4.67.1", "ninja==1.13.0", "gdown==5.2.0",
    "pandas==2.3.3", "setuptools==80.9.0", "thop==0.1.1.post2209072238",
]
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
TORCH_PACKAGES = ["torch==2.10.0", "torchvision==0.25.0"]


def run(command: list[str], env: dict[str, str]) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=env, check=True)


def runtime_env(cuda: Path, python_bin: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    drivers = [p for p in ("/usr/lib64-nvidia", "/usr/lib/x86_64-linux-gnu") if Path(p).is_dir()]
    # Preserve caller paths, but never choose CUDA link-time stubs as a driver.
    existing = [p for p in env.get("LD_LIBRARY_PATH", "").split(":") if p and "/stubs" not in p]
    env["LD_LIBRARY_PATH"] = ":".join(dict.fromkeys([*drivers, *existing, str(cuda / "lib64")]))
    env["CUDA_HOME"] = str(cuda)
    prefixes = ([str(python_bin)] if python_bin is not None else []) + [str(cuda / "bin")]
    env["PATH"] = ":".join([*prefixes, env.get("PATH", "")])
    env["MAX_JOBS"] = "2"
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if sys.version_info.major != 3 or not 10 <= sys.version_info.minor <= 13:
        raise RuntimeError("Use a Colab Python 3.10–3.13 GPU runtime")
    nvcc = args.cuda / "bin/nvcc"
    if not nvcc.is_file():
        raise RuntimeError("CUDA 13.0 toolkit not found; use a current GPU runtime, not an old CUDA override")
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "release 13.0," not in version:
        raise RuntimeError(f"This prospective recipe needs nvcc 13.0; found:\n{version}")
    env = runtime_env(args.cuda, args.venv.resolve() / "bin")
    run(["nvidia-smi"], env)
    venv = args.venv.resolve()
    python = venv / "bin/python"
    if not python.is_file():
        bootstrap = venv.parent / "m64_virtualenv_bootstrap"
        run([sys.executable, "-m", "pip", "install", "--target", str(bootstrap), "virtualenv==20.35.4"], env)
        bootstrap_env = dict(env, PYTHONPATH=str(bootstrap))
        run([sys.executable, "-m", "virtualenv", "--no-download", str(venv)], bootstrap_env)
    elif not (venv / "pyvenv.cfg").is_file():
        raise RuntimeError(f"Not an isolated virtual environment: {venv}")
    marker = venv / "m64_installer_identity.json"
    expected = dict(torch=TORCH_PACKAGES, index=TORCH_INDEX, packages=PACKAGES)
    if marker.exists() and json.loads(marker.read_text()) != expected:
        raise RuntimeError("Existing M64 venv belongs to another recipe; use a fresh venv, do not mutate it")
    # A first interrupted install is repairable. A completed recipe is reused.
    if not marker.exists():
        run([str(python), "-m", "pip", "install", *TORCH_PACKAGES, "--index-url", TORCH_INDEX], env)
        run([str(python), "-m", "pip", "install", *PACKAGES], env)
        marker.write_text(json.dumps(expected, indent=2) + "\n")
    run([str(python), "-m", "pip", "check"], env)
    probe = (
        "import torch,json; torch.cuda.init(); "
        "assert torch.__version__.startswith('2.10.0') and torch.version.cuda=='13.0'; "
        "print(json.dumps(dict(torch=torch.__version__,cuda=torch.version.cuda,"
        "gpu=torch.cuda.get_device_name(0))))"
    )
    run([str(python), "-c", probe], env)
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True).splitlines()
    receipt = dict(schema_version=1, complete=True, python=str(python), recipe=expected,
                   cuda_home=str(args.cuda.resolve()), nvcc=version, pip_freeze=freeze,
                   historical_environment_equivalence_claimed=False)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    if args.receipt.exists() and json.loads(args.receipt.read_text()) != receipt:
        raise RuntimeError("Saved runtime receipt differs; keep it and choose a new output RUN_ID")
    args.receipt.write_text(json.dumps(receipt, indent=2) + "\n")
    print("Prospective runtime ready. No historical manifest was edited.", flush=True)


if __name__ == "__main__":
    main()
