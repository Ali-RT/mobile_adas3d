"""Create M68's isolated CUDA reference and ONNX Runtime CPU environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

import m64_teacher_qualification as q
from setup_m64_runtime import PACKAGES, TORCH_INDEX, TORCH_PACKAGES, runtime_env

REVISION = "M68-A2-STANDALONE-ONNX-RUNTIME-2026-10-09-r2"
ONNX_PACKAGES = ["onnx==1.20.1", "onnxruntime==1.30.0"]
RECIPE = dict(torch=TORCH_PACKAGES, index=TORCH_INDEX,
              packages=PACKAGES, onnx=ONNX_PACKAGES)


def run(command, env):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=env, check=True)


def validate_receipt(value):
    body = {key: item for key, item in value.items() if key != "signature_sha256"}
    if (value.get("revision") != REVISION or value.get("complete") is not True
            or value.get("recipe") != RECIPE
            or q.signature(body) != value.get("signature_sha256")):
        raise RuntimeError("Invalid M68 runtime receipt; preserve it and use a new RUN_ID")
    # A saved inventory may supply package pins, never pip options or URLs.
    pins = value.get("pip_freeze", [])
    if not pins or any(not re.fullmatch(r"[A-Za-z0-9_.-]+==[A-Za-z0-9_.+!-]+", pin) for pin in pins):
        raise RuntimeError("M68 runtime inventory must contain exact package/version pins")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--cuda", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    venv, cuda = args.venv.resolve(), args.cuda.resolve()
    python = venv / "bin/python"
    nvcc = cuda / "bin/nvcc"
    if not nvcc.is_file():
        raise FileNotFoundError(f"Select a CUDA 13.0 Colab GPU runtime: missing {nvcc}")
    version = subprocess.check_output([str(nvcc), "--version"], text=True)
    if "release 13.0," not in version:
        raise RuntimeError(f"The tested A2 build recipe requires CUDA 13.0; found:\n{version}")
    env = runtime_env(cuda, python.parent)
    run(["nvidia-smi"], env)
    previous = validate_receipt(q.read_json(args.receipt)) if args.receipt.exists() else None
    if previous and Path(previous["python"]) != python:
        raise RuntimeError("M68 receipt belongs to a different virtual environment")
    if not python.is_file():
        bootstrap = venv.parent / "m68_virtualenv_bootstrap"
        run([sys.executable, "-m", "pip", "install", "--target", bootstrap,
             "virtualenv==20.35.4"], env)
        run([sys.executable, "-m", "virtualenv", "--no-download", venv],
            dict(env, PYTHONPATH=str(bootstrap)))
    if not (venv / "pyvenv.cfg").is_file():
        raise RuntimeError(f"M68 requires an isolated virtual environment: {venv}")
    if previous:
        # Restore this M68 run's own pins after a reset. Historical M67 packages
        # and receipts never enter this installer.
        run([python, "-m", "pip", "install", "--no-deps", *previous["pip_freeze"],
             "--extra-index-url", TORCH_INDEX], env)
    else:
        run([python, "-m", "pip", "install", *TORCH_PACKAGES, "--index-url", TORCH_INDEX], env)
        run([python, "-m", "pip", "install", *PACKAGES, *ONNX_PACKAGES], env)
    run([python, "-m", "pip", "check"], env)
    probe = (
        "import sys,json,torch,onnx,onnxruntime as ort;sys.path.insert(0,sys.argv[1]);"
        "from m64_teacher_qualification import environment;torch.cuda.init();"
        "assert torch.__version__.startswith('2.10.0') and torch.version.cuda=='13.0';"
        "assert onnx.__version__=='1.20.1' and ort.__version__=='1.30.0';"
        "print(json.dumps(environment(),sort_keys=True))"
    )
    current = json.loads(subprocess.check_output(
        [str(python), "-c", probe, str(Path(__file__).parent)], env=env, text=True))
    freeze = subprocess.check_output([str(python), "-m", "pip", "freeze"], env=env, text=True).splitlines()
    receipt = dict(schema_version=1, complete=True, revision=REVISION, recipe=RECIPE,
                   python=str(python), cuda_home=str(cuda), nvcc=version,
                   environment=current, onnx="1.20.1", onnxruntime="1.30.0", pip_freeze=freeze)
    receipt["signature_sha256"] = q.signature(receipt)
    if previous and previous != receipt:
        raise RuntimeError("This M68 run's runtime changed; preserve its results and use a new RUN_ID")
    q.write_json(args.receipt, receipt)
    print("M68 CUDA reference + ONNX Runtime CPU environment ready.", flush=True)


if __name__ == "__main__":
    main()
