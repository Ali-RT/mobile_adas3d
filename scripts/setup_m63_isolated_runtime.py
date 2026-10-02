"""Create a private M63 runtime; never install into Colab's shared interpreter."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import venv

PINS = ["numba==0.61.2", "llvmlite==0.44.0", "opencv-python-headless==5.0.0.93",
        "pyyaml==6.0.3", "scikit-image==0.25.2", "scikit-learn==1.6.1",
        "tqdm==4.67.3", "ninja==1.13.2", "pandas==2.2.3", "setuptools==80.10.2"]
PROBE = """import importlib.metadata as m,json,sys
print(json.dumps(dict(prefix=sys.prefix,base_prefix=sys.base_prefix,
 packages={d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()})))
"""
def inventory(python):
    return json.loads(subprocess.check_output([str(python),"-c",PROBE],text=True))
def validate_inventory(info):
    if info["prefix"]==info["base_prefix"]:
        raise RuntimeError("Not an isolated virtual environment")
    packages=info["packages"]
    banned=[n for n in packages if n in ("numba-cuda","cuda-python","cuda-core","nvidia-nvvm") or "cu13" in n]
    if banned:
        raise RuntimeError("Unexpected CUDA backend/packages: "+str(banned))
    if packages.get("numba")!="0.61.2" or packages.get("llvmlite")!="0.44.0":
        raise RuntimeError("Unexpected Numba/LLVM version")
def ensure_private_python(path):
    """Recover a partial venv without ensurepip or shared site-packages."""
    path=Path(path)
    config=path/"pyvenv.cfg"
    if config.exists():
        if "include-system-site-packages = false" not in config.read_text().lower():
            raise RuntimeError("Refusing an environment with shared packages")
    elif path.exists() and any(path.iterdir()):
        raise RuntimeError("Existing nonempty directory is not a venv; preserving "+str(path))
    python=path/"bin/python"
    if not config.exists() or not python.exists():
        venv.EnvBuilder(with_pip=False,system_site_packages=False,symlinks=True).create(path)
    if "include-system-site-packages = false" not in config.read_text().lower():
        raise RuntimeError("Venv isolation check failed")
    probe=subprocess.run([str(python),"-c",
        "import sys; assert sys.prefix != sys.base_prefix"],check=True)
    ready=subprocess.run([str(python),"-m","pip","--version"],
                         capture_output=True,text=True)
    if ready.returncode:
        # Host pip manages only the explicitly selected private interpreter.
        # This supported pip feature works when target Python has no ensurepip.
        command=[sys.executable,"-m","pip","--python",str(python),
                 "install","pip==25.2"]
        print("+"," ".join(command),flush=True)
        subprocess.run(command,check=True)
        subprocess.run([str(python),"-m","pip","--version"],check=True)
    return python

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--venv",type=Path,required=True)
    p.add_argument("--receipt",type=Path,required=True)
    a=p.parse_args()
    e=json.loads(a.manifest.read_text())["environment"]
    if sys.version.split()[0]!=e["python"] or e["cuda"]!="12.8" or e["opencv"]!="5.0.0":
        raise RuntimeError("Unexpected frozen Python/CUDA/OpenCV; stop for review")
    identity=dict(revision="M63H-ISOLATED-2026-10-02-r1",environment=e,
                  installer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  venv=str(a.venv),historical_numba_identity_known=False)
    python=ensure_private_python(a.venv)
    old=json.loads(a.receipt.read_text()) if a.receipt.exists() else None
    if old and old["identity"]!=identity:
        raise RuntimeError("Existing runtime receipt differs; preserve it for review")
    def pip(*args):
        subprocess.run([str(python),"-m","pip",*args],check=True)
    if old:
        # Recreate exactly the recorded package versions after a Colab reset.
        if inventory(python)["packages"]!=old["inventory"]["packages"]:
            lock=a.venv/"resolved-lock.txt"
            lock.write_text("\n".join(n+"=="+v for n,v in sorted(old["inventory"]["packages"].items())))
            pip("install","--extra-index-url","https://download.pytorch.org/whl/cu128","-r",str(lock))
    else:
        pip("install","--index-url","https://download.pytorch.org/whl/cu128",
            "torch=="+e["torch"],"torchvision=="+e["torchvision"])
        constraints=a.venv/"constraints.txt"
        constraints.write_text("torch=="+e["torch"]+"\ntorchvision=="+e["torchvision"]+"\n")
        req=PINS+[n+"=="+e[k] for n,k in (("numpy","numpy"),("scipy","scipy"),("pillow","pillow"),("timm","timm"))]
        pip("install","-c",str(constraints),*req)
    pip("check")
    info=inventory(python);validate_inventory(info)
    if old and info["packages"]!=old["inventory"]["packages"]:
        raise RuntimeError("Resolved runtime differs from recorded lock")
    receipt=dict(identity=identity,inventory=info)
    a.receipt.parent.mkdir(parents=True,exist_ok=True)
    if not old:
        temp=a.receipt.with_suffix(".tmp")
        temp.write_text(json.dumps(receipt,indent=2)+"\n");temp.replace(a.receipt)
    print("Isolated runtime ready:",python)
    print("Runtime receipt:",a.receipt)

if __name__=="__main__": main()
