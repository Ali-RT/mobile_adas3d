"""Restore M63's recorded package versions without changing frozen manifests."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

OPENCV_PACKAGES = ("opencv-python", "opencv-python-headless",
                   "opencv-contrib-python", "opencv-contrib-python-headless")

def torch_index(env):
    tag = "cu" + env["cuda"].replace(".", "")
    if not env["torch"].endswith("+" + tag) or not env["torchvision"].endswith("+" + tag):
        raise RuntimeError("Inconsistent frozen torch/CUDA versions")
    return "https://download.pytorch.org/whl/" + tag

def call(args):
    print("+", " ".join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)

def cv_version():
    r = subprocess.run([sys.executable, "-c", "import cv2; print(cv2.__version__)"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    a = p.parse_args()
    env = json.loads(a.manifest.read_text())["environment"]
    if sys.version.split()[0] != env["python"]:
        raise RuntimeError("Python differs from frozen runtime: need " + env["python"])
    index = torch_index(env)
    pip = [sys.executable, "-m", "pip"]
    with tempfile.TemporaryDirectory(prefix="m63_packages_") as tmp:
        wheel = None
        if cv_version() != env["opencv"]:
            # cv2 records three components; the distribution adds a wheel-build suffix.
            requirement = "opencv-python-headless==" + env["opencv"] + ".*"
            try:
                call(pip + ["download", "--pre", "--only-binary=:all:", "--no-deps",
                            "--dest", tmp, requirement])
            except subprocess.CalledProcessError as exc:
                raise RuntimeError("Cannot recover frozen cv2=" + env["opencv"] +
                    ". No package changes performed. Recover the original wheel/package "
                    "provenance; do not replace it with OpenCV4 or edit the manifest.") from exc
            wheels = list(Path(tmp).glob("*.whl"))
            if len(wheels) != 1:
                raise RuntimeError("Expected exactly one OpenCV wheel")
            wheel = wheels[0]
        requirements = ["pyyaml", "numba", "scikit-image", "scikit-learn",
                        "tqdm", "ninja", "pandas"]
        requirements += [name + "==" + env[key] for name, key in
                         (("timm", "timm"), ("numpy", "numpy"), ("scipy", "scipy"), ("pillow", "pillow"))]
        call(pip + ["install", *requirements])
        call(pip + ["install", "--index-url", index,
                    "torch==" + env["torch"], "torchvision==" + env["torchvision"]])
        if wheel:
            call(pip + ["uninstall", "-y", *OPENCV_PACKAGES])
            call(pip + ["install", "--no-deps", wheel])
        if cv_version() != env["opencv"]:
            raise RuntimeError("Installed wheel does not reproduce frozen cv2; stop before source/build")
    print("Package restoration complete. Notebook validates all imported versions in a fresh process.")

if __name__ == "__main__":
    main()
