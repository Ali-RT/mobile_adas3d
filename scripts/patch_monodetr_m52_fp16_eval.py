from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(new) == 1:
        print(f"already patched {label}")
        return
    if text.count(old) != 1:
        raise RuntimeError(f"Unexpected {label}: old={text.count(old)}, new={text.count(new)}")
    path.write_text(text.replace(old, new), encoding="utf-8")
    print(f"patched {label}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodetr_repo.resolve()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDETR {PINNED_COMMIT}, found {commit}")
    tester = repo / "lib/helpers/tester_helper.py"
    replace_once(
        tester,
        "        self.model_name = model_name\n",
        "        self.model_name = model_name\n"
        "        self.inference_precision = cfg.get('inference_precision', 'fp32')\n"
        "        if self.inference_precision not in ('fp32', 'fp16_autocast'):\n"
        "            raise ValueError('inference_precision must be fp32 or fp16_autocast')\n"
        "        if self.inference_precision == 'fp16_autocast' and self.device.type != 'cuda':\n"
        "            raise RuntimeError('fp16_autocast evaluation requires CUDA')\n",
        "M52 precision policy",
    )
    replace_once(
        tester,
        "            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)\n",
        "            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=self.inference_precision == 'fp16_autocast'):\n"
        "                outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)\n",
        "M52 autocast inference",
    )
    print("MonoDETR M52 FP16 evaluation patch ready")


if __name__ == "__main__":
    main()
