from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PINNED_COMMIT = "aa059a18214aebf644510e7f0793971b403f9d14"


def replace_exactly_once(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    old_count = text.count(old)
    new_count = text.count(new)
    if old_count == 1 and new_count == 0:
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"patched {label}")
        return
    if old_count == 0 and new_count == 1:
        print(f"already patched {label}")
        return
    raise RuntimeError(
        f"Unexpected {label} source in {path}: old={old_count}, new={new_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply fail-closed current-Colab compatibility fixes to pinned MonoDGP."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodgp_repo.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP commit {PINNED_COMMIT}, found {commit}")

    cuda_source = repo / "lib/models/monodgp/ops/src/cuda/ms_deform_attn_cuda.cu"
    cuda_text = cuda_source.read_text(encoding="utf-8")
    old_dispatch = "AT_DISPATCH_FLOATING_TYPES(value.type(),"
    new_dispatch = "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"
    old_count = cuda_text.count(old_dispatch)
    new_count = cuda_text.count(new_dispatch)
    if old_count == 2 and new_count == 0:
        cuda_source.write_text(
            cuda_text.replace(old_dispatch, new_dispatch), encoding="utf-8"
        )
        print("patched CUDA ScalarType dispatch")
    elif old_count == 0 and new_count == 2:
        print("already patched CUDA ScalarType dispatch")
    else:
        raise RuntimeError(
            f"Unexpected CUDA dispatch source: old={old_count}, new={new_count}"
        )

    setup = repo / "lib/models/monodgp/ops/setup.py"
    old_arches = '''            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-arch=sm_60",
            "-gencode=arch=compute_60,code=sm_60",
            "-gencode=arch=compute_61,code=sm_61",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
'''
    new_arches = '''            "-D__CUDA_NO_HALF2_OPERATORS__",
            # Let current PyTorch target the active Colab GPU instead of
            # restricting the extension to the upstream Pascal/Turing list.
'''
    replace_exactly_once(setup, old_arches, new_arches, "active CUDA architecture")

    save_helper = repo / "lib/helpers/save_helper.py"
    old_imports = "import os\nimport torch\nimport torch.nn as nn\n"
    new_imports = '''import codecs
import os

import numpy as np
import torch
import torch.nn as nn


def _safe_checkpoint_globals():
    # The pinned public checkpoint contains only tensors plus NumPy scalar
    # metadata. Explicitly allow those constructors while keeping PyTorch's
    # restricted weights-only unpickler enabled.
    numpy_core = getattr(np, "_core", None)
    if numpy_core is None:
        numpy_core = np.core
    safe = [
        (numpy_core.multiarray.scalar, "numpy.core.multiarray.scalar"),
        (np.dtype, "numpy.dtype"),
        (codecs.encode, "_codecs.encode"),
        type(np.dtype(np.float32)),
        type(np.dtype(np.float64)),
    ]
    numpy_dtypes = getattr(np, "dtypes", None)
    if numpy_dtypes is not None:
        for name in ("Float32DType", "Float64DType"):
            dtype = getattr(numpy_dtypes, name, None)
            if dtype is not None:
                safe.append(dtype)
    return safe


def load_checkpoint_safely(filename, map_location):
    with torch.serialization.safe_globals(_safe_checkpoint_globals()):
        return torch.load(filename, map_location, weights_only=True)
'''
    replace_exactly_once(save_helper, old_imports, new_imports, "safe imports")
    replace_exactly_once(
        save_helper,
        "checkpoint = torch.load(filename, map_location)",
        "checkpoint = load_checkpoint_safely(filename, map_location)",
        "restricted checkpoint load",
    )


if __name__ == "__main__":
    main()
