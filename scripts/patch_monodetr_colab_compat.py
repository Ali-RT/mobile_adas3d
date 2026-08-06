from __future__ import annotations

import argparse
from pathlib import Path


def replace_exactly_once(
    path: Path, old: str, new: str, label: str, new_count_inside_old: int = 0
) -> str:
    text = path.read_text()
    old_count = text.count(old)
    new_count = text.count(new)
    if old_count == 1 and new_count == new_count_inside_old:
        path.write_text(text.replace(old, new))
        return f"patched {label}"
    if old_count == 0 and new_count == 1:
        return f"already patched {label}"
    raise RuntimeError(
        f"Unexpected {label} source in {path}: old={old_count}, new={new_count}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply narrowly-scoped current-PyTorch fixes to pinned MonoDETR."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodetr_repo.resolve()
    ops = repo / "lib/models/monodetr/ops"

    cuda_source = ops / "src/cuda/ms_deform_attn_cuda.cu"
    cuda_text = cuda_source.read_text()
    old_dispatch = "AT_DISPATCH_FLOATING_TYPES(value.type(),"
    new_dispatch = "AT_DISPATCH_FLOATING_TYPES(value.scalar_type(),"
    old_count = cuda_text.count(old_dispatch)
    new_count = cuda_text.count(new_dispatch)
    if old_count == 2 and new_count == 0:
        cuda_source.write_text(cuda_text.replace(old_dispatch, new_dispatch))
        print("patched CUDA ScalarType dispatch")
    elif old_count == 0 and new_count == 2:
        print("already patched CUDA ScalarType dispatch")
    else:
        raise RuntimeError(
            f"Unexpected CUDA dispatch source: old={old_count}, new={new_count}"
        )

    attention = ops / "modules/ms_deform_attn.py"
    old_linear = (
        "if float(torch.__version__.split('.')[0]) == 0 or "
        "(float(torch.__version__.split('.')[0]) == 1 and "
        "float(torch.__version__.split('.')[1])) < 9:\n"
        "    from torch.nn.modules.linear import _LinearWithBias\n"
        "else:\n"
        "    from torch.nn.modules.linear import NonDynamicallyQuantizableLinear as _LinearWithBias"
    )
    print(
        replace_exactly_once(
            attention, old_linear, "from torch.nn import Linear as _LinearWithBias", "linear import"
        )
    )
    old_overrides = (
        "if float(torch.__version__.split('.')[0]) == 0 or "
        "(float(torch.__version__.split('.')[0]) == 1 and "
        "float(torch.__version__.split('.')[1])) < 7:\n"
        "    from torch._overrides import has_torch_function, handle_torch_function\n"
        "else:\n"
        "    from torch.overrides import has_torch_function, handle_torch_function"
    )
    print(
        replace_exactly_once(
            attention,
            old_overrides,
            "from torch.overrides import has_torch_function, handle_torch_function",
            "torch.overrides import",
            new_count_inside_old=1,
        )
    )
    print(
        replace_exactly_once(
            repo / "lib/helpers/save_helper.py",
            "checkpoint = torch.load(filename, map_location)",
            "checkpoint = torch.load(filename, map_location, weights_only=False)",
            "PyTorch 2.6 checkpoint load",
        )
    )


if __name__ == "__main__":
    main()
