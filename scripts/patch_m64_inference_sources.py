"""Narrow prospective M64 patches; never modify the frozen M62/M63 checkouts."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess

A2_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
TEACHER_COMMIT = "884b7d8562e528031616c4773170bfc4fe211bf0"
MARKER = "# M64: AP evaluation is deliberately lazy and separate from training."
BUILD_MARKER = "# M64: PyTorch selects the current GPU via TORCH_CUDA_ARCH_LIST."
LEGACY_ARCH_FLAGS = (
    "-arch=sm_60",
    "-gencode=arch=compute_60,code=sm_60",
    "-gencode=arch=compute_61,code=sm_61",
    "-gencode=arch=compute_70,code=sm_70",
    "-gencode=arch=compute_75,code=sm_75",
)


def cuda_build_targets(text: str) -> str:
    """Remove only the pinned release's legacy nvcc targets, not kernel code."""
    tree = ast.parse(text)
    assignments = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Subscript)
                           and isinstance(target.value, ast.Name)
                           and target.value.id == "extra_compile_args"
                           and isinstance(target.slice, ast.Constant)
                           and target.slice.value == "nvcc" for target in node.targets)]
    if len(assignments) != 1 or not isinstance(assignments[0].value, ast.List):
        raise RuntimeError("Unexpected CUDA compile-argument layout")
    values = assignments[0].value.elts
    if any(not isinstance(value, ast.Constant) or not isinstance(value.value, str) for value in values):
        raise RuntimeError("CUDA compile arguments must be the reviewed literal list")
    architecture = [value for value in values if "arch" in value.value]
    if architecture and tuple(value.value for value in architecture) != LEGACY_ARCH_FLAGS:
        raise RuntimeError("Unreviewed explicit CUDA architecture flags; do not silently remove them")
    lines = text.splitlines(keepends=True)
    removed = set()
    for value in architecture:
        if value.lineno != value.end_lineno:
            raise RuntimeError("Unexpected multiline CUDA architecture flag")
        line = lines[value.lineno - 1].strip()
        if line not in {repr(value.value) + ",", '"' + value.value + '",'}:
            raise RuntimeError("Architecture flag shares a line with other build settings")
        removed.add(value.lineno - 1)
    result = "".join(line for index, line in enumerate(lines) if index not in removed)
    if BUILD_MARKER not in result:
        old = '        extra_compile_args["nvcc"] = [\n'
        if result.count(old) != 1:
            raise RuntimeError("Unexpected nvcc list indentation")
        result = result.replace(old, old + "            " + BUILD_MARKER + "\n")
    ast.parse(result)
    return result


def lazy_evaluator(text: str) -> str:
    if MARKER in text:
        ast.parse(text)
        return text
    lines = text.splitlines(keepends=True)
    imports = [line for line in lines if line.startswith((
        "from lib.datasets.kitti.kitti_eval_python.eval import ",
        "import lib.datasets.kitti.kitti_eval_python.kitti_common as ",
    ))]
    if not imports or text.count("    def eval(self, results_dir, logger):\n") != 1:
        raise RuntimeError("Unexpected KITTI evaluator import/function layout")
    for line in imports:
        lines.remove(line)
    result = "".join(lines).replace(
        "    def eval(self, results_dir, logger):\n",
        "    def eval(self, results_dir, logger):\n        " + MARKER + "\n"
        + "".join("        " + line for line in imports),
    )
    ast.parse(result)
    return result


def teacher_model_syntax(text: str) -> str:
    old = 'class MonoPRIO(nn.Module):\n  """ This is the MonoDGP module that performs monocualr 3D object detection """'
    new = old.replace('\n  """', '\n    """')
    if old in text:
        if text.count(old) != 1:
            raise RuntimeError("Unexpected MonoPRIO class-docstring layout")
        text = text.replace(old, new)
    elif new not in text:
        raise RuntimeError("Unreviewed MonoPRIO model source")
    ast.parse(text)
    return text


def teacher_backbone(text: str) -> str:
    # Strictly loading the full released state replaces all initialization.
    old = "pretrained=is_main_process(), norm_layer=norm_layer)"
    new = "pretrained=False, norm_layer=norm_layer)"
    if text.count(old) == 1:
        text = text.replace(old, new)
    elif text.count(new) != 1:
        raise RuntimeError("Unexpected teacher backbone initialization")
    ast.parse(text)
    return text


def patch(repo: Path, role: str) -> None:
    expected = A2_COMMIT if role == "a2" else TEACHER_COMMIT
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    if commit != expected:
        raise RuntimeError(f"Wrong {role} source commit: {commit}")
    if repo.name in {"MonoDETR_M62", "MonoDETR_M63"}:
        raise RuntimeError("Do not patch a historical experiment checkout")
    model_name = "monodetr" if role == "a2" else "monoprio"
    paths = {repo / "lib/datasets/kitti/kitti_dataset.py": lazy_evaluator,
             repo / f"lib/models/{model_name}/ops/setup.py": cuda_build_targets}
    if role == "teacher":
        paths[repo / "lib/models/monoprio/monoprio.py"] = teacher_model_syntax
        paths[repo / "lib/models/monoprio/backbone.py"] = teacher_backbone
        router = repo / "lib/models/monoprio_router.py"
        text = router.read_text()
        old, new = "np.load(prior_path, allow_pickle=True)", "np.load(prior_path, allow_pickle=False)"
        if text.count(old) == 1:
            text = text.replace(old, new)
        elif text.count(new) != 1:
            raise RuntimeError("Unexpected prior-bank load implementation")
        ast.parse(text)
        paths[router] = lambda _: text
    # Validate all replacements before writing any of this patch's files.
    replacements = {path: transform(path.read_text()) for path, transform in paths.items()}
    for path, text in replacements.items():
        path.write_text(text)
        print(f"M64 verified patch: {path.relative_to(repo)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--role", choices=("a2", "teacher"), required=True)
    args = parser.parse_args()
    patch(args.repo.resolve(), args.role)


if __name__ == "__main__":
    main()
