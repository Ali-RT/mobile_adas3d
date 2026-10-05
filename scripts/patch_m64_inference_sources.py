"""Narrow prospective M64 patches; never modify the frozen M62/M63 checkouts."""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import subprocess

A2_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
TEACHER_COMMIT = "884b7d8562e528031616c4773170bfc4fe211bf0"
MARKER = "# M64: AP evaluation is deliberately lazy and separate from training."


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
    paths = {repo / "lib/datasets/kitti/kitti_dataset.py": lazy_evaluator}
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
