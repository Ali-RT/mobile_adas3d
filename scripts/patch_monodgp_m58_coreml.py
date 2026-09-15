from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


PINNED_COMMIT = "aa059a18214aebf644510e7f0793971b403f9d14"
MARKER = "M58_COREML_EXPORT_COMPAT"


def replace_once(path: Path, old: str, new: str, label: str) -> None:
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


def replace_count(path: Path, old: str, new: str, expected: int, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    old_count = text.count(old)
    new_count = text.count(new)
    if old_count == expected and new_count == 0:
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"patched {label} ({expected} occurrences)")
        return
    if old_count == 0 and new_count == expected:
        print(f"already patched {label}")
        return
    raise RuntimeError(
        f"Unexpected {label} source in {path}: old={old_count}, new={new_count}"
    )


def patch_decoder(path: Path, label: str) -> None:
    marker = "self.m58_coreml_export_compat = False"
    init_anchor = "        self.class_embed = None\n"
    text = path.read_text(encoding="utf-8")
    if marker not in text:
        replace_once(path, init_anchor, init_anchor + f"        {marker}\n", f"{label} flag")

    old = (
        "                    new_reference_points = tmp\n"
        "                    new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)\n"
    )
    new = (
        "                    if self.m58_coreml_export_compat:\n"
        "                        new_reference_points = torch.cat([\n"
        "                            tmp[..., :2] + inverse_sigmoid(reference_points),\n"
        "                            tmp[..., 2:]\n"
        "                        ], dim=-1)\n"
        "                    else:\n"
        "                        new_reference_points = tmp\n"
        "                        new_reference_points[..., :2] = tmp[..., :2] + inverse_sigmoid(reference_points)\n"
    )
    replace_once(path, old, new, f"{label} sliced reference update")


def patch_repo(repo: Path) -> None:
    repo = repo.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP {PINNED_COMMIT}, found {commit}")

    det2d = repo / "lib/models/monodgp/det2d_transformer.py"
    det3d = repo / "lib/models/monodgp/det3d_transformer.py"
    model = repo / "lib/models/monodgp/monodgp.py"
    for path in (det2d, det3d, model):
        if not path.is_file():
            raise FileNotFoundError(path)

    patch_decoder(det2d, "2D decoder")
    patch_decoder(det3d, "3D decoder")

    model_marker = "self.m58_coreml_export_compat = False"
    text = model.read_text(encoding="utf-8")
    if model_marker not in text:
        replace_once(
            model,
            "        self.num_classes = num_classes\n",
            "        self.num_classes = num_classes\n"
            "        self.m58_coreml_export_compat = False\n",
            "MonoDGP export flag",
        )

    old_update = "                tmp[..., :2] += reference\n"
    new_update = (
        "                if self.m58_coreml_export_compat:\n"
        "                    tmp = torch.cat([tmp[..., :2] + reference, tmp[..., 2:]], dim=-1)\n"
        "                else:\n"
        "                    tmp[..., :2] += reference\n"
    )
    replace_count(model, old_update, new_update, 2, "MonoDGP sliced reference updates")

    old_full = "                tmp += reference\n"
    new_full = (
        "                if self.m58_coreml_export_compat:\n"
        "                    tmp = tmp + reference\n"
        "                else:\n"
        "                    tmp += reference\n"
    )
    replace_count(model, old_full, new_full, 2, "MonoDGP full reference updates")
    print(f"MonoDGP M58 Core ML compatibility patch applied: {repo}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch pinned MonoDGP sliced updates for Core ML export."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    args = parser.parse_args()
    patch_repo(args.monodgp_repo)


if __name__ == "__main__":
    main()
