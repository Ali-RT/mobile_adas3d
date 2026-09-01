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
    if text.count(old) == 1:
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"patched {label}")
        return
    raise RuntimeError(
        f"Unexpected {label} source in {path}: old={text.count(old)}, new={text.count(new)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodetr_repo.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDETR {PINNED_COMMIT}, found {commit}")

    model = repo / "lib/models/monodetr/monodetr.py"
    replace_once(
        model,
        "        self.pedestrian_refinement_scale = float(pedestrian_refinement.get('residual_scale', 0.1))\n",
        "        self.pedestrian_refinement_scale = float(pedestrian_refinement.get('residual_scale', 0.1))\n"
        "        self.pedestrian_refinement_gate_mode = pedestrian_refinement.get('gate_mode', 'soft')\n"
        "        self.pedestrian_refinement_freeze_base = bool(pedestrian_refinement.get('freeze_base', False))\n"
        "        if self.pedestrian_refinement_gate_mode not in ('soft', 'hard'):\n"
        "            raise ValueError('pedestrian refinement gate_mode must be soft or hard')\n",
        "R2b refinement controls",
    )
    replace_once(
        model,
        "                pedestrian_probability = outputs_class[..., 0:1].sigmoid()\n"
        "                refined_edges = (outputs_coord[..., 2:6] + self.pedestrian_refinement_scale * pedestrian_probability * residual).clamp(0, 1)\n",
        "                if self.pedestrian_refinement_gate_mode == 'hard':\n"
        "                    pedestrian_gate = (outputs_class.argmax(dim=-1, keepdim=True) == 0).to(outputs_coord.dtype).detach()\n"
        "                else:\n"
        "                    pedestrian_gate = outputs_class[..., 0:1].sigmoid()\n"
        "                refined_edges = (outputs_coord[..., 2:6] + self.pedestrian_refinement_scale * pedestrian_gate * residual).clamp(0, 1)\n",
        "hard Pedestrian-only refinement gate",
    )
    replace_once(
        model,
        "        if two_stage:\n"
        "            # hack implementation for two-stage\n"
        "            self.depthaware_transformer.decoder.class_embed = self.class_embed\n"
        "            for box_embed in self.bbox_embed:\n"
        "                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)\n\n\n"
        "    def forward(self, images, calibs, targets, img_sizes, dn_args=None):",
        "        if two_stage:\n"
        "            # hack implementation for two-stage\n"
        "            self.depthaware_transformer.decoder.class_embed = self.class_embed\n"
        "            for box_embed in self.bbox_embed:\n"
        "                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)\n\n"
        "        if self.pedestrian_refinement_enabled and self.pedestrian_refinement_freeze_base:\n"
        "            trainable_prefixes = ('pedestrian_refinement_proj.', 'pedestrian_refinement_head.')\n"
        "            for name, parameter in self.named_parameters():\n"
        "                parameter.requires_grad_(name.startswith(trainable_prefixes))\n\n\n"
        "    def forward(self, images, calibs, targets, img_sizes, dn_args=None):",
        "freeze R0 outside refinement modules",
    )
    print("MonoDETR R2b frozen hard-gated refinement patch ready")


if __name__ == "__main__":
    main()
