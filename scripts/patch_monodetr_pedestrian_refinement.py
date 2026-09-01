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


def upgrade_if_present(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(new) == 1:
        print(f"already upgraded {label}")
        return
    if text.count(old) == 1:
        path.write_text(text.replace(old, new), encoding="utf-8")
        print(f"upgraded {label}")
        return
    if text.count(old) == 0 and text.count(new) == 0:
        print(f"upgrade pending {label}")
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

    backbone = repo / "lib/models/monodetr/backbone.py"
    replace_once(
        backbone,
        "        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)\n",
        "        self.refinement_feature = None\n"
        "        self._refinement_hook = backbone.layer1.register_forward_hook(self._capture_refinement_feature)\n"
        "        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)\n\n"
        "    def _capture_refinement_feature(self, module, inputs, output):\n"
        "        self.refinement_feature = output\n",
        "stride-4 refinement feature hook",
    )

    model = repo / "lib/models/monodetr/monodetr.py"
    replace_once(
        model,
        "                 aux_loss=True, with_box_refine=False, two_stage=False, init_box=False, use_dab=False, group_num=11, two_stage_dino=False):",
        "                 aux_loss=True, with_box_refine=False, two_stage=False, init_box=False, use_dab=False, group_num=11, two_stage_dino=False, pedestrian_refinement=None):",
        "model Pedestrian refinement argument",
    )
    replace_once(
        model,
        "        self.use_dab = use_dab\n",
        "        self.use_dab = use_dab\n"
        "        pedestrian_refinement = pedestrian_refinement or {}\n"
        "        self.pedestrian_refinement_enabled = bool(pedestrian_refinement.get('enabled', False))\n"
        "        self.pedestrian_refinement_grid_size = int(pedestrian_refinement.get('grid_size', 3))\n"
        "        self.pedestrian_refinement_scale = float(pedestrian_refinement.get('residual_scale', 0.1))\n"
        "        if self.pedestrian_refinement_enabled:\n"
        "            if self.pedestrian_refinement_grid_size < 2:\n"
        "                raise ValueError('pedestrian refinement grid_size must be >= 2')\n"
        "            self.pedestrian_refinement_proj = nn.Sequential(\n"
        "                nn.Conv2d(256, hidden_dim, kernel_size=1),\n"
        "                nn.GroupNorm(32, hidden_dim),\n"
        "                nn.ReLU(inplace=True))\n"
        "            self.pedestrian_refinement_head = MLP(hidden_dim * 2, hidden_dim, 4, 3)\n"
        "            nn.init.constant_(self.pedestrian_refinement_head.layers[-1].weight, 0)\n"
        "            nn.init.constant_(self.pedestrian_refinement_head.layers[-1].bias, 0)\n",
        "model Pedestrian refinement modules",
    )
    replace_once(
        model,
        "        outputs_angles = []\n\n        for lvl in range(hs.shape[0]):",
        "        outputs_angles = []\n"
        "        refinement_feature = None\n"
        "        if self.pedestrian_refinement_enabled:\n"
        "            refinement_feature = self.pedestrian_refinement_proj(self.backbone[0].refinement_feature)\n\n"
        "        for lvl in range(hs.shape[0]):",
        "project captured stride-4 feature",
    )
    upgrade_if_present(
        model,
        "                refined_edges = (inverse_sigmoid(outputs_coord[..., 2:6]) + self.pedestrian_refinement_scale * pedestrian_probability * residual).sigmoid()\n",
        "                refined_edges = (outputs_coord[..., 2:6] + self.pedestrian_refinement_scale * pedestrian_probability * residual).clamp(0, 1)\n",
        "bitwise-zero Pedestrian residual",
    )
    replace_once(
        model,
        "            outputs_class = self.class_embed[lvl](hs[lvl])\n            outputs_classes.append(outputs_class)\n",
        "            outputs_class = self.class_embed[lvl](hs[lvl])\n"
        "            outputs_classes.append(outputs_class)\n\n"
        "            if self.pedestrian_refinement_enabled and lvl == hs.shape[0] - 1:\n"
        "                cx, cy = outputs_coord[..., 0], outputs_coord[..., 1]\n"
        "                left, right = outputs_coord[..., 2], outputs_coord[..., 3]\n"
        "                top, bottom = outputs_coord[..., 4], outputs_coord[..., 5]\n"
        "                steps = torch.linspace(0, 1, self.pedestrian_refinement_grid_size, device=outputs_coord.device, dtype=outputs_coord.dtype)\n"
        "                xs = (cx - left).unsqueeze(-1) * (1 - steps) + (cx + right).unsqueeze(-1) * steps\n"
        "                ys = (cy - top).unsqueeze(-1) * (1 - steps) + (cy + bottom).unsqueeze(-1) * steps\n"
        "                grid_x = xs.unsqueeze(-2).expand(-1, -1, self.pedestrian_refinement_grid_size, -1)\n"
        "                grid_y = ys.unsqueeze(-1).expand(-1, -1, -1, self.pedestrian_refinement_grid_size)\n"
        "                grid = torch.stack((grid_x, grid_y), dim=-1).clamp(0, 1) * 2 - 1\n"
        "                batch_size, query_count = outputs_coord.shape[:2]\n"
        "                sampled = F.grid_sample(\n"
        "                    refinement_feature,\n"
        "                    grid.reshape(batch_size, query_count * self.pedestrian_refinement_grid_size, self.pedestrian_refinement_grid_size, 2),\n"
        "                    mode='bilinear', padding_mode='border', align_corners=True)\n"
        "                sampled = sampled.reshape(batch_size, self.hidden_dim, query_count, self.pedestrian_refinement_grid_size, self.pedestrian_refinement_grid_size).mean(dim=(-1, -2)).transpose(1, 2)\n"
        "                residual = self.pedestrian_refinement_head(torch.cat((hs[lvl], sampled), dim=-1))\n"
        "                pedestrian_probability = outputs_class[..., 0:1].sigmoid()\n"
        "                refined_edges = (outputs_coord[..., 2:6] + self.pedestrian_refinement_scale * pedestrian_probability * residual).clamp(0, 1)\n"
        "                outputs_coord = torch.cat((outputs_coord[..., 0:2], refined_edges), dim=-1)\n"
        "                outputs_coords[-1] = outputs_coord\n",
        "Pedestrian local-feature residual refinement",
    )
    replace_once(
        model,
        "        use_dab = cfg['use_dab'],\n        two_stage_dino=cfg['two_stage_dino'])",
        "        use_dab = cfg['use_dab'],\n"
        "        two_stage_dino=cfg['two_stage_dino'],\n"
        "        pedestrian_refinement=cfg.get('pedestrian_refinement', {}))",
        "model Pedestrian refinement config",
    )
    print("MonoDETR stride-4 Pedestrian refinement patch ready")


if __name__ == "__main__":
    main()
