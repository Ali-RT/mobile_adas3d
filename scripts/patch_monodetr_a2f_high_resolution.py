from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path


PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
MOBILE_BACKBONE_SHA256 = "89da029e45636a6d4d258ef75fc42aff5e5d87eb3d33fc4840c8b25dace64cb6"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        f"Unexpected {label} source in {path}: "
        f"old={text.count(old)}, new={text.count(new)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Enable an explicit stride-4 MobileNetV4 feature in MonoDETR."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodetr_repo.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDETR {PINNED_COMMIT}, found {commit}")

    backbone = repo / "lib/models/monodetr/backbone.py"
    current = sha256_file(backbone)
    already_patched = "backbone_expected_strides" in backbone.read_text(encoding="utf-8")
    if current != MOBILE_BACKBONE_SHA256 and not already_patched:
        raise RuntimeError(
            "Apply patch_monodetr_mobilenetv4.py first; "
            f"unexpected backbone SHA-256 {current}"
        )
    replace_once(
        backbone,
        "        expected_strides = [8, 16, 32] if return_interm_layers else [32]\n",
        "        expected_strides = list(cfg.get(\n"
        "            'backbone_expected_strides', [8, 16, 32]))\n"
        "        if not return_interm_layers:\n"
        "            expected_strides = [expected_strides[-1]]\n",
        "configurable MobileNetV4 feature strides",
    )
    model = repo / "lib/models/monodetr/monodetr.py"
    replace_once(
        model,
        "    def __init__(self, backbone, depthaware_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,\n"
        "                 aux_loss=True, with_box_refine=False, two_stage=False, init_box=False, use_dab=False, group_num=11, two_stage_dino=False):",
        "    def __init__(self, backbone, depthaware_transformer, depth_predictor, num_classes, num_queries, num_feature_levels,\n"
        "                 aux_loss=True, with_box_refine=False, two_stage=False, init_box=False, use_dab=False, group_num=11, two_stage_dino=False, depth_feature_start_index=0):",
        "configurable depth feature argument",
    )
    replace_once(
        model,
        "        self.num_feature_levels = num_feature_levels\n"
        "        self.two_stage_dino = two_stage_dino",
        "        self.num_feature_levels = num_feature_levels\n"
        "        self.depth_feature_start_index = int(depth_feature_start_index)\n"
        "        if not 0 <= self.depth_feature_start_index <= num_feature_levels - 3:\n"
        "            raise ValueError('depth_feature_start_index must leave at least three feature levels')\n"
        "        self.two_stage_dino = two_stage_dino",
        "configurable depth feature state",
    )
    replace_once(
        model,
        "        pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = self.depth_predictor(srcs, masks[1], pos[1])",
        "        depth_start = self.depth_feature_start_index\n        depth_features = srcs[depth_start:]\n        depth_index = depth_start + 1\n"
        "        pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = self.depth_predictor(\n"
        "            depth_features, masks[depth_index], pos[depth_index])",
        "stride-preserving depth feature routing",
    )
    replace_once(
        model,
        "        use_dab = cfg['use_dab'],\n"
        "        two_stage_dino=cfg['two_stage_dino'])",
        "        use_dab = cfg['use_dab'],\n"
        "        two_stage_dino=cfg['two_stage_dino'],\n"
        "        depth_feature_start_index=cfg.get('depth_feature_start_index', 0))",
        "depth feature config",
    )
    replace_once(
        model,
        "            srcs, masks, pos, query_embeds, depth_pos_embed, depth_pos_embed_ip)#, attn_mask)",
        "            srcs, masks, pos, query_embeds, depth_pos_embed, depth_pos_embed_ip,\n"
        "            depth_mask_index=depth_index)#, attn_mask)",
        "transformer depth mask call",
    )
    transformer = repo / "lib/models/monodetr/depthaware_transformer.py"
    replace_once(
        transformer,
        "    def forward(self, srcs, masks, pos_embeds, query_embed=None, depth_pos_embed=None, depth_pos_embed_ip=None, attn_mask=None):",
        "    def forward(self, srcs, masks, pos_embeds, query_embed=None, depth_pos_embed=None, depth_pos_embed_ip=None, attn_mask=None, depth_mask_index=1):",
        "configurable transformer depth mask argument",
    )
    replace_once(
        transformer,
        "        mask_depth = masks[1].flatten(1)",
        "        mask_depth = masks[depth_mask_index].flatten(1)",
        "stride-preserving transformer depth mask",
    )
    depth_predictor = repo / "lib/models/monodetr/depth_predictor/depth_predictor.py"
    replace_once(
        depth_predictor,
        "        assert len(feature) == 4\n",
        "        assert len(feature) >= 3\n",
        "depth predictor accepts routed three-level pyramid",
    )
    print("MonoDETR A2f stride-4 feature patch ready")


if __name__ == "__main__":
    main()
