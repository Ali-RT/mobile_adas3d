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


def replace_exact_count(path: Path, old: str, new: str, count: int, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    if text.count(new) == count:
        print(f"already patched {label}")
        return
    if text.count(old) != count:
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
    model = repo / "lib/models/monodetr/monodetr.py"
    replace_once(
        model,
        "        features, pos = self.backbone(images)\n",
        "        def m52_fp32(module, value):\n"
        "            if torch.is_autocast_enabled():\n"
        "                with torch.autocast(device_type='cuda', enabled=False):\n"
        "                    return module(value.float())\n"
        "            return module(value)\n"
        "\n"
        "        features, pos = m52_fp32(self.backbone, images)\n",
        "M52 feature-extraction FP32 island",
    )
    replace_once(
        model,
        "            srcs.append(self.input_proj[l](src))\n",
        "            srcs.append(m52_fp32(self.input_proj[l], src))\n",
        "M52 backbone feature projection FP32 island",
    )
    replace_once(
        model,
        "                    src = self.input_proj[l](features[-1].tensors)\n",
        "                    src = m52_fp32(self.input_proj[l], features[-1].tensors)\n",
        "M52 first extra feature projection FP32 island",
    )
    replace_once(
        model,
        "                    src = self.input_proj[l](srcs[-1])\n",
        "                    src = m52_fp32(self.input_proj[l], srcs[-1])\n",
        "M52 remaining feature projection FP32 island",
    )
    replace_once(
        model,
        "        pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = self.depth_predictor(srcs, masks[1], pos[1])\n",
        "        self.last_backbone_feature_dtypes = [feature.tensors.dtype for feature in features]\n"
        "        self.last_projected_feature_dtypes = [feature.dtype for feature in srcs]\n"
        "        if torch.is_autocast_enabled():\n"
        "            with torch.autocast(device_type='cuda', enabled=False):\n"
        "                depth_features = [feature.float() for feature in srcs]\n"
        "                pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = self.depth_predictor(depth_features, masks[1], pos[1].float())\n"
        "        else:\n"
        "            pred_depth_map_logits, depth_pos_embed, weighted_depth, depth_pos_embed_ip = self.depth_predictor(srcs, masks[1], pos[1])\n"
        "        self.last_depth_predictor_dtype = pred_depth_map_logits.dtype\n",
        "M52 depth-predictor FP32 island",
    )
    attention = repo / "lib/models/monodetr/ops/modules/ms_deform_attn.py"
    replace_exact_count(
        attention,
        "        N, Len_q, _ = query.shape\n        N, Len_in, _ = input_flatten.shape\n",
        "        if torch.is_autocast_enabled():\n"
        "            with torch.autocast(device_type='cuda', enabled=False):\n"
        "                return self.forward(query.float(), reference_points.float(), input_flatten.float(), input_spatial_shapes, input_level_start_index, input_padding_mask)\n"
        "        N, Len_q, _ = query.shape\n        N, Len_in, _ = input_flatten.shape\n",
        2,
        "M52 deformable-attention FP32 islands",
    )
    replace_exact_count(
        attention,
        "        output = MSDeformAttnFunction.apply(\n            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)\n",
        "        self.last_kernel_dtype = value.dtype\n"
        "        output = MSDeformAttnFunction.apply(\n            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)\n",
        2,
        "M52 deformable-attention dtype evidence",
    )
    print("MonoDETR M52 FP16 evaluation patch ready")


if __name__ == "__main__":
    main()
