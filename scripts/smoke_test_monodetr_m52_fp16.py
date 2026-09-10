from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_checkpoint(torch, path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(description="One-batch M52 mixed-precision CUDA preflight.")
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M52 smoke test requires CUDA")
    repo = args.monodetr_repo.resolve()
    sys.path.insert(0, str(repo))
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.model_helper import build_model

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    config = yaml.safe_load(Path(manifest["config"]).read_text(encoding="utf-8"))
    if config["tester"].get("inference_precision") != "fp16_autocast":
        raise RuntimeError("M52 config is not fp16_autocast")
    _, loader = build_dataloader(config["dataset"], workers=0)
    inputs, calibs, targets, info = next(iter(loader))
    device = torch.device("cuda")
    model, _ = build_model(config["model"])
    model.load_state_dict(load_checkpoint(torch, Path(manifest["evaluation_checkpoint"]))["model_state"], strict=True)
    model = model.to(device).eval()
    inputs, calibs = inputs.to(device), calibs.to(device)
    img_sizes = info["img_size"].to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        outputs = model(inputs, calibs, targets, img_sizes, dn_args=0)
    torch.cuda.synchronize()
    tensors = {key: value for key, value in outputs.items() if isinstance(value, torch.Tensor)}
    output_dtypes = {key: str(value.dtype) for key, value in tensors.items()}
    if not all(bool(torch.isfinite(value).all()) for value in tensors.values()):
        raise RuntimeError("M52 smoke produced non-finite output")
    backbone_dtypes = [str(dtype) for dtype in getattr(model, "last_backbone_feature_dtypes", [])]
    projected_dtypes = [str(dtype) for dtype in getattr(model, "last_projected_feature_dtypes", [])]
    if not backbone_dtypes or any(dtype != "torch.float32" for dtype in backbone_dtypes):
        raise RuntimeError(f"Unsafe backbone feature dtypes: {backbone_dtypes}")
    if not projected_dtypes or any(dtype != "torch.float32" for dtype in projected_dtypes):
        raise RuntimeError(f"Unsafe projected feature dtypes: {projected_dtypes}")
    depth_dtype = str(getattr(model, "last_depth_predictor_dtype", "missing"))
    if depth_dtype != "torch.float32":
        raise RuntimeError(f"Unsafe depth-predictor dtype: {depth_dtype}")
    kernel_dtypes = [str(module.last_kernel_dtype) for module in model.modules() if hasattr(module, "last_kernel_dtype")]
    if not kernel_dtypes or any(dtype != "torch.float32" for dtype in kernel_dtypes):
        raise RuntimeError(f"Unsafe deformable-attention kernel dtypes: {kernel_dtypes}")
    if not any(dtype == "torch.float16" for dtype in output_dtypes.values()):
        raise RuntimeError(f"Autocast did not produce any FP16 model outputs: {output_dtypes}")
    report = {"schema_version": 1, "complete": True, "device": torch.cuda.get_device_name(0), "precision": "fp16_autocast_with_fp32_feature_depth_and_deformable_attention", "batch_size": int(inputs.shape[0]), "output_dtypes": output_dtypes, "backbone_feature_dtypes": backbone_dtypes, "projected_feature_dtypes": projected_dtypes, "depth_predictor_dtype": depth_dtype, "deformable_attention_kernel_dtypes": kernel_dtypes, "finite_outputs": True, "optimizer_steps": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
