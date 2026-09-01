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


def make_targets(raw_targets, batch_size):
    mask = raw_targets["mask_2d"]
    keys = [
        "labels", "boxes", "calibs", "depth", "size_3d",
        "heading_bin", "heading_res", "boxes_3d",
    ]
    return [
        {key: raw_targets[key][index][mask[index]] for key in keys}
        for index in range(batch_size)
    ]


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(description="R2 real-data parity and gradient preflight.")
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodetr_repo.resolve()
    sys.path.insert(0, str(repo))

    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.model_helper import build_model

    if not torch.cuda.is_available():
        raise RuntimeError("R2 smoke test requires CUDA")
    device = torch.device("cuda")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    control_variant = manifest["variants"]["control_refine_off"]
    treatment_variant = manifest["variants"]["ped_refine_stride4"]
    control_cfg = yaml.safe_load(Path(control_variant["config"]).read_text())
    treatment_cfg = yaml.safe_load(Path(treatment_variant["config"]).read_text())
    treatment_cfg["dataset"]["batch_size"] = 1
    loader, _ = build_dataloader(treatment_cfg["dataset"], workers=0)
    inputs, calibs, raw_targets, _ = next(iter(loader))
    inputs, calibs = inputs.to(device), calibs.to(device)
    for key in raw_targets:
        raw_targets[key] = raw_targets[key].to(device)
    targets = make_targets(raw_targets, inputs.shape[0])
    img_sizes = raw_targets["img_size"]

    control, _ = build_model(control_cfg["model"])
    control.load_state_dict(
        load_checkpoint(torch, Path(control_variant["pretrain_model"]))["model_state"],
        strict=True,
    )
    control = control.to(device).eval()
    with torch.no_grad():
        control_outputs = {
            key: value.detach().cpu()
            for key, value in control(inputs, calibs, targets, img_sizes, dn_args=None).items()
            if isinstance(value, torch.Tensor)
        }
    del control
    torch.cuda.empty_cache()

    treatment, criterion = build_model(treatment_cfg["model"])
    treatment.load_state_dict(
        load_checkpoint(torch, Path(treatment_variant["pretrain_model"]))["model_state"],
        strict=True,
    )
    treatment = treatment.to(device).eval()
    with torch.no_grad():
        treatment_outputs = treatment(inputs, calibs, targets, img_sizes, dn_args=None)
    parity = {
        key: float((treatment_outputs[key].detach().cpu() - control_outputs[key]).abs().max())
        for key in ("pred_logits", "pred_boxes", "pred_3d_dim", "pred_depth", "pred_angle")
    }
    if max(parity.values()) > 1e-6:
        raise RuntimeError(f"R2 zero-initialization parity failed: {parity}")

    treatment.train()
    treatment.zero_grad(set_to_none=True)
    outputs = treatment(inputs, calibs, targets, img_sizes, dn_args=None)
    losses = criterion(outputs, targets, None)
    total = sum(
        value * criterion.weight_dict[key]
        for key, value in losses.items()
        if key in criterion.weight_dict
    )
    if not torch.isfinite(total):
        raise RuntimeError(f"Non-finite R2 smoke loss: {float(total)}")
    total.backward()
    head = treatment.pedestrian_refinement_head.layers[-1]
    gradient_sum = float(head.weight.grad.detach().abs().sum())
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in treatment.parameters()
    )
    if not finite_gradients or gradient_sum <= 0:
        raise RuntimeError(
            f"R2 refinement gradient failed: finite={finite_gradients}, sum={gradient_sum}"
        )
    report = {
        "schema_version": 1,
        "complete": True,
        "device": str(device),
        "sample_count": int(inputs.shape[0]),
        "initialization_max_abs_deltas": parity,
        "parity_tolerance": 1e-6,
        "total_loss": float(total.detach()),
        "refinement_final_weight_gradient_l1": gradient_sum,
        "finite_gradients": finite_gradients,
        "optimizer_steps": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
