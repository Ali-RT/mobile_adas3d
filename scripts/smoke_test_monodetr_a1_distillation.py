from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_torch_checkpoint(torch, path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(
        description="Run one real-data A1 distillation forward/backward preflight."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = args.monodetr_repo.resolve()
    sys.path.insert(0, str(repo))

    from lib.helpers.a1_distillation_loss import compute_a1_distillation_losses
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.model_helper import build_model

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("distillation", {}).get("enabled") is not True:
        raise RuntimeError("Distillation must be explicitly enabled")
    config["dataset"]["batch_size"] = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("A1 distillation smoke test requires CUDA")

    train_loader, _ = build_dataloader(config["dataset"])
    student, criterion = build_model(config["model"])
    teacher, _ = build_model(config["distillation"]["teacher_model"])
    student_payload = load_torch_checkpoint(
        torch, Path(config["trainer"]["pretrain_model"])
    )
    teacher_payload = load_torch_checkpoint(
        torch, Path(config["distillation"]["teacher_checkpoint"])
    )
    student.load_state_dict(student_payload["model_state"], strict=True)
    teacher.load_state_dict(teacher_payload["model_state"], strict=True)
    student = student.to(device).train()
    teacher = teacher.to(device).eval()
    criterion = criterion.to(device).train()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    inputs, calibs, raw_targets, _ = next(iter(train_loader))
    inputs = inputs.to(device)
    calibs = calibs.to(device)
    for key in raw_targets:
        raw_targets[key] = raw_targets[key].to(device)
    img_sizes = raw_targets["img_size"]
    mask = raw_targets["mask_2d"]
    keys = [
        "labels", "boxes", "calibs", "depth", "size_3d",
        "heading_bin", "heading_res", "boxes_3d",
    ]
    targets = [
        {key: raw_targets[key][batch_index][mask[batch_index]] for key in keys}
        for batch_index in range(inputs.shape[0])
    ]
    with torch.no_grad():
        teacher_outputs = teacher(inputs, calibs, targets, img_sizes, dn_args=None)
    student_outputs = student(inputs, calibs, targets, img_sizes, dn_args=None)
    supervised = criterion(student_outputs, targets, None)
    supervised_total = sum(
        value * criterion.weight_dict[key]
        for key, value in supervised.items()
        if key in criterion.weight_dict
    )
    distillation = compute_a1_distillation_losses(
        student_outputs,
        teacher_outputs,
        targets,
        criterion.matcher,
        config["distillation"],
        student_group_num=criterion.group_num,
    )
    total = supervised_total + distillation["distill_total"]
    if not torch.isfinite(total):
        raise RuntimeError(f"Non-finite smoke loss: {float(total)}")
    total.backward()
    finite_gradients = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in student.parameters()
    )
    if not finite_gradients:
        raise RuntimeError("A1 distillation produced non-finite gradients")
    if int(distillation["distill_pairs"].item()) <= 0:
        raise RuntimeError("A1 distillation smoke batch produced no approved pairs")

    report = {
        "schema_version": 1,
        "complete": True,
        "device": str(device),
        "batch_size": int(inputs.shape[0]),
        "supervised_total": float(supervised_total.detach()),
        "distill_total": float(distillation["distill_total"].detach()),
        "combined_total": float(total.detach()),
        "approved_query_pairs": int(distillation["distill_pairs"].item()),
        "distillation_losses": {
            key: float(value.detach())
            for key, value in distillation.items()
            if key != "distill_pairs"
        },
        "finite_gradients": finite_gradients,
        "optimizer_steps": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
