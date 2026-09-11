from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PINNED_COMMIT = "aa059a18214aebf644510e7f0793971b403f9d14"
CHECKPOINT_SHA256 = "1d5f30b34b8bef49638079a8b07f05ebf11bb5f85d6a9a11c7b028c69396f05d"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_targets(targets, batch_size):
    mask = targets["mask_2d"]
    keys = ["labels", "boxes", "calibs", "depth", "size_3d", "heading_bin", "heading_res", "boxes_3d"]
    rows = []
    for batch_index in range(batch_size):
        row = {}
        for key, value in targets.items():
            if key in keys:
                row[key] = value[batch_index][mask[batch_index]]
            elif key in {"depth_map", "obj_region"}:
                row[key] = value[batch_index]
        rows.append(row)
    return rows


class Logger:
    def info(self, message):
        print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one real CUDA forward/loss/backward/step for M54."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("M54 training smoke requires CUDA")
    repo = args.monodgp_repo.resolve()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP commit {PINNED_COMMIT}, found {commit}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if (
        not manifest.get("complete")
        or manifest.get("training_authorized") is not True
        or manifest.get("distillation_enabled") is not False
        or manifest.get("architecture_changed") is not False
    ):
        raise RuntimeError("Invalid M54 manifest")
    checkpoint = Path(manifest["initial_checkpoint"]).resolve()
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("M54 initialization checkpoint SHA-256 mismatch")

    sys.path.insert(0, str(repo))
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.model_helper import build_model
    from lib.helpers.optimizer_helper import build_optimizer
    from lib.helpers.save_helper import load_checkpoint

    config = yaml.safe_load(Path(manifest["runtime_config"]).read_text(encoding="utf-8"))
    config["dataset"]["batch_size"] = 2
    train_loader, _ = build_dataloader(config["dataset"], workers=0)
    model, criterion = build_model(config["model"])
    device = torch.device("cuda:0")
    model = model.to(device)
    criterion = criterion.to(device)
    load_checkpoint(model, None, checkpoint, device, Logger())
    optimizer = build_optimizer(config["optimizer"], model)
    model.train()
    criterion.train()

    selected_batch = None
    source_counts = {0: 0, 1: 0}
    for batch_index, batch in enumerate(train_loader):
        inputs, calibs, targets, info = batch
        labels = targets["labels"][targets["mask_2d"]].long()
        counts = {class_id: int((labels == class_id).sum().item()) for class_id in (0, 1)}
        if counts[0] > 0 and counts[1] > 0:
            selected_batch = batch
            source_counts = counts
            break
        if batch_index >= 63:
            break
    if selected_batch is None:
        raise RuntimeError("Could not find a smoke batch containing both Vehicle and Pedestrian")

    inputs, calibs, targets, info = selected_batch
    inputs = inputs.to(device)
    calibs = calibs.to(device)
    for key in targets:
        targets[key] = targets[key].to(device)
    image_sizes = targets["img_size"]
    target_rows = prepare_targets(targets, inputs.shape[0])

    optimizer.zero_grad()
    outputs = model(inputs, calibs, target_rows, image_sizes, dn_args=None)
    losses = criterion(outputs, target_rows, None)
    weighted = {
        key: value * criterion.weight_dict[key]
        for key, value in losses.items()
        if key in criterion.weight_dict
    }
    total_loss = sum(weighted.values())
    if not torch.isfinite(total_loss):
        raise RuntimeError(f"M54 smoke produced non-finite loss: {total_loss}")
    total_loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    finite_gradients = bool(gradients) and all(
        torch.isfinite(gradient).all().item() for gradient in gradients
    )
    if not finite_gradients:
        raise RuntimeError("M54 smoke produced missing or non-finite gradients")
    optimizer.step()
    torch.cuda.synchronize()

    output_tensors = {
        key: {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "finite": bool(torch.isfinite(value).all().item()),
        }
        for key, value in outputs.items()
        if isinstance(value, torch.Tensor)
    }
    finite_outputs = bool(output_tensors) and all(
        item["finite"] for item in output_tensors.values()
    )
    if not finite_outputs:
        raise RuntimeError("M54 smoke produced missing or non-finite outputs")

    report = {
        "schema_version": 1,
        "complete": True,
        "device": torch.cuda.get_device_name(0),
        "upstream_commit": commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "batch_size": int(inputs.shape[0]),
        "vehicle_targets": source_counts[1],
        "pedestrian_targets": source_counts[0],
        "total_loss": float(total_loss.detach().cpu()),
        "weighted_losses": {
            key: float(value.detach().cpu()) for key, value in weighted.items()
        },
        "finite_outputs": finite_outputs,
        "finite_gradients": finite_gradients,
        "optimizer_steps": 1,
        "output_tensors": output_tensors,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
