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


def tensor_inventory(value, prefix="output"):
    import torch

    rows = []
    if isinstance(value, torch.Tensor):
        rows.append(
            {
                "name": prefix,
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "finite": bool(torch.isfinite(value).all().item()),
            }
        )
    elif isinstance(value, dict):
        for key, item in value.items():
            rows.extend(tensor_inventory(item, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            rows.extend(tensor_inventory(item, f"{prefix}[{index}]"))
    return rows


class Logger:
    def info(self, message):
        print(message, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one safe CUDA forward pass for M53 MonoDGP.")
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import yaml

    if not torch.cuda.is_available():
        raise RuntimeError("M53 smoke test requires CUDA")
    repo = args.monodgp_repo.resolve()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if not manifest.get("complete") or manifest.get("training_authorized") is not False:
        raise RuntimeError("Invalid M53 manifest")
    if manifest.get("upstream_commit") != PINNED_COMMIT:
        raise RuntimeError("M53 upstream commit mismatch")
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP commit {PINNED_COMMIT}, found {commit}")
    checkpoint = Path(manifest["evaluation_checkpoint"])
    if sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("M53 checkpoint SHA-256 mismatch")

    sys.path.insert(0, str(repo))
    from lib.helpers.dataloader_helper import build_dataloader
    from lib.helpers.model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint

    config = yaml.safe_load(Path(manifest["runtime_config"]).read_text(encoding="utf-8"))
    config["dataset"]["batch_size"] = 1
    _, val_loader = build_dataloader(config["dataset"], workers=0)
    model, _ = build_model(config["model"])
    device = torch.device("cuda:0")
    model = model.to(device)
    epoch, _, _ = load_checkpoint(model, None, checkpoint, device, Logger())
    model.eval()
    inputs, calibs, targets, info = next(iter(val_loader))
    with torch.inference_mode():
        outputs = model(
            inputs.to(device),
            calibs.to(device),
            targets,
            info["img_size"].to(device),
            dn_args=0,
        )
        torch.cuda.synchronize()
    tensors = tensor_inventory(outputs)
    finite_outputs = bool(tensors) and all(item["finite"] for item in tensors)
    if not finite_outputs:
        raise RuntimeError("MonoDGP M53 smoke produced missing or non-finite tensors")
    report = {
        "schema_version": 1,
        "complete": True,
        "device": torch.cuda.get_device_name(0),
        "upstream_commit": commit,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_epoch": int(epoch),
        "sample_id": int(info["img_id"][0]),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "output_tensor_count": len(tensors),
        "finite_outputs": finite_outputs,
        "output_tensors": tensors,
        "safe_weights_only_load": True,
        "optimizer_steps": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
