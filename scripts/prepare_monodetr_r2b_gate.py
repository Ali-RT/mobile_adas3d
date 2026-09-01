from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


R0_EPOCH = 185
R0_SHA256 = "fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59"
VARIANTS = {"ped_refine_frozen_hard": True}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_r0_checkpoint(selection_path: Path) -> Path:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selected = selection.get("selected", {})
    if not selection.get("complete") or int(selected.get("epoch", -1)) != R0_EPOCH:
        raise RuntimeError("Expected complete frozen R0 epoch-185 selection")
    if selected.get("checkpoint_sha256") != R0_SHA256:
        raise RuntimeError("R0 selection records an unexpected checkpoint SHA-256")
    checkpoint = Path(selected["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual = sha256_file(checkpoint)
    if actual != R0_SHA256:
        raise RuntimeError(f"R0 checkpoint SHA-256 mismatch: {actual}")
    return checkpoint


def load_checkpoint(path: Path) -> dict:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def create_refinement_initialization(repo: Path, config: dict, r0: Path, output: Path, seed: int) -> dict:
    import torch

    sys.path.insert(0, str(repo))
    from lib.models.monodetr import build_monodetr

    torch.manual_seed(seed)
    model, _ = build_monodetr(config["model"])
    payload = load_checkpoint(r0)
    incompatible = model.load_state_dict(payload["model_state"], strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    allowed_prefixes = (
        "pedestrian_refinement_proj.",
        "pedestrian_refinement_head.",
    )
    disallowed = [key for key in missing if not key.startswith(allowed_prefixes)]
    trainable = sorted(name for name, parameter in model.named_parameters() if parameter.requires_grad)
    invalid_trainable = [name for name in trainable if not name.startswith(allowed_prefixes)]
    if disallowed or unexpected or not missing or invalid_trainable or not trainable:
        raise RuntimeError(
            f"Unsafe R2b initialization: missing={missing}, unexpected={unexpected}, trainable={trainable}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": 0,
            "model_state": model.state_dict(),
            "optimizer_state": None,
            "best_result": payload.get("best_result", 0.0),
            "best_epoch": payload.get("best_epoch", R0_EPOCH),
            "r2_initialization": {
                "source_checkpoint": str(r0),
                "source_checkpoint_sha256": R0_SHA256,
                "missing_initialized_parameters": missing,
                "policy": "exact frozen R0 state plus deterministic zero-output hard-gated Pedestrian refinement",
                "trainable_parameters": trainable,
            },
        },
        output,
    )
    return {"checkpoint": str(output), "checkpoint_sha256": sha256_file(output), "new_parameters": missing, "trainable_parameters": trainable}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare bounded R2b frozen hard-gated Pedestrian refinement.")
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--r0-selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20268)
    args = parser.parse_args()
    if not 0 <= args.seed <= 65535:
        raise ValueError("MonoDETR requires 0 <= seed <= 65535 because it calls np.random.seed(seed ** 2)")

    import yaml

    repo = args.monodetr_repo.resolve()
    output_root = args.output_root.resolve()
    base_path = args.base_config.resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    r0 = resolve_r0_checkpoint(args.r0_selection.resolve())
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    base["dataset"].update({"train_split": "train", "test_split": "val"})
    configs = {}
    refinement_init = None
    for name, enabled in VARIANTS.items():
        config = yaml.safe_load(yaml.safe_dump(base))
        run_name = f"monodetr_r2b_{name}_gate{args.gate_epochs}"
        config["model_name"] = run_name
        config["random_seed"] = args.seed
        config["model"]["pedestrian_refinement"] = {
            "enabled": enabled,
            "grid_size": 3,
            "residual_scale": 0.1,
            "gate_mode": "hard",
            "freeze_base": True,
        }
        pretrain = r0
        if enabled:
            init_path = output_root / "initialization/r2b_ped_refinement_init.pth"
            refinement_init = create_refinement_initialization(
                repo, config, r0, init_path, args.seed
            )
            pretrain = init_path
        config["optimizer"]["lr"] = args.learning_rate
        config["trainer"].update(
            {
                "max_epoch": args.gate_epochs,
                "save_frequency": args.gate_epochs,
                "save_all": True,
                "save_path": os.path.relpath(output_root, repo),
                "pretrain_model": str(pretrain),
                "resume_model": False,
                "log_frequency": 20,
            }
        )
        config["tester"].update(
            {"mode": "single", "checkpoint": args.gate_epochs, "threshold": 0.001, "topk": 50}
        )
        config_path = repo / f"configs/{run_name}.yaml"
        config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        configs[name] = {
            "pedestrian_refinement_enabled": enabled,
            "config": str(config_path),
            "run_name": run_name,
            "run_dir": str(output_root / run_name),
            "pretrain_model": str(pretrain),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "experiment": "R2b frozen hard-gated stride-4 Pedestrian refinement gate",
        "controlled_change": "train only refinement projection/head; hard-gate residual to frozen native Pedestrian predictions",
        "r0_checkpoint": str(r0),
        "r0_checkpoint_sha256": R0_SHA256,
        "refinement_initialization": refinement_init,
        "gate_epochs": args.gate_epochs,
        "learning_rate": args.learning_rate,
        "random_seed": args.seed,
        "transformer_changed": False,
        "base_model_frozen": True,
        "gate_mode": "hard",
        "backbone_weights_changed_at_initialization": False,
        "depth_dimension_yaw_heads_changed": False,
        "sampling_changed": False,
        "distillation_enabled": False,
        "temperature_scaling_enabled": False,
        "variants": configs,
    }
    path = output_root / "r2b_gate_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
