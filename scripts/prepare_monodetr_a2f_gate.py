from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


A2_EPOCH = 130
A2_SHA256 = "ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict:
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def resolve_a2_checkpoint(selection_path: Path) -> Path:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("complete"):
        raise RuntimeError("A2 selection is incomplete")
    if int(selection.get("selected_epoch", -1)) != A2_EPOCH:
        raise RuntimeError(f"Expected A2 epoch {A2_EPOCH}")
    checkpoint = Path(selection["selected_checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual = sha256_file(checkpoint)
    if actual != A2_SHA256:
        raise RuntimeError(f"A2 checkpoint SHA-256 mismatch: {actual}")
    return checkpoint


def main() -> None:
    import torch
    import yaml

    parser = argparse.ArgumentParser(description="Prepare the paired A2f stride-4 gate.")
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--a2-selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20268)
    args = parser.parse_args()
    if not 0 <= args.seed <= 65535:
        raise ValueError("MonoDETR requires 0 <= seed <= 65535")

    repo = args.monodetr_repo.resolve()
    base_path = args.base_config.resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    a2_checkpoint = resolve_a2_checkpoint(args.a2_selection.resolve())
    a2_payload = load_checkpoint(a2_checkpoint)
    a2_state = a2_payload.get("model_state")
    if a2_state is None:
        raise RuntimeError("A2 checkpoint has no model_state")

    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    base["dataset"]["train_split"] = "train"
    base["dataset"]["test_split"] = "val"
    variants: dict[str, dict] = {}

    control = yaml.safe_load(yaml.safe_dump(base))
    control_name = f"monodetr_a2f_control_gate{args.gate_epochs}"
    control["model_name"] = control_name
    control["random_seed"] = args.seed
    control["optimizer"]["lr"] = args.learning_rate
    control["trainer"].update(
        {
            "max_epoch": args.gate_epochs,
            "save_frequency": args.gate_epochs,
            "save_all": True,
            "save_path": os.path.relpath(output_root, repo),
            "pretrain_model": str(a2_checkpoint),
            "resume_model": False,
            "log_frequency": 20,
        }
    )
    control["tester"].update(
        {"mode": "single", "checkpoint": args.gate_epochs, "threshold": 0.001, "topk": 50}
    )
    control_path = repo / f"configs/{control_name}.yaml"
    control_path.write_text(yaml.safe_dump(control, sort_keys=False), encoding="utf-8")
    variants["control_stride8"] = {
        "architecture": "A2 strides 8/16/32 plus synthetic 64",
        "config": str(control_path),
        "run_name": control_name,
        "run_dir": str(output_root / control_name),
    }

    high = yaml.safe_load(yaml.safe_dump(base))
    high_name = f"monodetr_a2f_stride4_gate{args.gate_epochs}"
    high["model_name"] = high_name
    high["random_seed"] = args.seed
    high["model"]["backbone_out_indices"] = [1, 2, 3, 4]
    high["model"]["backbone_expected_strides"] = [4, 8, 16, 32]
    high["optimizer"]["lr"] = args.learning_rate

    sys.path.insert(0, str(repo))
    from lib.helpers.model_helper import build_model  # noqa: PLC0415

    torch.manual_seed(args.seed)
    high_model, _ = build_model(high["model"])
    high_state = high_model.state_dict()
    copied_exact: list[str] = []
    new_stride4_projection: list[str] = []
    remapped_projections: list[dict] = []
    for name, target in list(high_state.items()):
        if name.startswith("input_proj."):
            if name.startswith("input_proj.0."):
                new_stride4_projection.append(name)
                continue
            source_name = "input_proj." + str(int(name.split(".")[1]) - 1) + "." + ".".join(name.split(".")[2:])
            source = a2_state.get(source_name)
            if source is None or source.shape != target.shape:
                raise RuntimeError(
                    f"Cannot remap A2 projection {source_name} to {name}: "
                    f"source={None if source is None else list(source.shape)}, target={list(target.shape)}"
                )
            high_state[name] = source
            remapped_projections.append({"source": source_name, "target": name})
            continue
        source = a2_state.get(name)
        if source is None or source.shape != target.shape:
            raise RuntimeError(
                f"Unexpected A2f incompatibility for {name}: "
                f"source={None if source is None else list(source.shape)}, target={list(target.shape)}"
            )
        high_state[name] = source
        copied_exact.append(name)
    high_model.load_state_dict(high_state, strict=True)
    init_checkpoint = output_root / "checkpoint_a2f_stride4_init.pth"
    torch.save(
        {
            "epoch": 0,
            "model_state": high_model.state_dict(),
            "optimizer_state": None,
            "best_result": 0.0,
            "best_epoch": 0,
            "mobileadas3d_initialization": {
                "source_a2_sha256": A2_SHA256,
                "copied_exact_tensor_count": len(copied_exact),
                "remapped_projection_tensor_count": len(remapped_projections),
                "new_stride4_projection_tensor_count": len(new_stride4_projection),
            },
        },
        init_checkpoint,
    )
    high["trainer"].update(
        {
            "max_epoch": args.gate_epochs,
            "save_frequency": args.gate_epochs,
            "save_all": True,
            "save_path": os.path.relpath(output_root, repo),
            "pretrain_model": str(init_checkpoint),
            "resume_model": False,
            "log_frequency": 20,
        }
    )
    high["tester"].update(
        {"mode": "single", "checkpoint": args.gate_epochs, "threshold": 0.001, "topk": 50}
    )
    high_path = repo / f"configs/{high_name}.yaml"
    high_path.write_text(yaml.safe_dump(high, sort_keys=False), encoding="utf-8")
    variants["stride4_feature"] = {
        "architecture": "real MobileNetV4 strides 4/8/16/32",
        "config": str(high_path),
        "run_name": high_name,
        "run_dir": str(output_root / high_name),
        "initial_checkpoint": str(init_checkpoint),
    }

    manifest = {
        "schema_version": 1,
        "complete": True,
        "experiment": "A2f paired higher-resolution feature gate",
        "controlled_change": "replace synthetic stride-64 feature with real MobileNetV4 stride-4 feature",
        "a2_checkpoint": str(a2_checkpoint),
        "a2_checkpoint_sha256": A2_SHA256,
        "gate_epochs": args.gate_epochs,
        "learning_rate": args.learning_rate,
        "random_seed": args.seed,
        "num_feature_levels": 4,
        "transformer_changed": False,
        "heads_changed": False,
        "losses_changed": False,
        "sampling_changed": False,
        "distillation_enabled": False,
        "temperature_scaling_enabled": False,
        "variants": variants,
    }
    path = output_root / "a2f_gate_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
