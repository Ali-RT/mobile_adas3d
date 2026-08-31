from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


R0_EPOCH = 185
R0_SHA256 = "fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59"
VARIANTS = {
    "control_match_w1_0": 1.0,
    "ped_match_w2_0": 2.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_r0_checkpoint(selection_path: Path) -> Path:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if not selection.get("complete"):
        raise RuntimeError("R0 selection is incomplete")
    selected = selection.get("selected", {})
    if int(selected.get("epoch", -1)) != R0_EPOCH:
        raise RuntimeError(f"Expected R0 epoch {R0_EPOCH}")
    if selected.get("checkpoint_sha256") != R0_SHA256:
        raise RuntimeError("R0 selection records an unexpected checkpoint SHA-256")
    checkpoint = Path(selected["checkpoint"]).resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    actual = sha256_file(checkpoint)
    if actual != R0_SHA256:
        raise RuntimeError(f"R0 checkpoint SHA-256 mismatch: {actual}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare paired R1 Pedestrian matcher-cost gate.")
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--r0-selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20268)
    args = parser.parse_args()
    if not 0 <= args.seed <= 65535:
        raise ValueError("MonoDETR requires 0 <= seed <= 65535 because it calls np.random.seed(seed ** 2)")

    import yaml

    repo = args.monodetr_repo.resolve()
    base_path = args.base_config.resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    output_root = args.output_root.resolve()
    checkpoint = resolve_r0_checkpoint(args.r0_selection.resolve())
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    base["dataset"]["train_split"] = "train"
    base["dataset"]["test_split"] = "val"
    configs = {}
    for name, weight in VARIANTS.items():
        config = yaml.safe_load(yaml.safe_dump(base))
        run_name = f"monodetr_r1_{name}_gate{args.gate_epochs}"
        config["model_name"] = run_name
        config["random_seed"] = args.seed
        config["model"]["pedestrian_localization_cost_weight"] = weight
        config["optimizer"]["lr"] = args.learning_rate
        config["trainer"].update(
            {
                "max_epoch": args.gate_epochs,
                "save_frequency": args.gate_epochs,
                "save_all": True,
                "save_path": os.path.relpath(output_root, repo),
                "pretrain_model": str(checkpoint),
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
            "pedestrian_localization_cost_weight": weight,
            "config": str(config_path),
            "run_name": run_name,
            "run_dir": str(output_root / run_name),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "experiment": "R1 paired Pedestrian Hungarian localization-cost gate",
        "controlled_change": "Pedestrian 2D box L1 and GIoU assignment costs only",
        "r0_checkpoint": str(checkpoint),
        "r0_checkpoint_sha256": R0_SHA256,
        "gate_epochs": args.gate_epochs,
        "learning_rate": args.learning_rate,
        "random_seed": args.seed,
        "matcher_changed": True,
        "classification_changed": False,
        "geometry_3d_changed": False,
        "sampling_changed": False,
        "distillation_enabled": False,
        "temperature_scaling_enabled": False,
        "variants": configs,
    }
    path = output_root / "r1_gate_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
