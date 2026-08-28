from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


A2_EPOCH = 130
A2_SHA256 = "ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4"
VARIANTS = {
    "control_w1_0": 1.0,
    "smallped_w1_5": 1.5,
    "smallped_w2_0": 2.0,
    "smallped_w2_5": 2.5,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser = argparse.ArgumentParser(description="Prepare paired A2e small-Pedestrian box-loss gates.")
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--a2-selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--gate-epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=20268)
    parser.add_argument("--small-height-px", type=float, default=64.0)
    args = parser.parse_args()
    if not 0 <= args.seed <= 65535:
        raise ValueError("MonoDETR requires 0 <= seed <= 65535 because it calls np.random.seed(seed ** 2)")
    if not 0.0 < args.small_height_px <= 384.0:
        raise ValueError("small-height-px must be in (0, 384]")

    import yaml

    repo = args.monodetr_repo.resolve()
    base_path = args.base_config.resolve()
    if not base_path.is_file():
        raise FileNotFoundError(base_path)
    output_root = args.output_root.resolve()
    checkpoint = resolve_a2_checkpoint(args.a2_selection.resolve())
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    base["dataset"]["train_split"] = "train"
    base["dataset"]["test_split"] = "val"
    configs = {}
    for name, weight in VARIANTS.items():
        config = yaml.safe_load(yaml.safe_dump(base))
        run_name = f"monodetr_a2e_{name}_gate{args.gate_epochs}"
        config["model_name"] = run_name
        config["random_seed"] = args.seed
        config["model"]["small_pedestrian_box_loss_weight"] = weight
        config["model"]["small_pedestrian_box_height_norm"] = args.small_height_px / 384.0
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
            "small_pedestrian_box_loss_weight": weight,
            "small_pedestrian_box_height_px": args.small_height_px,
            "config": str(config_path),
            "run_name": run_name,
            "run_dir": str(output_root / run_name),
        }

    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "experiment": "A2e paired small-Pedestrian 2D box/GIoU loss gate",
        "controlled_change": "matched Pedestrian boxes at or below the frozen height threshold",
        "a2_checkpoint": str(checkpoint),
        "a2_checkpoint_sha256": A2_SHA256,
        "gate_epochs": args.gate_epochs,
        "learning_rate": args.learning_rate,
        "random_seed": args.seed,
        "small_pedestrian_box_height_px": args.small_height_px,
        "size_signal_source": "training target box height after canonical 1280x384 transform",
        "matcher_changed": False,
        "classification_changed": False,
        "geometry_3d_changed": False,
        "sampling_changed": False,
        "distillation_enabled": False,
        "temperature_scaling_enabled": False,
        "variants": configs,
    }
    path = output_root / "a2e_gate_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
