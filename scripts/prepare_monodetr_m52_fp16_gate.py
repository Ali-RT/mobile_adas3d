from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


R0_EPOCH = 185
R0_SHA256 = "fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="Prepare evaluation-only M52 R0 FP16 gate.")
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--r0-selection", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    selection = json.loads(args.r0_selection.read_text(encoding="utf-8"))
    selected = selection.get("selected", {})
    if not selection.get("complete") or int(selected.get("epoch", -1)) != R0_EPOCH:
        raise RuntimeError("Expected complete frozen R0 epoch-185 selection")
    source = Path(selected["checkpoint"]).resolve()
    if selected.get("checkpoint_sha256") != R0_SHA256 or sha256_file(source) != R0_SHA256:
        raise RuntimeError("Frozen R0 checkpoint SHA-256 mismatch")
    repo = args.monodetr_repo.resolve()
    output = args.output_root.resolve()
    run_name = "monodetr_m52_r0_fp16_autocast"
    run_dir = output / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = run_dir / f"checkpoint_epoch_{R0_EPOCH}.pth"
    if not checkpoint.is_file() or sha256_file(checkpoint) != R0_SHA256:
        shutil.copy2(source, checkpoint)
    config = yaml.safe_load(args.base_config.read_text(encoding="utf-8"))
    config["model_name"] = run_name
    config["tester"].update({"mode": "single", "checkpoint": R0_EPOCH, "threshold": 0.001, "topk": 50, "inference_precision": "fp16_autocast"})
    config["trainer"].update({"save_path": os.path.relpath(output, repo), "save_all": True, "pretrain_model": None, "resume_model": False})
    config_path = repo / "configs/monodetr_m52_r0_fp16_autocast.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    manifest = {
        "schema_version": 1, "complete": True,
        "experiment": "M52 R0 selective mixed-precision evaluation-only gate",
        "training_authorized": False, "architecture_changed": False,
        "precision": "fp16_autocast_with_fp32_feature_depth_and_deformable_attention", "r0_epoch": R0_EPOCH,
        "r0_checkpoint": str(source), "r0_checkpoint_sha256": R0_SHA256,
        "evaluation_checkpoint": str(checkpoint), "config": str(config_path),
        "run_dir": str(run_dir), "output_root": str(output),
    }
    path = output / "m52_fp16_gate_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
