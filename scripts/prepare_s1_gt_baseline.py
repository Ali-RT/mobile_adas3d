from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


EXPECTED_R0_SHA256 = "fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59"
EXPECTED_R0_EPOCH = 185
EXPECTED_VEHICLE_3D = 17.634769196266316
EXPECTED_PEDESTRIAN_3D = 5.721371354710236


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_r0_selection(path: Path, verify_checkpoint: bool = True) -> dict[str, Any]:
    selection = json.loads(path.read_text(encoding="utf-8"))
    if selection.get("complete") is not True or selection.get("evaluated_checkpoints") != 39:
        raise RuntimeError("R0 product sweep is incomplete")
    selected = selection.get("selected", {})
    expected = {
        "epoch": EXPECTED_R0_EPOCH,
        "checkpoint_sha256": EXPECTED_R0_SHA256,
        "vehicle_3d_moderate": EXPECTED_VEHICLE_3D,
        "pedestrian_3d_moderate": EXPECTED_PEDESTRIAN_3D,
    }
    for key, value in expected.items():
        if selected.get(key) != value:
            raise RuntimeError(
                f"Frozen R0 selection mismatch for {key}: {selected.get(key)!r} != {value!r}"
            )
    checkpoint = Path(selected["checkpoint"])
    if verify_checkpoint:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual_hash = sha256_file(checkpoint)
        if actual_hash != EXPECTED_R0_SHA256:
            raise RuntimeError(
                f"R0 checkpoint SHA-256 mismatch: {actual_hash} != {EXPECTED_R0_SHA256}"
            )
    return selection


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(
        description="Freeze R0 provenance and prepare GT-only S1 gate/full configs."
    )
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--r0-selection", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="mobileadas3d_s1_gt_baseline")
    parser.add_argument("--gate-epochs", type=int, default=20)
    parser.add_argument("--full-epochs", type=int, default=100)
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    args = parser.parse_args()

    from tools.config import load_config

    selection = validate_r0_selection(
        args.r0_selection.resolve(),
        verify_checkpoint=not args.skip_checkpoint_hash,
    )
    config = load_config(str(args.base_config.resolve()))
    if config["model"]["name"] != "MobileADAS3D-S1":
        raise RuntimeError("GT baseline config must build MobileADAS3D-S1")
    yaw_encoding = str(config["model"].get("yaw_encoding", "axis_direction"))
    if yaw_encoding not in ("axis_direction", "continuous_sincos"):
        raise RuntimeError(f"Unsupported S1 yaw encoding: {yaw_encoding!r}")
    if yaw_encoding == "continuous_sincos":
        heads = config["model"].get("heads", {})
        if heads.get("yaw") is not True:
            raise RuntimeError("Continuous-yaw S1 requires heads.yaw=true")
        if heads.get("yaw_axis") is not False or heads.get("yaw_direction") is not False:
            raise RuntimeError(
                "Continuous-yaw S1 must disable yaw_axis and yaw_direction heads"
            )
        if float(config["loss"].get("yaw_direction_weight", 0.0)) != 0.0:
            raise RuntimeError("Continuous-yaw S1 requires yaw_direction_weight=0")
        if not bool(config["loss"].get("detach_yaw_in_corner3d", False)):
            raise RuntimeError(
                "Continuous-yaw S1 requires detach_yaw_in_corner3d=true"
            )
        if not bool(config["loss"].get("yaw_pred_is_direct_sincos", False)):
            raise RuntimeError(
                "Continuous-yaw S1 requires yaw_pred_is_direct_sincos=true"
            )
        yaw_norm_floor = float(config["loss"].get("yaw_norm_floor", 0.0))
        if not 0.0 < yaw_norm_floor <= 1.0:
            raise RuntimeError(
                "Continuous-yaw S1 requires yaw_norm_floor in (0, 1]"
            )
    if config.get("distillation", {}).get("enabled") is not False:
        raise RuntimeError("GT baseline must keep distillation.enabled=false")
    if config["dataset"]["classes"] != ["Vehicle", "Pedestrian"]:
        raise RuntimeError("GT baseline classes must be Vehicle/Pedestrian")
    if args.gate_epochs < 1 or args.full_epochs < args.gate_epochs:
        raise ValueError("Require 1 <= gate_epochs <= full_epochs")

    selected = selection["selected"]
    config["logging"]["run_name"] = args.run_name
    config["outputs"]["profile_output_dirs"]["colab_drive"] = str(
        args.output_dir.resolve()
    )
    config["outputs"]["output_dir"] = str(args.output_dir.resolve())
    config["distillation"] = {"enabled": False}
    config["reference"] = {
        "name": "MonoDETR-R0",
        "selected_epoch": selected["epoch"],
        "checkpoint": selected["checkpoint"],
        "checkpoint_sha256": selected["checkpoint_sha256"],
        "vehicle_3d_moderate": selected["vehicle_3d_moderate"],
        "pedestrian_3d_moderate": selected["pedestrian_3d_moderate"],
        "vehicle_75pct_minimum": selected["vehicle_3d_moderate"] * 0.75,
        "pedestrian_75pct_minimum": selected["pedestrian_3d_moderate"] * 0.75,
        "vehicle_bev_moderate_minimum": 20.0,
    }

    args.config_dir.mkdir(parents=True, exist_ok=True)
    gate_config = dict(config)
    gate_config["training"] = dict(config["training"])
    gate_config["training"]["epochs"] = args.gate_epochs
    gate_path = args.config_dir / "mobileadas3d_s1_gt_gate20.yaml"
    gate_path.write_text(yaml.safe_dump(gate_config, sort_keys=False), encoding="utf-8")

    full_config = dict(config)
    full_config["training"] = dict(config["training"])
    full_config["training"]["epochs"] = args.full_epochs
    full_path = args.config_dir / "mobileadas3d_s1_gt_full100.yaml"
    full_path.write_text(yaml.safe_dump(full_config, sort_keys=False), encoding="utf-8")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": (
            "MobileADAS3D-S1-V2b bounded-yaw GT-only baseline"
            if yaw_encoding == "continuous_sincos" and yaw_norm_floor == 0.1
            else "MobileADAS3D-S1-V2 continuous-yaw GT-only baseline"
            if yaw_encoding == "continuous_sincos"
            else "MobileADAS3D-S1 GT-only baseline"
        ),
        "run_name": args.run_name,
        "architecture": config["model"]["name"],
        "backbone": config["model"]["backbone"],
        "yaw_encoding": yaw_encoding,
        "yaw_norm_floor": (
            yaw_norm_floor if yaw_encoding == "continuous_sincos" else None
        ),
        "classes": config["dataset"]["classes"],
        "distillation_enabled": False,
        "seed": config["training"]["seed"],
        "gate_epochs": args.gate_epochs,
        "full_epochs": args.full_epochs,
        "gate_config": str(gate_path),
        "full_config": str(full_path),
        "r0_selection_file": str(args.r0_selection.resolve()),
        "r0": config["reference"],
    }
    manifest_path = args.output_dir / "s1_gt_baseline_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
