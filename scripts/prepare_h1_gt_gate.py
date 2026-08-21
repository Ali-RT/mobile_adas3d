from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_s1_gt_baseline import validate_r0_selection
from tools.config import load_config


def main() -> None:
    import yaml

    parser = argparse.ArgumentParser(description="Prepare the GT-only H1 20-epoch gate.")
    parser.add_argument("--base-config", type=Path, required=True)
    parser.add_argument("--r0-selection", type=Path, required=True)
    parser.add_argument("--edge-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="mobileadas3d_h1_gt_gate20")
    parser.add_argument("--skip-checkpoint-hash", action="store_true")
    args = parser.parse_args()

    selection = validate_r0_selection(
        args.r0_selection.resolve(),
        verify_checkpoint=not args.skip_checkpoint_hash,
    )
    edge = json.loads(args.edge_evidence.read_text(encoding="utf-8"))
    if edge.get("complete") is not True or edge.get("architecture") != "MobileADAS3D-H1":
        raise RuntimeError("H1 edge preflight evidence is incomplete or mismatched")
    if float(edge["coreml_max_abs_delta"]) > 0.002:
        raise RuntimeError("H1 edge evidence fails Core ML parity")
    if float(edge["device_gate"]["p95_ms"]) > 35.0:
        raise RuntimeError("H1 edge evidence fails physical-device latency")

    config = load_config(str(args.base_config.resolve()))
    if config["model"]["name"] != "MobileADAS3D-H1":
        raise RuntimeError("Gate config must build MobileADAS3D-H1")
    if config.get("distillation", {}).get("enabled") is not False:
        raise RuntimeError("H1 GT gate requires distillation.enabled=false")
    if config.get("loss", {}).get("type") != "h1_hungarian_set":
        raise RuntimeError("H1 GT gate requires the Hungarian set criterion")
    if config["dataset"]["classes"] != ["Vehicle", "Pedestrian"]:
        raise RuntimeError("H1 GT gate requires Vehicle/Pedestrian taxonomy")
    if int(config["training"]["epochs"]) != 20:
        raise RuntimeError("H1 health gate must stop at epoch 20")

    config["logging"]["run_name"] = args.run_name
    config["outputs"]["output_dir"] = str(args.output_dir.resolve())
    config["outputs"]["profile_output_dirs"]["colab_drive"] = str(
        args.output_dir.resolve()
    )
    selected = selection["selected"]
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
    runtime_config = args.config_dir / "mobileadas3d_h1_gt_gate20.yaml"
    runtime_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "MobileADAS3D-H1 GT-only 20-epoch health gate",
        "run_name": args.run_name,
        "architecture": "MobileADAS3D-H1",
        "classes": ["Vehicle", "Pedestrian"],
        "criterion": "h1_hungarian_set",
        "distillation_enabled": False,
        "epochs": 20,
        "runtime_config": str(runtime_config),
        "edge_evidence": str(args.edge_evidence.resolve()),
        "edge_evidence_summary": edge,
        "r0_selection_file": str(args.r0_selection.resolve()),
        "r0": config["reference"],
    }
    manifest_path = args.output_dir / "h1_gt_gate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
