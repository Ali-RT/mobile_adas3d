from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

try:
    from scripts.prepare_monodgp_m55_feasibility import (
        PARENT_CHECKPOINT_SHA256,
        PINNED_COMMIT,
    )
    from scripts.prepare_monodgp_m56_fp16_storage import sha256_file
except ModuleNotFoundError:
    from prepare_monodgp_m55_feasibility import (
        PARENT_CHECKPOINT_SHA256,
        PINNED_COMMIT,
    )
    from prepare_monodgp_m56_fp16_storage import sha256_file


M56D_MANIFEST_SHA256 = "ce2cf03c37530f924f75f0f44ea94fe233d8aac916c5c51fb26cd8bfa84d0bc0"
M56D_SMOKE_SHA256 = "56678d529fda787441f842453e5f198c3d5017b4a0d8cb82373110713f737f87"
M56D_GATE_SHA256 = "4012d800c3972e15922ca0d3dd530cad8437f575f433d93b3708182a062227ad"
M56D_COMPARISON_SHA256 = "d64529315414d841f2658f69442722c2548e783a375f5bfcb9a46bdb2937c95f"
M56D_CANDIDATE_SHA256 = "7d18883d6f998e7616beaa45b92d348b22728f3fad87a891618b5ce5a1cde17c"
M56D_RUNTIME_CONFIG_SHA256 = "4ad6d50241e5a6dd552e5d8b9c043241a11377c26c59b84ab3d2f7f58c7a42af"
M57_PATCHED_SOURCE_SHA256 = "517cbdb0ab3686f8f708f633c816b971ed1ed99d0be270d7d8d27bc1195f3fbd"
M56D_POLICY_ID = "m56d_det2d_transformer_fp32"
M57_POLICY_ID = "m57_rank5_grid_sample_decomposition"
PORTABLE_ENV = "MONODGP_PORTABLE_DEFORM_ATTN"
PORTABLE_SPATIAL_SHAPES = ((48, 160), (24, 80), (12, 40), (6, 20))
EXPECTED_PATCHED_FILES = {
    "lib/datasets/kitti/kitti_dataset.py",
    "lib/helpers/save_helper.py",
    "lib/helpers/trainer_helper.py",
    "lib/models/monodgp/ops/modules/ms_deform_attn.py",
    "lib/models/monodgp/ops/setup.py",
    "lib/models/monodgp/ops/src/cuda/ms_deform_attn_cuda.cu",
    "tools/train_val.py",
}
EXPECTED_MODULES = (
    "det2d_transformer.encoder.layers.0.self_attn",
    "det2d_transformer.encoder.layers.1.self_attn",
    "det2d_transformer.encoder.layers.2.self_attn",
    "det2d_transformer.decoder.layers.0.cross_attn",
    "det2d_transformer.decoder.layers.1.cross_attn",
    "det2d_transformer.decoder.layers.2.cross_attn",
    "det3d_transformer.decoder.layers.0.cross_attn",
    "det3d_transformer.decoder.layers.1.cross_attn",
    "det3d_transformer.decoder.layers.2.cross_attn",
)
EXPECTED_M56D_METRICS = {
    "vehicle_3d_moderate": 19.46660516354734,
    "pedestrian_3d_moderate": 6.184783472196555,
    "mean_3d_moderate": 12.825694317871946,
    "vehicle_bev_moderate": 25.734914255182563,
    "pedestrian_bev_moderate": 6.765056067096485,
    "vehicle_near_recall": 0.9099254609650843,
    "pedestrian_near_recall": 0.7239858906525574,
    "pedestrian_localization_failure_rate": 0.23853615520282187,
}


def same_numbers(actual: dict, expected: dict, tolerance: float = 1e-12) -> bool:
    return set(actual) == set(expected) and all(
        abs(float(actual[key]) - float(value)) <= tolerance
        for key, value in expected.items()
    )


def validate_m56d_evidence(
    manifest_path: Path,
    smoke_path: Path,
    gate_path: Path,
    comparison_path: Path,
) -> tuple[dict, dict, dict, list[dict]]:
    expected_hashes = {
        manifest_path: M56D_MANIFEST_SHA256,
        smoke_path: M56D_SMOKE_SHA256,
        gate_path: M56D_GATE_SHA256,
        comparison_path: M56D_COMPARISON_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"M56d SHA-256 mismatch for {path}: {actual} != {expected}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    smoke = json.loads(smoke_path.read_text(encoding="utf-8"))
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    with comparison_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    comparison_values = {
        row["metric"]: float(row["candidate"])
        for row in rows
        if row["metric"] != "prediction_files"
    }
    prediction_rows = [row for row in rows if row["metric"] == "prediction_files"]

    if (
        manifest.get("complete") is not True
        or manifest.get("compression_policy", {}).get("policy_id") != M56D_POLICY_ID
        or manifest.get("parent_checkpoint_sha256") != PARENT_CHECKPOINT_SHA256
        or manifest.get("candidate_checkpoint_sha256") != M56D_CANDIDATE_SHA256
        or float(manifest.get("model_only_checkpoint_size_ratio", 1.0)) > 0.60
        or manifest.get("training_performed") is not False
        or manifest.get("graph_changed") is not False
        or smoke.get("complete") is not True
        or smoke.get("manifest_sha256") != M56D_MANIFEST_SHA256
        or smoke.get("candidate_checkpoint_sha256") != M56D_CANDIDATE_SHA256
        or smoke.get("all_smoke_gates_passed") is not True
        or not smoke.get("gate_results")
        or not all(smoke["gate_results"].values())
        or smoke.get("full_evaluation_authorized") is not True
        or gate.get("complete") is not True
        or gate.get("candidate_checkpoint_sha256") != M56D_CANDIDATE_SHA256
        or gate.get("prediction_files") != 3769
        or gate.get("all_preservation_gates_passed") is not True
        or not gate.get("preservation_gate_results")
        or not all(gate["preservation_gate_results"].values())
        or gate.get("offline_compression_candidate_selected") is not True
        or gate.get("direct_coreml_conversion_authorized") is not False
        or gate.get("product_safety_qualified") is not False
        or not same_numbers(gate.get("candidate_metrics", {}), EXPECTED_M56D_METRICS)
        or len(rows) != 9
        or any(row.get("passed") != "True" for row in rows)
        or len(prediction_rows) != 1
        or int(float(prediction_rows[0]["candidate"])) != 3769
        or not same_numbers(comparison_values, EXPECTED_M56D_METRICS)
    ):
        raise RuntimeError("M57 is not bound to the exact selected M56d result")
    return manifest, smoke, gate, rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare the M57 portable deformable-attention parity gate."
    )
    parser.add_argument("--monodgp-repo", type=Path, required=True)
    parser.add_argument("--m56d-manifest", type=Path, required=True)
    parser.add_argument("--m56d-smoke", type=Path, required=True)
    parser.add_argument("--m56d-gate", type=Path, required=True)
    parser.add_argument("--m56d-comparison", type=Path, required=True)
    parser.add_argument(
        "--runtime-config",
        type=Path,
        help=(
            "Hash-identical durable copy of the M56d runtime YAML. "
            "Defaults to the historical path recorded in the M56d manifest."
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    import torch
    import yaml

    repo = args.monodgp_repo.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDGP {PINNED_COMMIT}, found {commit}")

    m56d_manifest, _, m56d_gate, _ = validate_m56d_evidence(
        args.m56d_manifest.resolve(),
        args.m56d_smoke.resolve(),
        args.m56d_gate.resolve(),
        args.m56d_comparison.resolve(),
    )
    checkpoint = Path(m56d_manifest["candidate_checkpoint"]).resolve()
    recorded_runtime_config = Path(m56d_manifest["runtime_config"]).resolve()
    runtime_config = (
        args.runtime_config.resolve()
        if args.runtime_config is not None
        else recorded_runtime_config
    )
    attention_source = repo / "lib/models/monodgp/ops/modules/ms_deform_attn.py"
    if not checkpoint.is_file() or sha256_file(checkpoint) != M56D_CANDIDATE_SHA256:
        raise RuntimeError("Selected M56d checkpoint is missing or changed")
    if (
        not runtime_config.is_file()
        or m56d_manifest["runtime_config_sha256"] != M56D_RUNTIME_CONFIG_SHA256
        or sha256_file(runtime_config) != M56D_RUNTIME_CONFIG_SHA256
    ):
        raise RuntimeError("M56d runtime config is missing or changed")
    if (
        not attention_source.is_file()
        or sha256_file(attention_source) != M57_PATCHED_SOURCE_SHA256
    ):
        raise RuntimeError("Exact M57 attention patch is not present")
    patched_files = set(
        subprocess.run(
            ["git", "diff", "--name-only"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    if patched_files != EXPECTED_PATCHED_FILES:
        raise RuntimeError(f"Unexpected patched MonoDGP files: {patched_files}")

    os.environ.pop(PORTABLE_ENV, None)
    sys.path.insert(0, str(repo))
    from lib.helpers.model_helper import build_model
    from lib.helpers.save_helper import load_checkpoint_safely
    from lib.models.monodgp.ops.modules.ms_deform_attn import MSDeformAttn

    config = yaml.safe_load(runtime_config.read_text(encoding="utf-8"))
    model, _ = build_model(config["model"])
    payload = load_checkpoint_safely(checkpoint, torch.device("cpu"))
    if int(payload.get("epoch", -1)) != 100:
        raise RuntimeError("M56d checkpoint epoch changed")
    model.load_state_dict(payload["model_state"], strict=True)
    modules = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, MSDeformAttn)
    }
    module_names = tuple(modules)
    default_native = all(
        module.use_portable_deform_attn is False for module in modules.values()
    )
    spatial_shapes_match = all(
        tuple(module.portable_spatial_shapes) == PORTABLE_SPATIAL_SHAPES
        for module in modules.values()
    )
    source_text = attention_source.read_text(encoding="utf-8")
    preparation_gates = {
        "m56d_manifest_sha256": sha256_file(args.m56d_manifest.resolve()) == M56D_MANIFEST_SHA256,
        "m56d_smoke_sha256": sha256_file(args.m56d_smoke.resolve()) == M56D_SMOKE_SHA256,
        "m56d_gate_sha256": sha256_file(args.m56d_gate.resolve()) == M56D_GATE_SHA256,
        "m56d_comparison_sha256": sha256_file(args.m56d_comparison.resolve()) == M56D_COMPARISON_SHA256,
        "m56d_selected": m56d_gate.get("offline_compression_candidate_selected") is True,
        "candidate_checkpoint_sha256": sha256_file(checkpoint) == M56D_CANDIDATE_SHA256,
        "runtime_config_sha256": sha256_file(runtime_config) == M56D_RUNTIME_CONFIG_SHA256,
        "pinned_upstream_commit": commit == PINNED_COMMIT,
        "exact_patched_source_sha256": sha256_file(attention_source) == M57_PATCHED_SOURCE_SHA256,
        "exact_patched_file_set": patched_files == EXPECTED_PATCHED_FILES,
        "portable_grid_sample_present": "F.grid_sample(" in source_text,
        "portable_rank5_sampling_present": "level_points, 2" in source_text,
        "native_cuda_path_retained": source_text.count("MSDeformAttnFunction.apply(") == 2,
        "nine_target_modules_exact": module_names == EXPECTED_MODULES,
        "portable_mode_default_false": default_native,
        "fixed_spatial_shapes_exact": spatial_shapes_match,
        "no_training_or_weight_change": True,
        "coreml_and_product_claims_false": True,
    }
    report = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M57 rank-five grid-sample deformable-attention replacement",
        "training_performed": False,
        "weights_changed": False,
        "native_path_default": True,
        "portable_path_opt_in": True,
        "portable_environment_variable": f"{PORTABLE_ENV}=1",
        "replacement_policy": {
            "policy_id": M57_POLICY_ID,
            "target_class": "MSDeformAttn",
            "target_modules": list(EXPECTED_MODULES),
            "target_module_count": len(EXPECTED_MODULES),
            "spatial_shapes": [list(shape) for shape in PORTABLE_SPATIAL_SHAPES],
            "sampling_layout": "rank-five [N,Q,H,L*P,2]",
            "portable_kernel": "torch.nn.functional.grid_sample",
            "native_kernel": "MultiScaleDeformableAttention CUDA extension",
        },
        "upstream_commit": commit,
        "patched_source": str(attention_source),
        "patched_source_sha256": sha256_file(attention_source),
        "patched_files": sorted(patched_files),
        "m56d_manifest": str(args.m56d_manifest.resolve()),
        "m56d_manifest_sha256": M56D_MANIFEST_SHA256,
        "m56d_smoke": str(args.m56d_smoke.resolve()),
        "m56d_smoke_sha256": M56D_SMOKE_SHA256,
        "m56d_gate": str(args.m56d_gate.resolve()),
        "m56d_gate_sha256": M56D_GATE_SHA256,
        "m56d_comparison": str(args.m56d_comparison.resolve()),
        "m56d_comparison_sha256": M56D_COMPARISON_SHA256,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": M56D_CANDIDATE_SHA256,
        "checkpoint_epoch": 100,
        "m56d_metrics": EXPECTED_M56D_METRICS,
        "runtime_config": str(runtime_config),
        "runtime_config_sha256": M56D_RUNTIME_CONFIG_SHA256,
        "m56d_recorded_runtime_config": str(recorded_runtime_config),
        "dataset_root": m56d_manifest["dataset_root"],
        "split_protocol": "chen_3712_3769",
        "preparation_gate_results": preparation_gates,
        "smoke_authorized": all(preparation_gates.values()),
        "full_evaluation_authorized": False,
        "direct_coreml_conversion_authorized": False,
        "product_safety_qualified": False,
        "expected_artifacts": [
            "m57_deformable_attention_manifest.json",
            "m57_deformable_attention_smoke.json",
        ],
    }
    report_path = output / "m57_deformable_attention_manifest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["smoke_authorized"]:
        raise RuntimeError("M57 preparation failed; do not run the CUDA smoke")


if __name__ == "__main__":
    main()
