from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

try:
    from scripts.prepare_monodgp_m55_feasibility import (
        PARENT_CHECKPOINT_SHA256,
        PARENT_METRICS,
        PRESERVATION_GATES,
    )
except ModuleNotFoundError:
    from prepare_monodgp_m55_feasibility import (
        PARENT_CHECKPOINT_SHA256,
        PARENT_METRICS,
        PRESERVATION_GATES,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def same_numbers(actual: dict, expected: dict) -> bool:
    if set(actual) != set(expected):
        return False
    return all(abs(float(actual[key]) - float(value)) <= 1e-12 for key, value in expected.items())


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize the fail-closed M55 feasibility decision.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--operator-audit", type=Path, required=True)
    parser.add_argument("--latency-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    paths = [
        args.manifest.resolve(),
        args.profile.resolve(),
        args.operator_audit.resolve(),
        args.latency_csv.resolve(),
    ]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest = json.loads(paths[0].read_text(encoding="utf-8"))
    profile = json.loads(paths[1].read_text(encoding="utf-8"))
    audit = json.loads(paths[2].read_text(encoding="utf-8"))
    with paths[3].open("r", encoding="utf-8", newline="") as handle:
        latency_rows = list(csv.DictReader(handle))
    latency_values = [float(row["cuda_event_ms"]) for row in latency_rows]

    required_outputs = {
        "pred_logits",
        "pred_boxes",
        "pred_3d_dim",
        "pred_depth",
        "pred_angle",
        "pred_depth_map_logits",
        "pred_region_prob",
    }
    profile_outputs = {
        key.split(".", 1)[0].split("[", 1)[0]
        for key in profile.get("outputs", {})
    }
    inventory = profile.get("parameter_inventory", {})
    memory = profile.get("memory", {})
    latency = profile.get("latency", {})
    compute = profile.get("compute", {})
    gates = {
        "manifest_complete": manifest.get("complete") is True,
        "no_training_or_compression": (
            manifest.get("training_performed") is False
            and manifest.get("compression_performed") is False
            and profile.get("training_performed") is False
            and profile.get("compression_performed") is False
        ),
        "parent_checkpoint_sha256": (
            manifest.get("parent_checkpoint_sha256") == PARENT_CHECKPOINT_SHA256
            and profile.get("checkpoint_sha256") == PARENT_CHECKPOINT_SHA256
            and audit.get("checkpoint_sha256") == PARENT_CHECKPOINT_SHA256
        ),
        "checkpoint_payload_epoch": profile.get("checkpoint_epoch") == 100,
        "parent_metrics_frozen": same_numbers(manifest.get("parent_metrics", {}), PARENT_METRICS),
        "preservation_gates_frozen": same_numbers(
            manifest.get("preservation_gates", {}), PRESERVATION_GATES
        ),
        "profile_complete": profile.get("complete") is True,
        "manifest_profile_binding": profile.get("manifest_sha256") == sha256_file(paths[0]),
        "runtime_config_binding": (
            profile.get("runtime_config_sha256") == manifest.get("runtime_config_sha256")
        ),
        "cuda_device_recorded": bool(profile.get("device", {}).get("name")),
        "finite_required_outputs": (
            profile.get("finite_outputs") is True and required_outputs.issubset(profile_outputs)
        ),
        "five_warmups_100_predictions": (
            latency.get("warmup_runs") == 5
            and latency.get("timed_runs") == 100
            and len(latency_rows) == 100
            and all(math.isfinite(value) and value > 0 for value in latency_values)
        ),
        "latency_summary_finite": all(
            math.isfinite(float(latency.get(key, float("nan"))))
            and float(latency.get(key, 0)) > 0
            for key in ("mean_ms", "median_ms", "p95_ms", "min_ms", "max_ms")
        ),
        "model_size_recorded": (
            int(profile.get("checkpoint_bytes", 0)) > 0
            and int(inventory.get("total_parameters", 0)) > 0
            and int(inventory.get("parameter_bytes", 0)) > 0
            and int(inventory.get("state_dict_tensor_bytes", 0)) > 0
        ),
        "cuda_memory_recorded": all(
            int(memory.get(key, 0)) > 0
            for key in (
                "loaded_allocated_bytes",
                "loaded_reserved_bytes",
                "peak_allocated_bytes",
                "peak_reserved_bytes",
            )
        ),
        "partial_compute_accounting_documented": (
            compute.get("partial_lower_bound") is True
            and bool(compute.get("note"))
            and (
                (compute.get("complete") is True and int(compute.get("profiled_flops", -1)) >= 0)
                or (compute.get("complete") is False and bool(compute.get("error")))
            )
        ),
        "operator_audit_complete": audit.get("complete") is True,
        "ordinary_weight_scope_available": (
            audit.get("offline_weight_compression_scope_available") is True
            and int(audit.get("eligible_parameter_bytes", 0)) > 0
            and 0 < float(audit.get("eligible_parameter_fraction", 0)) <= 1
        ),
        "custom_attention_blocker_documented": (
            audit.get("custom_cuda_extension_required") is True
            and bool(audit.get("custom_deformable_attention_modules"))
            and bool(audit.get("known_export_blockers"))
            and audit.get("direct_coreml_export_ready") is False
            and audit.get("direct_coreml_conversion_authorized") is False
        ),
    }
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "complete": True,
        "experiment": "M55 M54-parent compression baseline and export feasibility",
        "parent_epoch": 100,
        "parent_checkpoint_sha256": PARENT_CHECKPOINT_SHA256,
        "parent_metrics": PARENT_METRICS,
        "compression_preservation_gates": PRESERVATION_GATES,
        "gate_results": gates,
        "all_feasibility_gates_passed": passed,
        "offline_weight_compression_authorized": passed,
        "direct_coreml_conversion_authorized": False,
        "direct_coreml_blocker": (
            "custom MultiScaleDeformableAttention must be decomposed or replaced with raw-output parity"
        ),
        "latency_baseline": latency,
        "parameter_inventory": inventory,
        "weight_compression_scope": {
            "families": audit.get("ordinary_weight_operator_families"),
            "eligible_parameter_bytes": audit.get("eligible_parameter_bytes"),
            "eligible_parameter_fraction": audit.get("eligible_parameter_fraction"),
        },
        "compute_profile": profile.get("compute"),
        "product_safety_qualified": False,
        "unmet_product_target": {
            "metric": "pedestrian_near_recall",
            "parent": PARENT_METRICS["pedestrian_near_recall"],
            "target": 0.80,
        },
        "next_step_if_passed": (
            "M56 controlled offline weight-compression sensitivity with complete M54 preservation evaluation"
        ),
        "artifacts": {
            "manifest": str(paths[0]),
            "profile": str(paths[1]),
            "profile_sha256": sha256_file(paths[1]),
            "operator_audit": str(paths[2]),
            "operator_audit_sha256": sha256_file(paths[2]),
            "latency_csv": str(paths[3]),
            "latency_csv_sha256": sha256_file(paths[3]),
        },
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError("M55 feasibility gate failed; do not start weight compression")


if __name__ == "__main__":
    main()
