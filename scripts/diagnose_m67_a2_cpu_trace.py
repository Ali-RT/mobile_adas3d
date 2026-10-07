"""Isolate M67 A2 portable-export drift across CUDA, CPU eager, and CPU trace."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import m64_teacher_qualification as q
from audit_m67_a2_coreml import (
    OUTPUT_KEYS,
    OUTPUT_NAMES,
    REVISION,
    arrays,
    compare_outputs,
    load_manifest,
    set_export,
    signed,
    verify_signed,
)

DIAGNOSTIC_REVISION = "M67B-A2-CPU-TRACE-DIAGNOSTIC-2026-10-07-r2"
PAIR_NAMES = (
    "cuda_export_vs_native_cuda",
    "cpu_export_vs_cuda_export",
    "trace_vs_cpu_export",
    "cpu_export_vs_native_cuda",
    "trace_vs_native_cuda",
)


def flatten_tensors(value, torch, path="root"):
    """Return named NumPy leaves from tensors, containers, and KITTI feature bundles."""
    if torch.is_tensor(value):
        return {path: value.detach().cpu().numpy().copy()}
    if isinstance(value, dict):
        leaves = {}
        for key in sorted(value, key=str):
            leaves.update(flatten_tensors(value[key], torch, f"{path}.{key}"))
        return leaves
    if isinstance(value, (tuple, list)):
        leaves = {}
        for index, item in enumerate(value):
            leaves.update(flatten_tensors(item, torch, f"{path}[{index}]"))
        return leaves
    # MonoDETR's backbone returns NestedTensor-like values rather than plain tuples.
    leaves = {}
    for attr in ("tensors", "mask"):
        child = getattr(value, attr, None)
        if child is not None:
            leaves.update(flatten_tensors(child, torch, f"{path}.{attr}"))
    return leaves


def attach_boundary_hooks(model, torch):
    """Capture direct model-component outputs for one forward pass at a time."""
    events = []
    handles = []
    for name, module in model.named_children():
        def capture(_module, _inputs, output, component=name):
            leaves = flatten_tensors(output, torch)
            if leaves:
                events.append({"component": component, "leaves": leaves})

        handles.append(module.register_forward_hook(capture))
    return events, handles


def compare_boundaries(reference_events, actual_events):
    """Compare matching top-level component outputs without imposing product tolerances."""
    rows = []
    count = max(len(reference_events), len(actual_events))
    for event_index in range(count):
        ref = reference_events[event_index] if event_index < len(reference_events) else None
        got = actual_events[event_index] if event_index < len(actual_events) else None
        if ref is None or got is None or ref["component"] != got["component"]:
            rows.append({
                "call_index": event_index,
                "reference_component": ref["component"] if ref else None,
                "actual_component": got["component"] if got else None,
                "structure_match": False,
                "leaves": [],
            })
            continue

        leaf_rows = []
        ref_leaves, got_leaves = ref["leaves"], got["leaves"]
        for path in sorted(set(ref_leaves) | set(got_leaves)):
            left, right = ref_leaves.get(path), got_leaves.get(path)
            if left is None or right is None or left.shape != right.shape:
                leaf_rows.append({
                    "path": path,
                    "shape_match": False,
                    "reference_shape": list(left.shape) if left is not None else None,
                    "actual_shape": list(right.shape) if right is not None else None,
                })
                continue
            if not np.isfinite(left).all() or not np.isfinite(right).all():
                leaf_rows.append({
                    "path": path,
                    "shape_match": True,
                    "finite": False,
                    "max_abs": None,
                    "mean_abs": None,
                    "bitwise_equal": False,
                })
                continue
            delta = np.abs(left.astype(np.float64) - right.astype(np.float64))
            leaf_rows.append({
                "path": path,
                "shape_match": True,
                "finite": True,
                "max_abs": float(delta.max()) if delta.size else 0.0,
                "mean_abs": float(delta.mean()) if delta.size else 0.0,
                "bitwise_equal": bool(np.array_equal(left, right)),
            })
        rows.append({
            "call_index": event_index,
            "component": ref["component"],
            "structure_match": set(ref_leaves) == set(got_leaves)
            and all(
                ref_leaves[path].shape == got_leaves[path].shape
                for path in set(ref_leaves) & set(got_leaves)
            ),
            "leaves": leaf_rows,
        })
    return rows


def first_divergent_boundary(rows):
    """Find the first recorded component with a shape change or nonzero numeric drift."""
    for row in rows:
        if not row.get("structure_match"):
            return row.get("component", row.get("actual_component"))
        for leaf in row["leaves"]:
            if (
                not leaf.get("shape_match")
                or leaf.get("finite") is False
                or (leaf.get("max_abs") is not None and leaf.get("max_abs", 0.0) > 0.0)
            ):
                return row["component"]
    return None


def summarize_pairs(rows):
    summary = {}
    for pair_name in PAIR_NAMES:
        pair_summary = {"passed_samples": 0, "failed_samples": 0, "outputs": {}}
        for row in rows:
            comparison = row["comparisons"][pair_name]
            pair_summary["passed_samples"] += int(comparison["passed"])
            pair_summary["failed_samples"] += int(not comparison["passed"])
            for output in OUTPUT_NAMES:
                item = comparison["outputs"][output]
                result = pair_summary["outputs"].setdefault(
                    output, {"max_abs_global": 0.0, "mean_abs_per_image": [], "failed_samples": 0}
                )
                result["max_abs_global"] = max(result["max_abs_global"], item["max_abs"])
                result["mean_abs_per_image"].append(item["mean_abs"])
                result["failed_samples"] += int(not item["passed"])
        for output in pair_summary["outputs"].values():
            values = output.pop("mean_abs_per_image")
            output["mean_abs_across_images"] = float(np.mean(values))
        summary[pair_name] = pair_summary
    return summary


def validate_prior_failure(path, manifest):
    failure = q.read_json(path)
    verify_signed(failure)
    if (
        failure.get("revision") != REVISION
        or failure.get("stage") != "CPU export and traced graph versus native"
        or failure.get("export_gate_passed") is not False
        or failure.get("optimizer_steps") != 0
        or [row.get("sample_id") for row in failure.get("rows", [])] != manifest["sample_ids"]
        or any(not row.get("cuda_export_vs_native", {}).get("passed") for row in failure["rows"])
    ):
        raise RuntimeError("Prior report is not the exact M67 CPU/trace failure for these fixed inputs")
    return failure


def run(args):
    import torch
    from torch import nn

    manifest = load_manifest(args.manifest)
    run_output = Path(manifest["output"]).resolve()
    if args.failed_report.resolve() != run_output / "m67_parity_failure.json":
        raise RuntimeError("Use the CPU/trace failure report from this exact M67 output folder")
    if args.output.resolve() != run_output / "m67_cpu_trace_diagnostic.json":
        raise RuntimeError("Write the diagnostic to this M67 run's m67_cpu_trace_diagnostic.json")
    failure = validate_prior_failure(args.failed_report, manifest)
    output_path = args.output.resolve()
    if output_path.exists():
        previous = q.read_json(output_path)
        verify_signed(previous)
        if (
            previous.get("diagnostic_revision") == DIAGNOSTIC_REVISION
            and previous.get("manifest_signature") == manifest["signature_sha256"]
            and previous.get("failure_report_signature") == failure["signature_sha256"]
            and previous.get("checkpoint_sha256") == manifest["checkpoint_sha256"]
            and previous.get("model_state_unchanged") is True
        ):
            print(f"Verified existing diagnostic: {output_path}", flush=True)
            return
        raise RuntimeError(f"Preserve existing diagnostic and choose a new output path: {output_path}")

    q.seed()
    repo = Path(manifest["repo"])
    sys.path[:0] = [str(repo), str(repo / "lib/models/monodetr/ops")]
    import MultiScaleDeformableAttention as extension

    if Path(extension.__file__).resolve() != Path(manifest["attention_binary"]).resolve():
        raise RuntimeError("Attention import resolved outside the frozen M67 checkout")
    from lib.datasets.kitti.kitti_dataset import KITTI_Dataset
    from lib.helpers.model_helper import build_model

    cfg = copy.deepcopy(manifest["config"])
    dataset = KITTI_Dataset("val", cfg["dataset"])
    dataset.data_augmentation = False
    if dataset.cls2id != {"Pedestrian": 0, "Car": 1, "Cyclist": 2}:
        raise RuntimeError("Native class order changed")

    model, _ = build_model(cfg["model"])
    payload = q.safe_payload(Path(manifest["checkpoint"]))
    if int(payload.get("epoch", -1)) != 130:
        raise RuntimeError("Expected original A2 epoch-130 checkpoint")
    model.load_state_dict(payload["model_state"], strict=True)
    del payload
    model.cuda().eval().requires_grad_(False)
    state_before = q.tensor_hash(model.state_dict())

    class Wrapper(nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base

        def forward(self, image, calibration, image_size):
            values = self.base(image, calibration, None, image_size)
            return tuple(values[key] for key in OUTPUT_KEYS)

    cuda_wrapper = Wrapper(model).eval()
    cpu_model = copy.deepcopy(model).cpu().eval().requires_grad_(False)
    cpu_wrapper = Wrapper(cpu_model).eval()
    if q.tensor_hash(cpu_model.state_dict()) != state_before:
        raise RuntimeError("CPU diagnostic copy did not preserve the exact checkpoint state")

    # Trace only the CPU portable-export path, without hooks that could affect tracing.
    set_export(cpu_model, True)
    first_image, first_calibration, _, first_info = dataset[0]
    first_cpu_inputs = (
        torch.from_numpy(first_image).unsqueeze(0).float(),
        torch.from_numpy(first_calibration).unsqueeze(0).float(),
        torch.as_tensor(first_info["img_size"]).unsqueeze(0).float(),
    )
    with torch.no_grad():
        traced = torch.jit.trace(cpu_wrapper, first_cpu_inputs, strict=True, check_trace=False)

    cuda_events, cuda_handles = attach_boundary_hooks(model, torch)
    cpu_events, cpu_handles = attach_boundary_hooks(cpu_model, torch)
    rows = []
    try:
        with torch.no_grad():
            for index, sample in enumerate(manifest["sample_ids"]):
                image, calibration, _, info = dataset[index]
                if f'{int(info["img_id"]):06d}' != sample:
                    raise RuntimeError("Fixed16 input order changed")
                cpu_inputs = (
                    torch.from_numpy(image).unsqueeze(0).float(),
                    torch.from_numpy(calibration).unsqueeze(0).float(),
                    torch.as_tensor(info["img_size"]).unsqueeze(0).float(),
                )
                cuda_inputs = tuple(value.cuda() for value in cpu_inputs)

                set_export(model, False)
                native_cuda = arrays(cuda_wrapper(*cuda_inputs))
                set_export(model, True)
                cuda_events.clear()
                cuda_export = arrays(cuda_wrapper(*cuda_inputs))
                captured_cuda = list(cuda_events)

                cpu_events.clear()
                cpu_export = arrays(cpu_wrapper(*cpu_inputs))
                captured_cpu = list(cpu_events)
                traced_cpu = arrays(traced(*cpu_inputs))

                boundary_rows = compare_boundaries(captured_cuda, captured_cpu)
                comparisons = {
                    "cuda_export_vs_native_cuda": compare_outputs(cuda_export, native_cuda),
                    "cpu_export_vs_cuda_export": compare_outputs(cpu_export, cuda_export),
                    "trace_vs_cpu_export": compare_outputs(traced_cpu, cpu_export),
                    "cpu_export_vs_native_cuda": compare_outputs(cpu_export, native_cuda),
                    "trace_vs_native_cuda": compare_outputs(traced_cpu, native_cuda),
                }
                rows.append({
                    "sample_id": sample,
                    "comparisons": comparisons,
                    "first_divergent_top_level_component_cuda_export_vs_cpu_export":
                        first_divergent_boundary(boundary_rows),
                    "component_boundaries_cuda_export_vs_cpu_export": boundary_rows,
                })
                print(
                    f"M67b {index + 1}/{len(manifest['sample_ids'])} {sample}: "
                    f"CUDA-export/native={comparisons['cuda_export_vs_native_cuda']['passed']}, "
                    f"CPU-export/CUDA-export={comparisons['cpu_export_vs_cuda_export']['passed']}, "
                    f"trace/CPU-export={comparisons['trace_vs_cpu_export']['passed']}",
                    flush=True,
                )
    finally:
        for handle in cuda_handles + cpu_handles:
            handle.remove()

    unchanged = (
        q.tensor_hash(model.state_dict()) == state_before
        and q.tensor_hash(cpu_model.state_dict()) == state_before
        and not any(parameter.grad is not None for parameter in model.parameters())
        and not any(parameter.grad is not None for parameter in cpu_model.parameters())
    )
    if not unchanged:
        raise RuntimeError("Diagnostic changed model state or left gradient buffers")

    report = signed({
        "schema_version": 1,
        "diagnostic_revision": DIAGNOSTIC_REVISION,
        "m67_revision": REVISION,
        "manifest_signature": manifest["signature_sha256"],
        "failure_report_signature": failure["signature_sha256"],
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "checkpoint_epoch": manifest["checkpoint_epoch"],
        "fixed_sample_ids": manifest["sample_ids"],
        "component_boundary_scope": "direct child modules of MonoDETR; numeric drift is diagnostic only",
        "raw_output_limits": manifest["raw_limits"],
        "comparisons": summarize_pairs(rows),
        "rows": rows,
        "model_state_unchanged": unchanged,
        "optimizer_steps": 0,
        "training_performed": False,
        "quantization_performed": False,
        "coreml_conversion_performed": False,
        "iphone_performance_measured": False,
        "deployment_authorized": False,
        "diagnostic_complete": True,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    q.write_json(output_path, report)
    print(json.dumps({"diagnostic_complete": True, "comparisons": report["comparisons"],
                      "output": str(output_path)}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--failed-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
