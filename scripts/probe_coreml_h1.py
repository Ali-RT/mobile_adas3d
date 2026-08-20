from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.build import build_model
from models.mobile_adas3d_h1 import MobileADAS3DH1TupleWrapper
from tools.config import load_config


def operation_counts(program) -> Counter:
    counts = Counter()

    def visit(block) -> None:
        for operation in block.operations:
            counts[operation.op_type] += 1
            for child in operation.blocks:
                visit(child)

    visit(program.functions["main"])
    return counts


def package_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def profiler_macs(model: torch.nn.Module, image: torch.Tensor) -> int:
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU],
        with_flops=True,
    ) as profile:
        with torch.inference_mode():
            model(image)
    flops = sum(int(event.flops or 0) for event in profile.key_averages())
    return flops // 2


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace, convert, and validate the random-weight MobileADAS3D-H1 graph."
    )
    parser.add_argument("--config", default="configs/kitti_mobileadas3d_h1.yaml")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--predict", action="store_true")
    parser.add_argument(
        "--compute-precision",
        choices=("fp16", "mixed", "fp32"),
        default="fp16",
        help="Core ML precision; mixed keeps depth/query reasoning and heads in FP32.",
    )
    args = parser.parse_args()

    import coremltools as ct

    torch.manual_seed(11)
    torch.backends.mha.set_fastpath_enabled(False)
    config = load_config(args.config)
    config["model"]["pretrained"] = False
    model = build_model(config).eval()
    wrapper = MobileADAS3DH1TupleWrapper(model).eval()
    output_names = model.export_output_names
    image = torch.rand(1, 3, 384, 1280)

    with torch.inference_mode():
        reference = wrapper(image)
        traced = torch.jit.trace(wrapper, image, strict=True, check_trace=False)
        traced_outputs = traced(image)
    trace_delta = max(
        float((actual - expected).abs().max())
        for actual, expected in zip(traced_outputs, reference)
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "MobileADAS3D_H1_random.pt"
    package_path = output_dir / "MobileADAS3D_H1_random.mlpackage"
    traced.save(str(trace_path))

    fp32_selected_operations = []
    if args.compute_precision == "fp16":
        compute_precision = ct.precision.FLOAT16
    elif args.compute_precision == "fp32":
        compute_precision = ct.precision.FLOAT32
    else:
        fp16_feature_ops = {"conv", "relu", "upsample_bilinear"}
        def select_fp16(operation):
            use_fp16 = operation.op_type in fp16_feature_ops
            if not use_fp16 and operation.op_type != "const":
                fp32_selected_operations.append(
                    {"type": operation.op_type, "name": operation.name}
                )
            return use_fp16

        compute_precision = ct.transform.FP16ComputePrecision(op_selector=select_fp16)

    conversion_start = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=compute_precision,
        compute_units=ct.ComputeUnit.ALL,
        skip_model_load=not args.predict,
        inputs=[ct.TensorType(name="image", shape=image.shape, dtype=np.float32)],
        outputs=[ct.TensorType(name=name) for name in output_names],
    )
    conversion_seconds = time.perf_counter() - conversion_start
    mlmodel.save(str(package_path))
    counts = operation_counts(mlmodel._mil_program)

    coreml_delta = None
    coreml_output_deltas = None
    coreml_prediction_ms = None
    if args.predict:
        prediction_start = time.perf_counter()
        prediction = mlmodel.predict({"image": image.numpy()})
        coreml_prediction_ms = (time.perf_counter() - prediction_start) * 1000.0
        coreml_output_deltas = {
            name: float(
                np.max(np.abs(prediction[name].astype(np.float32) - expected.numpy()))
            )
            for name, expected in zip(output_names, reference)
        }
        coreml_delta = max(coreml_output_deltas.values())

    parameters = sum(parameter.numel() for parameter in model.parameters())
    macs = profiler_macs(wrapper, image)
    size_bytes = package_size(package_path)
    parity_pass = coreml_delta is not None and coreml_delta <= 2e-3
    fp16_artifact_gate = args.compute_precision == "fp16"
    complete = (
        parameters <= 10_000_000
        and macs <= 15_000_000_000
        and size_bytes <= 25 * 1024 * 1024
        and trace_delta <= 1e-6
        and "custom" not in counts
        and parity_pass
        and fp16_artifact_gate
    )
    report = {
        "schema_version": 1,
        "complete": complete,
        "scope": "random-weight MobileADAS3D-H1 pre-training graph gate",
        "torch_version": torch.__version__,
        "coremltools_version": ct.__version__,
        "compute_precision": args.compute_precision,
        "fp16_artifact_gate": fp16_artifact_gate,
        "fp32_selected_operations": fp32_selected_operations,
        "input_shape": list(image.shape),
        "output_shapes": {
            name: list(value.shape)
            for name, value in zip(output_names, reference)
        },
        "parameters": parameters,
        "parameter_gate": parameters <= 10_000_000,
        "macs": macs,
        "gmacs": macs / 1e9,
        "mac_measurement": "torch.profiler FLOPs / 2",
        "mac_gate": macs <= 15_000_000_000,
        "trace_max_abs_delta": trace_delta,
        "coreml_predict_requested": args.predict,
        "coreml_max_abs_delta": coreml_delta,
        "coreml_output_max_abs_deltas": coreml_output_deltas,
        "coreml_prediction_ms": coreml_prediction_ms,
        "coreml_parity_gate": parity_pass,
        "conversion_seconds": conversion_seconds,
        "mil_operation_counts": dict(sorted(counts.items())),
        "mil_has_custom_op": "custom" in counts,
        "mlpackage_size_bytes": size_bytes,
        "mlpackage_size_mb": size_bytes / (1024.0 * 1024.0),
        "package_size_gate": size_bytes <= 25 * 1024 * 1024,
        "trace": str(trace_path),
        "mlpackage": str(package_path),
    }
    report_path = output_dir / "mobileadas3d_h1_coreml_gate.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not complete:
        raise RuntimeError("MobileADAS3D-H1 random graph gate failed")


if __name__ == "__main__":
    main()
