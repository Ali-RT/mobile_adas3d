from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


REAL_SPATIAL_SHAPES = ((48, 160), (24, 80), (12, 40), (6, 20))
PARITY_MAX_ABS_TOLERANCE = 2e-5


class StaticMSDeformAttnCore(nn.Module):
    """MonoDETR's reference deformable-attention kernel with fixed geometry."""

    def __init__(self, query_count: int, spatial_shapes=REAL_SPATIAL_SHAPES):
        super().__init__()
        self.query_count = query_count
        self.spatial_shapes = tuple(tuple(x) for x in spatial_shapes)
        self.split_sizes = tuple(height * width for height, width in self.spatial_shapes)

    def forward(self, value, sampling_locations, attention_weights):
        values = value.split(self.split_sizes, dim=1)
        sampling_grids = 2 * sampling_locations - 1
        sampled = []
        for level, (height, width) in enumerate(self.spatial_shapes):
            value_level = (
                values[level]
                .flatten(2)
                .transpose(1, 2)
                .reshape(8, 32, height, width)
            )
            grid_level = (
                sampling_grids[:, :, :, level * 4:(level + 1) * 4]
                .transpose(1, 2)
                .flatten(0, 1)
            )
            sampled.append(
                F.grid_sample(
                    value_level,
                    grid_level,
                    mode="bilinear",
                    padding_mode="zeros",
                    align_corners=False,
                )
            )
        weights = attention_weights.transpose(1, 2).reshape(8, 1, self.query_count, 16)
        output = (
            torch.stack(sampled, dim=-2).flatten(-2) * weights
        ).sum(-1).view(1, 256, self.query_count)
        return output.transpose(1, 2).contiguous()


def operation_counts(program) -> Counter:
    counts = Counter()

    def visit(block):
        for operation in block.operations:
            counts[operation.op_type] += 1
            for child in operation.blocks:
                visit(child)

    visit(program.functions["main"])
    return counts


def run_probe(query_count: int, output_dir: Path, skip_predict: bool) -> dict:
    import coremltools as ct

    torch.manual_seed(7)
    batch, heads, channels, levels, points = 1, 8, 32, 4, 4
    flattened = sum(height * width for height, width in REAL_SPATIAL_SHAPES)
    value = torch.randn(batch, flattened, heads, channels)
    locations = torch.rand(batch, query_count, heads, levels * points, 2)
    raw_weights = torch.randn(batch, query_count, heads, levels * points)
    weights = raw_weights.softmax(-1)
    model = StaticMSDeformAttnCore(query_count).eval()

    with torch.no_grad():
        reference = model(value, locations, weights)
        trace_start = time.perf_counter()
        traced = torch.jit.trace(
            model, (value, locations, weights), strict=True, check_trace=True
        )
        trace_seconds = time.perf_counter() - trace_start
        traced_delta = float((traced(value, locations, weights) - reference).abs().max())

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"ms_deform_attn_q{query_count}.pt"
    traced.save(str(trace_path))

    convert_start = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        inputs=[
            ct.TensorType(name="value", shape=value.shape, dtype=np.float32),
            ct.TensorType(
                name="sampling_locations", shape=locations.shape, dtype=np.float32
            ),
            ct.TensorType(
                name="attention_weights", shape=weights.shape, dtype=np.float32
            ),
        ],
    )
    convert_seconds = time.perf_counter() - convert_start
    package_path = output_dir / f"ms_deform_attn_q{query_count}.mlpackage"
    mlmodel.save(str(package_path))
    counts = operation_counts(mlmodel._mil_program)

    predict_delta = None
    predict_seconds = None
    prediction_error = None
    if not skip_predict:
        try:
            predict_start = time.perf_counter()
            prediction = mlmodel.predict(
                {
                    "value": value.numpy(),
                    "sampling_locations": locations.numpy(),
                    "attention_weights": weights.numpy(),
                }
            )
            predict_seconds = time.perf_counter() - predict_start
            converted = np.asarray(next(iter(prediction.values())))
            predict_delta = float(np.max(np.abs(converted - reference.numpy())))
        except Exception as error:  # conversion remains a useful independent gate
            prediction_error = f"{type(error).__name__}: {error}"

    conversion_passed = "custom" not in counts
    parity_passed = (
        predict_delta is not None and predict_delta <= PARITY_MAX_ABS_TOLERANCE
    )
    report = {
        "schema_version": 1,
        "complete": conversion_passed and (skip_predict or parity_passed),
        "conversion_passed": conversion_passed,
        "parity_passed": None if skip_predict else parity_passed,
        "parity_max_abs_tolerance": PARITY_MAX_ABS_TOLERANCE,
        "query_count": query_count,
        "spatial_shapes": [list(x) for x in REAL_SPATIAL_SHAPES],
        "flattened_tokens": flattened,
        "heads": heads,
        "channels_per_head": channels,
        "points_per_level": points,
        "torch_version": torch.__version__,
        "coremltools_version": ct.__version__,
        "trace_seconds": trace_seconds,
        "conversion_seconds": convert_seconds,
        "prediction_seconds": predict_seconds,
        "trace_max_abs_delta": traced_delta,
        "coreml_max_abs_delta": predict_delta,
        "coreml_prediction_error": prediction_error,
        "mil_operation_counts": dict(sorted(counts.items())),
        "mil_has_custom_op": "custom" in counts,
        "trace": str(trace_path),
        "mlpackage": str(package_path),
    }
    report_path = output_dir / f"ms_deform_attn_q{query_count}_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["complete"]:
        raise RuntimeError(f"Core ML feasibility probe failed for Q={query_count}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--query-count", type=int, action="append", default=None,
        help="Repeat for multiple real graph cases; defaults to decoder=50 and encoder=10200.",
    )
    parser.add_argument("--skip-predict", action="store_true")
    args = parser.parse_args()
    query_counts = args.query_count or [50, 10200]
    reports = [
        run_probe(count, args.output_dir, args.skip_predict) for count in query_counts
    ]
    summary = {
        "schema_version": 1,
        "complete": all(report["complete"] for report in reports),
        "reports": reports,
    }
    (args.output_dir / "ms_deform_attn_coreml_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
