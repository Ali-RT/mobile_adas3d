from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn


PINNED_COMMIT = "6994b9f512400b258c6edb75f77423beb9c126f2"
OUTPUT_NAMES = ("logits", "boxes", "dimensions", "depth", "angle")


class CoreMLInferenceWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image, calibration, image_size):
        output = self.model(image, calibration, None, image_size)
        return (
            output["pred_logits"],
            output["pred_boxes"],
            output["pred_3d_dim"],
            output["pred_depth"],
            output["pred_angle"],
        )


def operation_counts(program) -> Counter:
    counts = Counter()

    def visit(block):
        for operation in block.operations:
            counts[operation.op_type] += 1
            for child in operation.blocks:
                visit(child)

    visit(program.functions["main"])
    return counts


def build_model(monodetr_repo: Path):
    config_path = monodetr_repo / "configs/monodetr.yaml"
    config = yaml.safe_load(config_path.read_text())["model"]
    config.update(
        {
            "device": "cpu",
            "num_classes": 2,
            "aux_loss": False,
            "backbone_source": "timm",
            "backbone": "mobilenetv4_conv_small.e2400_r224_in1k",
            "backbone_out_indices": [2, 3, 4],
            "backbone_pretrained": False,
        }
    )
    sys.path.insert(0, str(monodetr_repo))
    from lib.models.monodetr.backbone import build_backbone  # noqa: PLC0415
    from lib.models.monodetr.depth_predictor import DepthPredictor  # noqa: PLC0415
    from lib.models.monodetr.depthaware_transformer import (  # noqa: PLC0415
        build_depthaware_transformer,
    )
    from lib.models.monodetr.monodetr import MonoDETR  # noqa: PLC0415

    transformer = build_depthaware_transformer(config)
    return MonoDETR(
        build_backbone(config),
        transformer,
        DepthPredictor(config),
        num_classes=config["num_classes"],
        num_queries=config["num_queries"],
        aux_loss=False,
        num_feature_levels=config["num_feature_levels"],
        with_box_refine=config["with_box_refine"],
        two_stage=config["two_stage"],
        init_box=config["init_box"],
        use_dab=config["use_dab"],
        two_stage_dino=config["two_stage_dino"],
    )


def enable_coreml_export(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if hasattr(module, "coreml_export"):
            module.coreml_export = True
            count += 1
    return count


def validate_attention_export() -> float:
    from lib.models.monodetr.depth_predictor.transformer import (  # noqa: PLC0415
        TransformerEncoderLayer,
    )
    from lib.models.monodetr.depthaware_transformer import (  # noqa: PLC0415
        coreml_multi_head_attention,
    )

    encoder = TransformerEncoderLayer(256, 8, dim_feedforward=256, dropout=0.0)
    encoder.eval()
    source = torch.randn(64, 1, 256)
    position = torch.randn(64, 1, 256)
    mask = torch.zeros(1, 64, dtype=torch.bool)
    with torch.no_grad():
        expected_encoder = encoder(source, mask, position)
        encoder.coreml_export = True
        actual_encoder = encoder(source, mask, position)

        attention = nn.MultiheadAttention(256, 8, dropout=0.0).eval()
        query = torch.randn(50, 1, 256)
        key = torch.randn(64, 1, 256)
        value = torch.randn(64, 1, 256)
        expected_attention = attention(query, key, value, need_weights=False)[0]
        actual_attention = coreml_multi_head_attention(
            attention, query, key, value, 8, 32
        )
    return max(
        float((expected_encoder - actual_encoder).abs().max()),
        float((expected_attention - actual_attention).abs().max()),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trace and convert the fixed-shape random-weight MobileMonoDETR graph."
    )
    parser.add_argument("--monodetr-repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--compile-model",
        action="store_true",
        help="Ask Core ML to compile/load the model; may take many minutes.",
    )
    args = parser.parse_args()

    import coremltools as ct

    repo = args.monodetr_repo.resolve()
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    if commit != PINNED_COMMIT:
        raise RuntimeError(f"Expected MonoDETR {PINNED_COMMIT}, found {commit}")
    marker_path = repo / "lib/models/monodetr/ops/modules/ms_deform_attn.py"
    if "def ms_deform_attn_core_coreml(" not in marker_path.read_text():
        raise RuntimeError("Apply scripts/patch_monodetr_coreml_export.py first")

    torch.manual_seed(11)
    model = build_model(repo).eval()
    attention_delta = validate_attention_export()
    export_modules = enable_coreml_export(model)
    wrapper = CoreMLInferenceWrapper(model).eval()
    image = torch.rand(1, 3, 384, 1280)
    calibration = torch.tensor(
        [[[721.5, 0.0, 609.5, 44.9], [0.0, 721.5, 172.9, 0.2], [0.0, 0.0, 1.0, 0.0]]]
    )
    image_size = torch.tensor([[1280.0, 384.0]])

    with torch.no_grad():
        forward_start = time.perf_counter()
        reference = wrapper(image, calibration, image_size)
        forward_seconds = time.perf_counter() - forward_start
        trace_start = time.perf_counter()
        traced = torch.jit.trace(
            wrapper,
            (image, calibration, image_size),
            strict=True,
            check_trace=False,
        )
        trace_seconds = time.perf_counter() - trace_start
        traced_output = traced(image, calibration, image_size)
        trace_delta = max(
            float((actual - expected).abs().max())
            for actual, expected in zip(traced_output, reference)
        )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "MobileMonoDETRVP1_random.pt"
    package_path = output_dir / "MobileMonoDETRVP1_random.mlpackage"
    traced.save(str(trace_path))

    convert_start = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
        skip_model_load=not args.compile_model,
        inputs=[
            ct.TensorType(name="image", shape=image.shape, dtype=np.float32),
            ct.TensorType(
                name="calibration", shape=calibration.shape, dtype=np.float32
            ),
            ct.TensorType(name="image_size", shape=image_size.shape, dtype=np.float32),
        ],
        outputs=[ct.TensorType(name=name) for name in OUTPUT_NAMES],
    )
    conversion_seconds = time.perf_counter() - convert_start
    mlmodel.save(str(package_path))
    counts = operation_counts(mlmodel._mil_program)
    complete = (
        trace_delta == 0.0
        and attention_delta <= 1e-5
        and "custom" not in counts
    )
    report = {
        "schema_version": 1,
        "complete": complete,
        "scope": "random-weight fixed-shape graph feasibility; not model quality",
        "pinned_monodetr_commit": commit,
        "torch_version": torch.__version__,
        "coremltools_version": ct.__version__,
        "input_shape": list(image.shape),
        "output_shapes": {
            name: list(value.shape) for name, value in zip(OUTPUT_NAMES, reference)
        },
        "coreml_export_modules": export_modules,
        "forward_seconds": forward_seconds,
        "trace_seconds": trace_seconds,
        "conversion_seconds": conversion_seconds,
        "trace_max_abs_delta": trace_delta,
        "native_attention_max_abs_delta": attention_delta,
        "native_compile_requested": args.compile_model,
        "mil_operation_counts": dict(sorted(counts.items())),
        "mil_has_custom_op": "custom" in counts,
        "trace_size_bytes": trace_path.stat().st_size,
        "mlpackage_size_bytes": sum(
            path.stat().st_size for path in package_path.rglob("*") if path.is_file()
        ),
        "trace": str(trace_path),
        "mlpackage": str(package_path),
    }
    report_path = output_dir / "full_graph_coreml_feasibility.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not complete:
        raise RuntimeError("Full MobileMonoDETR Core ML graph feasibility failed")


if __name__ == "__main__":
    main()
