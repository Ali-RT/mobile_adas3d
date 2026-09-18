"""Test the M59e positional-channel layout hypothesis without changing the model."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_monodgp_m59d_2d_transformer import PROBE_LIMIT, sha256_file
from scripts.export_monodgp_m59e_encoder_internals import verify_source

SPATIAL_SHAPES = ((48, 160), (24, 80), (12, 40), (6, 20))


def error_summary(reference, candidate):
    delta = np.abs(reference - candidate)
    maximum = float(delta.max())
    return {"max_abs_delta": maximum, "mean_abs_delta": float(delta.mean()),
            "elements_over_limit": int((delta > PROBE_LIMIT).sum()),
            "passed": maximum <= PROBE_LIMIT}


def analyze_layout(reference, candidate, level_embedding, shapes=SPATIAL_SHAPES):
    """Compare each scale, then test grouped sin/cos channels after removing level bias.

    This is an offline hypothesis test, NOT a correction to the exported model.
    The learned embedding stays in its original channel order in both tensors.
    """
    tokens = sum(height * width for height, width in shapes)
    channels = level_embedding.shape[-1]
    if (reference.shape != candidate.shape or reference.shape != (1, tokens, channels)
            or level_embedding.shape != (len(shapes), channels) or channels % 4):
        raise ValueError("Unexpected positional tensor, level embedding, or spatial shapes")
    if not all(np.isfinite(value).all() for value in (reference, candidate, level_embedding)):
        raise ValueError("Non-finite positional evidence")
    half = channels // 2
    permutation = np.concatenate([np.arange(start + offset, start + half, 2)
                                 for start in (0, half) for offset in (0, 1)])
    rows = []
    start = 0
    for index, (height, width) in enumerate(shapes):
        end = start + height * width
        ref, pred = reference[:, start:end], candidate[:, start:end]
        ref_sine = ref - level_embedding[index]
        pred_sine = pred - level_embedding[index]
        raw = error_summary(ref, pred)
        grouped = error_summary(ref_sine[..., permutation], pred_sine)
        rows.append({"level_index": index, "spatial_shape": [height, width],
                     "token_slice_exclusive": [start, end], "original_order": raw,
                     "grouped_sine_cosine_hypothesis": grouped,
                     "grouped_layout_explains_failure": not raw["passed"] and grouped["passed"]})
        start = end
    return {"probe_limit_max_abs": PROBE_LIMIT,
            "hypothesis": "within each y/x block, interleaved sin/cos becomes all sine then all cosine",
            "learned_level_embedding_subtracted_before_permutation": True,
            "permutation_of_reference_channels": permutation.tolist(), "scales": rows,
            "explained_failing_levels": [r["level_index"] for r in rows
                                          if r["grouped_layout_explains_failure"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m59d-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_gate = verify_source(args.m59d_dir)
    export_path = args.artifact_dir / "m59e_encoder_internal_export_gate.json"
    export = json.loads(export_path.read_text())
    runtime = json.loads(args.runtime_report.read_text())
    if export.get("all_export_gates_passed") is not True or runtime.get("complete") is not True:
        raise RuntimeError("Completed M59e export and runtime reports are required")
    if runtime.get("export_gate_sha256") != sha256_file(export_path):
        raise RuntimeError("Runtime report belongs to a different export")
    if export.get("source_torchscript_sha256") != source_gate["artifacts"]["torchscript_sha256"]:
        raise RuntimeError("M59e and M59d source traces differ")
    if runtime.get("predictions_sha256") != sha256_file(args.predictions):
        raise RuntimeError("Saved predictions are missing or changed; rerun validator with --save-predictions")
    reference_path = args.artifact_dir / "m59e_encoder_internal_reference_io.npz"
    if export["artifacts"]["reference_io_sha256"] != sha256_file(reference_path):
        raise RuntimeError("M59e reference tensors changed")
    import torch
    model = torch.jit.load(str(args.m59d_dir / "MonoDGP_M59d_2d_transformer_fp32.pt"), map_location="cpu")
    level_embedding = model.wrapped.det2d_transformer.level_embed.detach().numpy()
    with np.load(reference_path, allow_pickle=False) as ref, np.load(args.predictions, allow_pickle=False) as pred:
        analysis = analyze_layout(ref["enc0_pos"], pred["enc0_pos"], level_embedding)
    report = {"schema_version": 1, "complete": True,
              "experiment": "M59e frozen positional channel-order hypothesis test",
              "scope": "one frozen input; offline tensor comparison; model NOT repaired",
              "runtime_report_sha256": sha256_file(args.runtime_report),
              "predictions_sha256": sha256_file(args.predictions),
              "source_torchscript_sha256": source_gate["artifacts"]["torchscript_sha256"],
              "training_performed": False, "weights_changed": False,
              "full_model_parity_passed": False, **analysis}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
