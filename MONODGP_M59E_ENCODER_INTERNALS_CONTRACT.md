# M59e: first 2D encoder layer internals

M59d passed the six region/depth tensor checks and failed first at encoder
layer 0. M59e exposes that layer's inputs, deformable-attention intermediates,
residual additions, LayerNorms, and feed-forward stages. It keeps the trained
weights and the absolute diagnostic limit of 0.001 unchanged.

The diagnostic reuses the hash-verified M59d TorchScript and reference inputs.
Only an in-memory copy of its output graph is instrumented. The original
M59d TorchScript, Core ML package, checkpoint, and reference tensors are not
overwritten. All original M59d output comparisons must pass after instrumenting
the graph before an internal diagnostic is exported.

This is a one-sample numerical investigation, not an AP evaluation or runtime
benchmark. It does not authorize FP16, quantization, device testing, or deployment.

## Executed result — 2026-09-18

The hash-verified M59d trace was replayed on the local Mac. Original-output
preservation and all 18 diagnostic trace comparisons passed before conversion.
The FP32 Core ML package exported without custom MIL operators. Actual Core ML
execution with ALL compute units completed, but the unchanged runtime parity
gate correctly **failed**.

| First encoder tensor | Maximum absolute delta | Result at 0.001 |
| --- | ---: | --- |
| Source features | 0.00001276 | Pass |
| Positional input | 1.99901617 | Fail — earliest observed |
| Reference coordinates | 0 | Pass |
| Projected attention values | 0.00002098 | Pass |
| Encoder output | 1.29778898 | Fail |

The positional input contains four scales: 48×160, 24×80, 12×40, and 6×20.
The first three scales pass (maximum error about 2.38e-7). The fourth scale,
tokens `[10080:10200]`, fails. After subtracting its learned level embedding,
Core ML's positional channels match **grouped** sine/cosine channels within
each y/x block, instead of the expected **interleaved** channels:

```
Expected: sin0, cos0, sin1, cos1, ...
Observed: sin0, sin1, ..., sin63, cos0, cos1, ..., cos63
```

Applying that permutation to the reference tensor in an **offline hypothesis
test** leaves max error **4.917383e-7**, with zero elements above 0.001. No
model output or trained weight was corrected in this experiment. This strongly
localizes the discrepancy to the fourth-scale positional construction/channel
layout. It does not yet prove which converter optimization caused it or that
repairing it will eliminate all downstream drift.

Prior backbone-position taps covered the three backbone scales, not this
additional fourth scale, so this finding does not contradict those checks.
The source uses a sine/cosine `stack` followed by `flatten` and a permutation;
the next experiment should preserve that mathematical ordering explicitly in
an export-only rewrite.

Evidence:

- `artifacts/m59e_encoder_internal_export_gate_20260918.json`
- `artifacts/m59e_encoder_internal_macos_all_20260918.json`
- `artifacts/m59e_position_layout_20260918.json`
- Parent: `artifacts/m59d_macos_2d_transformer_all_20260918.json`

Environment: macOS 26.6.2, Torch 2.12.0, Core ML Tools 9.0. Export used isolated
NumPy 2.3.5 because NumPy 2.4 rejects an older scalar cast in Core ML Tools 9.
Torch 2.12 is newer than the converter's advertised tested range; versions are
recorded in the evidence, and any converter-specific attribution remains a
hypothesis until a controlled rewrite/re-export is tested.

## Reproduce

Colab entry point: `notebooks/MonoDGP_M59e_Encoder_Internals_Colab.ipynb`, revision
`M59e-2026-09-18-r1`. Run its three code cells in order on **CPU**. It needs only
the passed M59d export gate, TorchScript, and reference NPZ already in Drive.
It does not clone MonoDGP, load the KITTI dataset, or compile CUDA extensions.
New timestamped directories preserve prior evidence. After failure, rerun from
Cell 1 to create a fresh output directory. Logs capture stdout and stderr.

For local export, use a Python environment with Torch, `coremltools==9.0`, and
`numpy>=2.0,<2.4`. From the project checkout:

```sh
python scripts/export_monodgp_m59e_encoder_internals.py \
  --m59d-dir /absolute/path/to/m59d \
  --output-dir /absolute/path/to/new_m59e
```

On macOS, execute the package and save predictions for layout analysis:

```sh
python scripts/validate_monodgp_m59e_macos.py \
  --artifact-dir /absolute/path/to/new_m59e --compute-units ALL \
  --output /absolute/path/to/m59e_runtime.json \
  --save-predictions /absolute/path/to/m59e_predictions.npz
```

The validator exits nonzero on parity failure **after** writing its diagnostic
report. `complete: true` means execution completed, not that parity passed.
Then test the specific layout hypothesis without altering the model:

```sh
python scripts/diagnose_monodgp_m59e_position_layout.py \
  --m59d-dir /absolute/path/to/m59d \
  --artifact-dir /absolute/path/to/new_m59e \
  --runtime-report /absolute/path/to/m59e_runtime.json \
  --predictions /absolute/path/to/m59e_predictions.npz \
  --output /absolute/path/to/m59e_position_layout.json
```

## Next controlled experiment

M59f: rewrite only the export-time positional encoding's sine/cosine
interleaving. Verify PyTorch equivalence, all four positional scales, every
encoder output, and full-model raw/decoded parity before considering a full
validation run. Keep weights, inputs, precision, and tolerances fixed. If the
first mismatch moves elsewhere, instrument that boundary rather than widening
the thresholds. Do not retrain or redesign the detector based on this export bug.
