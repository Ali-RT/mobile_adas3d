# MonoDGP M59b Core ML Intermediate-Tensor Diagnostic Contract

## Purpose

M59 macOS prediction parity rejected the exact M58 FP32 package: the depth-map
and region-pyramid outputs passed, but the query/head outputs did not. M59b is
the next investigation gate. It exports a separate fixed-shape diagnostic
package that returns causal tensors from the same frozen M56d/M57/M58 graph so
the first numerical divergence can be located.

M59b performs no training and changes no weights. It does not replace the M58
product package and does not authorize iPhone execution, FP16, quantization,
deployment, or product-safety work.

## Frozen inputs and provenance

Use only the exact M58 checkout, checkpoint, runtime configuration, M57
evidence, dataset view, and Chen split. The exporter requires:

- a complete, all-gates-passed `m58_coreml_export_gate.json`;
- the reviewed M57 manifest, smoke report, complete gate, and comparison CSV;
- input tensors `image [1,3,384,1280]`, `calibration [1,3,4]`, and
  `image_size [1,2]`, all FP32;
- `coremltools==9.0`, iOS 17 ML Program conversion, and FP32 compute
  precision.

The diagnostic package must be generated in a new Drive directory. Never
overwrite `MonoDGP_M58_fixed_fp32.mlpackage` or its reference I/O.

## Causal tap order

The output order is frozen in the generated gate and follows the model path:

1. backbone feature and positional tensors;
2. input projections;
3. region-enhanced features, segmentation embeddings, and region levels;
4. depth-map logits, depth embedding, and weighted depth;
5. final 2D-transformer query state;
6. final 3D-transformer query state;
7. raw class/box/dimension/depth/angle heads before reference fusion;
8. final semantic outputs and region outputs.

The macOS validator compares every tap with the frozen PyTorch reference using
the diagnostic probe limit `1e-3` maximum absolute error. It reports
`first_diverging_tensor` as the first failed name in this order. A missing,
non-finite, or shape-changed tensor is a failure.

## Colab export

Run `notebooks/MonoDGP_M59b_CoreML_Intermediate_Diagnostic_Colab.ipynb` on a
Colab GPU. The main command is:

```bash
python3 -u scripts/export_monodgp_m59b_coreml_diagnostics.py \
  --monodgp-repo /content/MonoDGP_M59b \
  --m58-export-gate /content/drive/MyDrive/mobile_adas3d_outputs/compression/monodgp_m58_coreml_conversion/m58_coreml_export_gate.json \
  --m57-manifest /content/drive/MyDrive/mobile_adas3d_outputs/compression/monodgp_m57_deformable_attention/m57_deformable_attention_manifest.json \
  --m57-smoke /content/drive/MyDrive/mobile_adas3d_outputs/compression/monodgp_m57_deformable_attention/m57_deformable_attention_smoke.json \
  --m57-gate /content/drive/MyDrive/mobile_adas3d_outputs/compression/monodgp_m57_deformable_attention/m57_portable_attention_gate.json \
  --m57-comparison /content/drive/MyDrive/mobile_adas3d_outputs/compression/monodgp_m57_deformable_attention/m57_portable_attention_comparison.csv \
  --dataset-root /content/monodgp_kitti_m53 \
  --split-dir /content/drive/MyDrive/mobile_adas3d_splits/kitti_chen \
  --output-dir /content/drive/MyDrive/mobile_adas3d_outputs/compression/monodgp_m59b_coreml_diagnostic
```

The exact M57 artifact paths may differ; use the paths recorded by the passed
M58 gate and do not substitute a new checkpoint.

## macOS diagnostic

Copy the generated `.mlpackage`, `m59b_diagnostic_reference_io.npz`, and
`m59b_coreml_diagnostic_export_gate.json` to macOS. Run:

```bash
python3 scripts/validate_monodgp_m59b_macos_diagnostics.py \
  --mlpackage MonoDGP_M59b_intermediate_diagnostic_fp32.mlpackage \
  --reference-io m59b_diagnostic_reference_io.npz \
  --export-gate m59b_coreml_diagnostic_export_gate.json \
  --compute-units ALL \
  --output m59b_macos_coreml_diagnostic.json
```

If compilation is bounded or expensive, `CPU_AND_GPU` may be run as a second
diagnostic, but it does not replace the recorded ALL-units result. A report
with `first_diverging_tensor` is useful even though the investigation gate is
rejected.

## Decision map

- divergence at backbone/input projection: investigate Core ML convolution,
  normalization, or input preprocessing before touching attention;
- divergence at region/depth tensors: investigate resampling, interpolation,
  or depth-position construction;
- divergence at `det2d_hs_last`: investigate portable deformable attention or
  2D transformer numerical semantics;
- divergence at `det3d_hs_last`: investigate depth cross-attention or 3D
  transformer sampling;
- divergence only at raw heads/final outputs: investigate reference fusion,
  sigmoid/activation, or head export ordering.

No deployment or precision-compression experiment may start until the causal
source is understood and a revised raw-output parity gate passes.
