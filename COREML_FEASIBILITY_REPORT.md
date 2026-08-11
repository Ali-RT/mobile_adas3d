# MobileMonoDETR-VP1 Core ML feasibility report

Status: microkernel gate passed; full-model and physical-iPhone gates pending
Date: 2026-08-11

## Decision

Continue with MobileMonoDETR-VP1. Its multi-scale deformable-attention math can
be represented by native Core ML ML Program operations at both decoder and
encoder scale. No custom Core ML operation or host callback is required for
this kernel.

This result does not approve deployment. It proves the highest-risk operator,
not the complete model. Full graph conversion, checkpoint parity, decoded KITTI
parity, and physical-iPhone latency remain mandatory.

## Required export rewrite

The training implementation uses a custom CUDA extension and a rank-six
sampling tensor shaped as:

```text
[batch, queries, heads, levels, points, xy]
```

Neither belongs in the deployed graph. The equivalent export path must:

1. retain the CUDA implementation for training;
2. flatten `levels x points` before tracing, producing the rank-five tensor
   `[batch, queries, heads, levels*points, xy]`;
3. slice each level's four points and use bilinear `grid_sample`;
4. let coremltools lower the four samples to native MIL `resample` operations;
5. use fixed 1280x384 feature geometry for the first deployment model.

This is an export-only layout rewrite. It does not change attention values,
weights, sampling coordinates, or learned parameters.

## Reproducible probe

[`scripts/probe_coreml_ms_deform_attn.py`](scripts/probe_coreml_ms_deform_attn.py)
uses the locked graph geometry:

```text
feature maps:       48x160, 24x80, 12x40, 6x20
flattened tokens:   10,200
heads:              8
channels/head:      32
points/level:       4
decoder queries:    50
encoder queries:    10,200
```

Environment used: Python 3.13, PyTorch 2.7.1, coremltools 9.0, ML Program,
iOS 17 minimum target, Float32, and CPU-only execution on the development Mac.

| Case | Trace | Convert | Predict | Max absolute delta | Result |
|---|---:|---:|---:|---:|---|
| Decoder, Q=50 | 0.038 s | 0.460 s | 0.0247 s | 2.22e-6 | Pass |
| Encoder, Q=10,200 | 0.264 s | 0.447 s | 0.171 s | 1.11e-5 | Pass |

Both graphs contain four native `resample` operations and no MIL `custom`
operation. The probe enforces an FP32 maximum-absolute-error tolerance of
`2e-5`. Mac CPU timings are feasibility evidence only and are not iPhone
latency claims.

Run both cases with:

```bash
python scripts/probe_coreml_ms_deform_attn.py --output-dir /tmp/mobileadas3d-coreml-probe
```

The script writes TorchScript, `.mlpackage`, per-case JSON, and a combined JSON
summary. Generated model packages are intentionally not committed.

## Remaining gate sequence

1. Add the rank-five export branch to the pinned MonoDETR source patch while
   keeping CUDA training unchanged.
2. Trace and convert the complete randomly initialized fixed-shape graph.
3. Export the selected trained Vehicle + Pedestrian checkpoint.
4. Compare PyTorch and Core ML raw tensors and decoded detections on fixed KITTI
   images.
5. Require the parity thresholds in `PRODUCT_MODEL_CONTRACT.md`.
6. Install on the target iPhone and measure model-only and end-to-end latency,
   memory, thermals, and 30-minute stability.

If the complete graph fails despite this microkernel result, MonoDETR remains
the accuracy teacher and the fallback is a Core-ML-native student.
