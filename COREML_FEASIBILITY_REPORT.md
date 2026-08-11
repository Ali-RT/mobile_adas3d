# MobileMonoDETR-VP1 Core ML feasibility report

Status: native compile/load/predict passed; physical-iPhone latency gate failed
Date: 2026-08-11

## Decision

MobileMonoDETR-VP1 is Core ML compatible, but it is not the production iPhone
architecture at the locked 1280x384 geometry. Its complete graph compiles,
loads, and predicts on an iPhone 16 Pro Max without a custom operation or host
callback, but steady model-only predictions take 161-177 ms. This is more than
three times the 50 ms physical-iPhone inference gate.

This result does not approve deployment. Random weights are sufficient for this
compute-feasibility measurement because parameter values do not change graph
cost. Retain MobileMonoDETR as the accuracy/reference model and proceed with a
Core-ML-native student unless a separately versioned architecture ablation can
meet the same quality and runtime contract. Trained-checkpoint parity and
decoded KITTI parity are still required for any future deployment candidate.

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

1. **Completed 2026-08-11:** add the rank-five export branch while preserving
   the original CUDA training paths.
2. **Completed 2026-08-11:** trace and convert the complete randomly initialized
   fixed-shape graph to an ML Program.
3. **Completed 2026-08-11:** compile, load, and predict with the random-weight
   package on the target Xcode/iOS runtime and iPhone 16 Pro Max.
4. Export the selected trained Vehicle + Pedestrian checkpoint.
5. Compare PyTorch and Core ML raw tensors and decoded detections on fixed KITTI
   images.
6. Require the parity thresholds in `PRODUCT_MODEL_CONTRACT.md`.
7. Install on the target iPhone and measure model-only and end-to-end latency,
   memory, thermals, and 30-minute stability.

If the complete graph fails despite this microkernel result, MonoDETR remains
the accuracy teacher and the fallback is a Core-ML-native student.

## Complete random-weight graph result

The exact pinned MonoDETR source plus MobileNetV4 patch now receives
[`third_party/monodetr/coreml_export.patch`](third_party/monodetr/coreml_export.patch).
All export changes are guarded by `coreml_export`; normal CUDA training and
checkpoint parameters remain unchanged. Besides deformable attention, the
export path replaces PyTorch multi-head-attention internals and in-place box
updates with mathematically equivalent native tensor operations.

[`scripts/probe_coreml_full_monodetr.py`](scripts/probe_coreml_full_monodetr.py)
successfully traced and converted the complete two-class random-weight model:

```text
input:                    [1, 3, 384, 1280]
outputs:                  logits [1,50,2], boxes [1,50,6]
                          dimensions [1,50,3], depth [1,50,2]
                          angle [1,50,24]
export-enabled modules:   12
TorchScript max delta:    0.0
native MHA max delta:     7.15e-7
Torch frontend ops:       2,322
ML Program custom ops:    0
ML Program resample ops:  27
ML Program matmul ops:    14
TorchScript size:         50.4 MB
ML package size:          55.8 MB
conversion time:          4.60 s
```

The reproducible default uses `skip_model_load=True`, which validates and saves
the complete ML Program without asking macOS to compile it. An initial native
load attempt completed every Core ML conversion/backend pass but did not return
within ten minutes and was stopped. The active command-line-tools selection did
not expose `coremlcompiler`/`coremlc`; native compile/load was subsequently
completed by selecting full Xcode and using a physical iPhone, as recorded
below.

## Native Xcode and physical-iPhone result

The generated package compiled successfully with Xcode's iOS Core ML compiler,
was embedded in a signed app, loaded with `MLComputeUnits.all`, and completed a
warmup plus three predictions on an iPhone 16 Pro Max running iOS 26.6.

| Measurement | Result |
|---|---:|
| Native Xcode compile | Pass |
| Model load | 566.0 ms |
| First warmup prediction | 5,422.2 ms |
| Timed prediction 1 | 176.6 ms |
| Timed prediction 2 | 161.3 ms |
| Timed prediction 3 | 161.2 ms |
| Output tensors returned | 5/5 |
| Runtime compatibility | Pass |
| `<= 50 ms` inference gate | **Fail** |

The three-sample probe is a feasibility measurement, not a statistically valid
p95 benchmark. It is already decisive for architecture selection because every
steady sample exceeds the 50 ms limit by more than 3x. The raw device report is
[`reports/coreml/mobilemonodetr_vp1_random_iphone16promax.json`](reports/coreml/mobilemonodetr_vp1_random_iphone16promax.json).
