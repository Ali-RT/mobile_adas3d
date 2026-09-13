# MonoDGP M55 Compression Feasibility Contract

Status: complete; all feasibility gates passed on 2026-09-13.

## Purpose

M55 establishes whether the selected M54 accuracy parent is ready for a
controlled compression ladder. It measures the untouched parent, freezes the
accuracy-preservation denominators, inventories quantizable parameters, and
identifies export blockers before any weights or graph operators are changed.

M55 performs no training, pruning, quantization, Core ML conversion, or product
safety qualification. A pass authorizes only a separate M56 offline
weight-compression experiment.

## Immutable parent and protocol

- Parent: M54 MonoDGP epoch 100.
- Parent checkpoint SHA-256:
  `8e79f3921d96e1de70cbb4219245e3fcc3fa1fb67ae675468b4ebca90e579847`.
- Upstream repository: https://github.com/PuFanqi23/MonoDGP
- Upstream commit: `aa059a18214aebf644510e7f0793971b403f9d14`.
- Split protocol: Chen 3,712 train / 3,769 validation.
- Validation split SHA-256:
  `6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8`.
- Product classes: Vehicle and Pedestrian.
- Evaluation: score threshold 0.001, TopK 50, complete 3,769-image split.

The M54 selection report must identify epoch 100, the exact checkpoint hash,
all eight R0-comparability gates passing, and `accuracy_parent_candidate=true`.
The source checkpoint is never copied over or modified in place.

## Frozen M54 denominators

| Metric | M54 parent |
| --- | ---: |
| Vehicle moderate 3D AP_R40 | 19.451854 |
| Pedestrian moderate 3D AP_R40 | 6.174879 |
| Balanced moderate 3D mean | 12.813366 |
| Vehicle moderate BEV AP_R40 | 25.775087 |
| Pedestrian moderate BEV AP_R40 | 6.767616 |
| Vehicle nearby recall | 0.909925 |
| Pedestrian nearby recall | 0.724868 |
| Pedestrian localization-failure rate | 0.237654 |

These values are preservation denominators, not claims that the parent is
product-safe. Pedestrian nearby recall remains below the separate 0.80 target.

## Compression preservation gates

Every future compressed candidate derived from M54 must satisfy all rows:

| Gate | Required |
| --- | ---: |
| Vehicle moderate 3D AP_R40 | >= 18.479261 |
| Pedestrian moderate 3D AP_R40 | >= 5.866135 |
| Balanced moderate 3D mean | >= 12.172698 |
| Vehicle moderate BEV AP_R40 | >= 24.486333 |
| Pedestrian moderate BEV AP_R40 | >= 6.429235 |
| Vehicle nearby recall | >= 0.899925 |
| Pedestrian nearby recall | >= 0.714868 |
| Pedestrian localization-failure rate | <= 0.247654 |
| Prediction completeness | exactly 3,769 / 3,769 |

The AP floors retain 95% of M54. Nearby recall may fall by at most 0.01
absolute per class. Pedestrian localization failures may increase by at most
0.01 absolute. No aggregate pass can compensate for a failed individual row.

## Native baseline profile

M55 runs the exact parent in evaluation mode on one real Chen-validation image
using CUDA, batch size one, five warmups, and 100 timed model-only predictions.
It records:

- device, software versions, checkpoint and configuration hashes;
- total/trainable parameters, raw state-dict bytes, checkpoint bytes, and dtype;
- parameter bytes by top-level component and operator family;
- fixed input and output tensor shapes plus finite-output checks;
- mean, median, p95, minimum, and maximum CUDA-event latency;
- loaded, peak allocated, and peak reserved CUDA memory;
- a PyTorch-profiler FLOP count explicitly labeled as a partial lower bound,
  because the custom deformable-attention CUDA kernel is not FLOP-accounted.

Latency is an environment-specific baseline, not an iPhone speed claim. A
future speed comparison must use the same GPU/runtime, sample, batch size,
warmup count, timed count, and synchronization method.

## Quantization and export audit

M55 inventories unique Conv2d and Linear parameter bytes as the initial
weight-compression scope. It also records all custom deformable-attention
modules and the fraction of parent parameters covered by ordinary weight
operators.

The pinned graph uses the custom `MultiScaleDeformableAttention` CUDA extension.
Therefore direct Core ML conversion is not authorized by M55. The eventual
deployment path must first replace or decompose that operator and prove raw
tensor parity before Core ML conversion. Dynamic calibration-matrix and image-
size inputs must remain explicit and tested; decoder/post-processing behavior
must not be silently moved outside the parity contract.

## M55 decision

M55 passes only when all provenance checks pass, the exact checkpoint loads,
all required outputs are finite, the 5/100 CUDA benchmark completes, size and
memory fields are positive, and the operator audit is complete.

A pass sets `offline_weight_compression_authorized=true`. It does not set
`direct_coreml_conversion_authorized` or `product_safety_qualified` to true.
M56 must be a separate versioned experiment and must rerun the complete AP,
nearby-recall, localization, and prediction-completeness gates above.

## Required durable artifacts

- `m55_feasibility_manifest.json`
- `monodgp_m55_profile.yaml`
- `m55_native_baseline_profile.json`
- `m55_native_baseline_latency.csv`
- `m55_operator_export_audit.json`
- `m55_feasibility_gate.json`
- durable Colab logs for preparation, profiling, and finalization

The first files to return are `m55_feasibility_gate.json`,
`m55_native_baseline_profile.json`, and `m55_operator_export_audit.json`.

## Execution result (2026-09-13)

All 19 M55 feasibility checks passed against the exact M54 epoch-100
checkpoint and SHA-256. The untouched FP32 parent contains 42,163,648
parameters (168.65 MB of parameter tensors; 173.34 MB state-dict tensors) and
the serialized checkpoint is 495.99 MB. The profiler recorded a partial
143.56 GFLOP lower bound; custom deformable-attention kernels are explicitly
excluded from that count.

On an NVIDIA RTX PRO 6000 Blackwell Server Edition with CUDA 12.8 and PyTorch
2.11.0, batch-one 1280×384 model-only inference measured 12.67 ms mean,
12.67 ms median, and 12.72 ms p95 over 100 CUDA-event-timed runs after five
warmups. Peak allocated/reserved CUDA memory was 492.97 MB/1.086 GB. These are
environment-specific baseline values, not iPhone claims.

Ordinary Conv2d and Linear weights account for 156,331,452 bytes, or 92.69%
of all parameter bytes, so `offline_weight_compression_authorized=true` for a
separate M56 experiment. The audit found nine custom `MSDeformAttn` modules;
therefore `direct_coreml_conversion_authorized=false`. Direct export still
requires deformable-attention decomposition/replacement and raw-output parity.
Product safety remains false because Pedestrian nearby recall is 0.72487,
below the 0.80 target.

Durable review copies:

- `artifacts/m55_feasibility_gate_20260913.json`
- `artifacts/m55_native_baseline_profile_20260913.json`
- `artifacts/m55_operator_export_audit_20260913.json`
