# MonoDGP M56d Det2D-Transformer-FP32 Storage Contract

Status: complete; offline compressed checkpoint selected on 2026-09-14.

## Purpose

M56d tests the single storage policy isolated by M56c. It retains the complete
alias-consistent `det2d_transformer` Conv2d/Linear parameter group in FP32 and
stores every other eligible Conv2d/Linear parameter in FP16. It performs no
training and changes no graph, operator, activation dtype, decoding rule, or
metric.

This is a controlled offline checkpoint-storage experiment. Loading the
candidate into the ordinary MonoDGP model restores FP32 runtime parameters, so
M56d does not predict execution-speed or runtime-memory improvement.

## Immutable evidence

- Parent: selected M54 MonoDGP epoch 100.
- Parent checkpoint SHA-256:
  `8e79f3921d96e1de70cbb4219245e3fcc3fa1fb67ae675468b4ebca90e579847`.
- Upstream MonoDGP commit:
  `aa059a18214aebf644510e7f0793971b403f9d14`.
- M56c manifest SHA-256:
  `0c501fa2e0f13e8ad952e492ef44d45a93fa72d3655e15ee4dfca5ae77417796`.
- M56c diagnostic JSON SHA-256:
  `7f1ca25ec2270642c508bb36a8e109bdcf8cdf09d8b05d5749a23a0129321ba1`.
- M56c diagnostic CSV SHA-256:
  `8d9eea64c0bf67eb7f4b597e76344d67ceb5e5a392074259050fa51ae32c5abe`.
- Fixed smoke sample: first Chen validation image, `000001`.
- Raw-output thresholds: unchanged from M56, M56b, and M56c.

Preparation fails closed if any evidence hash, source checkpoint, source
commit, runtime configuration, architecture invariant, split, or parameter
inventory changes.

## M56c decision evidence

M56c executed all 16 planned policies and passed every diagnostic-integrity
gate. The complete 2D transformer was the only group that failed in isolation:

- `only_det2d_transformer`: final-depth maximum delta
  `0.5688076 > 0.50`.
- `all_except_det2d_transformer`: every output family passed and final-depth
  maximum delta was `0.0300064`.
- The latter policy's projected parameter-size ratio was `0.5646269`, below
  the unchanged `0.60` gate.

Holding the backbone or input projection in FP32 also rescued parity, but
their projected ratios were `0.814675` and `0.603382`. They are strictly worse
storage choices for this rung. M56d therefore contains exactly one candidate;
it is not a sweep.

## Frozen storage policy

All floating-point parameters directly owned by ordinary Conv2d or Linear
modules are grouped by unique parameter identity so shared state-dict aliases
cannot receive conflicting dtypes.

- FP32: all 80 aliases/unique parameters owned by `det2d_transformer`, totaling
  `9,476,104` original FP32 bytes.
- FP16: 295 state aliases representing 247 unique parameters outside
  `det2d_transformer`, totaling `146,855,348` original FP32 bytes.
- Expected FP16 state-name SHA-256:
  `6afced8a5d40e0829f8b5b5cc06c2e55b1161a2bdfe22a7d17a61deb5de10255`.
- Noneligible parameters and buffers: bitwise unchanged.
- Optimizer/scheduler state: removed from both paired model-only artifacts.

The candidate model-only checkpoint must be no larger than 60% of the paired
FP32 model-only checkpoint.

## Stop point 1: real-CUDA smoke

The smoke must verify:

- exact source, M55, and M56c evidence bindings;
- exact state keys and epoch;
- reproduced alias-consistent FP16 and FP32 policy sets and name hashes;
- stored FP16 dtypes and bitwise-exact FP32/noneligible tensors;
- unchanged FP32 runtime parameters;
- unchanged output paths and shapes;
- finite output tensors;
- maximum raw-output deltas within every unchanged limit;
- paired model-only size ratio no larger than `0.60`;
- five warmups and 100 timed CUDA predictions;
- no training, graph-change, Core ML, deployment, or product-safety claim.

The frozen raw-output limits are:

| Output | Maximum absolute delta |
| --- | ---: |
| `pred_logits` | 0.10 |
| `pred_boxes` | 0.01 |
| `pred_3d_dim` | 0.10 |
| `pred_depth` | 0.50 m |
| `pred_angle` | 0.10 |
| `pred_depth_map_logits` | 0.10 |
| `pred_region_prob` | 0.01 |

Stop and return `m56d_compression_manifest.json` and
`m56d_det2d_fp32_smoke.json`. Complete validation is forbidden until this
exact pair is reviewed.

## Complete validation barrier

Only after an approved smoke may M56d run one complete 3,769-image Chen
validation. It must enforce the existing M54 preservation gates:

- Vehicle moderate 3D AP_R40 at least `18.4793`;
- Pedestrian moderate 3D AP_R40 at least `5.8661`;
- balanced moderate 3D mean at least `12.1727`;
- Vehicle moderate BEV AP_R40 at least `24.4863`;
- Pedestrian moderate BEV AP_R40 at least `6.4292`;
- Vehicle nearby recall at least `0.89993`;
- Pedestrian nearby recall at least `0.71487`;
- Pedestrian localization-failure rate at most `0.24765`;
- exactly 3,769 / 3,769 prediction files.

Passing selects an offline compressed storage artifact only. It does not
authorize direct Core ML conversion or product deployment. Custom deformable
attention remains an M57 export blocker, and product safety remains false
because Pedestrian nearby recall is still below the separate `0.80` target.

## Required artifacts

After Stop point 1:

- `m56d_compression_manifest.json`
- `m56d_det2d_fp32_smoke.json`
- durable preparation and smoke logs

Only after an approved complete evaluation:

- `m56d_det2d_fp32_gate.json`
- `m56d_det2d_fp32_comparison.csv`
- checkpoint-bound prediction manifest and durable evaluation logs

## Frozen outcome

The exact M56d manifest and CUDA smoke passed every required check. The
candidate model-only checkpoint is `98,048,696` bytes versus `173,610,189`
bytes for the paired FP32 artifact, a `0.5647635` ratio and `75,561,493` bytes
saved. The smoke retained the unchanged output structure and passed all seven
raw-output limits; final decoded-depth maximum absolute delta was `0.326031 m`
against the `0.50 m` limit.

Complete Chen validation produced all `3,769/3,769` prediction files and
passed every frozen preservation gate:

| Metric | Parent | M56d | Change |
| --- | ---: | ---: | ---: |
| Vehicle moderate 3D AP_R40 | 19.4519 | 19.4666 | +0.0148 |
| Pedestrian moderate 3D AP_R40 | 6.1749 | 6.1848 | +0.0099 |
| Balanced moderate 3D mean | 12.8134 | 12.8257 | +0.0123 |
| Vehicle moderate BEV AP_R40 | 25.7751 | 25.7349 | -0.0402 |
| Pedestrian moderate BEV AP_R40 | 6.7676 | 6.7651 | -0.0026 |
| Vehicle nearby recall | 0.90993 | 0.90993 | 0.00000 |
| Pedestrian nearby recall | 0.72487 | 0.72399 | -0.00088 |
| Pedestrian localization-failure rate | 0.23765 | 0.23854 | +0.00088 |

The negligible signed metric differences are expected consequences of FP16
checkpoint rounding and satisfy all predeclared tolerances. M56d is selected
as the offline compressed artifact. Its runtime still restores FP32 weights,
so it establishes no speed or runtime-memory benefit. The reported M55/M56d
timings used non-comparable environments and cannot support a regression or
speedup claim. M57 must address the custom deformable-attention export blocker;
direct Core ML conversion and product-safety qualification remain false.
