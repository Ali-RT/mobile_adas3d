# MonoDGP M56b Selective FP16-Storage Contract

Status: frozen before M56b artifact generation or complete validation.

## Purpose

M56b is a controlled follow-up to the rejected M56 all-eligible FP16-storage
experiment. It asks one narrow question: does preserving the directly
depth-sensitive prediction heads in FP32 restore the frozen raw-output parity
while retaining a useful model-only storage reduction?

M56b performs no training, fine-tuning, pruning, activation conversion, graph
replacement, Core ML conversion, or deployment qualification. Stored FP16
parameters are loaded into the unchanged FP32 MonoDGP graph. Therefore M56b
does not predict lower execution latency or runtime memory.

## Immutable parent and evidence

- Parent: selected M54 MonoDGP epoch 100.
- Parent checkpoint SHA-256:
  `8e79f3921d96e1de70cbb4219245e3fcc3fa1fb67ae675468b4ebca90e579847`.
- Upstream MonoDGP commit:
  `aa059a18214aebf644510e7f0793971b403f9d14`.
- M55 gate/profile/audit SHA-256:
  `684338ae7be11f5394aff76b9e3115c22f583fb06103db02c56091c4208050fc`,
  `8ef15616614c9852ed6b51f171b69a7a0f6994a5a04b79789f0ebbdb1a3d0751`,
  and `798746036e50d37729adca3e2ef9a966a65386ab000f97feb516962b3e1ee1a3`.
- Rejected M56 manifest SHA-256:
  `a9cfbd8e41ed020b4d49ee44594b81966b27601750fe4167454cbbdecffde046`.
- Rejected M56 candidate SHA-256:
  `8da64f181e5c09b17a29e09eb41b1133159313f2d09eeb1ac8218bf1d1ea4404`.
- M56 failed exactly one raw-output family: final `pred_depth` maximum
  absolute delta `0.7107391357421875` exceeded the frozen `0.50` limit.
- Chen validation protocol: exactly 3,769 images, score threshold 0.001,
  TopK 50.

Preparation must fail closed if the M56 evidence does not reproduce that exact
rejection. M56b is not permission to reinterpret or relax the M56 result.

## Frozen storage policy

M55 identified 156,331,452 unique FP32 parameter bytes directly owned by
ordinary Conv2d and Linear modules. M56b partitions those exact eligible
parameters by parameter identity:

1. Retain every parameter alias associated with `bbox_embed`,
   `dim_embed_3d`, or `depth_embed` in FP32 storage.
2. Store every remaining eligible Conv2d/Linear parameter alias in FP16.
3. Leave every noneligible tensor and buffer bitwise unchanged.
4. Remove optimizer state from both the FP32 model-only control and selective
   FP16 model-only candidate.

The policy is alias-aware. MonoDGP reuses some prediction modules across
decoder layers, so one tensor can have several state-dict names. If any alias
belongs to a preserved head, all aliases for that shared parameter remain
FP32. This prevents checkpoint load order from silently undoing the policy.

The three preserved roots are fixed before running M56b. The MonoDGP depth
path combines predicted 3D height, decoded 2D box height, focal length, and a
learned depth residual. Preserving all three associated prediction heads is a
single mechanistic intervention aimed at M56's isolated depth-parity failure.

The candidate model-only checkpoint must be no larger than 60% of the paired
FP32 model-only control. If preserving these heads exceeds that limit, M56b
fails before CUDA smoke testing.

## CUDA smoke barrier

The notebook must stop after one fixed Chen-validation image unless every
smoke gate passes. The smoke must verify:

- exact parent, M55, M56, candidate, runtime-config, and epoch bindings;
- exact state-dict key equality;
- disjoint, complete FP16 and preserved-FP32 policy lists;
- every listed FP16 tensor is stored as FP16;
- every listed preserved-FP32 tensor is bitwise equal to its source;
- every tensor outside the FP16 policy is bitwise unchanged;
- all runtime model parameters are FP32 after loading;
- identical output paths and shapes with finite candidate values;
- the same frozen M56 maximum absolute raw-output limits:
  - logits 0.10;
  - normalized boxes 0.01;
  - dimensions 0.10;
  - depth 0.50;
  - angle 0.10;
  - depth-map logits 0.10;
  - region probabilities 0.01;
- separate final-depth and log-variance channel diagnostics;
- five warmups and 100 CUDA-event-timed predictions.

Timing is comparable with M55 only when GPU, PyTorch, CUDA, and sample match.
No speedup is required or claimed.

## Complete preservation gate

Only a passing smoke authorizes one complete 3,769-image evaluation. The
complete-evaluation cell defaults `AUTHORIZE_COMPLETE_EVALUATION=False`; it
must be changed manually only after review, so Colab "Run all" cannot cross
the barrier silently.

After explicit authorization, every frozen M54-relative row must pass:

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

No aggregate improvement can compensate for an individual failure.

## Decision

- If preparation or smoke fails, reject M56b without complete inference.
- If smoke passes but any complete gate fails, reject M56b and retain the
  exact M54 FP32 parent.
- If every gate passes, select the M56b offline storage artifact and authorize
  M57 deformable-attention decomposition/replacement as a separate experiment.

Regardless of outcome, direct Core ML conversion remains unauthorized and
product safety remains false. The separate Pedestrian nearby-recall target is
0.80, and external-domain plus physical-device qualification remain pending.

## Required durable artifacts

Before review at stop point 1:

- `m56b_compression_manifest.json`
- `m56b_parent_model_only_fp32.pth`
- `monodgp_m56b_selective_fp16_storage/checkpoint_epoch_100.pth`
- `m56b_selective_fp16_storage_smoke.json`
- durable preparation and smoke logs

Only after an approved smoke pass:

- `m56b_selective_fp16_storage_prediction_manifest.json`
- `m56b_selective_fp16_storage_gate.json`
- `m56b_selective_fp16_storage_comparison.csv`
- durable inference and evaluation logs
