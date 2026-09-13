# MonoDGP M56 FP16 Parameter-Storage Contract

Status: frozen before M56 artifact generation or full validation.

## Purpose

M56 is the first controlled compression experiment on the selected M54
accuracy parent. It asks one narrow question: can ordinary Conv2d and Linear
parameters be rounded to FP16 for storage, then loaded into the unchanged FP32
MonoDGP graph, without violating the frozen M54 preservation gates?

This is an offline storage-sensitivity experiment. It performs no training,
fine-tuning, pruning, activation quantization, graph replacement, Core ML
conversion, or product-safety qualification. Because runtime execution remains
FP32, M56 does not predict an inference speedup or runtime-memory reduction.

## Immutable evidence and parent

- Parent: M54 MonoDGP epoch 100.
- Parent checkpoint SHA-256:
  `8e79f3921d96e1de70cbb4219245e3fcc3fa1fb67ae675468b4ebca90e579847`.
- M55 gate SHA-256:
  `684338ae7be11f5394aff76b9e3115c22f583fb06103db02c56091c4208050fc`.
- M55 profile SHA-256:
  `8ef15616614c9852ed6b51f171b69a7a0f6994a5a04b79789f0ebbdb1a3d0751`.
- M55 audit SHA-256:
  `798746036e50d37729adca3e2ef9a966a65386ab000f97feb516962b3e1ee1a3`.
- Upstream MonoDGP commit:
  `aa059a18214aebf644510e7f0793971b403f9d14`.
- Chen validation: exactly 3,769 images, score threshold 0.001, TopK 50.

M55 measured 156,331,452 bytes of unique parameters directly owned by ordinary
Conv2d and Linear modules, equal to 92.69% of all parameter bytes. Direct Core
ML conversion remains unauthorized because the graph contains nine custom
`MSDeformAttn` modules.

## Frozen storage policy

M56 creates two model-only checkpoints with identical metadata and no optimizer
state:

1. An FP32 model-only control containing the exact M54 state.
2. A candidate where every floating-point parameter directly owned by Conv2d
   or Linear is stored as FP16. This includes both weight and bias parameters.

All noneligible tensors and buffers remain bitwise unchanged. Loading the
candidate into the ordinary FP32 model must cast the stored FP16 parameters
back to FP32. Architecture, operators, inputs, decoding, class mapping, score
threshold, and TopK are unchanged.

The candidate model-only checkpoint must be no larger than 60% of the FP32
model-only control. The original 495.99 MB training checkpoint is not the size
denominator because it contains training state; the paired model-only control
prevents optimizer removal from being misreported as weight compression.

## CUDA smoke barrier

Before complete validation, one fixed Chen-validation image must pass:

- exact source/candidate/config hashes and epoch 100;
- every eligible stored parameter is FP16;
- every noneligible tensor is bitwise unchanged;
- all runtime model parameters are FP32 after loading;
- identical output paths and tensor shapes with finite values;
- maximum absolute raw-output deltas no greater than:
  - logits 0.10;
  - normalized boxes 0.01;
  - dimensions 0.10;
  - depth 0.50;
  - angle 0.10;
  - depth-map logits 0.10;
  - region probabilities 0.01;
- five warmups and 100 CUDA-event-timed candidate predictions.

Latency and CUDA memory are recorded. A speed comparison is valid only if GPU,
PyTorch, CUDA, and sample match M55. No speedup is required or claimed because
the candidate runs in FP32.

## Complete preservation gate

Only a passing smoke authorizes all 3,769 validation images. Every row must
pass:

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

No aggregate gain can compensate for an individual failure.

## Decision

If the size, smoke, and all complete-validation gates pass, the FP16-storage
checkpoint becomes the first selected compressed M54 artifact and M57 may
address deformable-attention export. If any gate fails, M56 is rejected and
the exact M54 parent remains selected.

Regardless of the M56 outcome:

- direct Core ML conversion remains unauthorized;
- the result is not an iPhone latency claim;
- product safety remains false because the Pedestrian nearby-recall target is
  0.80 and external/device qualification remains outstanding.

## Required durable artifacts

- `m56_compression_manifest.json`
- `m56_parent_model_only_fp32.pth`
- `monodgp_m56_fp16_parameter_storage/checkpoint_epoch_100.pth`
- `m56_fp16_storage_smoke.json`
- `m56_prediction_manifest.json`
- `m56_fp16_storage_gate.json`
- `m56_fp16_storage_comparison.csv`
- durable preparation, smoke, inference, and evaluation logs

## Frozen outcome

M56 was rejected at its real-CUDA smoke barrier on 2026-09-13. Every gate
passed except raw final-depth parity: maximum `pred_depth` absolute delta was
`0.710739` against the frozen `0.500000` limit. The threshold is not
changed after observing the result. Consequently the complete validation was
not authorized, and no M56 prediction manifest, final gate, or comparison CSV
should exist.

The next controlled experiment is M56b. It may retain the directly
depth-sensitive `bbox_embed`, `dim_embed_3d`, and `depth_embed`
parameters in FP32 while applying the same FP16 policy elsewhere. It must use
the same parent, sample, paired accounting, and parity thresholds. This outcome
does not itself authorize M56b's complete validation.
