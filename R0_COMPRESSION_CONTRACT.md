# R0 Compression Acceptance Contract

## Status

Frozen on 2026-09-02. This contract governs compression experiments derived
from R0 ResNet50 MonoDETR epoch 185. Passing it means **R0 accuracy is
preserved**; it does not mean the model meets the product safety target.

## Immutable parent and protocol

- Checkpoint: `checkpoint_epoch_185.pth`
- SHA-256: `fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59`
- Split: Chen validation, all 3,769 images
- Taxonomy: Vehicle and Pedestrian
- Input: 1280x384
- Score threshold / TopK: 0.001 / 50
- Dataset, preprocessing, decoding, NMS, and evaluator may not change.

## Accuracy-preservation gates

Every gate must pass. AP floors retain at least 95% of frozen R0. Recall may
drop by at most one absolute percentage point, and Pedestrian localization
failure may increase by at most one absolute percentage point.

| Gate | R0 | Candidate requirement |
| --- | ---: | ---: |
| Vehicle moderate 3D AP_R40 | 17.6348 | >= 16.7530 |
| Pedestrian moderate 3D AP_R40 | 5.7214 | >= 5.4353 |
| Balanced moderate 3D mean | 11.6781 | >= 11.0942 |
| Vehicle moderate BEV AP_R40 | 23.6816 | >= 22.4975 |
| Pedestrian moderate BEV AP_R40 | 6.5961 | >= 6.2663 |
| Vehicle nearby recall | 0.88246 | >= 0.87246 |
| Pedestrian nearby recall | 0.68342 | >= 0.67342 |
| Pedestrian localization-failure rate | 0.24603 | <= 0.25603 |
| Complete prediction set | 3,769 | exactly 3,769 non-missing files |

Any NaN, crash, missing prediction, checkpoint/protocol mismatch, or failed
gate rejects the candidate. The R0 denominator never moves to a compressed
candidate.

## Product-status boundary

Pedestrian nearby recall >= 0.80 remains an unmet aspirational product target.
No compression result may be described as safety-qualified, production-ready,
or externally qualified merely because it preserves R0. External nuScenes and
deployment qualification remain blocked until explicitly authorized.

## Controlled compression ladder

Change one variable per experiment:

1. M52: FP16/mixed-precision evaluation with unchanged architecture.
2. Weight-only quantization sensitivity.
3. Structured width/depth/token reduction with retraining if required.
4. QAT only after a post-training quantization baseline exists.
5. Core ML conversion and physical-device qualification last.

Each experiment requires immutable provenance, a complete validation run, the
gates above, and a versioned result manifest before the next rung starts.

M52 completed through `notebooks/MonoDETR_M52_R0_FP16_Gate_Colab.ipynb`. It uses
selective CUDA mixed precision without changing stored R0 weights or model
structure. The backbone feature extraction, input projections that feed depth,
depth prediction/position path, and legacy custom deformable-attention extension
remain FP32. The extension's pinned CUDA dispatch supports float/double, not half;
only the remaining compatible transformer and prediction-head operators use FP16
autocast. A one-batch CUDA smoke test must prove those dtype boundaries and finite
outputs before full validation. Only the complete JSON comparison may authorize
the next compression rung.

The exact R0 epoch-185 checkpoint produced all 3,769 prediction files. M52
Vehicle/Pedestrian moderate 3D AP_R40 was `17.6399/5.6952`, balanced mean was
`11.6676`, and moderate BEV was `23.6395/6.6603`. Vehicle/Pedestrian nearby
recall was `0.88215/0.68519`; Pedestrian localization-failure rate was `0.24559`.
All nine frozen preservation gates passed, authorizing weight-only quantization
sensitivity as the next rung. This result does not satisfy the aspirational
Pedestrian safety target and contains no controlled FP32-versus-M52 runtime or
memory comparison, so it does not authorize a deployment-speed claim.
