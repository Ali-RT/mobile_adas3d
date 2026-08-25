# MobileADAS3D model milestones and architecture

Last updated: 2026-08-24

This document is the concise reporting reference for model selection,
teacher/student status, metrics, edge constraints, completed evidence, and the
next experiment. The canonical execution status remains in `PROJECT_TRACKER.md`.

## Product targets

1. Detect and reconstruct road objects from one RGB camera image, including 2D
   box, 3D position, depth, dimensions, orientation, and confidence.
2. First train a teacher-compatible student to at least 90% of the frozen R0
   metrics, without rejecting it for current iPhone limits.
3. Validate the frozen accurate model outside KITTI, then compress and qualify
   the selected descendant for a deployment target.

The frozen production classes are:

- **Vehicle:** Car, Van, Truck, and Tram.
- **Pedestrian:** Pedestrian and Person_sitting.

Cyclist, Misc, and DontCare are excluded from the first product model.

## Model-selection milestones

| Milestone | Outcome | Decision |
| --- | --- | --- |
| Lightweight MobileNetV4 v1-v5 experiments | Fast and Core-ML-friendly, but insufficient 3D AP after geometry, scoring, and yaw experiments | Retain MobileNetV4 as the mobile backbone; replace the original student design |
| Published MonoDETR feasibility | Published ResNet50 checkpoint reproduced on all 3,769 Chen-validation images; Car moderate 2D/BEV/3D AP_R40 was about 88.10/27.1/20.35 | Accept MonoDETR as the accuracy source and teacher family |
| Two-class R0 teacher training | Fine-tuned original ResNet50 MonoDETR for Vehicle/Pedestrian through 195 epochs | Teacher/reference available; not an iPhone candidate |
| R0 checkpoint sweep | Evaluated 39 checkpoints; selected epoch 185 using balanced Vehicle/Pedestrian moderate 3D AP_R40 | Freeze epoch 185 and its SHA-256 as the denominator |
| MonoDETR MobileNetV4 backbone ablation | Car-only epoch 40 reached moderate 3D AP_R40 14.065, retaining 69.2% of the ResNet50 teacher | Use as evidence for a fresh two-class teacher-compatible baseline, not as a product checkpoint |
| Full transformer Core ML gate | Graph conversion was possible after compatibility work, but full-model iPhone latency was unacceptable | Preserve evidence; suspend this rejection criterion during accuracy development |
| MobileADAS3D-S1/H1/H2 | Edge-efficient graphs, but supervised learning gates failed | Freeze as negative experiments; do not resume |
| S1 GT-only 20-epoch gate | Training was stable, but Vehicle/Pedestrian moderate 3D AP_R40 was only 0.024/0.519 | Reject continuation to 100 epochs |
| S1 checkpoint and geometry diagnosis | Epoch 20 was best; 12,105 Vehicle detections matched at mean 2D IoU 0.675, but yaw/shape/placement were poor | 2D detection is not the first failure |
| S1 yaw diagnosis | Vehicle axis error was good at 9.63 degrees mean, but final yaw was 72.28 degrees with a 35.8% flip-candidate rate | Reject independent axis plus hard direction-bit yaw |
| A1 two-class GT-only baseline | MobileNetV4-MonoDETR trained for 195 epochs; epoch 140 reached Vehicle/Pedestrian moderate 3D AP_R40 12.860/7.267 and BEV 18.785/8.510 | Freeze as a healthy paired-experiment baseline; Pedestrian exceeds R0, but Vehicle prevents final qualification |
| A1 paired R0 distillation | Online R0 query guidance changed Vehicle moderate 3D AP_R40 by only +0.082 while Pedestrian fell 1.937, balanced mean fell 0.927, and both BEV metrics regressed | Reject the distilled branch; the evidence points to A1 representation capacity, not missing teacher supervision, as the next bottleneck |

## Teacher/reference status

Two results must remain separate.

### Published MonoDETR checkpoint reproduction

This validates the upstream implementation and checkpoint on the Chen split:

| Car AP_R40 | Easy | Moderate | Hard |
| --- | ---: | ---: | ---: |
| 2D bounding box | 96.26 | 88.10 | 83.31 |
| Bird's-eye view | 37.65 | 27.10 | 23.33 |
| 3D bounding box | 28.08 | 20.33 | 16.96 |

Our independent BEV/3D evaluator produced consistent moderate values of
27.35/20.35. These are official-style **Car** diagnostics, not the product
Vehicle/Pedestrian denominator.

### Frozen two-class R0 teacher

- Architecture: original ResNet50 MonoDETR.
- Selected checkpoint: epoch 185 of 195.
- Evaluated checkpoints: 39.
- Validation images per checkpoint: 3,769.
- Vehicle moderate 3D AP_R40: **17.6348**.
- Pedestrian moderate 3D AP_R40: **5.7214**.
- Vehicle moderate BEV AP_R40: **23.6816**.
- Pedestrian moderate BEV AP_R40: **6.5961**.

R0 is trained, evaluated, frozen, and available for later knowledge
distillation. It is an accuracy reference, not a deployment model.

## How to interpret AP_R40

AP_R40 is not the percentage of frames reconstructed correctly. Predictions
are confidence-ranked; KITTI computes precision and recall while requiring a
minimum 3D or BEV intersection-over-union, samples the curve at 40 recall
positions, and reports its average area. Vehicle uses a strict 0.70 3D/BEV IoU
threshold and Pedestrian uses 0.50 in the product protocol.

The accuracy-first student gates are 90% of the frozen R0 moderate metrics:

- Vehicle moderate 3D AP_R40 >= **15.8713**.
- Pedestrian moderate 3D AP_R40 >= **5.1493**.
- Balanced moderate 3D mean >= **10.5103**.
- Vehicle moderate BEV AP_R40 >= **21.3134**.
- Pedestrian moderate BEV AP_R40 >= **5.9365**.
- Complete evaluation of all **3,769** validation images.

The exact 2D product AP of R0 epoch 185 is not in the frozen sweep summary.
The often-quoted 88.10 moderate 2D AP belongs to the reproduced published Car
checkpoint. Measuring epoch-185 product 2D AP requires an evaluator extension,
but no retraining.

## Accuracy-first student architecture

The active A1 candidate preserves MonoDETR and replaces only its backbone:

```text
RGB image [1, 3, 384, 1280]
  -> MobileNetV4 Conv Small backbone
  -> MonoDETR multi-scale projections and LID depth predictor
  -> three-layer depth-aware deformable encoder
  -> three-layer, 50-query decoder
  -> two-class MonoDETR prediction heads
```

Resolution, product taxonomy, transformer, query count, heads, decoding, and
evaluation remain aligned with frozen R0. Compatible R0 epoch-185 tensors are
transferred; only the new backbone/projection interface requires its defined
pretrained/new initialization. See `ACCURACY_FIRST_STUDENT_CONTRACT.md`.

## Core ML and physical-iPhone evidence

Core ML can represent many transformer operations, but support does not imply
efficient execution. MonoDETR's deformable attention, large intermediate
tensors, and transformer decoder caused unacceptable full-model latency on the
target iPhone. The full transformer remains the teacher; the convolutional S1
is Plan A for deployment.

The original random-weight S1 graph measured:

- 1.403 million parameters.
- 2.155 GMAC per 1280x384 image.
- 2.73 MB FP16 Core ML package.
- iPhone 16 Pro Max model-only p50: 1.878 ms.
- iPhone 16 Pro Max model-only p95: 3.788 ms.
- Five warmups followed by 100 timed predictions.

GMAC means billions of multiply-accumulate operations per prediction. It is a
hardware-independent estimate of compute, not a latency measurement; the S1
architecture ceiling is 15 GMAC, while physical-device milliseconds remain the
deployment decision metric.

Those measurements preceded restoration of the distinct 2D `center_offset`
head. The corrected graph uses the same supported operator types, but final
size, parity, and physical-device latency must be refreshed before deployment.

The iPhone application also passed KITTI Golden Parity, custom-image and live
camera paths, frame dropping, the 30-second no-saving pipeline, full recording,
required artifacts, clean ZIP layout, and iOS share-sheet export. Optimized
preprocessing removed the original 540-570 ms bottleneck and measured around
1 ms in the validated device run.

## Current teacher and student structure

- **Teacher R0:** frozen ResNet-50 MonoDETR, 80-bin depth predictor, three-layer
  depth-aware deformable encoder, three-layer query decoder, epoch 185.
- **Retired student S1:** MobileNetV4 + Lite-FPN + dense stride-8 heads. It is
  exceptionally fast but failed Vehicle/Pedestrian 3D AP in every supervised
  variant. The optional checkpoint sweep was waived by the decision to move on.
- **Retired students H1/H2:** fast MobileNetV4 hybrid query graphs; frozen after
  supervised capacity/assignment/localization gates failed.
- **Frozen student A1:** MobileNetV4 Conv Small backbone inside the otherwise
  preserved two-class MonoDETR architecture; epoch 140 is healthy but not
  accuracy-qualified.
- **Prepared student A2:** replaces only A1's backbone/projections with
  `mobilenetv4_conv_medium.e500_r256_in1k`; no trained A2 weights exist yet.
- **Qualified student weights:** none yet.
- **Distillation:** completed and rejected for A1; GT-only epoch 140 remains the
  stronger A1 baseline and the distilled checkpoint is not a product candidate.

The active architecture and experiment gates are frozen in
`ACCURACY_FIRST_STUDENT_CONTRACT.md`; the older S1/H1/H2 contracts remain as
historical, reproducible evidence.

## Next step

Run `MonoDETR_A2_MobileNetV4_Medium_Two_Class_GT_Colab.ipynb` top-to-bottom
on a Colab GPU. A2 uses the verified `timm==1.0.20` identifier
`mobilenetv4_conv_medium.e500_r256_in1k`. It preserves the transformer, query
count, heads, input resolution, R0 initialization policy, Chen split,
augmentation, optimizer/schedule, evaluator, exact resume, and all five gates.
Do not add teacher losses or begin compression. Compare its product sweep
directly with A1 epoch 140 and complete nearby-recall review only if all five
accuracy gates pass.
