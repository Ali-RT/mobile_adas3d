# MobileADAS3D model milestones and architecture

Last updated: 2026-08-19

This document is the concise reporting reference for model selection,
teacher/student status, metrics, edge constraints, completed evidence, and the
next experiment. The canonical execution status remains in `PROJECT_TRACKER.md`.

## Product targets

1. Detect and reconstruct road objects from one RGB camera image, including 2D
   box, 3D position, depth, dimensions, orientation, and confidence.
2. Deploy the selected model through Core ML for stable real-time inference on
   iPhone without blocking the camera pipeline.
3. Retain a useful fraction of the high-accuracy teacher and validate the frozen
   model outside KITTI before deployment approval.

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
| MonoDETR MobileNetV4 backbone ablation | Reduced backbone cost but lost too much accuracy | Do not use as the high-accuracy reference |
| Full transformer Core ML gate | Graph conversion was possible after compatibility work, but full-model iPhone latency was unacceptable | Do not deploy full MonoDETR on iPhone |
| MobileADAS3D-S1 graph gate | Core-ML-native MobileNetV4 student was very small and fast on physical iPhone | Keep the convolutional student direction |
| S1 GT-only 20-epoch gate | Training was stable, but Vehicle/Pedestrian moderate 3D AP_R40 was only 0.024/0.519 | Reject continuation to 100 epochs |
| S1 checkpoint and geometry diagnosis | Epoch 20 was best; 12,105 Vehicle detections matched at mean 2D IoU 0.675, but yaw/shape/placement were poor | 2D detection is not the first failure |
| S1 yaw diagnosis | Vehicle axis error was good at 9.63 degrees mean, but final yaw was 72.28 degrees with a 35.8% flip-candidate rate | Reject independent axis plus hard direction-bit yaw |

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

The initial student-retention gates are 75% of the frozen R0 moderate 3D AP:

- Vehicle moderate 3D AP_R40 >= **13.2261**.
- Pedestrian moderate 3D AP_R40 >= **4.2910**.
- Vehicle moderate BEV AP_R40 >= **20.0**.
- Complete evaluation of all **3,769** validation images.

The exact 2D product AP of R0 epoch 185 is not in the frozen sweep summary.
The often-quoted 88.10 moderate 2D AP belongs to the reproduced published Car
checkpoint. Measuring epoch-185 product 2D AP requires an evaluator extension,
but no retraining.

## Student architecture

The deployment candidate is MobileADAS3D-S1, a fixed-shape convolutional model:

```text
RGB image [1, 3, 384, 1280]
  -> MobileNetV4 Conv Small backbone
  -> stride-8, stride-16, and stride-32 features
  -> lightweight top-down feature pyramid
  -> shared stride-8 prediction feature [1, 96, 48, 160]
  -> independent 1x1 dense prediction heads
```

The corrected S1 learned heads are:

| Head | Channels | Purpose |
| --- | ---: | --- |
| `cls_logits` | 2 | Vehicle and Pedestrian confidence |
| `quality` | 1 | localization-quality ranking |
| `box2d` | 4 | 2D left/top/right/bottom distances |
| `center_offset` | 2 | 2D box-center offset from the owned cell |
| `projected_center_offset` | 2 | projected 3D bottom-center offset |
| `log_depth` | 1 | camera-space depth |
| `depth_uncertainty` | 1 | depth uncertainty |
| `dim` | 3 | 3D dimension residuals |
| `yaw_axis` | 2 | 180-degree-invariant orientation axis |
| `yaw_direction` | 1 | front/back direction decision |
| `loc_xy` | 2 | auxiliary X/Z and Y/Z location ratios |

The first S1 version implemented yaw as a double-angle axis plus a separate
front/back direction bit. Its axis was learned, but the discontinuous direction
decision caused frequent 180-degree flips. S1-V2 now replaces only this yaw
representation with continuous `[sin(yaw), cos(yaw)]` regression and decoder
normalization; its dedicated
config and resumable 20-epoch Colab workflow are ready to run.

Dynamic TopK, calibration back-projection, non-maximum suppression, tracking,
and artifact writing remain outside the neural network. The graph intentionally
avoids deformable attention, custom Core ML operators, recurrent state, and
dynamic tensor shapes.

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

## Current status and next step

- **Teacher:** available, validated, frozen, and ready for later distillation.
- **Student architecture:** available, Core-ML-native, and fast.
- **Qualified student weights:** not yet available; S1-V1 was rejected.
- **Distillation:** intentionally pending until a healthy GT-only student is
  frozen.

S1-V2's epoch-20 health gate was rejected. Vehicle/Pedestrian moderate 3D
AP_R40 was `0.133/0.256`, versus S1-V1's `0.024/0.519`, and moderate BEV was
`0.801/0.652`. Vehicle yaw improved to `37.77°` mean with a `17.45%` >90° flip
rate, but Pedestrian yaw remained `74.25°`. More importantly, the direct
dot-product cosine objective was not scale-invariant: yaw cosine loss reached
approximately `-1.29` and total train loss became negative. This invalidates
all S1-V2 checkpoints, including the unevaluated continuation through epoch
100.

The next controlled task is **S1-V2b bounded continuous yaw**:

1. **Prepared:** scale-invariant yaw normalization uses a `0.1` norm floor
   inside training loss, bounding its local Jacobian by 10.
2. Keep MobileNetV4, feature pyramid, input, taxonomy, other heads, seed, data,
   optimizer, and 20-epoch gate unchanged.
3. Run `notebooks/MobileADAS3D_S1_V2b_Bounded_Yaw_Colab.ipynb` from a fresh
   seed-42 initialization; do not resume S1-V1 or S1-V2.

S1-V2b subsequently completed its epoch-20 gate with a healthy bounded loss,
but failed accuracy. Vehicle/Pedestrian moderate 3D AP_R40 was `0.103/0.178`
and moderate BEV was `0.581/0.454`, versus required 3D retention
`13.226/4.291` and Vehicle BEV `20.0`. Mean matched yaw error was
`40.20°/72.76°`; Vehicle flip rate was `18.73%`. Training total loss stayed
positive (`0.853` at epoch 20), confirming the prior numerical defect is fixed.

The immediate next step is a no-retraining sweep of saved epochs 5/10/15/20.
If there is no hidden product-AP peak, the dense convolutional S1 family is
closed as a fast but inadequate baseline. Plan B becomes a fixed-shape
MobileNetV4 student with a small depth-aware attention encoder/query decoder,
structurally closer to MonoDETR while remaining far smaller than the R0
ResNet-50 teacher.
4. Run preflight, the 20-epoch GT-only gate, complete product AP evaluation,
   checkpoint sweep, and geometry/yaw diagnostics.
5. Continue toward a full schedule only if Vehicle and Pedestrian improve
   materially without numerical or generalization failure.
6. After freezing a healthy GT-only student, run the paired MonoDETR
   distillation experiment from the identical initialization.

If the convolutional student cannot meet accuracy retention, Plan B is a small
fixed-shape Core-ML-native hybrid attention module without deformable attention.
Plan C is separate optimized 2D detection and monocular depth models with
deterministic geometry and temporal tracking.
