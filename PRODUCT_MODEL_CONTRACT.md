# MobileADAS3D product model contract

Status: architecture candidate locked; deployment approval pending Core ML gate  
Decision date: 2026-08-11

## 1. Product objective and priority order

MobileADAS3D is an on-device monocular safety-perception system. It consumes one
RGB camera frame and reports safety-relevant objects, their 2D boxes, distance,
and 3D geometry without cloud processing.

The priorities are ordered and must not be traded silently:

1. **Nearby-object detection.** Maximize recall for vehicles and pedestrians,
   especially in the near field. A missed nearby object is the primary failure.
2. **Distance and position.** Estimate longitudinal range and lateral position;
   minimize dangerous overestimation of distance.
3. **Edge reliability and latency.** Run continuously on the iPhone without
   blocking the camera queue, freezing, overheating prematurely, or exceeding
   the latency budget.
4. **2D and 3D localization.** Produce stable 2D boxes, camera-space centers,
   BEV boxes, and 3D boxes.
5. **Dimensions and orientation.** Estimate physical dimensions and yaw. These
   are retained, but cannot take priority over detection, distance, or runtime.

This is a perception and warning component, not a safety-certified autonomous
driving controller.

## 2. Production class contract

The first production model has exactly two foreground classes:

1. `Vehicle`
2. `Pedestrian`

### Source-label mapping

| Production class | Training labels | Notes |
|---|---|---|
| `Vehicle` | KITTI `Car`, `Van`, `Truck`, `Tram` | Preserve source subtype in metadata. Exclude `Misc`. |
| `Pedestrian` | KITTI `Pedestrian`, `Person_sitting` | Preserve pose/source label for analysis. |

Motorcycle, cyclist, trailer, stroller, wheelchair, animal, cone, and unknown
obstacle are not silently relabeled. They remain ignored/diagnostic categories
until representative training data and explicit acceptance criteria exist.

Official KITTI reporting must still follow the official `Car` and `Pedestrian`
definitions. The merged production taxonomy is an additional product metric,
not a replacement for standard KITTI AP.

## 3. Dataset roles and isolation

| Role | Dataset | Permitted use |
|---|---|---|
| Training | KITTI Chen train, 3,712 images | Optimization and augmentation only. |
| Model selection | KITTI Chen val, 3,769 images | Checkpoint, threshold, and ablation selection. It is validation, not an unseen test set. |
| Pipeline smoke test | nuScenes mini | Adapter and geometry debugging only; never report as final generalization. |
| Locked external test | nuScenes official validation, `CAM_FRONT` | One zero-shot evaluation after checkpoint, preprocessing, taxonomy, threshold, NMS, and TopK are frozen. LiDAR-derived boxes are ground truth; RGB remains the only model input. |
| Second external confirmation | Locked Waymo front-camera subset | Future zero-shot confirmation with camera/LiDAR-associated 2D and 3D labels. |
| Edge runtime | iPhone street recordings | Latency, stability, thermal, artifact, and qualitative failure analysis. Not 3D ground truth unless independently annotated. |

If nuScenes results are used to change the model or thresholds, nuScenes is no
longer an unseen test. Record it as external development data and reserve Waymo
for the final zero-shot claim.

Split all video datasets by scene/session, never by adjacent frames. Record the
dataset release, split manifest, class mapping, calibration policy, and content
hash with every report.

### nuScenes zero-shot visibility policy

Use `CAM_FRONT` only. Transform each LiDAR-derived 3D annotation through global,
ego, and calibrated-camera coordinates, then through the exact 10:3 crop and
1280x384 resize. Report:

- all camera-visible ground truth;
- crop-visible ground truth;
- LiDAR-supported subsets with `num_lidar_pts > 0` and `>= 5`.

Only crop-visible objects are eligible for the primary model-input score. The
all-visible score quantifies objects removed by the fixed KITTI crop.

## 4. Locked architecture candidate

The production candidate is **MobileMonoDETR-VP1**:

```text
RGB Float32 /255 input [1, 3, 384, 1280]
  -> MobileNetV4 Conv Small ImageNet backbone
  -> features at strides 8, 16, 32 with channels 64, 96, 960
  -> learned 1x1 projections to 256 channels
  -> MonoDETR depth predictor, LID 80 bins, 0.001-60 m
  -> depth-aware deformable transformer, 3 encoder + 3 decoder layers
  -> 50 object queries
  -> two-class classification plus no-object
  -> 2D box, projected 3D center, depth, dimensions, yaw, 3D location
```

The backbone replacement is the only approved structural change from the
validated MonoDETR teacher. Do not change the depth predictor, transformer,
query count, heads, or loss definitions in the same experiment.

The Car-only backbone ablation reached its best KITTI Car 3D moderate AP_R40 at
epoch 40:

```text
MobileNetV4-MonoDETR epoch 40: 14.065
ResNet50 MonoDETR teacher:      20.328
retained performance:          69.2%
```

This validates the research candidate, not the two-class product model. A new
Vehicle + Pedestrian training run is required because the current checkpoint
does not establish pedestrian performance.

### Deployment condition

MonoDETR uses multi-scale deformable attention implemented by a custom CUDA
extension. MobileMonoDETR-VP1 is not deployment-final until the exact graph:

1. exports without unsupported or host-fallback operations;
2. passes PyTorch-to-Core-ML numerical and decoded-detection parity;
3. meets the physical-iPhone latency, memory, and stability gates below.

If that gate fails, retain MonoDETR as the accuracy teacher and build a
Core-ML-native student. Do not weaken the runtime gate or describe a Python/CUDA
model as the deployed architecture.

## 5. Metrics and acceptance gates

All metrics are reported per class, per distance bucket, and in aggregate.
Always include sample counts and confidence intervals where practical.

### A. KITTI model-quality gate

Report official 2D, BEV, and 3D AP_R40 for easy/moderate/hard. The primary
checkpoint-selection metric is moderate 3D AP_R40, not validation loss.

Initial product gate:

- each production class retains at least 75% of its same-protocol ResNet50
  MonoDETR teacher moderate 3D AP_R40;
- Vehicle moderate BEV AP_R40 >= 20;
- no checkpoint is selected from KITTI loss alone;
- all 3,769 Chen-val prediction files must exist.

The current Car-only epoch-40 model retains 69.2%, so it has not passed the
75% production-accuracy gate. Before applying the per-class retention rule, run
one frozen-protocol ResNet50 Vehicle + Pedestrian reference training/evaluation;
the validated published checkpoint is Car-only and cannot supply a legitimate
Pedestrian denominator.

### B. External zero-shot gate

On the frozen nuScenes crop-visible, LiDAR-supported subset:

- Vehicle 2D recall at IoU 0.5 within 40 m >= 85%;
- Pedestrian 2D recall at IoU 0.5 within 30 m >= 80%;
- 0-20 m longitudinal range MAE <= 2.0 m;
- dangerous range overestimation greater than 3 m occurs in <= 5% of matched
  0-40 m objects;
- report 2D AP50, center-distance error, lateral error, BEV IoU, 3D IoU,
  dimension MAE, yaw error, false positives/frame, and recall by distance.

These are product gates, not nuScenes leaderboard claims. Full nuScenes NDS is
supplemental because the model does not predict all nuScenes attributes and
velocity fields.

### C. Core ML parity gate

- decoded class/box/depth comparisons use identical preprocessing and NMS;
- KITTI moderate 3D AP_R40 degradation from PyTorch to Core ML <= 1% relative;
- matched-object mean absolute depth delta <= 0.10 m;
- no unsupported CPU-host callback or custom CUDA dependency remains.

### D. Physical-iPhone end-to-end gate

- preprocessing p95 <= 20 ms;
- Core ML inference p95 <= 50 ms;
- capture-to-decoded-result p95 <= 100 ms;
- sustained processed rate >= 10 FPS;
- camera queue remains nonblocking and frame dropping remains enabled;
- zero freezes/crashes in a 30-minute pipeline-no-saving run;
- zero freezes/crashes in a 30-minute full-artifact run;
- report p50/p90/p95/p99/max latency, capture and processed FPS, dropped-frame
  rate, peak memory, model-load time, thermal state, and battery use;
- every full run exports the complete artifact contract and clean session ZIP.

## 6. Training and decision sequence

1. **Completed 2026-08-11:** repair resumable-checkpoint ordering so finalized
   best-result metadata is written after validation while retaining a
   provisional pre-validation crash-recovery save.
2. Prove MobileMonoDETR-VP1 Core ML graph feasibility before another long run.
3. Implement the two-class KITTI mapping with unit-tested counts and manifests.
4. Establish the same-protocol ResNet50 Vehicle + Pedestrian reference.
5. Train a fresh Vehicle + Pedestrian mobile model. Evaluate at epochs 20, 40, 50,
   75, and 100; continue farther only while moderate 3D AP improves.
6. Select the checkpoint using per-class moderate 3D AP plus the nearby recall
   gate, not a single aggregate loss.
7. Freeze the checkpoint and all inference parameters.
8. Validate the nuScenes adapter on mini, then run the locked zero-shot test.
9. Export to Core ML, verify parity, and run physical-iPhone benchmarks.
10. Approve deployment only if model quality, external generalization, parity,
   and edge-runtime gates all pass.

Any architecture, taxonomy, dataset-role, or threshold change requires a new
version of this contract and a new evaluation manifest.
