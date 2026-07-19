# MobileADAS3D iPhone Labeling and Benchmarking Action Plan

_Historical implementation plan, updated 2026-07-19._ The iPhone v7 pipeline,
recording artifacts, no-saving benchmark, model-only benchmark, ZIP export, and
share sheet are now implemented and device-tested. This document remains the
product and schema rationale. New model training uses MobileNetV4 Conv Small;
it does not retroactively change the deployed v7/MobileNetV3 recording model.

## 1. Goal

Build an iPhone-based prototype that can:

1. Capture a short driving scene, for example 10-20 seconds.
2. Run MobileADAS3D on-device using Core ML.
3. Detect cars, pedestrians, and cyclists.
4. Estimate 2D box, depth, 3D dimensions, yaw/orientation, center offset, and uncertainty.
5. Optionally project estimated 3D cuboids into the image.
6. Export a frame-by-frame label file with camera intrinsics, camera/device pose metadata, detections, and benchmark timing.
7. Benchmark the full product pipeline, not only model forward latency.

This should be described as:

> On-device monocular 3D object pseudo-label generation for ADAS-style scene understanding.

It should not be described as:

> Ground-truth 3D reconstruction.

The model estimates 3D object attributes from a single camera image. Labels should be considered pseudo-labels unless externally validated.

---

## 2. Important terminology

### 2.1 3D detection

In this project, "3D" means the model predicts object-level 3D attributes:

- depth
- 3D dimensions: height, width, length
- yaw / orientation
- approximate 3D center in camera coordinates
- optionally projected 3D cuboid corners

It does not mean full dense 3D reconstruction, lidar-like point clouds, or mesh generation.

### 2.2 3D cuboid

A 3D cuboid can be generated from:

- predicted object depth
- predicted object dimensions
- predicted yaw
- predicted/projected object center
- camera intrinsics
- coordinate-system assumptions

The cuboid is estimated, not ground truth.

### 2.3 Intrinsics

Camera intrinsics describe the camera's internal projection model:

- fx
- fy
- cx
- cy
- image/reference dimensions

Use intrinsics to project between camera-coordinate 3D points and image pixels.

### 2.4 Extrinsics

Camera extrinsics describe the camera pose relative to another coordinate frame.

For this project, there are several possible extrinsic frames:

1. Camera-to-device frame
2. Device-to-vehicle frame
3. Camera-to-vehicle frame
4. Camera-to-ARKit-world frame
5. Vehicle-to-world / GPS frame

The most useful MVP target is:

> camera-to-vehicle approximate extrinsics for a fixed dashboard mount.

These should be stored as metadata and treated as approximate unless calibrated carefully.

---

## 3. Deployed validated reference

### Model

- Model: MobileADAS3D v7 deployment lineage
- Backbone: MobileNetV3-Small
- Neck: stride-16 FPN
- Input: [1, 3, 384, 1280]
- Core ML artifact: MobileADAS3D_v7_cuboid_fp16.mlpackage
- Core ML package size: about 12.18 MB
- Output tensors:
  - cls_logits: [1, 3, 24, 80]
  - box2d: [1, 4, 24, 80]
  - log_depth: [1, 1, 24, 80]
  - dim: [1, 3, 24, 80]
  - yaw: [1, 2, 24, 80]
  - center_offset: [1, 2, 24, 80]
  - depth_uncertainty: [1, 1, 24, 80]

### Accuracy baseline

- IoU >= 0.50 F1: about 0.79
- Depth MAE on matched detections: about 1.88 m
- Yaw MAE on matched detections: about 10.63 degrees
- Dimension MAE: about 0.143 m

### PyTorch/TorchScript benchmark

- TorchScript CUDA end-to-end latency: about 9.94 ms mean
- TorchScript CUDA end-to-end FPS: about 100.6
- Vectorized decoder is enabled
- Deployment topK: 50
- Score threshold: 0.55
- NMS IoU threshold: 0.5

### iPhone Core ML forward-only benchmark

Device:

- iPhone model identifier: iPhone17,2
- Model: MobileADAS3D_fp16.mlpackage
- Input: MLMultiArray Float32 [1, 3, 384, 1280]
- Benchmark type: forward-only Core ML model.prediction
- Random input generated once and reused

Best compute mode:

- computeUnits: .cpuAndNeuralEngine
- mean latency: 1.934 ms
- p50 latency: 1.827 ms
- p95 latency: 2.546 ms
- p99 latency: 2.798 ms
- FPS mean: 517.04
- thermal state: nominal before and after benchmark

This is a strong forward-only deployment signal, but it does not include camera preprocessing, decode/NMS, cuboid projection, label writing, or live capture overhead.

---

## 4. Product pipeline target

Final prototype pipeline:

```text
AVCapture camera frame
    ↓
timestamp + frame index
    ↓
camera intrinsics
    ↓
device/camera pose metadata
    ↓
resize/crop/normalize to [1, 3, 384, 1280]
    ↓
Core ML MobileADAS3D prediction
    ↓
Swift decode/NMS
    ↓
optional 3D cuboid projection
    ↓
label JSONL export
    ↓
benchmark report
```

---

## 5. Coordinate systems to track

### 5.1 Original image coordinate system

Coordinates in the captured frame resolution.

Example:

```json
"image_size": {
  "width": 1920,
  "height": 1080
}
```

### 5.2 Model input coordinate system

Coordinates in the resized model input:

```json
"model_input_size": {
  "width": 1280,
  "height": 384
}
```

Need to store resize/crop metadata so boxes can be mapped back to original image coordinates.

### 5.3 Camera coordinate system

Use for predicted 3D object center:

```json
"center_3d_camera_m": {
  "x": 1.23,
  "y": 0.42,
  "z": 18.6
}
```

### 5.4 Vehicle coordinate system

Approximate and optional in MVP.

Requires camera-to-vehicle extrinsics:

```json
"camera_to_vehicle": {
  "translation_m": [x, y, z],
  "rotation_rpy_deg": [roll, pitch, yaw]
}
```

### 5.5 ARKit/world coordinate system

Optional. Useful if using ARKit for pose tracking.

Do not confuse ARKit local world coordinates with true vehicle/world ground truth.

---

## 6. Intrinsics capture plan

### 6.1 AVFoundation path

Use AVFoundation camera capture.

Enable camera intrinsic delivery when supported:

```swift
if connection.isCameraIntrinsicMatrixDeliverySupported {
    connection.isCameraIntrinsicMatrixDeliveryEnabled = true
}
```

Store per-frame intrinsics if available:

```json
"camera_intrinsics": {
  "available": true,
  "fx": 1234.5,
  "fy": 1230.2,
  "cx": 960.0,
  "cy": 540.0,
  "matrix_3x3": [
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
  ],
  "reference_dimensions": [width, height]
}
```

### 6.2 Fallback path

If intrinsics are unavailable:

1. Store `available=false`.
2. Store approximate intrinsics from device/camera metadata if available.
3. Mark 3D cuboid projection as approximate.
4. Do not claim calibrated 3D labels.

---

## 7. Extrinsics capture plan

### 7.1 What extrinsics are needed?

For dashboard-mounted driving, the important extrinsic is:

```text
camera frame → vehicle frame
```

This tells us where the phone camera is mounted and how it is rotated relative to the car.

Required approximate values:

- camera height above road
- lateral offset from vehicle center
- longitudinal offset from vehicle front/reference
- pitch angle
- roll angle
- yaw angle relative to vehicle forward direction

### 7.2 MVP manual mount calibration

For the first implementation, use manual calibration fields:

```json
"mount_calibration": {
  "method": "manual",
  "camera_height_m": 1.25,
  "camera_lateral_offset_m": 0.0,
  "camera_longitudinal_offset_m": 0.0,
  "pitch_deg": -5.0,
  "roll_deg": 0.0,
  "yaw_deg": 0.0,
  "notes": "Phone mounted near windshield centerline."
}
```

This is simple and transparent.

### 7.3 Core Motion attitude metadata

Use Core Motion to record device attitude:

- roll
- pitch
- yaw
- quaternion
- rotation matrix
- gravity vector
- timestamp

Store it per frame or at a lower rate with interpolation.

Example:

```json
"device_motion": {
  "available": true,
  "timestamp_s": 12.345,
  "roll_rad": 0.01,
  "pitch_rad": -0.08,
  "yaw_rad": 1.57,
  "quaternion": [x, y, z, w],
  "gravity": [gx, gy, gz],
  "user_acceleration": [ax, ay, az]
}
```

Caution:

- Core Motion attitude is not the same as calibrated camera-to-vehicle extrinsic.
- Yaw may drift depending on reference frame.
- Use it as metadata first, not as ground truth.

### 7.4 ARKit pose metadata

Optional path:

- Run ARKit session.
- Capture ARCamera transform per frame.
- Store camera pose in ARKit local world coordinates.

Example:

```json
"arkit_pose": {
  "available": true,
  "camera_transform_4x4": [[...], [...], [...], [...]]
}
```

Caution:

- ARKit local world is not necessarily vehicle frame.
- Outdoor driving ARKit tracking may be unstable depending on motion, lighting, and scene.
- Use as optional metadata, not primary ground truth.

### 7.5 Better calibration later

Later improvements:

1. Horizon-line calibration for pitch/roll.
2. Lane/road-plane calibration.
3. Calibration board while parked.
4. Use known camera height and road plane.
5. Fuse IMU + visual odometry + GPS.
6. Add manual UI for pitch/yaw adjustment.

---

## 8. 3D cuboid projection plan

### 8.1 Inputs

For each detection:

- class
- score
- bbox_2d
- center_2d
- depth
- dimensions h/w/l
- yaw
- camera intrinsics K

### 8.2 Back-project 2D center to camera coordinates

Given:

```text
u = center x pixel
v = center y pixel
Z = predicted depth
K = [[fx, 0, cx],
     [0, fy, cy],
     [0, 0, 1]]
```

Compute:

```text
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy
Z = Z
```

### 8.3 Build cuboid

Use object dimensions:

- height h
- width w
- length l

Create 8 corners in local object coordinates, rotate by yaw, translate to center_3d_camera_m, then project each corner to image pixels.

### 8.4 Store cuboid output

```json
"cuboid_3d_camera_m": [
  [x1, y1, z1],
  ...
],
"cuboid_2d_original_image": [
  [u1, v1],
  ...
]
```

---

## 9. Label file schema

Use JSONL: one JSON object per frame.

Recommended file name:

```text
mobileadas3d_labels_<session_id>.jsonl
```

### 9.1 Session metadata file

Save once:

```json
{
  "session_id": "2026-06-24_iphone17_2_drive_001",
  "model": {
    "name": "MobileADAS3D",
    "version": "v7",
    "artifact": "MobileADAS3D_fp16.mlpackage",
    "precision": "fp16",
    "input_shape": [1, 3, 384, 1280],
    "score_threshold": 0.55,
    "topk": 50,
    "nms_iou_threshold": 0.5
  },
  "device": {
    "model_identifier": "iPhone17,2",
    "ios_version": "26.5"
  },
  "capture": {
    "mode": "live_camera",
    "duration_s": 20,
    "target_fps": 30,
    "camera_position": "back_wide"
  },
  "mount_calibration": {
    "method": "manual",
    "camera_height_m": null,
    "pitch_deg": null,
    "roll_deg": null,
    "yaw_deg": null
  }
}
```

### 9.2 Per-frame JSONL record

```json
{
  "session_id": "2026-06-24_iphone17_2_drive_001",
  "frame_index": 42,
  "timestamp_s": 1.400,
  "image_size": {
    "width": 1920,
    "height": 1080
  },
  "model_input_size": {
    "width": 1280,
    "height": 384
  },
  "preprocess": {
    "resize_mode": "resize_stretch",
    "scale_x": 0.6667,
    "scale_y": 0.3556,
    "crop": null
  },
  "camera_intrinsics": {
    "available": true,
    "fx": 1234.5,
    "fy": 1230.2,
    "cx": 960.0,
    "cy": 540.0,
    "reference_dimensions": [1920, 1080],
    "matrix_3x3": [[1234.5, 0.0, 960.0], [0.0, 1230.2, 540.0], [0.0, 0.0, 1.0]]
  },
  "extrinsics": {
    "mount_calibration_available": true,
    "coordinate_system": "camera",
    "camera_to_vehicle": {
      "translation_m": [0.0, 0.0, 1.25],
      "rotation_rpy_deg": [0.0, -5.0, 0.0]
    }
  },
  "device_motion": {
    "available": true,
    "roll_rad": 0.01,
    "pitch_rad": -0.08,
    "yaw_rad": 1.57,
    "gravity": [0.0, -0.1, -0.99]
  },
  "timing_ms": {
    "preprocess": 3.2,
    "coreml_forward": 2.0,
    "decode_nms": 4.1,
    "cuboid_projection": 0.5,
    "label_write": 0.3,
    "total": 10.1
  },
  "detections": [
    {
      "class_id": 0,
      "class_name": "Car",
      "score": 0.91,
      "bbox_2d_model_input": [475.0, 120.0, 600.0, 240.0],
      "bbox_2d_original_image": [712.5, 337.5, 900.0, 675.0],
      "center_2d_original_image": [806.0, 510.0],
      "depth_m": 18.6,
      "dimensions_hwl_m": [1.50, 1.72, 4.05],
      "yaw_rad": -0.12,
      "center_3d_camera_m": [1.2, 0.4, 18.6],
      "cuboid_2d_original_image": [[0, 0]],
      "cuboid_3d_camera_m": [[0, 0, 0]],
      "depth_uncertainty": 0.21
    }
  ]
}
```

---

## 10. Benchmarking plan

### 10.1 Benchmark levels

#### Level 0: Forward-only random input

Status: complete.

Measures only:

```text
MLModel.prediction
```

Use this as hardware/model baseline.

#### Level 1: Static image full pipeline

Input:

- one real road image bundled with app

Measure:

- image load
- resize/preprocess
- MLMultiArray conversion
- Core ML forward
- decode/NMS
- cuboid projection
- label JSON generation

Output:

- one JSON label file
- optional overlay image

#### Level 2: Recorded video replay

Input:

- 10-20 second video file

Measure:

- frame extraction
- preprocessing
- Core ML forward
- decode/NMS
- cuboid projection
- JSONL writing
- FPS
- dropped/skipped frames

This is the best reproducible product benchmark.

#### Level 3: Live camera, no writing

Input:

- AVCaptureVideoDataOutput

Measure:

- live frame callback interval
- preprocessing
- Core ML forward
- decode/NMS
- optional overlay

Purpose:

- isolate live camera performance without file I/O.

#### Level 4: Live camera with label export

Input:

- dashboard-mounted live camera

Measure:

- full product pipeline
- label export
- dropped frames
- thermal state
- memory growth
- battery change

This is the actual product benchmark.

---

## 11. Benchmark metrics to collect

Per frame:

```json
"timing_ms": {
  "camera_callback_gap": 33.3,
  "preprocess": 3.2,
  "coreml_forward": 2.0,
  "decode_nms": 4.1,
  "cuboid_projection": 0.5,
  "label_serialization": 0.2,
  "label_write": 0.3,
  "total": 10.1
}
```

Per session:

```json
"benchmark_summary": {
  "duration_s": 20.0,
  "frames_captured": 600,
  "frames_processed": 600,
  "frames_skipped": 0,
  "effective_fps": 30.0,
  "latency_mean_ms": 10.1,
  "latency_p50_ms": 9.8,
  "latency_p90_ms": 12.4,
  "latency_p95_ms": 14.1,
  "latency_p99_ms": 18.0,
  "thermal_start": "nominal",
  "thermal_end": "nominal",
  "battery_start_percent": 75,
  "battery_end_percent": 74
}
```

---

## 12. Repo structure

Recommended additions:

```text
docs/
  mobileadas3d_iphone_labeling_action_plan.md
  label_schema.md
  calibration_notes.md
  benchmark_protocol.md

ios/
  MobileADAS3DBenchmarkApp/
    MobileADAS3DBenchmarkApp.xcodeproj
    MobileADAS3DBenchmarkApp/
      ContentView.swift
      MobileADAS3DModelRunner.swift
      MobileADAS3DDecoder.swift
      MobileADAS3DLabelWriter.swift
      CameraCaptureManager.swift
      CameraCalibrationProvider.swift
      DeviceMotionProvider.swift
      BenchmarkRecorder.swift
      CuboidProjector.swift
      LabelSchema.swift
      Assets/
        MobileADAS3D_fp16.mlpackage
```

---

## 13. Implementation milestones

### Milestone 1: Repo docs

- [ ] Add this action plan to `docs/mobileadas3d_iphone_labeling_action_plan.md`
- [ ] Add label schema to `docs/label_schema.md`
- [ ] Add benchmark protocol to `docs/benchmark_protocol.md`
- [ ] Update README with iPhone deployment status

### Milestone 2: Static image pipeline

- [ ] Add static test road image
- [ ] Implement image resize/preprocess to MLMultiArray
- [ ] Run Core ML inference
- [ ] Implement Swift decode/NMS
- [ ] Write one-frame JSON label
- [ ] Benchmark static-image full pipeline

Acceptance criteria:

- outputs valid detections
- JSON label file is written
- timing contains preprocess/forward/decode/write
- no crash
- average total static-image latency reported

### Milestone 3: Intrinsics capture

- [ ] Enable camera intrinsic matrix delivery when supported
- [ ] Extract/store intrinsic matrix
- [ ] Store intrinsic reference dimensions
- [ ] Add fallback if unavailable
- [ ] Validate intrinsics are attached to frames

Acceptance criteria:

- per-frame label includes intrinsics or `available=false`
- no silent missing calibration fields

### Milestone 4: Extrinsics metadata

- [ ] Add manual mount calibration UI/config
- [ ] Store camera height, pitch, roll, yaw, offsets
- [ ] Add Core Motion logging
- [ ] Store device attitude metadata
- [ ] Clearly mark coordinate system and calibration method

Acceptance criteria:

- session metadata includes mount calibration
- per-frame labels include device motion when available
- labels distinguish camera coordinates vs vehicle coordinates

### Milestone 5: Cuboid projection

- [ ] Back-project center_2d + depth to camera 3D
- [ ] Build 3D cuboid corners from h/w/l/yaw
- [ ] Project cuboid corners to image using intrinsics
- [ ] Store cuboid_2d and cuboid_3d in labels
- [ ] Optional overlay for visual sanity check

Acceptance criteria:

- JSON label includes cuboid fields
- overlay roughly aligns with visible objects for sample frames

### Milestone 6: Recorded video replay benchmark

- [ ] Add local video replay mode
- [ ] Process 10-20 second video
- [ ] Export JSONL labels
- [ ] Export benchmark summary
- [ ] Report dropped/skipped frames

Acceptance criteria:

- processes complete video
- label count matches processed frames
- benchmark summary created

### Milestone 7: Live dashboard logging

- [ ] Implement AVCapture live capture
- [ ] Process frames at target FPS or every Nth frame
- [ ] Export JSONL labels
- [ ] Record benchmark summary
- [ ] Verify thermal/memory/battery observations

Acceptance criteria:

- records 10-20 seconds
- app remains responsive
- labels are saved
- benchmark summary is saved
- safe/passive logging workflow documented

---

## 14. Safety and scope constraints

This app is for research/prototyping only.

Rules:

- Do not interact with the app while driving.
- Mount phone before driving.
- Start/stop only while parked.
- Prefer a passenger/operator.
- Do not use detections for real-time driving decisions.
- Do not block windshield or driver view.
- Follow local laws and company safety policies.
- Treat exported detections as pseudo-labels, not ground truth.

---

## 15. Presentation wording

Use:

> MobileADAS3D is an on-device monocular 3D object detection and pseudo-labeling prototype for ADAS-style scene understanding.

Avoid:

> The system generates ground-truth 3D labels.

Better:

> The system estimates object-level 3D labels from a single iPhone camera stream and stores them with camera calibration and benchmark metadata.

---

## 16. Current next work item

The static-image, recorded-video, live-camera, artifact-export, and benchmark
milestones below were the original implementation sequence and are complete.
The next project action is now:

```text
Train fresh MobileNetV4 Conv Small baseline in Google Colab
```

Run `notebooks/MobileADAS3D_MobileNetV4_Colab_Baseline.ipynb` from top to
bottom. Keep the iPhone v7 deployment unchanged until the new model completes
full Chen-split AP_R40 evaluation, Core ML parity, and device benchmarking.
