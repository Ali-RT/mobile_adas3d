# MobileADAS3D TODO and Benchmarking Plan

_Original plan: 2026-06-16. Status updated: 2026-07-19._

> **Historical deployment plan.** The v6/v7 MobileNetV3 model has since been
> exported, integrated, and benchmarked on a physical iPhone. The active work
> is fresh MobileNetV4 Conv Small training through
> `notebooks/MobileADAS3D_MobileNetV4_Colab_Baseline.ipynb`. Sections 2–8 are
> retained as the deployment benchmark record; their F1 figures are not results
> for the new model and are not KITTI AP_R40.

## 1. Project purpose

MobileADAS3D is a compact ADAS-oriented monocular 3D detector. Given a single front-camera RGB image, the model predicts:

- object class: `Car`, `Pedestrian`, `Cyclist`
- 2D bounding box
- depth / distance from camera
- 3D dimensions: height, width, length
- yaw / orientation
- optional depth uncertainty

The long-term goal is to keep the model small enough for edge/mobile deployment while preserving useful ADAS-style scene understanding.

---

## 2. Historical v6 model

Historical version:

```text
v6_stride16_fpn_ltrb_center_sampling_class_balance
```

Architecture:

```text
Input: RGB image [B, 3, 384, 1280]

MobileNetV3-Small backbone
  stride-16 feature
  stride-32 feature

Lightweight FPN:
  project stride-16
  project stride-32
  upsample stride-32
  concat + conv fusion

Prediction heads at stride 16:
  cls_head
  box2d_head using local l/t/r/b encoding
  depth_head
  dim_head
  yaw_head
  center_offset_head
  depth_uncertainty_head
```

Key training/target changes already implemented:

- stride-16 FPN output instead of stride-32-only output
- local `l/t/r/b` 2D box encoding instead of absolute `[x1, y1, x2, y2]`
- center-sampling target assignment
- class-balanced loss for minority classes
- scheduled learning-rate decay
- early stopping
- IoU sweep evaluation
- 3D metric evaluation
- yaw diagnostic evaluation

---

## 3. Historical v6 metrics

### 2D detection

Best test threshold:

```text
score_threshold = 0.55
```

Test result at `IoU >= 0.50`:

```text
Precision = 0.8188
Recall    = 0.7686
F1        = 0.7929
mIoU      = 0.7584
TP / FP / FN = 3953 / 875 / 1190
```

Per-class test result at `score_threshold=0.55`, `IoU >= 0.50`:

```text
Car:
  Precision = 0.8723
  Recall    = 0.7970
  F1        = 0.8330
  mIoU      = 0.7678

Pedestrian:
  Precision = 0.6454
  Recall    = 0.6216
  F1        = 0.6333
  mIoU      = 0.7045

Cyclist:
  Precision = 0.5601
  Recall    = 0.7254
  F1        = 0.6321
  mIoU      = 0.7197
```

### 3D metrics on matched test detections

```text
matched true positives: 3953
2D IoU mean:            0.758
depth MAE:              1.876 m
depth relative error:   7.64%
yaw MAE:                10.63 deg
dimension MAE:          0.143 m
```

Depth by distance:

```text
0-20m:
  depth MAE = 0.886 m

20-40m:
  depth MAE = 2.012 m

40-60m:
  depth MAE = 3.676 m

60m+:
  depth MAE = 5.730 m
```

Yaw diagnostic:

```text
ALL:
  standard yaw mean = 10.63 deg
  standard yaw p50  = 3.29 deg
  standard yaw p90  = 21.97 deg
  axis yaw mean     = 7.90 deg
  flip rate         = 1.2%

Car:
  standard yaw mean = 8.56 deg

Cyclist:
  standard yaw mean = 12.39 deg

Pedestrian:
  standard yaw mean = 25.07 deg
```

Conclusion: yaw front/back flips are not the main issue. Pedestrian yaw remains the weakest orientation case.

---

## 4. Open TODO list

### A. Benchmarking and deployment readiness

Historical priority at the time this plan was written.

- [ ] Static model complexity benchmark
  - parameter count
  - trainable parameter count
  - model checkpoint size
  - estimated MACs/FLOPs
  - per-module parameter/MAC breakdown if possible

- [ ] PyTorch inference latency benchmark
  - CPU latency
  - MPS latency on Mac, if available
  - CUDA latency in Colab, if available
  - batch size 1 latency
  - warmup iterations
  - median / p90 / p95 / p99 latency
  - throughput FPS

- [ ] End-to-end inference benchmark
  - image loading
  - resize/preprocessing
  - model forward
  - decode
  - NMS
  - total latency

- [ ] Runtime memory benchmark
  - peak CPU memory
  - peak GPU memory on CUDA
  - model memory footprint
  - activation memory estimate

- [ ] Accuracy/performance tradeoff table
  - baseline stride-32 model
  - v6 stride-16 FPN model
  - future quantized model
  - future Core ML model

- [ ] Export readiness
  - TorchScript trace/export test
  - ONNX export test
  - Core ML conversion test
  - numerical parity check between PyTorch and exported model

- [ ] Edge/mobile readiness
  - Core ML `.mlpackage` generation
  - Xcode model performance report
  - iPhone device benchmark
  - Instruments profiling
  - memory and thermal/power observation

### B. Detector quality improvements

Do only after benchmarking current model.

- [ ] Class-specific threshold sweep
  - Car threshold
  - Pedestrian threshold
  - Cyclist threshold
  - optimize F1 and precision/recall by class

- [ ] False positive / false negative mining
  - save worst FP/FN images
  - group FP/FN by class, distance, size bucket
  - inspect whether FP are duplicate boxes, wrong class, or background hallucinations

- [ ] Small-object analysis
  - compare performance by object height bucket
  - identify if Pedestrian/Cyclist still fail below 32 px height

### C. 3D output improvements

- [ ] Depth uncertainty loss
  - re-enable with safe low weight, for example `0.02` or `0.05`
  - clamp predicted log scale
  - compare depth MAE by distance bucket
  - verify total loss remains stable and non-negative

- [ ] Depth calibration
  - plot predicted vs GT depth
  - residual by distance
  - residual by class
  - evaluate whether model underestimates far objects

- [ ] Yaw improvements if needed
  - keep current yaw head for now
  - optionally downweight Pedestrian yaw
  - optionally add class-specific yaw weights
  - only consider yaw bin + residual if Car/Cyclist yaw blocks downstream use

- [ ] Dimension improvements if needed
  - dimensions are currently good
  - no immediate change recommended

### D. Training and experiment hygiene

- [ ] Keep experiment registry
  - config
  - checkpoint path
  - dataset split
  - metrics
  - benchmark numbers
  - notes

- [ ] Save all generated metrics CSVs per run
  - IoU sweep
  - 3D metrics
  - yaw diagnostics
  - benchmark reports

- [ ] Add reproducible benchmark command section to README

---

## 5. Benchmarking strategy

Benchmarking should happen in layers.

### Layer 1: static model complexity

Purpose:

```text
How big is the model independent of hardware?
```

Metrics:

- parameter count
- trainable parameter count
- checkpoint size
- estimated MACs/FLOPs
- model output tensor sizes
- input size
- output stride
- head resolution

Suggested script:

```text
scripts/benchmark_model_complexity.py
```

Output:

```text
benchmark_complexity.json
benchmark_complexity.csv
```

### Layer 2: pure model forward latency

Purpose:

```text
How fast is the neural network forward pass alone?
```

Exclude:

- image loading
- preprocessing
- decode
- NMS
- visualization

Metrics:

- mean latency
- median latency
- p90, p95, p99 latency
- FPS
- device
- dtype: fp32 / fp16 if available
- batch size
- warmup iterations
- benchmark iterations

Suggested script:

```text
scripts/benchmark_inference_latency.py
```

### Layer 3: full pipeline latency

Purpose:

```text
How fast is the complete inference path?
```

Include:

- image resize
- tensor preparation
- model forward
- decode
- NMS

Metrics:

- preprocessing latency
- model forward latency
- decode/NMS latency
- total latency
- FPS

Suggested script:

```text
scripts/benchmark_end_to_end.py
```

### Layer 4: resource consumption

Purpose:

```text
How much memory and compute does the model consume?
```

Metrics:

- model memory footprint
- peak CUDA memory if available
- process RSS memory if available
- activation memory approximation
- exported model size

Suggested script:

```text
scripts/benchmark_resources.py
```

### Layer 5: export and edge benchmark

Purpose:

```text
Can the model run on mobile/edge runtimes?
```

Steps:

1. Export PyTorch model.
2. Convert to Core ML.
3. Validate numerical parity.
4. Open Core ML model in Xcode.
5. Review Xcode performance report.
6. Run on real iPhone if available.
7. Profile with Instruments.
8. Compare iPhone latency to Mac/Colab latency.

Suggested scripts:

```text
scripts/export_torchscript.py
scripts/export_onnx.py
scripts/export_coreml.py
scripts/validate_export_parity.py
```

---

## 6. Important note about iPhone / simulator testing

An iOS Simulator running on a Mac is useful for app integration testing, but it is not a reliable benchmark for iPhone Neural Engine, GPU, thermal behavior, or real device latency.

For real edge performance, use:

```text
actual iPhone hardware
+ Core ML model
+ Xcode performance report
+ Instruments profiling
```

A simulator can help validate app flow, but not final model performance.

---

## 7. Completed deployment implementation order

### Step 1

Create:

```text
scripts/benchmark_model_complexity.py
```

Goal:

```text
parameter count
checkpoint size
MACs/FLOPs estimate
output tensor shapes
```

### Step 2

Create:

```text
scripts/benchmark_inference_latency.py
```

Goal:

```text
CPU / MPS / CUDA forward latency
batch size 1
median / p95 / FPS
```

### Step 3

Create:

```text
scripts/benchmark_end_to_end.py
```

Goal:

```text
preprocess + forward + decode + NMS total latency
```

### Step 4

Create:

```text
scripts/export_coreml.py
```

Goal:

```text
convert model to Core ML for iPhone testing
```

### Step 5

Create a small iOS test app or minimal Xcode project.

Goal:

```text
run model on actual iPhone and profile latency/memory
```

---

## 8. Benchmark report template

Each benchmark run should produce a row like:

```text
run_name:
checkpoint:
git_commit:
device:
runtime:
input_size:
precision:
batch_size:
params_m:
macs_g:
checkpoint_size_mb:
model_size_mb:
latency_mean_ms:
latency_p50_ms:
latency_p90_ms:
latency_p95_ms:
latency_p99_ms:
fps:
peak_memory_mb:
iou50_f1:
depth_mae_m:
yaw_mae_deg:
notes:
```

The historical benchmark table compared:

```text
baseline_stride32_absolute_box
v6_stride16_fpn_ltrb_center_sampling_class_balance
v6_torchscript
v6_onnx
v6_coreml_fp32
v6_coreml_fp16
future_quantized_coreml
```

## 9. Active training order

1. Run the MobileNetV4 Colab notebook from top to bottom on the complete KITTI
   data in Google Drive.
2. Require `best.pt` plus a `kitti_r40_summary.json` with
   `complete_split: true`.
3. Establish the untouched MobileNetV4 AP3D/BEV R40 baseline before any
   architecture or augmentation ablation.
4. Compare future candidates on the same split and evaluator, then measure
   Core ML parity and physical-iPhone latency for Pareto candidates only.
