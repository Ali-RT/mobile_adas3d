# MobileADAS3D Benchmark Vocabulary

> **Historical deployed-model explanation.** Numeric examples and “current”
> conclusions below describe the v6/v7 MobileNetV3 lineage. They are retained to
> explain the terms, not as results for the fresh MobileNetV4 Conv Small model.
> New training results must use the canonical Chen split and KITTI AP_R40.

# Big picture

Benchmarking answers four questions:

```text
1. Is the model accurate enough?
2. Is the model small enough?
3. Is the model fast enough?
4. Where is the bottleneck?
```

For MobileADAS3D, we found:

```text
Accuracy: good baseline
Model size: reasonable
GPU forward speed: good
CPU speed: slow
Current bottleneck: Python decode/NMS
```

---

# 1. Parameter count

**Meaning:** Number of learned weights in the model.

Your model:

```text
6.33M parameters
```

This is relatively small. It means the model is not huge in storage or memory.

**Decision impact:**

```text
Low parameter count:
  good for mobile / edge deployment

High parameter count:
  larger model file
  more RAM
  slower load time
  harder to deploy
```

For us, **parameters are not the main problem**. The model is compact enough.

---

# 2. Trainable parameters

**Meaning:** Parameters updated during training.

Your model:

```text
6.33M trainable parameters
0 frozen parameters
```

So the full model was trained.

**Decision impact:**

If we freeze the backbone later, training becomes faster and more stable, but accuracy may drop. For deployment, trainable vs frozen does not matter; all parameters become inference weights.

---

# 3. State dict size

**Meaning:** Size of model weights only.

Your model:

```text
FP32 state_dict size = 24.2 MB
```

This is approximately the deployable PyTorch weight size before compression.

**Decision impact:**

```text
FP32: ~24.2 MB
FP16: ~12.1 MB
INT8: ~6 MB approximate
```

For iPhone/Core ML, FP16 is likely a good first deployment format.

---

# 4. Checkpoint size

**Meaning:** Full training checkpoint file size.

Your checkpoint:

```text
72.7 MB
```

This is larger than the model weights because it may include optimizer state, scheduler state, epoch, metrics, and other training metadata.

**Decision impact:**

Do **not** judge deployability from checkpoint size. For deployment, we care more about exported model size.

---

# 5. MACs and FLOPs

## MACs

**MAC** means multiply-accumulate operation.

```text
a * b + c
```

Your model:

```text
10.85 GMACs
```

## FLOPs

**FLOPs** means floating-point operations. Often:

```text
FLOPs ≈ 2 × MACs
```

Your model:

```text
21.69 GFLOPs
```



**Decision impact:**

This tells us theoretical compute cost.

```text
High MACs/FLOPs:
  more computation
  slower inference
  more battery use
  more heat on phone

Low MACs/FLOPs:
  faster
  cheaper
  better for edge
```

Important: your model has only **6.33M params**, but still **10.85 GMACs** because the input image is large: `1280 x 384`.

So the model is:

```text
small in weights
moderate-heavy in computation
```

---

# 6. Input size

Your input:

```text
1280 x 384
```

This is large enough to preserve small objects, especially pedestrians and cyclists.

**Decision impact:**

Bigger input:

```text
better small-object detection
better far-object detection
higher MACs/FLOPs
slower inference
more memory
```

Smaller input:

```text
faster
less memory
better for iPhone
but may hurt pedestrian/cyclist accuracy
```

Possible future tradeoff:

```text
1280 x 384 → best accuracy
960 x 288  → faster, maybe acceptable
640 x 192  → much faster, likely weaker small-object detection
```

Do not reduce input size until we finish current benchmarking.

---

# 7. Output stride

Your current output stride:

```text
stride = 16
```

That means one prediction cell corresponds to roughly `16 x 16` input pixels.

Feature map:

```text
input: 1280 x 384
feature map: 80 x 24
```



**Decision impact:**

Smaller stride, such as `16`:

```text
better localization
better small-object detection
more compute
more prediction cells
slower decode
```

Larger stride, such as `32`:

```text
faster
fewer cells
worse for small pedestrians/cyclists
more cell collisions
```

We moved to stride-16 because stride-32 was hurting Pedestrian/Cyclist.

---

# 8. Feature map size

Your prediction grid:

```text
80 x 24 = 1920 cells
```

Each cell predicts class, box, depth, dimensions, yaw, offset, uncertainty.

**Decision impact:**

More cells:

```text
better spatial detail
more candidate predictions
more decode/NMS cost
```

This is one reason decode became expensive.

---

# 9. Output tensors

Your model outputs:

```text
cls_logits
box2d
log_depth
dim
yaw
center_offset
depth_uncertainty
```

Total output tensor size:

```text
0.117 MB
```



**Decision impact:**

The output tensor memory is tiny. Output size is not the issue. The issue is the **postprocessing logic** applied to those outputs.

---

# 10. Latency

**Latency** means how long one inference takes.

Example:

```text
33 ms per image
```

means roughly one image every 33 milliseconds.

**Decision impact:**

For real-time ADAS-style use:

```text
< 16.7 ms  ≈ 60 FPS
< 33.3 ms  ≈ 30 FPS
< 50 ms    ≈ 20 FPS
< 100 ms   ≈ 10 FPS
```

Your current CUDA end-to-end latency with `topk=50`:

```text
mean total latency = 29.07 ms
FPS = 34.40
```

So the current pipeline is around real-time on Colab GPU.

---

# 11. Mean latency

**Meaning:** Average latency across all benchmark iterations.

Example:

```text
CUDA end-to-end mean = 29.07 ms
```



**Decision impact:**

Good for a general estimate, but can hide spikes.

Use mean for:

```text
overall throughput estimate
high-level comparison
```

---

# 12. P50 latency

**Meaning:** Median latency. Half the runs are faster, half are slower.

Example:

```text
CUDA end-to-end P50 = 25.45 ms
```



**Decision impact:**

P50 tells us normal-case speed.

If mean is much higher than P50, there are latency spikes.

Your case:

```text
mean = 29.07 ms
p50  = 25.45 ms
```

So normal speed is good, but some iterations are slower.

---

# 13. P90 / P95 / P99 latency

**Meaning:** Tail latency.

Example:

```text
P95 = 52.42 ms
P99 = 56.80 ms
```

for `topk=50`.

**Decision impact:**

This matters for real-time systems. Even if average is fast, occasional slow frames can cause jitter.

```text
Low P95/P99:
  stable real-time behavior

High P95/P99:
  frame drops
  jitter
  worse user experience
```

For our model, P95/P99 are higher because decode/NMS has variable cost depending on number of candidates.

---

# 14. FPS

**FPS** means frames per second.

Formula:

```text
FPS = 1000 / latency_ms
```

Your CUDA full pipeline with `topk=50`:

```text
34.40 FPS mean
39.29 FPS P50
```



**Decision impact:**

FPS helps compare against application needs:

```text
10 FPS: slow but usable for offline/demo
20 FPS: acceptable for some perception prototypes
30 FPS: real-time target
60 FPS: strong real-time
```

Your current CUDA pipeline is around 30 FPS+.

---

# 15. Preprocessing latency

**Meaning:** Time to prepare image for the model.

Includes:

```text
resize
tensor formatting
move to device
dtype conversion
```

Your CUDA preprocessing:

```text
~2.69 ms at topk=50
```



**Decision impact:**

If preprocessing is high, optimize image resizing, avoid unnecessary copies, and keep memory layout efficient.

For us, preprocessing is small. Not the main bottleneck.

---

# 16. Forward latency

**Meaning:** Time for neural network forward pass only.

Your CUDA forward-only benchmark:

```text
7.52 ms
```



In end-to-end benchmark with `topk=50`:

```text
forward mean = 10.54 ms
```



**Decision impact:**

If forward latency is too high, optimize the model itself:

```text
reduce input size
reduce head channels
reduce FPN channels
use FP16
use INT8
use Core ML / TensorRT
```

For us, forward speed is acceptable on GPU. It is not the first bottleneck.

---

# 17. Decode latency

**Meaning:** Time to convert raw model outputs into final detections.

Decode includes:

```text
sigmoid / confidence scoring
topK candidate selection
box reconstruction
depth conversion
dimension conversion
yaw conversion
NMS
formatting final predictions
```

Your CUDA decode with `topk=50`:

```text
15.84 ms
```

Your CUDA forward with `topk=50`:

```text
10.54 ms
```



**Decision impact:**

This tells us the bottleneck has moved from the network to postprocessing.

For us:

```text
decode is slower than forward
```

So next optimization should be:

```text
vectorize decode
avoid .item() loops
move tensors CPU/GPU only once
reduce topK
optimize NMS
```

---

# 18. NMS

**NMS** means non-maximum suppression.

Purpose:

```text
remove duplicate overlapping boxes
keep the highest-confidence detection
```

Example:

If the model predicts five boxes around the same car, NMS keeps one.

**Decision impact:**

NMS IoU threshold controls how aggressively duplicates are removed.

```text
lower NMS IoU threshold:
  removes more boxes
  fewer false duplicates
  may remove valid nearby objects

higher NMS IoU threshold:
  keeps more boxes
  better for crowded scenes
  may increase duplicate false positives
```

Current setting:

```text
nms_iou_threshold = 0.5
```

This is reasonable.

---

# 19. topK

**Meaning:** Maximum number of candidate detections considered before NMS.

We tested:

```text
topk = 50, 100, 150, 300
```

Result:

```text
topk=50:
  total mean = 29.07 ms
  F1 = 0.7903

topk=300:
  total mean = 33.74 ms
  F1 = 0.7929
```

**Decision impact:**

Higher topK:

```text
slightly better recall
slower decode
more NMS work
```

Lower topK:

```text
faster
slightly more risk of missing objects
```

Our decision:

```text
topk=50 for deployment/demo
topk=100 for formal evaluation
```

Because `topk=50` is faster and loses only a tiny amount of F1.

---

# 20. Score threshold

**Meaning:** Minimum confidence needed to keep a candidate.

Current value:

```text
score_threshold = 0.55
```

**Decision impact:**

Higher threshold:

```text
fewer predictions
higher precision
lower recall
faster decode
```

Lower threshold:

```text
more predictions
higher recall
lower precision
slower decode
```

We selected `0.55` because it gave the best overall F1 in the sweep.

Later, we may use class-specific thresholds:

```text
Car:        0.55
Pedestrian: 0.55 or 0.60
Cyclist:    0.60 or 0.65
```

---

# 21. Number of predictions per image

Your average:

```text
~4.0 predictions/image
```

with `topk=50`.

**Decision impact:**

This tells us the final scene output is small, but the decoder still processes many candidates before reaching those final detections.

That is why decode optimization matters.

---

# 22. CPU vs CUDA

## CPU

Your CPU end-to-end result:

```text
159.45 ms
6.27 FPS
```



## CUDA

Your CUDA end-to-end result with `topk=50`:

```text
29.07 ms
34.40 FPS
```



**Decision impact:**

CPU is not good enough for real-time at this input size.

CUDA is good enough for prototype real-time.

For iPhone, we need Core ML because iPhone can use:

```text
CPU
GPU
Neural Engine
```

PyTorch CPU numbers do not predict iPhone performance well.

---

# 23. Memory allocated

**Meaning:** Actual memory used by tensors.

Your CUDA memory allocated:

```text
~24.27 MB after benchmark
```

Peak allocated:

```text
72.69 MB
```



**Decision impact:**

Memory looks good. This is promising for edge deployment.

If memory were too high, we would need:

```text
smaller input
FP16
smaller heads
activation checkpointing during training
simpler model
```

For inference, memory is not currently the major problem.

---

# 24. Memory reserved

**Meaning:** Memory PyTorch keeps in its CUDA cache.

Your CUDA reserved memory:

```text
92 MB
```



**Decision impact:**

Reserved memory is not exactly active usage. PyTorch keeps memory cached to avoid repeated allocation overhead.

For deployment, actual Core ML memory may differ.

---

# 25. Batch size

Your benchmarks use:

```text
batch_size = 1
```

**Decision impact:**

For real-time camera inference, batch size 1 is the correct benchmark.

Larger batch size may improve throughput but increases latency per frame and is less relevant for live ADAS/mobile inference.

---

# 26. Warmup iterations

**Meaning:** Runs before measuring latency.

Purpose:

```text
initialize kernels
fill caches
stabilize runtime
avoid first-run overhead
```

**Decision impact:**

Without warmup, benchmark numbers are misleading.

For CUDA/Core ML, first inference is usually slower than steady-state inference.

---

# 27. Benchmark iterations

**Meaning:** Number of measured runs.

You used:

```text
100 / 200 iterations
```

**Decision impact:**

More iterations give more stable latency estimates.

For final reporting, use at least:

```text
100 warmup
500 benchmark iterations
```

on the final chosen runtime.

---

# 28. Precision: FP32, FP16, INT8

## FP32

Default full precision.

```text
most accurate
largest model
slower
```

## FP16

Half precision.

```text
smaller model
faster on GPU/Neural Engine
usually little accuracy loss
good first mobile format
```

## INT8

Quantized 8-bit.

```text
smallest
fastest potential
may lose accuracy
needs calibration/validation
```

**Decision impact:**

For iPhone, first try:

```text
Core ML FP16
```

Then later:

```text
INT8 quantization
```

only after we have a strong FP16 baseline.

---

# 29. Precision / Recall / F1

These are accuracy metrics.

## Precision

Of the detections model made, how many were correct?

```text
Precision = TP / (TP + FP)
```

High precision means fewer false alarms.

## Recall

Of the real objects, how many did model find?

```text
Recall = TP / (TP + FN)
```

High recall means fewer missed objects.

## F1

Balanced score between precision and recall.

```text
F1 = 2 * precision * recall / (precision + recall)
```

**Decision impact:**

For ADAS:

```text
low precision:
  too many false warnings

low recall:
  missed road users
```

F1 helps choose a balanced threshold.

---

# 30. TP / FP / FN

## TP: true positive

Correct detection.

## FP: false positive

Model predicted an object that should not be there.

## FN: false negative

Model missed a real object.

**Decision impact:**

```text
many FP:
  increase score threshold
  improve NMS
  tune class thresholds

many FN:
  lower threshold
  improve small-object detection
  improve training targets
```

Your current model still has more pedestrian/cyclist challenges than car.

---

# 31. IoU

**IoU** means intersection over union.

It measures overlap between predicted box and ground-truth box.

```text
IoU = overlap area / union area
```

Common thresholds:

```text
IoU >= 0.25: loose detection
IoU >= 0.50: stronger detection
```

**Decision impact:**

Higher IoU means better localization.

Our key metric is:

```text
IoU >= 0.50 F1
```

because it requires reasonably accurate boxes.

---

# 32. mIoU

**mIoU** means mean IoU across matched detections.

**Decision impact:**

F1 says whether we detected the object.

mIoU says how well the box aligns when detected.

---

# 33. Depth MAE

**MAE** means mean absolute error.

Depth MAE:

```text
average absolute depth error in meters
```

Example:

```text
depth MAE = 1.876 m
```

**Decision impact:**

Important for ADAS because distance is critical.

If depth MAE is high:

```text
add depth calibration
add uncertainty loss
add depth bins
improve far-object training
```

---

# 34. Relative depth error

**Meaning:**

```text
abs(pred_depth - gt_depth) / gt_depth
```

This matters because 2 meters error at 10 meters is bad, but 2 meters error at 80 meters is less severe.

**Decision impact:**

Good for comparing near and far objects fairly.

---

# 35. Yaw MAE

**Meaning:** Average orientation error in degrees.

Example:

```text
yaw MAE = 10.63 degrees
```

**Decision impact:**

Important for object heading, tracking, and trajectory reasoning.

But for pedestrians, yaw is naturally ambiguous. We decided not to change the yaw head yet because car/cyclist yaw is acceptable.

---

# 36. Axis-aware yaw error

**Meaning:** Yaw error that treats front/back flipped orientation as equivalent.

It answers:

```text
Does the model know the object axis, even if it confuses front vs back?
```

Our result showed front/back flips are not the main issue.

**Decision impact:**

Because flip rate was low, we should not prioritize a special front/back yaw fix right now.

---

# 37. Dimension MAE

**Meaning:** Average error in predicted object size.

For 3D detection:

```text
height
width
length
```

Your dimension error was already good.

**Decision impact:**

No need to focus on dimensions now.

---

# Final decision map

Here is how the benchmark terms guide our next steps:

```text
Parameters are low:
  model size is okay

MACs/FLOPs are moderate:
  watch input size and head size later

CUDA forward is good:
  do not change model architecture yet

CPU latency is slow:
  not suitable for CPU-only real-time at current resolution

CUDA end-to-end is okay:
  real-time prototype is possible

Decode/NMS is too slow:
  optimize postprocessing next

topk=50 is best for deployment:
  use topk=50 for demo/inference

Memory is low:
  good sign for edge deployment

F1 drop from topk=300 to topk=50 is tiny:
  topk=50 is acceptable

iPhone needs Core ML:
  PyTorch CUDA/CPU numbers are only development benchmarks
```

So the next engineering decision is clear:

```text
Do not modify model architecture yet.
Optimize decode/NMS first.
Then export to Core ML FP16.
Then benchmark on iPhone hardware.
```
