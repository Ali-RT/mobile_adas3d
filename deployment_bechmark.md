# MobileADAS3D Deployment Benchmarking Guide

> **Historical deployed-model guide.** The measurements and “current” wording
> below refer to the v6/v7 MobileNetV3 deployment lineage. The iPhone pipeline
> described as future work was subsequently completed. The active model-training
> work is the fresh MobileNetV4 Conv Small Colab baseline; it must generate its
> own AP_R40 and deployment measurements before these numbers can be compared.

# 1. What are we trying to do?

We trained a PyTorch model:

```text id="0xedxc"
MobileADAS3D
```

It takes one image:

```text id="ohd26s"
[1, 3, 384, 1280]
```

and predicts:

```text id="6k5gln"
class
2D box
depth
3D dimensions
yaw
center offset
depth uncertainty
```

The goal now is no longer just training accuracy. The goal is:

```text id="alhupp"
Can this model run fast and efficiently on edge/mobile hardware?
```

Specifically, we want to eventually run it on an **iPhone**.

So we need to move through this chain:

```text id="boiwt5"
PyTorch training checkpoint
        ↓
PyTorch eager inference
        ↓
TorchScript export
        ↓
Core ML export
        ↓
Xcode / iPhone benchmark
```

Each step answers a different question.

---

# 2. PyTorch eager model

This is the normal model you train and run in Python:

```python id="re6gnf"
model = MobileADAS3D(...)
outputs = model(image)
```

This is called **eager mode** because PyTorch executes operations immediately using normal Python control flow.

## Why eager PyTorch is good

It is good for:

```text id="6lvuox"
training
debugging
experimenting
changing architecture quickly
printing tensors
using Python logic
```

## Why eager PyTorch is not ideal for deployment

It depends on:

```text id="2zm3ni"
Python
PyTorch runtime
your source code
your exact model class
Python dictionaries
Python loops
```

That is fine in Colab, but not what an iPhone app runs.

So eager PyTorch is mostly the **development format**, not the final mobile format.

---

# 3. Checkpoint vs model weights vs exported model

You had a checkpoint:

```text id="sfutry"
best.pt
```

A checkpoint usually contains more than the model:

```text id="113pc5"
model weights
optimizer state
scheduler state
epoch number
global step
validation metric
training metadata
```

That is why your checkpoint was large:

```text id="ybci7o"
checkpoint size ≈ 72.7 MB
```

But the actual FP32 model weights were:

```text id="ywxd5v"
state_dict size ≈ 24.2 MB
```

Then Core ML FP16 export became:

```text id="4h9fku"
Core ML FP16 package ≈ 12.18 MB
```

This is expected:

```text id="zcvi2i"
FP32 weights use 32 bits/value
FP16 weights use 16 bits/value
```

So FP16 is roughly half the size.

Your exported Core ML package is `12.18 MB`, which is good for a mobile model.

---

# 4. What is TorchScript?

**TorchScript** is an intermediate PyTorch representation. Instead of relying on your Python model code directly, TorchScript captures the model as a more static graph that can be saved and loaded independently. PyTorch provides `torch.jit.trace` / `torch.jit.save` / `torch.jit.load` for this workflow, and Core ML Tools also describes TorchScript as the PyTorch representation used for conversion. ([PyTorch Documentation][1]) ([Apple GitHub][2])

In our project, we created:

```text id="yxef2q"
mobileadas3d_torchscript.pt
```

That file represents the neural network in a more deployment-friendly form.

## Why we needed TorchScript

We needed TorchScript for three reasons.

First, it removes much of the Python overhead from the model forward pass.

Second, it creates a stable export artifact:

```text id="o60j21"
same input shape
same output tensors
same computation graph
```

Third, it is a good bridge to Core ML. Core ML Tools supports converting PyTorch models directly to Core ML, and its PyTorch conversion workflow uses TorchScript tracing. ([coremltools][3]) ([Apple GitHub][2])

## Tracing vs scripting

There are two common TorchScript paths:

```text id="fqbc0o"
torch.jit.trace
torch.jit.script
```

### Trace

Tracing runs one example input through the model and records the operations that happen. Apple’s Core ML Tools documentation describes tracing as running an example input tensor through the model and capturing the invoked operations. ([Apple GitHub][4])

This is what we used.

Good when:

```text id="pq6nvn"
model forward is mostly tensor operations
input shape is fixed
control flow does not change based on data
```

Our model is good for tracing because the inference path is fixed:

```text id="bd6563"
image → backbone → FPN → heads
```

### Script

Scripting tries to compile Python-like model code. It is more flexible for dynamic control flow, but often harder to make work with complex Python objects.

For us, tracing is cleaner.

---

# 5. Why did TorchScript fail first?

TorchScript failed because the model had this inside `forward()`:

```python id="kg0cyw"
stride = int(round(input_h / out.shape[-2]))
```

That is Python shape logic inside the forward pass.

During tracing, this became problematic because TorchScript could not safely use Python `round()` on that symbolic shape-like value.

So we changed the model from:

```text id="b7yveo"
Calculate stride dynamically during forward
```

to:

```text id="htptgw"
Find stride-16 and stride-32 feature layer indices once during __init__
Then use fixed layer indices during forward
```

That made the model export-friendly.

This is an important deployment lesson:

```text id="g4j81q"
Training-time Python flexibility is convenient.
Deployment usually wants static, predictable tensor computation.
```

---

# 6. Why did we use tuple output instead of dict output?

Your normal model returns:

```python id="fs410r"
{
    "cls_logits": tensor,
    "box2d": tensor,
    "log_depth": tensor,
    ...
}
```

That is nice in Python.

But export tools usually prefer fixed outputs like:

```python id="8edkq1"
(
    cls_logits,
    box2d,
    log_depth,
    dim,
    yaw,
    center_offset,
    depth_uncertainty,
)
```

So we created:

```text id="45kl44"
MobileADAS3DTupleWrapper
```

The wrapper does not change the model math. It only changes the output format.

This makes export easier for:

```text id="t8gqb1"
TorchScript
Core ML
ONNX later if needed
```

---

# 7. What is Core ML?

**Core ML** is Apple’s machine-learning framework for running models inside iOS, macOS, watchOS, and other Apple platforms. In an app, the model is loaded as an `MLModel`, and `MLModelConfiguration` controls things such as which compute devices are allowed. Apple’s documentation says `MLModelConfiguration` can designate or restrict the device used for prediction, such as CPU or GPU. ([Apple Developer][5])

For us, Core ML produces:

```text id="zqpmal"
MobileADAS3D_fp16.mlpackage
```

An `.mlpackage` is the deployment artifact that Xcode/iOS can use.

## Why Core ML instead of PyTorch on iPhone?

Because iPhone apps do not normally run your PyTorch training environment.

Core ML can use Apple hardware acceleration:

```text id="h5h2bg"
CPU
GPU
Neural Engine
```

Apple’s `MLComputeUnits` lets you choose allowed compute units. The `.all` option allows all available compute units, including the Neural Engine when available. ([Apple Developer][6])

So the chain is:

```text id="m1l4la"
PyTorch model
    ↓
TorchScript
    ↓
Core ML .mlpackage
    ↓
iPhone MLModel
```

---

# 8. What does `computeUnits = all` mean?

In Swift/Core ML:

```swift id="x46meq"
config.computeUnits = .all
```

means iOS can choose among available compute units.

Typical options are:

```text id="30y9jq"
.all
.cpuOnly
.cpuAndGPU
.cpuAndNeuralEngine
```

Apple documents `MLComputeUnits` as the set of processing-unit configurations the model can use for prediction. ([Apple Developer][6])

## How it affects our decision

We need to benchmark several settings:

```text id="t3skfm"
.all
.cpuAndGPU
.cpuOnly
```

Why?

Because `.all` may be fastest, but we want to understand:

```text id="jtzzi7"
Is the Neural Engine being used?
Is GPU faster?
Is CPU fallback too slow?
Does memory/thermal behavior change?
```

For final iPhone testing, `.all` is the main deployment setting, but comparison is useful.

---

# 9. What is FP32, FP16, INT8?

These are numerical precision formats.

## FP32

```text id="x0t4cx"
32-bit floating point
largest
most standard
usually safest numerically
```

Your PyTorch model weights were about:

```text id="uc3sk4"
24.2 MB FP32
```

## FP16

```text id="kwu9i9"
16-bit floating point
about half the size
often faster on GPU / Neural Engine
usually small accuracy impact
```

Your Core ML FP16 package is:

```text id="x3xulw"
12.18 MB
```



## INT8

```text id="pram9l"
8-bit integer quantized
smallest
can be fastest
can lose accuracy if not calibrated carefully
```

We are not doing INT8 yet. Correct order is:

```text id="grwc18"
1. FP32 PyTorch baseline
2. TorchScript parity
3. Core ML FP16
4. iPhone benchmark
5. Optional INT8 later
```

---

# 10. What is benchmarking?

Benchmarking means measuring the model systematically.

Not just:

```text id="lnd02q"
It feels fast
```

but:

```text id="6eq2uw"
How many milliseconds?
How much memory?
How much model size?
How stable is latency?
Which stage is bottleneck?
Does accuracy stay the same?
```

For deployment, benchmarking is as important as accuracy.

---

# 11. Why do we benchmark in stages?

Because “model speed” is not one number.

The complete pipeline is:

```text id="sfoagq"
image loading / camera frame
        ↓
preprocessing
        ↓
model forward
        ↓
decode raw outputs
        ↓
NMS
        ↓
final predictions
```

If total latency is bad, we need to know **which part is bad**.

In our case, we discovered the bottleneck changed over time:

```text id="pus6k5"
First:
  decode/NMS was too slow

After vectorized decode:
  model forward became the main cost

After TorchScript:
  full pipeline became very fast
```

That is exactly why staged benchmarking is useful.

---

# 12. Main benchmark terms

## Latency

Latency means time for one inference.

Example:

```text id="7p2p3g"
10 ms per image
```

Lower is better.

For real-time:

```text id="vyifxp"
33.3 ms ≈ 30 FPS
16.7 ms ≈ 60 FPS
10.0 ms ≈ 100 FPS
```

Your TorchScript full pipeline is:

```text id="7td0q8"
9.94 ms mean
```

That is about:

```text id="3wxapb"
100.57 FPS
```



---

## FPS

FPS means frames per second.

Formula:

```text id="i0bhox"
FPS = 1000 / latency_ms
```

So:

```text id="qki7mi"
10 ms → 100 FPS
20 ms → 50 FPS
33 ms → 30 FPS
```

Your current TorchScript end-to-end benchmark:

```text id="pb2zm8"
100.57 FPS
```



---

## Mean latency

Average latency across all runs.

Example:

```text id="xrsq60"
mean total latency = 9.94 ms
```

Good for overall throughput.

But it can hide slow spikes.

---

## P50 latency

Median latency.

Example:

```text id="lh2xac"
p50 total latency = 9.88 ms
```

Half the runs are faster than this, half are slower.

This is the normal-case latency.

---

## P90 / P95 / P99 latency

Tail latency.

Example from your TorchScript end-to-end result:

```text id="tyfl74"
p90 = 10.31 ms
p95 = 10.61 ms
p99 = 11.27 ms
```



This tells us stability.

Low tail latency means:

```text id="bge9rk"
less jitter
fewer dropped frames
more predictable real-time behavior
```

Your TorchScript tail latency is good. `p99 = 11.27 ms` is very stable.

---

# 13. Preprocess latency

Preprocessing includes:

```text id="jpm4ew"
resize image
convert to tensor
move to device
format input shape
```

Your TorchScript end-to-end preprocess:

```text id="8n2nr6"
2.32 ms
```



## Decision impact

Preprocess is not the biggest problem.

But on iPhone, we need to be careful because camera frames come as:

```text id="qfxw9k"
CVPixelBuffer
```

whereas our Core ML model currently expects:

```text id="251d1h"
MLMultiArray [1, 3, 384, 1280]
```

For a production app, image preprocessing can become significant. Later we may want Core ML input as an image type rather than raw NCHW tensor.

---

# 14. Forward latency

Forward latency means the neural network computation itself:

```text id="7n3v9c"
image tensor → raw model outputs
```

Your TorchScript forward-only result:

```text id="0u6csc"
5.69 ms mean
```



Your TorchScript end-to-end forward section:

```text id="kkrvqz"
5.08 ms mean
```



## Decision impact

This is now very good.

If forward latency were high, we would consider:

```text id="32yit4"
smaller input size
smaller FPN channels
smaller head channels
different backbone
FP16
INT8
Core ML / Neural Engine
```

But now forward pass is not urgent to optimize.

---

# 15. Decode latency

Decode means converting raw tensors into real detections.

Raw model outputs are not final boxes. They are tensors:

```text id="whj7sd"
cls_logits
box2d
log_depth
dim
yaw
center_offset
depth_uncertainty
```

Decode does:

```text id="irtnp9"
sigmoid class scores
topK candidate selection
box reconstruction
depth exp
dimension reconstruction
yaw atan2
NMS
format output dictionaries
```

Before optimization, decode was slow:

```text id="oad31a"
15.84 ms
```

After vectorization:

```text id="7jyql9"
2.54 ms in TorchScript full pipeline
```



## Decision impact

This was the biggest optimization win.

It taught us:

```text id="knn6mm"
Do not only optimize the neural network.
Postprocessing can dominate runtime.
```

For iPhone, we need to implement decode efficiently in Swift or integrate more of it into the model.

---

# 16. NMS

NMS means **non-maximum suppression**.

The model may predict multiple boxes around the same object.

NMS keeps the best one and removes overlapping duplicates.

Example:

```text id="hx466t"
5 boxes around same car
       ↓
NMS
       ↓
1 final box
```

Current NMS setting:

```text id="ysiu8u"
nms_iou_threshold = 0.5
```

## Decision impact

Lower threshold:

```text id="34pjq1"
more aggressive duplicate removal
may remove valid nearby objects
```

Higher threshold:

```text id="caq366"
keeps more boxes
may increase duplicate false positives
```

`0.5` is a reasonable default.

---

# 17. topK

TopK is the maximum number of candidate detections we keep before NMS.

We tested:

```text id="5yfazw"
topk = 50, 100, 150, 300
```

With the old decoder:

```text id="ahtzzw"
topk=50 was faster
accuracy drop was tiny
```

So we selected:

```text id="mxrws2"
topk = 50 for deployment
topk = 100 for formal evaluation if needed
```

## Decision impact

Higher topK:

```text id="rhxov3"
more candidates
possibly slightly better recall
slower decode/NMS
```

Lower topK:

```text id="bpba91"
faster
less postprocessing
small risk of missing low-score objects
```

For deployment, `topk=50` is the right choice.

---

# 18. Score threshold

Score threshold controls minimum confidence.

Current:

```text id="qnxjkn"
score_threshold = 0.55
```

Higher threshold:

```text id="1mv26z"
fewer predictions
higher precision
lower recall
faster postprocessing
```

Lower threshold:

```text id="v8c6qv"
more predictions
higher recall
more false positives
slower postprocessing
```

We picked `0.55` because the IoU sweep showed it was a good operating point.

---

# 19. Parameter count

Parameter count means learned weights.

Your model:

```text id="rbz8f2"
6.33M parameters
```

This is small-to-moderate.

## Decision impact

Parameter count affects:

```text id="flgmlm"
model file size
RAM needed for weights
load time
some compute cost
```

But parameter count alone does not tell speed.

A model can have few parameters but still be compute-heavy if the input is large.

---

# 20. MACs / FLOPs

MAC means multiply-accumulate.

FLOPs means floating-point operations.

Your model:

```text id="h2cc8q"
10.85 GMACs
21.69 GFLOPs
```

## Decision impact

This is theoretical compute.

The reason it is not tiny is the large input:

```text id="16zqpu"
1280 x 384
```

Large input gives better small-object detection, but costs more compute.

So the tradeoff is:

```text id="arcsdj"
high resolution:
  better pedestrian/cyclist/far object detection
  more compute

lower resolution:
  faster
  less battery
  possible accuracy loss
```

We should not reduce input size yet because the current runtime is already good.

---

# 21. Memory

Memory tells how much RAM/VRAM is used during inference.

Your TorchScript full pipeline:

```text id="kycjha"
CUDA peak allocated = 46.77 MB
CUDA reserved       = 68 MB
```



## Decision impact

This is good.

For mobile, low memory helps:

```text id="p41pmm"
avoid crashes
reduce pressure on app
reduce thermal/battery load
run alongside camera pipeline
```

But iPhone memory behavior must be measured separately because Core ML runtime is different from PyTorch CUDA.

---

# 22. Package size

Your Core ML FP16 package:

```text id="3tgz1k"
12.18 MB
```



## Decision impact

This is good for mobile.

Package size affects:

```text id="s96gdk"
app download size
load time
storage
memory pressure
```

A 12 MB model is acceptable for an iPhone prototype.

---

# 23. What is parity checking?

Parity means:

```text id="rrg4oo"
Does exported model produce almost the same output as original PyTorch model?
```

We checked:

```text id="zj4zjo"
PyTorch eager output
vs
TorchScript output
```

Result:

```text id="dl3hlr"
all outputs pass allclose at 1e-4
```



## Why parity matters

A model can export successfully but silently change outputs.

If parity fails, benchmarks are meaningless because we may be measuring a broken model.

Parity protects against:

```text id="bkp4zx"
wrong output order
unsupported operation conversion
precision error
shape mismatch
bad export wrapper
```

Our TorchScript parity passed, so TorchScript is trusted.

We still need Core ML parity later.

---

# 24. How to read our current benchmark story

Here is the timeline.

## Stage 1: Eager PyTorch, old decoder

```text id="8ru92q"
total = 29.07 ms
FPS   = 34.40
```

Problem:

```text id="x4emdf"
decode/NMS was too slow
```

## Stage 2: Eager PyTorch, vectorized decoder

```text id="bjk2mn"
total = 13.67 ms
FPS   = 73.16
```

Improvement:

```text id="94k1tc"
postprocessing fixed
```

## Stage 3: TorchScript, vectorized decoder

```text id="ctg82c"
total = 9.94 ms
FPS   = 100.57
```

Improvement:

```text id="hb3kdd"
forward pass became faster
memory improved
tail latency became stable
```



## Stage 4: Core ML FP16 export

```text id="myhczn"
package size = 12.18 MB
```

Now ready for iPhone testing.

---

# 25. Why Core ML benchmark is still needed

Even though Colab CUDA is fast, iPhone is different.

Colab CUDA uses:

```text id="9ui8cj"
NVIDIA GPU
PyTorch CUDA kernels
desktop/server-style memory
```

iPhone Core ML uses:

```text id="yxyetq"
Apple CPU
Apple GPU
Apple Neural Engine
Core ML runtime
mobile thermal and power limits
```

So CUDA speed does not directly predict iPhone speed.

The iPhone benchmark answers:

```text id="b5p7w2"
Can this run on actual mobile hardware?
What latency does Core ML achieve?
Does it use Neural Engine/GPU?
Does it heat up?
Does latency degrade over time?
How much memory does the app consume?
```

That is why we need real-device testing.

---

# 26. What does simulator mean here?

An iOS Simulator runs on your Mac.

It is useful for:

```text id="cgwg4a"
checking app UI
checking code compiles
checking model loads
checking input/output names
```

But it is **not** a valid iPhone performance benchmark.

Why?

Because the simulator does not reproduce:

```text id="fm6yvi"
iPhone Neural Engine behavior
iPhone thermal limits
iPhone memory bandwidth
iPhone camera pipeline
real mobile scheduling
```

So the simulator is okay for app integration, but final benchmark must be on a real iPhone.

---

# 27. How to read future iPhone benchmark

When we run iPhone Core ML, we should record:

```text id="9nxuus"
iPhone model
iOS version
Core ML model precision
computeUnits setting
mean latency
p50 latency
p95 latency
p99 latency
FPS
memory
thermal behavior
```

Example interpretation:

## Case A

```text id="uo77ky"
mean = 8 ms
p95 = 10 ms
```

Excellent. Model is very mobile-feasible.

## Case B

```text id="t4q80g"
mean = 20 ms
p95 = 30 ms
```

Still good. Around 30–50 FPS.

## Case C

```text id="s7cocy"
mean = 50 ms
p95 = 90 ms
```

Too slow for real-time. Then we optimize:

```text id="c72912"
smaller input
smaller head_channels
smaller fpn_channels
quantization
image input instead of MLMultiArray
move decode into optimized Swift/Metal/Core ML
```

## Case D

```text id="cm70zj"
first 30 seconds fast, then slow
```

Thermal throttling. Need lower compute.

---

# 28. What is the difference between model forward and full pipeline on iPhone?

First iPhone benchmark should test only:

```text id="pw109k"
Core ML model prediction
```

That is equivalent to model forward.

Later, full pipeline should include:

```text id="gpos2h"
camera frame
resize/crop
normalization
MLModel prediction
decode
NMS
drawing boxes
```

The full app latency is what matters eventually.

But we start with model forward only to isolate the Core ML model.

---

# 29. Why our current Core ML input is not perfect yet

Current input:

```text id="uzgasg"
MLMultiArray [1, 3, 384, 1280]
```

This is simple and matches PyTorch.

But camera frames on iPhone naturally come as:

```text id="9zjx7e"
CVPixelBuffer
```

So if we use `MLMultiArray`, we may pay extra cost converting camera image to tensor.

Later optimization:

```text id="mdq1jc"
Core ML image input
CVPixelBuffer directly
preprocessing inside Core ML or app pipeline
```

For now, `MLMultiArray` is fine for the first benchmark.

---

# 30. How each benchmark changed our decisions

## Static complexity

Found:

```text id="wlhita"
6.33M params
10.85 GMACs
```

Decision:

```text id="8p4pva"
Model size is okay.
Compute is moderate but acceptable.
Do not shrink architecture yet.
```

## Eager end-to-end

Found:

```text id="fbqq17"
decode was bottleneck
```

Decision:

```text id="g8tqp4"
Optimize decode/NMS before changing model.
```

## topK sweep

Found:

```text id="4362i0"
topk=50 gives almost same accuracy and faster runtime
```

Decision:

```text id="tcmb7y"
Use topk=50 for deployment.
```

## Vectorized decoder

Found:

```text id="cpm2ny"
decode dropped from ~15.84 ms to ~2.85 ms
```

Decision:

```text id="689lzm"
Keep vectorized decoder.
```

## TorchScript parity

Found:

```text id="efzqhc"
all outputs match at 1e-4
```

Decision:

```text id="e3j8t5"
TorchScript export is valid.
```

## TorchScript benchmark

Found:

```text id="jdbt05"
total latency = 9.94 ms
FPS = 100.57
```

Decision:

```text id="mw0tjb"
TorchScript is the best PyTorch-side runtime.
Proceed to Core ML.
```

## Core ML export

Found:

```text id="xee0ix"
FP16 mlpackage = 12.18 MB
```

Decision:

```text id="r7m3k5"
Ready for Xcode/iPhone test.
```

---

# 31. Final mental model

Think of the project like this:

```text id="d2kog0"
Training format:
  PyTorch checkpoint
  best for training/resuming

Development inference:
  PyTorch eager model
  best for debugging

Optimized PyTorch inference:
  TorchScript
  best for stable/exportable PyTorch runtime

Apple deployment:
  Core ML .mlpackage
  best for iPhone/macOS runtime

Real product:
  iOS app with camera + Core ML + decode/NMS + visualization
```

And think of benchmarking like this:

```text id="wevjrm"
Accuracy tells us:
  Is the model useful?

Latency tells us:
  Is it fast enough?

P95/P99 tells us:
  Is it stable enough?

Memory tells us:
  Can it fit on device?

Package size tells us:
  Is it deployable?

Parity tells us:
  Did export preserve the model?

Stage breakdown tells us:
  What should we optimize next?
```

# Current conclusion

Your current model is in a strong state:

```text id="1fmkvf"
Accuracy:
  IoU≥0.50 F1 ≈ 0.79

PyTorch/TorchScript speed:
  ~9.94 ms end-to-end on CUDA
  ~100 FPS

Model size:
  12.18 MB Core ML FP16

Export:
  TorchScript parity passed
  Core ML FP16 export succeeded
```

The next real question is no longer “can we export it?” It is:

```text id="b18jw8"
How fast does MobileADAS3D run on an actual iPhone with Core ML?
```

[1]: https://docs.pytorch.org/tutorials/beginner/basics/saveloadrun_tutorial.html?utm_source=chatgpt.com "Save and Load the Model — PyTorch Tutorials 2.12.0+ ..."
[2]: https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html?utm_source=chatgpt.com "PyTorch Conversion Workflow — Guide to Core ML Tools"
[3]: https://coremltools.readme.io/v6.3/docs/pytorch-conversion?utm_source=chatgpt.com "Converting from PyTorch"
[4]: https://apple.github.io/coremltools/docs-guides/source/model-tracing.html?utm_source=chatgpt.com "Model Tracing — Guide to Core ML Tools"
[5]: https://developer.apple.com/documentation/coreml/mlmodelconfiguration?utm_source=chatgpt.com "MLModelConfiguration | Apple Developer Documentation"
[6]: https://developer.apple.com/documentation/coreml/mlcomputeunits?utm_source=chatgpt.com "MLComputeUnits | Apple Developer Documentation"
