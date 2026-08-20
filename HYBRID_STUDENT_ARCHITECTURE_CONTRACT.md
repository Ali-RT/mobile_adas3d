# MobileADAS3D-H1 hybrid student architecture contract

Status: random graph implemented; FP16 parity and physical-device gates pending
Decision date: 2026-08-20

## Decision

MobileADAS3D-H1 is the next deployment candidate. It restores the teacher's
depth-aware, query-based reasoning while retaining a MobileNetV4 backbone and
using only fixed-shape Core ML operations. It replaces the accuracy-inadequate
dense S1 family; it does not modify or replace the frozen R0 teacher.

The S1 checkpoint sweep was deliberately waived when the project moved on:
S1-V1, invalid S1-V2, and corrected S1-V2b all missed the product gates by
orders of magnitude. S1 remains a documented speed baseline, not a training or
distillation candidate.

## Teacher and student side by side

```mermaid
flowchart TB
  subgraph R0["Frozen teacher: MonoDETR R0"]
    R0I["RGB 1280x384"] --> R0B["ResNet-50 backbone"]
    R0B --> R0M["Multi-scale features"]
    R0M --> R0D["LID depth predictor: 80 bins"]
    R0M --> R0E["Depth-aware deformable encoder: 3 layers"]
    R0D --> R0E
    R0E --> R0Q["Deformable query decoder: 3 layers"]
    R0Q --> R0H["Class, 2D box, projected center, depth, dimensions, yaw, 3D location"]
  end

  subgraph H1["Deployment student: MobileADAS3D-H1"]
    H1I["RGB 1280x384"] --> H1B["MobileNetV4 Conv Small"]
    H1B --> H1F["Lite-FPN: strides 8, 16, 32; 128 channels"]
    H1F --> H1D["Depth context: 40 bins at stride 16"]
    H1F --> H1E["Standard global encoder: stride 32, 2 layers"]
    H1D --> H1E
    H1E --> H1Q["Standard query decoder: 50 queries, 2 layers"]
    H1F --> H1Q
    H1Q --> H1H["Teacher-compatible query heads"]
  end

  R0H -. "GT-associated output and feature distillation after GT-only baseline" .-> H1H
```

## Locked H1 graph

```text
RGB Float32 /255 [1, 3, 384, 1280]
  -> embedded ImageNet normalization
  -> MobileNetV4 Conv Small ImageNet backbone
  -> native stride-8, stride-16, stride-32 features
  -> 1x1 projections + Lite-FPN, 128 channels
  -> stride-16 depth logits, 40 linearly increasing discretization bins
  -> expected-depth embedding added to multi-scale memory
  -> stride-32 tokens [1, 480, 192]
  -> 2 standard pre-norm Transformer encoder layers
  -> fixed multi-scale memory from strides 8/16/32
  -> 50 learned object queries, width 192
  -> 2 standard pre-norm Transformer decoder layers
  -> per-query product heads
```

Locked dimensions:

| Component | H1 value |
| --- | ---: |
| Input | 1280x384 |
| Backbone | MobileNetV4 Conv Small |
| FPN width | 128 |
| Transformer width | 192 |
| Attention heads | 6 |
| Encoder layers | 2 |
| Decoder layers | 2 |
| Feed-forward width | 768 |
| Object queries | 50 |
| Depth bins | 40 |
| Dropout | 0.0 for export determinism |

The encoder operates only on the 12x40 stride-32 map, limiting global
self-attention to 480 tokens. The decoder may cross-attend to fixed flattened
stride-8/16/32 memory so small Pedestrians retain high-resolution evidence.
All shapes are compile-time constants.

## Query output contract

Each of the 50 queries predicts:

| Output | Meaning |
| --- | --- |
| `class_logits` | Vehicle and Pedestrian sigmoid logits; unmatched queries are all-negative focal targets |
| `box2d_cxcywh` | normalized 2D box center, width, and height |
| `projected_center` | normalized projected 3D bottom center |
| `depth_logits` + `depth_residual` | depth distribution and local correction |
| `dimensions` | log residual from train-only class mean dimensions |
| `yaw` | direct sine/cosine vector, normalized only in loss/decoder |
| `location_xy` | X/Z and Y/Z auxiliary geometry ratios |
| `quality` | 3D localization quality used for ranking after calibration |

TopK, calibration back-projection, yaw normalization, score calibration, NMS,
tracking, and recording remain outside the Core ML graph.

## Teacher-to-student transfer map

H1 is structurally closer to R0, but it is not shape-identical. Exact R0 weight
loading is neither expected nor claimed.

| Teacher knowledge | H1 transfer method | Timing |
| --- | --- | --- |
| Image representation | ImageNet initialization, not R0 ResNet weights | initialization |
| Depth distribution | KL loss on GT-associated spatial depth distributions | after GT-only baseline |
| Query class/objectness scores | temperature-scaled query distillation after Hungarian association | after GT-only baseline |
| 2D box and projected center | GT remains primary; teacher is auxiliary on matched objects | after GT-only baseline |
| Depth, dimensions, location, yaw | confidence-gated regression distillation | after GT-only baseline |
| Transformer features | learned projection from R0 width 256 to H1 width 192, masked feature loss | optional ablation |

Ground truth remains authoritative. Teacher predictions never create unfiltered
pseudo-labels. Distillation is disabled until a complete GT-only H1 baseline is
trained, evaluated, and frozen.

## Explicit exclusions

H1 contains no multi-scale deformable attention, custom CUDA/Core ML operation,
dynamic query count, dynamic input shape, ROIAlign, grid sampling, recurrent
state, or in-model NMS. It does not use the full ResNet-50 teacher graph or the
dense S1 output contract.

## Gates before training

The random-weight H1 graph must pass:

1. parameters <= 10 million;
2. compute <= 15 GMAC at 1280x384;
3. FP16 Core ML package <= 25 MB;
4. no custom or host-fallback operation;
5. PyTorch-to-Core-ML raw-output maximum absolute delta <= 0.002;
6. physical iPhone 16 Pro Max model-only p95 <= 35 ms after 5 warmups and 100
   timed predictions.

If H1 misses compute, package, or device latency, reduce transformer width from
192 to 128 before reducing query count, input resolution, or stride-8 memory.
Only one fallback change is permitted per versioned experiment.

## Random-graph implementation evidence

The locked H1 graph is implemented in `models/mobile_adas3d_h1.py` with an
explicit fixed-shape attention implementation. It does not depend on PyTorch's
dynamic multi-head-attention export path. Forward, tuple-export, finite-value,
and full-graph backward tests pass.

Measured on the random-weight graph at 1280x384:

| Check | Result | Gate |
| --- | ---: | --- |
| Parameters | 3,619,457 | Pass (<=10M) |
| Compute | 4.9068 GMAC | Pass (<=15) |
| FP16 package | 10.35 MB | Pass (<=25 MB) |
| Core ML custom operations | none | Pass |
| Trace maximum absolute delta | 0.0 | Pass |
| FP16 Core ML maximum raw delta | 0.07133 | **Fail** (<=0.002) |
| FP32 Core ML control delta | 0.0000361 | Pass as diagnosis only |
| FP32 package | 20.56 MB | Size passes; not the required FP16 artifact |

The FP32 control proves that the fixed graph converts faithfully. The remaining
local blocker is accumulated FP16 numerical error, not an unsupported operator
or an incorrect trace. No H1 training is authorized yet. The next task is to
resolve the FP16 policy/graph numerics and then run the physical iPhone
5-warmup/100-prediction gate. Mac prediction time is diagnostic only and must
not be substituted for the device result.

## Training order

1. Implement and unit-test the fixed-shape random graph.
2. Convert to Core ML and pass local raw-output parity.
3. Run the physical-iPhone random graph gate.
4. Prepare a fresh GT-only 20-epoch health gate with Hungarian matching.
5. Review complete Vehicle/Pedestrian AP_R40 and geometry diagnostics.
6. Continue GT-only training only if both classes learn and AP is materially
   above the S1 baseline.
7. Freeze the GT-only checkpoint and then run one paired R0-distillation
   experiment from the identical initialization.

The frozen product gates, data splits, class mapping, external nuScenes test,
Core ML parity, and sustained iPhone qualification remain unchanged.
