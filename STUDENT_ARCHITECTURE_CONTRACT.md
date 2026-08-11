# MobileADAS3D-S1 student architecture contract

Status: locked for graph implementation and pre-training device gate
Decision date: 2026-08-11

## Decision

The first production student candidate is **MobileADAS3D-S1**, a fully
convolutional, anchor-free monocular 3D detector designed around native Core ML
operations. MonoDETR remains the accuracy teacher/reference and is not embedded
in the iPhone app.

S1 is the smallest controlled change that addresses both observed limits:

- MobileMonoDETR-VP1 reached useful KITTI accuracy but required 161-177 ms per
  steady iPhone prediction, failing the 50 ms gate.
- The deployed MobileNetV3-Small dense CNN reached 15.481 ms inference p95 on
  CPU-only Core ML, proving that the dense convolutional family has substantial
  device headroom, but its model quality is not sufficient.

## Locked graph

```text
RGB Float32 /255 input [1, 3, 384, 1280]
  -> embedded ImageNet normalization
  -> MobileNetV4 Conv Small ImageNet backbone
  -> native feature maps at strides 8, 16, and 32
  -> 1x1 lateral projections to 96 channels
  -> top-down Lite-FPN: bilinear 2x resize (align_corners=false) + add
  -> one depthwise-separable 3x3 refinement per level
  -> stride-8 prediction feature [1, 96, 48, 160]
  -> shared depthwise-separable 3x3 prediction tower, 96 channels
  -> independent 1x1 output projections
```

The output heads are:

| Head | Channels | Meaning |
|---|---:|---|
| `cls_logits` | 2 | Vehicle and Pedestrian logits |
| `quality` | 1 | learned 3D/localization quality for score ranking |
| `box2d` | 4 | positive LTRB distances from the owned cell |
| `projected_center_offset` | 2 | projected 3D bottom-center offset |
| `log_depth` | 1 | longitudinal camera-space depth |
| `depth_uncertainty` | 1 | heteroscedastic depth uncertainty |
| `dim` | 3 | log residual from per-class mean dimensions |
| `yaw_axis` | 2 | orientation axis as sine/cosine of doubled yaw |
| `yaw_direction` | 1 | front/back direction logit |
| `loc_xy` | 2 | auxiliary teacher-compatible X/Z and Y/Z ratios |

All heads use the same stride-8 feature and only a final 1x1 convolution. There
is no per-head 3x3 tower. Decoding, TopK, calibration back-projection, NMS, and
artifact writing remain outside the model so the existing Swift pipeline can
be adapted without embedding dynamic control flow in Core ML.

Every S1 depthwise-separable block is locked to depthwise 3x3 convolution,
BatchNorm, ReLU, pointwise 1x1 convolution, BatchNorm, and ReLU. The top-down
path computes `P16 = refine(lateral16 + resize(lateral32))`, followed by
`P8 = refine(lateral8 + resize(P16))`; the shared prediction tower consumes
only `P8`. Convolution biases are disabled when immediately followed by
BatchNorm. This removes implementation ambiguity from the device probe.

## Explicit exclusions

S1 contains no transformer encoder/decoder, object-query set prediction,
deformable attention, ROI operation, grid sampling, custom Core ML operation,
dynamic tensor shape, recurrent temporal state, or in-model NMS. These may be
researched later, but they are not part of the first student experiment.

Do not simultaneously change the backbone, width, input resolution, output
stride, loss family, and taxonomy after this contract. S1 is one versioned
architecture; any ablation receives a new identifier.

## Compatibility contracts

### Input

- fixed shape `[1, 3, 384, 1280]`;
- RGB channel order;
- Float32 values divided by 255 externally;
- exact 10:3 center crop and bilinear resize already used by the iPhone app;
- ImageNet mean/std normalization embedded in the graph.

### Training and teacher integration

- production classes are exactly `Vehicle` and `Pedestrian`;
- ground-truth objects own stride-8 cells through the existing center-sampling
  target builder;
- MonoDETR predictions may supervise only GT-associated cells through the
  validated teacher target adapter;
- GT remains authoritative for class and 2D box targets;
- teacher depth, dimensions, location, and yaw are auxiliary targets, never
  unfiltered pseudo ground truth;
- training and checkpoint selection follow `PRODUCT_MODEL_CONTRACT.md`.

### Deployment output

The first Core ML package exposes raw named tensors. A versioned Swift decoder
maps the stride-8 tensors to the same detection record used by the current app:
class, score, 2D box, depth, dimensions, yaw, projected 3D center, camera-space
location, and uncertainty. Existing recording artifacts and frame-dropping
behavior do not change.

## Pre-training gates

Training is not authorized until a random-weight S1 graph passes all of these:

1. parameter count <= 10 million;
2. estimated compute <= 15 GMAC at 1280x384;
3. Float16 Core ML package <= 25 MB;
4. conversion contains no custom/host-fallback operation;
5. PyTorch-to-Core-ML raw-output maximum absolute delta <= 2e-3;
6. physical iPhone 16 Pro Max model-only timing after five warmups:
   p95 <= 35 ms over 100 predictions.

The 35 ms architecture gate is intentionally tighter than the 50 ms product
limit, reserving margin for trained-weight variation, OS/runtime variation,
decode, thermal effects, and later output-contract adjustments. If S1 misses
the gate, reduce FPN/tower width from 96 to 64 before changing input resolution
or removing the stride-8 pedestrian path.

## Post-training gates

S1 must still pass every quality, parity, external-generalization, and sustained
runtime gate in `PRODUCT_MODEL_CONTRACT.md`. The random graph gate proves only
deployability and speed; it does not prove detection quality.

## Controlled implementation order

1. Implement S1 alongside the existing models under a distinct architecture
   name; do not replace or silently reinterpret old checkpoints.
2. Add output-shape, parameter-count, target/decode round-trip, TorchScript, and
   Core ML conversion tests.
3. Run the random-weight native iPhone 5-warmup/100-prediction gate.
4. Implement and audit the two-class KITTI mapping.
5. Establish the two-class teacher/reference protocol.
6. Train the GT-only S1 baseline.
7. Run a paired teacher-distillation experiment from the same initialization;
   continue distillation only if it improves frozen-protocol AP/nearby recall.
