# MobileADAS3D accuracy-first student contract

Status: active
Decision date: 2026-08-24

## Decision

Model development now prioritizes accuracy before deployment compression. The
physical-iPhone parameter, compute, package-size, operator, and latency gates
are suspended as model-selection gates. They remain recorded diagnostics and
will be reapplied only after an accuracy-qualified student is frozen.

This does not discard the iPhone application or its validated recording and
benchmark workflows. It prevents edge limits from forcing an architecture that
cannot first learn the two-class 3D task.

## Frozen reference and comparable-performance gate

The denominator is the frozen two-class R0 ResNet50 MonoDETR epoch-185
checkpoint and the unchanged Chen 3,712/3,769 product protocol.

| Metric | R0 | Student gate (90% of R0) |
| --- | ---: | ---: |
| Vehicle moderate 3D AP_R40 | 17.6348 | 15.8713 |
| Pedestrian moderate 3D AP_R40 | 5.7214 | 5.1493 |
| Balanced moderate 3D mean | 11.6781 | 10.5103 |
| Vehicle moderate BEV AP_R40 | 23.6816 | 21.3134 |
| Pedestrian moderate BEV AP_R40 | 6.5961 | 5.9365 |

All five gates and all 3,769 validation predictions are required. Checkpoint
selection also reports per-class nearby recall; aggregate loss or Vehicle AP
alone cannot select a model.

## A1 and A2 student architecture

Both accuracy-first students preserve the R0 MonoDETR depth/transformer path:

```text
RGB [1, 3, 384, 1280]
  -> MobileNetV4 backbone
  -> stride-8/16/32 features plus fourth projected feature level
  -> 256-channel feature projections
  -> 80-bin LID depth predictor, 0-60 m
  -> 3-layer depth-aware deformable encoder
  -> 3-layer decoder, 8 heads, 50 inference queries
  -> class, 2D box/projected-center, depth/uncertainty,
     dimensions, and 12-bin-plus-residual yaw heads
```

A1 uses `mobilenetv4_conv_small.e2400_r224_in1k` (about 3.8M backbone
parameters). A2 changes only the backbone and required projection shapes to
`mobilenetv4_conv_medium.e500_r256_in1k` (about 9.7M backbone parameters).
Resolution, taxonomy, Chen split, augmentation, transformer, grouped-query
training, heads, decoder, evaluator, threshold, TopK, NMS, and selection rule
remain frozen. Compatible tensors are initialized from R0 epoch 185; the new
backbone/projection interface uses ImageNet-pretrained/new initialization.

The upstream-compatible head retains Car, Pedestrian, and Cyclist logits, but
training targets map Car/Van/Truck/Tram to Car (product Vehicle), map
Pedestrian/Person_sitting to Pedestrian, and exclude Cyclist.

## Frozen supervised loss

A1 and A2 are GT-only and use the same Hungarian set criterion:

```text
total =
    2 * sigmoid focal classification
  + 5 * 2D box L1
  + 2 * generalized IoU
  + 10 * projected 3D-center L1
  + 1 * uncertainty-aware metric depth
  + 1 * dimension-aware L1
  + 1 * orientation (12-bin cross entropy + selected-residual L1)
  + 1 * dense 80-bin depth-map loss
```

Hungarian assignment uses the matching costs class/box/GIoU/projected-center =
`2/5/2/10`. Cardinality error is logging-only. Auxiliary classification, box,
GIoU, center, depth, dimension, and angle losses supervise the first two
decoder layers; the dense depth-map loss is final-only. During training,
11 groups of 50 queries provide repeated assignments; inference uses 50
queries. A2 uses AdamW, learning rate `1e-4`, weight decay `1e-4`, batch size
16, 195 epochs, and 0.1 learning-rate decays at epochs 125 and 165.
Distillation is disabled.

## Controlled experiment sequence

1. S1, H1, and H2 are frozen negative experiments.
2. A1 GT-only is frozen at epoch 140; paired A1 distillation was completed and
   rejected because balanced/Pedestrian accuracy regressed.
3. A2 changed only Conv Small to Conv Medium and completed all 195 epochs plus
   all 39 checkpoint evaluations.
4. Freeze A2 epoch 130 as the strongest current student. It passes four of five
   gates; Vehicle moderate 3D is `15.4573` versus the `15.8713` gate.
5. Run the frozen nearby-recall and geometry diagnostic before defining A2b.
6. Change one diagnosed variable in A2b; do not blindly enlarge the model or
   repeat rejected distillation.
7. Freeze the first student passing all five accuracy and nearby-recall gates,
   then run the locked external test.
8. Compress one variable at a time and restore Core ML parity and physical
   device qualification only after accuracy qualification.

### A3 capacity escalation

A2b-A2f did not close the remaining accuracy and nearby-Pedestrian gaps, so local A2 tuning is closed. A3 is the single next accuracy experiment. It replaces only A2s `mobilenetv4_conv_medium.e500_r256_in1k` backbone and required projection tensors with `mobilenetv4_conv_large.e500_r256_in1k`. It retains the R0/A2 depth predictor, deformable transformer, query count, heads, GT-only losses, taxonomy, Chen split, 1280x384 input, optimizer schedule, decoder, and five product gates.

A3 trained from ImageNet-pretrained Conv Large plus all shape-compatible frozen R0 epoch-185 downstream tensors for 195 epochs. Selected epoch 140 reached Vehicle/Pedestrian moderate 3D AP_R40 `14.9492/7.6859` and BEV `20.7324/8.8273`; it failed the Vehicle 3D and BEV gates. MobileNetV4 capacity escalation is therefore closed. Frozen R0 epoch 185 becomes the accuracy parent for locked nearby-recall/geometry and external validation before any compression. A2 epoch 130 remains the strongest MobileNetV4 diagnostic, not an accuracy-qualified product model.

Quantization is a later optimization, not a substitute for a learnable model.
Transformer activation and memory cost must be measured separately because
weight quantization alone may not make the graph sufficiently fast.

## Locked R0 qualification

Before compression, frozen R0 epoch 185 must be evaluated without training on all 3,769 Chen validation images at score threshold `0.001`, TopK `50`, and 2D diagnostic matching IoU `0.5`. The qualification records per-class overall and nearby recall, precision/false positives, depth and 3D center error, BEV/3D IoU, dimensions, yaw and front/back flips, distance/size buckets, and detailed nearby Pedestrian failure modes. The checkpoint SHA-256 must remain `fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59`.

## Experiment discipline

- Baseline and distillation runs must share initialization, data order,
  schedule, augmentation, and evaluator.
- Do not reuse the Car-only teacher cache for Vehicle/Pedestrian training.
- Do not apply the legacy dense-student target adapter without a new explicit
  compatibility test for MonoDETR query outputs.
- No compression result may become the new accuracy denominator.
- Preserve every prior edge result as evidence; do not interpret a random-graph
  latency pass as trained-model accuracy qualification.
