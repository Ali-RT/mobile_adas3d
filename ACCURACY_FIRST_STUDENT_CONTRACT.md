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

## Student A1 architecture

The first accuracy-first candidate is a teacher-compatible MonoDETR student:

```text
RGB [1, 3, 384, 1280]
  -> MobileNetV4 Conv Small backbone
  -> MonoDETR multi-scale feature projections
  -> LID depth predictor and depth positional encoding
  -> 3-layer depth-aware deformable encoder
  -> 3-layer, 50-query decoder
  -> Vehicle/Pedestrian 2D, depth, dimension, position, yaw, and confidence heads
```

The experiment changes only the ResNet50 backbone to MobileNetV4. Resolution,
taxonomy, split, augmentation, transformer, queries, heads, decoder, evaluator,
threshold, TopK, NMS, and selection rule remain frozen. Compatible tensors are
initialized from R0 epoch 185; the new backbone/projection interface uses its
defined pretrained/new initialization.

## Controlled experiment sequence

1. Freeze S1, H1, and H2 as negative experiments; do not resume them.
2. Train A1 GT-only with durable Google Drive checkpoints, verbose epoch/batch
   logs, exact-run automatic resume, and the frozen product evaluator.
3. Freeze the A1 initialization and baseline result.
4. Generate a two-class R0 train cache or implement an equivalent deterministic
   teacher path. The existing Car-only cache is not valid for this experiment.
5. Train one paired distilled A1 run from the identical initialization and
   schedule. Keep distillation only if it improves the frozen comparison
   without reducing Pedestrian or nearby recall.
6. Freeze the best accuracy-qualified student and run the locked external test.
7. Compress one variable at a time: FP16, INT8 weights, activation QAT,
   structured pruning, encoder/decoder depth, width, feature/token reduction,
   then low-rank substitutions where justified.
8. Measure each compressed candidate against the frozen accurate student, then
   restore Core ML parity and physical-device qualification for the selected
   deployment target.

Quantization is a later optimization, not a substitute for a learnable model.
Transformer activation and memory cost must be measured separately because
weight quantization alone may not make the graph sufficiently fast.

## Experiment discipline

- Baseline and distillation runs must share initialization, data order,
  schedule, augmentation, and evaluator.
- Do not reuse the Car-only teacher cache for Vehicle/Pedestrian training.
- Do not apply the legacy dense-student target adapter without a new explicit
  compatibility test for MonoDETR query outputs.
- No compression result may become the new accuracy denominator.
- Preserve every prior edge result as evidence; do not interpret a random-graph
  latency pass as trained-model accuracy qualification.
