# Two-class teacher/reference protocol

Status: frozen for the R0 reference run

## Purpose

R0 is the accuracy denominator for MobileADAS3D-H1 and future students. It is
not an iPhone deployment candidate. The published MonoDETR checkpoint remains useful evidence
for Car, but it is Car-only and therefore cannot supply the Pedestrian
denominator required by the product gate.

## Frozen R0 model and initialization

- Architecture: original ResNet50 MonoDETR at upstream commit
  `6994b9f512400b258c6edb75f77423beb9c126f2`.
- Input: RGB, ImageNet normalization, fixed 1280x384 affine resize, matching the
  original MonoDETR implementation.
- Initialization: published MonoDETR checkpoint. Load every shape-compatible
  tensor. The existing three-logit classifier is retained because MonoDETR's
  native IDs already contain Pedestrian and Car; Cyclist receives no targets.
- Trainable labels: native `Car` and `Pedestrian` IDs after applying the frozen
  production mapping below. All model parameters remain trainable.
- Schedule and augmentation: the published MonoDETR training recipe. Any
  deviation must create a new reference ID rather than overwrite R0.

The upstream dataset adapter must map labels before filtering and target
encoding:

| KITTI source | MonoDETR training ID | Product evaluation label |
| --- | --- | --- |
| Car, Van, Truck, Tram | Car | Vehicle |
| Pedestrian, Person_sitting | Pedestrian | Pedestrian |
| Cyclist, Misc, DontCare | excluded | excluded |

This preserves compatibility with the published three-logit checkpoint while
training on the exact production object set. Prediction text emitted as `Car`
is mapped to `Vehicle` only by the product evaluator.

## Frozen data protocol

- Dataset: KITTI 3D object detection training set only.
- Split: Chen `3,712 train / 3,769 validation` files already locked in the
  taxonomy manifest.
- Reference training never reads Chen validation labels for optimization,
  checkpoint selection, calibration, or threshold tuning.
- Train mapping count: 16,366 Vehicle and 2,263 Pedestrian objects.
- Validation mapping count: 16,895 Vehicle and 2,446 Pedestrian objects.
- The reference run must record the taxonomy, split-file, label-tree, source
  commit, config, and checkpoint hashes.

## Frozen product AP protocol

The primary comparison metric is **KITTI-difficulty product-taxonomy AP_R40**:

- difficulty filters are KITTI easy/moderate/hard height, occlusion, and
  truncation rules;
- Vehicle uses 0.70 BEV/3D IoU;
- Pedestrian uses 0.50 BEV/3D IoU;
- all mapped source labels are positive objects, not neighboring ignored
  classes;
- excluded source labels are irrelevant;
- predictions are globally score-ranked per class and AP uses 40 recall
  positions;
- all 3,769 validation prediction files are required.

This is deliberately named a product-taxonomy metric. It is **not** an official
KITTI leaderboard result because the Vehicle class merges KITTI categories.
Official KITTI Car/Pedestrian AP may be reported separately as a diagnostic but
must never replace the product denominator.

## Reference artifacts and selection

Every R0 run must export:

- resolved training config and run manifest;
- `checkpoint_best.pth` and its SHA-256;
- per-epoch training and validation losses;
- all 3,769 KITTI-format validation prediction files;
- product-taxonomy AP_R40 CSV and JSON;
- per-class easy/moderate/hard BEV and 3D AP_R40;
- nearby recall and range-error report using the same decoded predictions.

Select the R0 checkpoint by mean Vehicle/Pedestrian **moderate 3D AP_R40**.
Vehicle moderate 3D AP_R40 is the first tie-breaker, then nearby recall. The
selection rule and evaluation thresholds are fixed before training.

## Student comparison rule

H1 and R0 must be evaluated by the same product evaluator, split, mapping, IoU
thresholds, score threshold, top-k, and NMS settings. Per class, H1 must retain
at least 75% of R0 moderate 3D AP_R40. A result is not comparable if either run
is partial or if its manifest hashes differ.
