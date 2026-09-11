# MonoDGP M54 Two-Class Adaptation Contract

Status: frozen for the first M54 run.

## Purpose

M54 asks one controlled question: can the verified official MonoDGP checkpoint
be adapted to the MobileADAS3D Vehicle/Pedestrian taxonomy while matching or
beating the frozen R0 two-class reference?

M54 is an accuracy-parent experiment. It is not an iPhone model, a compressed
model, a knowledge-distillation run, or a product-safety qualification.

## Immutable provenance

- Upstream repository: https://github.com/PuFanqi23/MonoDGP
- Upstream commit: aa059a18214aebf644510e7f0793971b403f9d14
- Initialization checkpoint SHA-256:
  1d5f30b34b8bef49638079a8b07f05ebf11bb5f85d6a9a11c7b028c69396f05d
- M53 must report schema version 2, reference_reproduced=true,
  two_class_adaptation_authorized=true, and no failed gate.
- Split protocol: Chen 3,712 train / 3,769 validation.
- Training split SHA-256:
  e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c
- Validation split SHA-256:
  6e2394d97c866c3af1ffb049f828abdf3d5b707d9575d16885fd0de87b72b0c8

## Frozen architecture

The graph is identical to the reproduced M53 model:

- ResNet50 backbone
- four feature levels
- 80-bin depth predictor over 0.001 to 60 m
- region-segmentation enhancement
- decoupled 2D and 3D transformer paths
- hidden dimension 256
- three encoder and three decoder layers
- eight attention heads
- 50 inference queries
- 11 grouped training-query copies
- native three-logit head ordered Pedestrian, Car, Cyclist

Cyclist remains an unused native output. It receives negative focal
supervision because M54 has no Cyclist targets and is excluded by the product
prediction parser. The output head is not resized, preserving every M53 tensor.

## Frozen taxonomy

Source KITTI labels are mapped before target encoding:

| Source label | Native training label | Product label |
| --- | --- | --- |
| Car | Car | Vehicle |
| Van | Car | Vehicle |
| Truck | Car | Vehicle |
| Tram | Car | Vehicle |
| Pedestrian | Pedestrian | Pedestrian |
| Person_sitting | Pedestrian | Pedestrian |

Cyclist, Misc, and DontCare are excluded. Both primary and mixup target paths
must apply the same mapping.

## Initialization and optimization

- Initialization: exact M53 checkpoint; no tensor is reinitialized.
- Supervision: KITTI ground truth only.
- Distillation: disabled.
- Distillation temperature: not applicable and not tuned.
- Optimizer: upstream AdamW.
- Learning rate: 5e-5.
- Batch size: 8.
- Maximum epochs: 100.
- LR decay: multiply by 0.5 at epochs 40, 70, and 90.
- Save frequency: every 5 epochs.
- Seed: 54054, chosen to remain valid under the upstream squared-seed logic.
- In-training and automatic final validation: disabled. Selection uses the
  explicit restartable sweep only.
- Sweep epochs: 5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, and 100.

Training must pass a real CUDA forward/loss/backward/optimizer-step smoke test
containing both Vehicle and Pedestrian targets. Non-finite loss or gradients
stop training immediately. Resume accepts only a readable checkpoint whose
filename epoch matches its payload and whose model and optimizer states exist.

## Checkpoint selection

All selected metrics use the complete 3,769-image validation split, score
threshold 0.001, TopK 50, and the product Vehicle/Pedestrian evaluator.

Rank checkpoints by:

1. balanced mean of Vehicle and Pedestrian moderate 3D AP_R40;
2. Pedestrian moderate 3D AP_R40;
3. Vehicle moderate 3D AP_R40;
4. balanced moderate BEV AP_R40.

Loss and native Car-only validation are not checkpoint-selection metrics.

## R0-comparable accuracy-parent gates

The selected checkpoint must match or exceed every frozen R0 value:

| Metric | Gate |
| --- | ---: |
| Vehicle moderate 3D AP_R40 | >= 17.634769 |
| Pedestrian moderate 3D AP_R40 | >= 5.721371 |
| Balanced moderate 3D AP_R40 | >= 11.678070 |
| Vehicle moderate BEV AP_R40 | >= 23.681563 |
| Pedestrian moderate BEV AP_R40 | >= 6.596149 |
| Vehicle nearby recall | >= 0.882464 |
| Pedestrian nearby recall | >= 0.683422 |
| Pedestrian nearby localization-failure rate | <= 0.246032 |
| Prediction completeness | 3,769 / 3,769 |

Failure of any row means MonoDGP M54 does not replace R0 as the accuracy parent.

## Offline product targets

- Vehicle nearby recall: at least 0.85.
- Pedestrian nearby recall: at least 0.80.

These are reported separately from the R0-comparable gates. Passing them still
does not authorize a safety claim: external-domain evaluation, device runtime,
calibration, and deployment qualification remain required.

## Required durable artifacts

- m54_adaptation_manifest.json
- m54_training_smoke.json
- per-epoch checkpoints and combined training logs
- m54_product_checkpoint_sweep.csv and the cached generic sweep artifacts
- m54_product_selection.json
- complete selected-checkpoint KITTI predictions
- product AP summary/CSV
- nearby geometry summary/CSV
- Pedestrian false-negative diagnostics

The first report to return is m54_training_smoke.json. Full training starts only
after that smoke report is complete with finite outputs and gradients.
