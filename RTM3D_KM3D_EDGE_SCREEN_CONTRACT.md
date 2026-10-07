# RTM3D/KM3D ResNet-18 edge-candidate screen

Status: **implemented; awaiting the user's Colab run**
Revision: `RTM3D-KM3D-RES18-EDGE-SCREEN-2026-10-07-r6`
Notebook: `notebooks/RTM3D_KM3D_ResNet18_Edge_Screen_Colab.ipynb`

## Why this experiment

M67 established that the selected MonoDETR A2 checkpoint has unresolved
CPU/CUDA parity in its backbone path. That gate remains failed. This experiment
does not repair or supersede M67: it screens a different, smaller monocular
3D detector before we spend more time on A2 distillation or a custom YOLO 3D
head.

RTM3D/KM3D's official repository provides a multi-class KITTI ResNet-18
checkpoint, describes a keypoint-based model, and reports 46.7 FPS on a 1080Ti.
Those are upstream claims, not iPhone results. Its official install instructions
include DCNv2 and iou3d CUDA builds; code inspection shows the generic model
factory imports optional DLA/DCN modules even when the selected backbone is
ResNet-18. The inference screen therefore instantiates the plain ResNet-18
network directly. It uses the upstream 2D keypoint/3D geometry decoder, which
hard-codes CUDA. This skips the unused training/evaluation extensions, but
does not demonstrate that the full graph is Core ML compatible.

## Frozen candidate

- Official source URL: `https://github.com/Banconxuan/RTM3D.git`
- Source commit: `888c379e79d8a6d134f06a9b7d669118679e06dc`
- Public checkpoint: `model_res18_1.pth`, Google Drive ID
  `14ww6mxtitO9aDszZN3ai8N7U1doehvi8`
- Architecture: ResNet-18 plus deconvolution feature upsampling and CenterNet
  heads; heads cover three KITTI categories (Car, Pedestrian, Cyclist), 2D
  boxes/keypoints, dimensions, orientation, confidence, and geometry inputs.
- Fixed image size: `1 x 3 x 384 x 1280`; one scale, no flip test.
- Validation smoke: the first 16 IDs, in order, from the full 3,769-image
  `chen_3712_3769` validation split. It is only a numerical/inference smoke,
  not a representative accuracy estimate.
- Output root:
  `/content/drive/MyDrive/mobile_adas3d_outputs/edge_candidates/rtm3d_km3d_res18_r1`

The checkpoint is loaded with PyTorch `weights_only=True` and a strict state
dictionary match. No model weights are changed and no optimizer is created.

## Colab checkout recovery

The first Colab smoke attempt stopped at the source-cleanliness guard before
model inference and wrote no report. The existing RTM3D directory contained
modified or untracked files; the helper's earlier error message incorrectly
called these only "tracked modifications." Notebook revision r3 preserves that
directory and selects a fresh sibling checkout (or reuses a previously clean,
pinned sibling) rather than resetting or deleting anything. This is a setup
recovery, not a model-quality or performance result.

The next run's smoke cell streams combined stdout/stderr to the notebook and a
timestamped Drive log, then includes both that log and the report in a uniquely
named ZIP. This captures the underlying exception if the script exits nonzero;
the notebook should not rely on a bare `CalledProcessError` without the child's
output. Each retry gets a fresh report/log/ZIP name so previous evidence is
preserved. The first logged inference attempt reached report serialization but
failed because NumPy comparison results (`numpy.bool_`) entered the geometry
validity flag. Revision r5 converts that flag to a native Python `bool` before
writing JSON; it does not change the model or inference outputs.

Revision r6 keeps the strict gate unchanged and adds per-invalid-candidate
diagnostics: the decoder-grid box, upstream-postprocessed pixel box, class,
score components, top-K index, and per-image counts. This distinguishes an
inverted box already produced by decoding from one introduced by the affine
post-process; it does not drop or reorder predictions to force a pass.

## Pass/fail and boundaries

The smoke passes only if all 16 source images/calibration files are processed,
all checkpoint tensors load strictly, raw heads are finite, and the selected
decoded 2D boxes and 3D dimensions/yaw/camera locations are finite and
geometrically valid. It records class counts and GPU forward/decode timing for
diagnosis. It does **not** require every class to appear among predictions on
this small sample.

Regardless of result, this stage performs no full-val AP evaluation, Core ML
conversion, quantization, phone timing, training, distillation, checkpoint
promotion, or deployment approval. A pass permits a separately reviewed Core
ML export of the neural-network heads. The calibration-dependent 3D geometry
decode must be independently reproduced and validated for the iOS runtime; a
network package alone is not a complete 3D detector.

No phone connection is needed for this Colab screen. Phone testing is only
appropriate after a Core ML package exists and Mac parity passes.
