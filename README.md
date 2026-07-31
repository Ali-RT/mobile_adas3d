# MobileADAS3D

Mobile monocular-3D detector for Car, Pedestrian, and Cyclist objects. The
model consumes a `1x3x384x1280` RGB `/255.0` tensor and predicts class, 2D box,
depth, camera-space location, 3D dimensions, yaw, center offset, and depth
uncertainty on a stride-16 feature map.

## Model lineages

- **Deployed reference:** v7 with MobileNetV3-Small. This is the model already
  validated in the iPhone benchmark app and its decode contract remains frozen.
- **Active training baseline:** a fresh MobileNetV4 Conv Small model. It keeps
  the external input contract, preserves the v0/v1 eight-output path, and adds
  an optional v2 projected-center output for calibration-aware geometry.

Historical v6/v7 F1 and device results in the supporting documents describe
the deployed lineage; they are not claimed as MobileNetV4 results.

## Fresh MobileNetV4 baseline in Google Colab

Open
[`notebooks/MobileADAS3D_MobileNetV4_Colab_Baseline.ipynb`](notebooks/MobileADAS3D_MobileNetV4_Colab_Baseline.ipynb)
in Colab and run it from top to bottom with a GPU runtime. It:

- mounts Google Drive and optionally stages KITTI onto the Colab SSD with
  an archive fast path from `datasets/kitti/zips`, explicit notebook
  diagnostics, a per-folder progress bar, resumable `rsync`, file counts, and
  a completion manifest;
- installs the pinned Colab dependencies without replacing CUDA PyTorch;
- installs and validates the canonical KITTI Chen 3,712/3,769 split;
- preflights all 7,481 images, labels, calibration files, GPU, pretrained
  MobileNetV4 weights, output shapes, and one real loss;
- trains MobileNetV4 Conv Small with an effective batch size of eight;
- atomically checkpoints every epoch to Drive, automatically resumes, and
  writes live Colab training logs under `colab_logs/`;
- evaluates all 3,769 validation frames with BEV and 3D AP_R40.

The original reproducible baseline configuration is
[`configs/kitti_mnv4_conv_small_baseline.yaml`](configs/kitti_mnv4_conv_small_baseline.yaml).
The first AP-oriented follow-up configuration is
[`configs/kitti_mnv4_conv_small_ap_v1.yaml`](configs/kitti_mnv4_conv_small_ap_v1.yaml);
its run name is `mnv4_v1_long80_no_earlystop`, disables early stopping, and
saves checkpoints every five epochs for later checkpoint/AP comparison.

The calibrated-geometry follow-up configuration is
[`configs/kitti_mnv4_calibrated_geometry_v2.yaml`](configs/kitti_mnv4_calibrated_geometry_v2.yaml).
Its run name is `mnv4_v2_calibrated_geometry_quality`. It adds an optional
`projected_center_offset` head and decodes camera-frame X/Y by back-projecting
the predicted projected 3D bottom-center with KITTI `P2` plus predicted depth,
while keeping the legacy `loc_xy` head as a weak auxiliary/fallback path.

The active v3 configuration is
[`configs/kitti_mnv4_quality_scoring_v3.yaml`](configs/kitti_mnv4_quality_scoring_v3.yaml).
Its run name is `mnv4_v3_quality_scoring`. It keeps v2 geometry and adds an
optional `quality` head. The completed validation sweep found that multiplying
by the learned quality score reduced Car AP, so the selected inference default
uses `quality_score_power: 0.0` (class-only ranking) while retaining the head
and checkpoint for reproducibility.
Only results whose `kitti_r40_summary.json` contains `complete_split: true` are
reportable.

The next controlled experiment is
[`configs/kitti_mnv4_angular_yaw_v4.yaml`](configs/kitti_mnv4_angular_yaw_v4.yaml).
Its run name is `mnv4_v4_angular_yaw`. It preserves the deployed two-channel
`[sin(yaw), cos(yaw)]` contract, adds a cosine angular loss to emphasize
front/back errors, and returns to class-only scoring without the ineffective
quality head.

The active fresh-run experiment is
[`configs/kitti_mnv4_axis_direction_v5.yaml`](configs/kitti_mnv4_axis_direction_v5.yaml).
V4 improved balanced AP and BEV but worsened the yaw tail. Axis-aware
diagnostics showed that front/back selection is the dominant failure, so v5
regresses a double-angle orientation axis and separately classifies direction.
The model still emits the same final two-channel `yaw` tensor used by export
and the iPhone decoder.

## Pretrained teacher feasibility

Before adding distillation to the mobile student, run
[`notebooks/MonoDETR_Teacher_Feasibility_Colab.ipynb`](notebooks/MonoDETR_Teacher_Feasibility_Colab.ipynb).
It pins the official MonoDETR source revision, downloads the published
checkpoint, runs the exact 3,769-image Chen validation split, and evaluates its
KITTI text predictions with this repository's AP_R40 implementation. The
transfer-learning gate passes only when `complete_split` is true and Car 3D
moderate AP_R40 is at least 15%.

The canonical 2026-07-31 run passed on all 3,769 Chen validation images with
Car 3D moderate AP_R40 `20.35%` (`28.27 / 20.35 / 17.11` easy/moderate/hard)
and Car BEV moderate AP_R40 `27.35%`. The next step is a provenance-tracked
teacher prediction cache for the 3,712-image Chen training split; student
distillation should not begin until that cache is complete.

The notebook resolves KITTI from
`/content/drive/MyDrive/datasets/kitti` when `/content/kitti` has not already
been staged. It accepts both canonical `image_2`/`label_2` names and the Drive
aliases `image_02`/`label_02`, exposing them to MonoDETR through canonical
symlinks. Before compiling the pinned custom CUDA extension, it applies a
strict two-call compatibility patch for the current PyTorch `ScalarType`
dispatch API and replaces MonoDETR's removed private `_LinearWithBias` import
with public `torch.nn.Linear`. It also removes the obsolete
`torch._overrides` fallback in favor of `torch.overrides`. It clears only the
generated extension build cache and requires successful CUDA-extension and
full-model imports before inference can begin. For PyTorch 2.6 and newer, the
notebook scopes `weights_only=False` to the pinned official MonoDETR checkpoint
loader, records the downloaded file's SHA-256 digest, and validates that it
contains the expected `model_state` dictionary before the long evaluation.

External KITTI-format results can also be evaluated directly:

```bash
python scripts/evaluate_kitti_prediction_dir.py \
  --config configs/kitti_mnv4_quality_scoring_v3.yaml \
  --profile colab_drive \
  --dataset-root /content/kitti \
  --split-dir /content/drive/MyDrive/mobile_adas3d_splits/kitti_chen \
  --prediction-dir /path/to/kitti/result/data \
  --split val \
  --classes Car \
  --source-name MonoDETR_official \
  --output-dir /path/to/evaluation
```

The evaluator requires one prediction file per split image by default,
including empty files for frames with no detections. This prevents partial
teacher inference from being reported as a complete benchmark.

After a run finishes, sweep AP across saved checkpoints instead of trusting
`best.pt` by validation loss:

```bash
python scripts/sweep_kitti_r40_checkpoints.py \
  --config configs/kitti_mnv4_calibrated_geometry_v2.yaml \
  --profile colab_drive \
  --dataset-root /content/kitti \
  --split-dir /content/drive/MyDrive/mobile_adas3d_splits/kitti_chen \
  --run-dir /content/drive/MyDrive/mobile_adas3d_outputs/mnv4_conv_small_baseline/runs/<run_id> \
  --split val \
  --score-threshold 0.001 \
  --topk 300 \
  --nms-iou-threshold 0.5
```

The sweep writes `checkpoint_ap_summary.csv`,
`checkpoint_ap_metrics_long.csv`, and `checkpoint_ap_summary.json`, and reuses
completed per-checkpoint evaluations after interruptions.

For run `20260721_142002_mnv4_v2_calibrated_geometry_quality`, the sweep found
`epoch_040.pt` best for Car 3D moderate AP_R40 (`3.02`) and
`epoch_080.pt`/`latest.pt` best for mean all-class 3D moderate AP_R40 (`1.661`).

The Colab staging cell accepts Drive folders named `training/image_02` and
`training/label_02`, then stages them into the canonical `image_2` and
`label_2` names expected by the training loader.

For speed, keep KITTI archives under
`/content/drive/MyDrive/datasets/kitti/zips`; the notebook copies/extracts
those large archives locally before falling back to slower per-folder Drive
sync.

## Local verification

```bash
python -m unittest discover -s tests -v
```

The strict readiness check requires all 7,481 KITTI training files. Run it in
Colab after mounting Drive, or locally only when the complete dataset is
available:

```bash
python scripts/check_training_ready.py \
  --config configs/kitti_mnv4_conv_small_baseline.yaml \
  --profile colab_drive
```
