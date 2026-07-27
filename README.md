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
optional `quality` head. At decode time, v3 ranks detections with
`class_prob * quality_prob` to test whether AP improves when score ranking is
geometry/center-quality aware.
Only results whose `kitti_r40_summary.json` contains `complete_split: true` are
reportable.

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
