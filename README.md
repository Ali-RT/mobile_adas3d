# MobileADAS3D

Mobile monocular-3D detector for Car, Pedestrian, and Cyclist objects. The
model consumes a `1x3x384x1280` RGB `/255.0` tensor and predicts class, 2D box,
depth, camera-space location, 3D dimensions, yaw, center offset, and depth
uncertainty on a stride-16 feature map.

## Model lineages

- **Deployed reference:** v7 with MobileNetV3-Small. This is the model already
  validated in the iPhone benchmark app and its decode contract remains frozen.
- **Active training baseline:** a fresh MobileNetV4 Conv Small model. It keeps
  the external input and eight-output contracts but has no inherited
  MobileADAS3D checkpoint and must earn a new KITTI AP_R40 baseline.

Historical v6/v7 F1 and device results in the supporting documents describe
the deployed lineage; they are not claimed as MobileNetV4 results.

## Fresh MobileNetV4 baseline in Google Colab

Open
[`notebooks/MobileADAS3D_MobileNetV4_Colab_Baseline.ipynb`](notebooks/MobileADAS3D_MobileNetV4_Colab_Baseline.ipynb)
in Colab and run it from top to bottom with a GPU runtime. It:

- mounts Google Drive and optionally stages KITTI onto the Colab SSD with
  explicit notebook diagnostics, a per-folder progress bar, resumable `rsync`,
  file counts, and a completion manifest;
- installs the pinned Colab dependencies without replacing CUDA PyTorch;
- installs and validates the canonical KITTI Chen 3,712/3,769 split;
- preflights all 7,481 images, labels, calibration files, GPU, pretrained
  MobileNetV4 weights, output shapes, and one real loss;
- trains MobileNetV4 Conv Small with an effective batch size of eight;
- atomically checkpoints every epoch to Drive and automatically resumes;
- evaluates all 3,769 validation frames with BEV and 3D AP_R40.

The reproducible baseline configuration is
[`configs/kitti_mnv4_conv_small_baseline.yaml`](configs/kitti_mnv4_conv_small_baseline.yaml).
Only results whose `kitti_r40_summary.json` contains `complete_split: true` are
reportable.

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
