# MobileADAS3D Roadmap

_Updated: 2026-07-19_

## Active baseline

Train a fresh model in Google Colab with:

- `mobilenetv4_conv_small.e2400_r224_in1k` pretrained backbone;
- stride-16/32 lightweight FPN and the existing eight prediction heads;
- `1280x384` RGB `/255.0` external input contract;
- canonical KITTI Chen `3,712/3,769` train/validation split;
- KITTI BEV and 3D AP_R40 as the reportable evaluation;
- Drive-backed atomic checkpoints and automatic resume.

The deployed iPhone v7/MobileNetV3 model remains a validated deployment
reference. It is not the checkpoint or architecture being trained by the new
baseline notebook.

## Run next

Open
[`notebooks/MobileADAS3D_MobileNetV4_Colab_Baseline.ipynb`](notebooks/MobileADAS3D_MobileNetV4_Colab_Baseline.ipynb)
in a Colab GPU runtime and run every cell from top to bottom. The current
follow-up run is:

```text
experiment_id: mnv4_v1_long80_no_earlystop
config: configs/kitti_mnv4_conv_small_ap_v1.yaml
v0_reference_run: 20260720_212816_baseline_mnv4_conv_small_stride16
```

The expected KITTI root in Drive is:

```text
/content/drive/MyDrive/datasets/kitti/training/image_2
/content/drive/MyDrive/datasets/kitti/training/label_2
/content/drive/MyDrive/datasets/kitti/training/calib
```

The notebook also accepts raw KITTI folder aliases
`training/image_02` and `training/label_02` in Drive, then stages them as the
canonical `/content/kitti/training/image_2` and
`/content/kitti/training/label_2` folders expected by the training loader.

The notebook first tries the faster archive path from
`/content/drive/MyDrive/datasets/kitti/zips`, then falls back to folder `rsync`.
It stages files to `/content/kitti` with explicit path diagnostics, a
per-folder progress bar, resumable copy behavior, and
`/content/kitti/.mobileadas3d_stage_manifest.json`; rerun the staging cell if
Colab disconnects during staging.

The run is complete only when Drive contains both:

```text
mobile_adas3d_outputs/mnv4_conv_small_baseline/runs/<run>/checkpoints/best.pt
mobile_adas3d_outputs/mnv4_conv_small_baseline/runs/<run>/kitti_r40_val/kitti_r40_summary.json
```

and the summary reports `complete_split: true` for all 3,769 validation frames.
If training fails, inspect the matching
`mobile_adas3d_outputs/mnv4_conv_small_baseline/colab_logs/train_mnv4_v1_long80_no_earlystop_*.log`;
the notebook also prints the last log lines automatically.

## After the baseline

1. Record AP3D/BEV R40, 2D AP, model size, and latency without changing the
   baseline configuration.
2. Diagnose per-class depth, yaw, and localization failures.
3. Add geometry-safe augmentation and calibration propagation.
4. Compare EMA and validation-AP3D checkpoint selection.
5. Run controlled mobile ablations such as width, FPN channels, stride-8
   fusion, FastViT, or MobileNetV4 variants.
6. Export only a selected Pareto candidate to Core ML and verify numerical
   parity plus real-iPhone latency.

Do not compare experiments that use different splits, incomplete validation
runs, or legacy threshold-sweep F1 as if they were the same benchmark.
