# MobileADAS3D TODO

## Current implementation status

The current target builder is intentionally simple for the first MVP.

Current assignment strategy:

- Each object is assigned to the feature-map cell containing the resized 2D bounding-box center.
- Classification target is one-hot at that cell.
- 2D box target is stored as absolute resized image coordinates.
- If multiple objects land in the same cell, the closer object is kept.

This is acceptable for the initial training/debugging pipeline, but it should be improved before serious model training.

---

## Target Builder Improvements

### 1. Use projected 3D center instead of 2D box center

Current:

```text
object cell = center of 2D bounding box


# MobileADAS3D TODO

## Target Builder Improvements

1. Use projected 3D center instead of 2D box center.
2. Use Gaussian heatmap instead of one-hot cell.
3. Add multi-scale feature maps.
4. Add better collision handling.
5. Add normalized box encoding instead of absolute pixel box.

## Notes

The current target builder is intentionally simple for MVP debugging. It assigns each object to the feature-map cell containing the resized 2D bounding-box center. This is acceptable for initial pipeline validation, but should be improved before serious model training.

Recommended improvement order:

1. Gaussian heatmap
2. Normalized box encoding
3. Projected 3D center assignment
4. Better collision handling
5. Multi-scale feature maps


## Colab Training Plan

Local Mac is only for repository development, parser testing, visualization, and small debug runs.

Real training should run in Google Colab with GPU.

KITTI should not be stored on the local Mac because of disk-space limits. The real dataset should be stored in Google Drive:

```text
/content/drive/MyDrive/datasets/kitti

Expected KITTI Drive structure:

/content/drive/MyDrive/datasets/kitti/training/image_2
/content/drive/MyDrive/datasets/kitti/training/label_2
/content/drive/MyDrive/datasets/kitti/training/calib



---

## KITTI Download Plan for Colab

Real KITTI data should be downloaded/extracted in Google Colab directly into Google Drive.

Target location:

```text
/content/drive/MyDrive/datasets/kitti

Required for current RGB-only MobileADAS3D:

data_object_image_2.zip
data_object_label_2.zip
data_object_calib.zip

Expected extracted structure:

/content/drive/MyDrive/datasets/kitti/training/image_2
/content/drive/MyDrive/datasets/kitti/training/label_2
/content/drive/MyDrive/datasets/kitti/training/calib

Optional later for sparse-depth / LiDAR supervision:

data_object_velodyne.zip