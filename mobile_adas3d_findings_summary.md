# MobileADAS3D Evaluation Findings Summary

**Project:** MobileADAS3D monocular ADAS 3D detector  
**Dataset:** KITTI object data, classes: Car, Pedestrian, Cyclist  
**Input size:** 1280 × 384  
**Current best model version:** `v6_stride16_fpn_ltrb_center_sampling_class_balance`  
**Recommended operating threshold:** `score_threshold = 0.55`

---

## 1. Executive summary

The refactor from the original stride-32 detector to the stride-16 FPN detector produced a major improvement.

- The old stride-32 baseline had **IoU ≥ 0.50 F1 ≈ 0.3204**.
- The new stride-16/FPN model reaches **test IoU ≥ 0.50 F1 = 0.7929** at threshold `0.55`.
- Pedestrian and Cyclist performance improved dramatically after adding stride-16 features, local `l/t/r/b` box encoding, center sampling, and class-balanced loss.
- The detector now generalizes well: test F1 is slightly higher than validation F1 at the chosen threshold.
- Remaining weaknesses are mainly **depth absolute error at long range**, **yaw/orientation outliers**, and **minority-class false positive/false negative balance**.

---

## 2. Architecture progression

### Previous baseline

Previous stride-32 baseline used MobileNetV3-Small, dense heads on a 12x40 feature map, one positive cell per object, and absolute [x1, y1, x2, y2] box regression. Best IoU>=0.50 F1 from earlier sweep was about 0.3204.

### Current best architecture

```text
Input image [B, 3, 384, 1280]
  -> MobileNetV3-Small backbone
  -> stride-16 feature [24, 80]
  -> stride-32 feature [12, 40]
  -> upsample stride-32 and fuse with stride-16
  -> dense heads at stride 16 [24, 80]
```

Current heads:

```text
cls_head              -> class confidence for Car / Pedestrian / Cyclist
box2d_head            -> local l/t/r/b normalized box distances
depth_head            -> log depth
dim_head              -> 3D dimension residual from class mean dims
yaw_head              -> sin/cos yaw
center_offset_head    -> object center offset from feature-cell center
depth_uncertainty     -> present, but loss weight still 0
```

Key changes that helped:

1. **Stride-16 FPN:** improves small-object feature resolution.
2. **Local `l/t/r/b` box encoding:** improves box localization compared with absolute `[x1, y1, x2, y2]` regression.
3. **Center sampling:** assigns multiple positive cells around object center instead of one brittle positive cell.
4. **Class-balanced loss:** increases minority-class gradient contribution for Pedestrian and Cyclist.

---

## 3. Ground-truth diagnostics

The GT analysis confirmed that stride-32 was too coarse for Pedestrian/Cyclist.

| class_name | count | count_fraction | width_px_p50 | height_px_p50 | stride32_width_cells_p50 | stride32_height_cells_p50 | stride32_frac_width_lt_1_cell | stride32_frac_width_lt_2_cells | stride16_frac_width_lt_1_cell | stride16_frac_width_lt_2_cells |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ALL | 5364 | 1.000 | 71.291 | 52.372 | 2.228 | 1.637 | 0.157 | 0.450 | 0.019 | 0.157 |
| Car | 4458 | 0.831 | 81.020 | 48.026 | 2.532 | 1.501 | 0.097 | 0.391 | 0.001 | 0.097 |
| Pedestrian | 642 | 0.120 | 32.222 | 86.205 | 1.007 | 2.694 | 0.495 | 0.766 | 0.118 | 0.495 |
| Cyclist | 264 | 0.049 | 41.682 | 61.317 | 1.303 | 1.916 | 0.345 | 0.689 | 0.080 | 0.345 |

Interpretation:

- Pedestrian median width is about **32 px**, which is roughly **1 stride-32 cell**.
- Cyclist median width is about **42 px**, roughly **1.3 stride-32 cells**.
- At stride 16, those objects occupy much healthier feature-map support.

### Cell collisions

| stride | total_objects | total_positive_cells | collided_cells | collision_cell_fraction | object_collision_fraction | Car_object_collision_fraction | Pedestrian_object_collision_fraction | Cyclist_object_collision_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16.0000 | 5364.0000 | 5315.0000 | 49.0000 | 0.0092 | 0.0183 | 0.0121 | 0.0639 | 0.0114 |
| 32.0000 | 5364.0000 | 5129.0000 | 226.0000 | 0.0441 | 0.0859 | 0.0747 | 0.1713 | 0.0682 |

Collision rate dropped from stride 32 to stride 16:

- Object collision fraction: **8.59% → 1.83%**.
- Pedestrian collision fraction: **17.13% → 6.39%**.

This strongly justified the stride-16/FPN branch.

---

## 4. 2D IoU results

### Validation: best overall thresholds at IoU ≥ 0.50

| score_threshold | iou_threshold | tp | fp | fn | precision | recall | f1 | mean_matched_iou |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5500 | 0.5000 | 4057.0000 | 899.0000 | 1307.0000 | 0.8186 | 0.7563 | 0.7862 | 0.7582 |
| 0.6000 | 0.5000 | 3895.0000 | 692.0000 | 1469.0000 | 0.8491 | 0.7261 | 0.7828 | 0.7602 |
| 0.5000 | 0.5000 | 4195.0000 | 1161.0000 | 1169.0000 | 0.7832 | 0.7821 | 0.7826 | 0.7565 |
| 0.6500 | 0.5000 | 3672.0000 | 546.0000 | 1692.0000 | 0.8706 | 0.6846 | 0.7664 | 0.7628 |
| 0.4000 | 0.5000 | 4401.0000 | 1882.0000 | 963.0000 | 0.7005 | 0.8205 | 0.7557 | 0.7530 |

### Test: best overall thresholds at IoU ≥ 0.50

| score_threshold | iou_threshold | tp | fp | fn | precision | recall | f1 | mean_matched_iou |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.5500 | 0.5000 | 3953.0000 | 875.0000 | 1190.0000 | 0.8188 | 0.7686 | 0.7929 | 0.7584 |
| 0.5000 | 0.5000 | 4089.0000 | 1103.0000 | 1054.0000 | 0.7876 | 0.7951 | 0.7913 | 0.7563 |
| 0.6000 | 0.5000 | 3778.0000 | 663.0000 | 1365.0000 | 0.8507 | 0.7346 | 0.7884 | 0.7596 |
| 0.6500 | 0.5000 | 3594.0000 | 501.0000 | 1549.0000 | 0.8777 | 0.6988 | 0.7781 | 0.7615 |
| 0.4000 | 0.5000 | 4259.0000 | 1756.0000 | 884.0000 | 0.7081 | 0.8281 | 0.7634 | 0.7530 |

### Test result at recommended threshold

At `score_threshold = 0.55`, `IoU ≥ 0.50`:

```text
Precision = 0.8188
Recall    = 0.7686
F1        = 0.7929
mIoU      = 0.7584
TP / FP / FN = 3953 / 875 / 1190
```

### Per-class test result at `score_threshold = 0.55`, `IoU ≥ 0.50`

| class_name | tp | fp | fn | precision | recall | f1 | mean_matched_iou |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Car | 3321 | 486 | 846 | 0.8723 | 0.7970 | 0.8330 | 0.7678 |
| Pedestrian | 455 | 250 | 277 | 0.6454 | 0.6216 | 0.6333 | 0.7045 |
| Cyclist | 177 | 139 | 67 | 0.5601 | 0.7254 | 0.6321 | 0.7197 |

Validation vs test at the same threshold is consistent, so there is no obvious validation overfitting.

---

## 5. 3D metrics on matched test detections

Matched true positives: **3953**

Overall matched-detection quality:

```text
2D IoU mean:          0.7584
Depth MAE:            1.876 m
Depth relative error: 7.641%
Yaw MAE:              10.63 deg
Yaw median error:     3.29 deg
Yaw p90 error:        21.97 deg
Dimension MAE:        0.143 m
Dimension rel error:  6.476%
```

### Per-class 3D metrics

| group_value | count | iou_2d_mean | depth_mae_m | depth_rel_error_mean | yaw_abs_error_mean_deg | yaw_abs_error_p50_deg | yaw_abs_error_p90_deg | dim_mae_m | dim_mean_rel_error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Car | 3321 | 0.7678 | 2.0016 | 0.0776 | 8.5630 | 2.7384 | 16.9386 | 0.1526 | 0.0593 |
| Cyclist | 177 | 0.7197 | 1.5051 | 0.0683 | 12.3884 | 4.6765 | 24.9567 | 0.0763 | 0.0643 |
| Pedestrian | 455 | 0.7045 | 1.1045 | 0.0709 | 25.0667 | 10.7868 | 68.1810 | 0.0963 | 0.1051 |

Findings:

- Car is the strongest class overall.
- Cyclist detection improved strongly, but precision remains lower than Car.
- Pedestrian yaw is weak: mean yaw error is much larger than Car/Cyclist.
- Dimension regression is already good and is not the current priority.

---

## 6. Distance and size effects

### By distance bucket

| group_value | count | iou_2d_mean | depth_mae_m | depth_rel_error_mean | yaw_abs_error_mean_deg | yaw_abs_error_p50_deg | yaw_abs_error_p90_deg | dim_mae_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 00_20m | 1717 | 0.7954 | 0.8862 | 0.0819 | 14.0919 | 3.8274 | 31.5891 | 0.1355 |
| 20_40m | 1519 | 0.7485 | 2.0116 | 0.0695 | 7.2477 | 2.8109 | 14.7510 | 0.1450 |
| 40_60m | 618 | 0.6976 | 3.6762 | 0.0766 | 9.1265 | 2.9333 | 19.7183 | 0.1550 |
| 60m_plus | 99 | 0.6473 | 5.7296 | 0.0855 | 12.0267 | 3.2898 | 35.1123 | 0.1559 |

Depth absolute error grows with distance:

```text
0–20m:   0.886 m
20–40m:  2.012 m
40–60m:  3.676 m
60m+:    5.730 m
```

But relative depth error remains around 7–9%, which is reasonable for monocular depth.

### By object size bucket

| group_value | count | iou_2d_mean | depth_mae_m | depth_rel_error_mean | yaw_abs_error_mean_deg | yaw_abs_error_p50_deg | yaw_abs_error_p90_deg | dim_mae_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| large_h_ge_96px | 1081 | 0.8015 | 0.7418 | 0.0904 | 14.9933 | 3.9281 | 35.3504 | 0.1345 |
| medium_h_32_96px | 2053 | 0.7606 | 1.7028 | 0.0686 | 9.0463 | 3.1333 | 18.0992 | 0.1429 |
| small_h_lt_32px | 819 | 0.6958 | 3.8077 | 0.0777 | 8.8597 | 2.7937 | 18.2948 | 0.1531 |

Small objects still have lower 2D IoU and higher depth MAE, but stride-16 made the detector viable.

---

## 7. Class + distance breakdown

| class_name | distance_bucket | count | iou_2d_mean | depth_mae_m | depth_abs_error_p50_m | depth_abs_error_p90_m | yaw_abs_error_mean_deg | yaw_abs_error_p50_deg | yaw_abs_error_p90_deg | dim_mae_m |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Car | 00_20m | 1268 | 0.8187 | 0.9212 | 0.7110 | 1.9415 | 10.5082 | 2.9619 | 20.4219 | 0.1499 |
| Car | 20_40m | 1376 | 0.7571 | 2.0239 | 1.6000 | 4.3127 | 6.3130 | 2.6305 | 13.8957 | 0.1514 |
| Car | 40_60m | 583 | 0.7013 | 3.7015 | 3.1717 | 7.3099 | 9.0582 | 2.6843 | 18.1178 | 0.1601 |
| Car | 60m_plus | 94 | 0.6508 | 5.7062 | 4.8771 | 11.7450 | 12.1888 | 3.2576 | 35.8978 | 0.1604 |
| Cyclist | 00_20m | 93 | 0.7665 | 0.8940 | 0.6989 | 1.9691 | 17.1996 | 5.5186 | 29.3615 | 0.0790 |
| Cyclist | 20_40m | 63 | 0.6811 | 1.7538 | 1.0700 | 4.2038 | 6.3879 | 4.2688 | 10.8000 | 0.0750 |
| Cyclist | 40_60m | 19 | 0.6354 | 3.6022 | 3.3784 | 5.9361 | 9.9789 | 5.8549 | 23.2041 | 0.0684 |
| Cyclist | 60m_plus | 2 | 0.5584 | 2.1609 | 2.1609 | 2.7052 | 0.5752 | 0.5752 | 0.7814 | 0.0630 |
| Pedestrian | 00_20m | 356 | 0.7200 | 0.7594 | 0.5418 | 1.6293 | 26.0444 | 11.4275 | 67.8498 | 0.0989 |
| Pedestrian | 20_40m | 80 | 0.6532 | 2.0026 | 1.6851 | 4.2927 | 24.0021 | 8.3219 | 84.9490 | 0.0906 |
| Pedestrian | 40_60m | 16 | 0.6364 | 2.8424 | 1.8066 | 5.6432 | 10.6033 | 9.0949 | 23.2749 | 0.0710 |
| Pedestrian | 60m_plus | 3 | 0.5970 | 8.8398 | 9.3613 | 11.2032 | 14.5838 | 12.6763 | 17.3413 | 0.0777 |

Important findings:

- Car depth error increases steadily with distance.
- Pedestrian 60m+ has very few samples, so that bucket should not drive architecture decisions.
- Pedestrian yaw is poor mainly in the closer/mid-range buckets, where visual heading ambiguity may matter.
- Cyclist yaw is much better than Pedestrian but still has outliers.

---

## 8. False positives and false negatives

### FP/FN by class

| class_name | false_positives | false_negatives |
| --- | --- | --- |
| Car | 486 | 846 |
| Cyclist | 139 | 67 |
| Pedestrian | 250 | 277 |

### False negatives by class and distance

| class_name | distance_bucket | false_negatives |
| --- | --- | --- |
| Car | 00_20m | 240 |
| Car | 20_40m | 318 |
| Car | 40_60m | 201 |
| Car | 60m_plus | 87 |
| Cyclist | 00_20m | 26 |
| Cyclist | 20_40m | 28 |
| Cyclist | 40_60m | 9 |
| Cyclist | 60m_plus | 4 |
| Pedestrian | 00_20m | 151 |
| Pedestrian | 20_40m | 99 |
| Pedestrian | 40_60m | 26 |
| Pedestrian | 60m_plus | 1 |

Notes:

- Car has the largest absolute number of false negatives because it dominates the dataset.
- Pedestrian still has a meaningful number of false negatives, especially 0–40m.
- Cyclist recall is strong, but Cyclist precision remains lower than Car.

---

## 9. Current interpretation

The detector is no longer the main bottleneck. The largest remaining technical issues are:

1. **Yaw/orientation diagnostics**
   - Pedestrian yaw mean error is high.
   - Worst yaw outliers are suspiciously close to 180°.
   - We need an axis-aware yaw metric to determine whether this is a front/back ambiguity or a true orientation failure.

2. **Long-range depth**
   - Absolute depth error grows with distance.
   - Relative error remains reasonable.
   - Depth uncertainty loss may be useful later, but it should not be the immediate next step.

3. **Class-specific thresholding**
   - Cyclist has strong recall but lower precision.
   - Class-specific thresholds may improve deployment behavior without retraining.

4. **False-positive / false-negative mining**
   - Many FPs have non-trivial class-agnostic IoU, meaning some are localization/classification edge cases rather than pure hallucinations.

---

## 10. Recommended next steps

### Immediate next step: yaw diagnostic

Add a yaw diagnostic evaluator that reports:

```text
standard yaw error = direction-sensitive error in [0, 180]
axis yaw error     = treats yaw and yaw + pi as equivalent
front/back flip rate
class-wise yaw metrics
distance-wise yaw metrics
worst yaw samples
```

Decision rules:

- If axis-aware yaw is much lower than standard yaw, the model is learning the object axis but failing front/back direction.
- If both are high, the yaw head representation/training needs improvement.
- If only Pedestrian yaw is poor, consider ignoring Pedestrian yaw for ADAS scoring or using a class-specific yaw treatment.
- If Cyclist yaw remains noisy, consider yaw bin + residual or a class-specific yaw head.

### Later improvements

1. Class-specific thresholds.
2. Depth uncertainty with a small, clamped NLL loss.
3. Yaw bin + residual head if diagnostics confirm true orientation failure.
4. False positive / false negative overlay mining.

---

## 11. Current recommended defaults

```yaml
model_version: v6_stride16_fpn_ltrb_center_sampling_class_balance
score_threshold: 0.55
match_iou_threshold: 0.50
nms_iou_threshold: 0.50
topk: 300
```

Use `best.pt` from the v6 run for all current evaluations and visualizations.
