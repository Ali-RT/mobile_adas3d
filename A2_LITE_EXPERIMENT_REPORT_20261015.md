# A2 student experiments and deployment status

Updated 15 October 2026. This report covers the MonoDETR/A2 development path,
compact-student screening, A2-Lite recovery and distillation experiments.
MonoDGP is parked background, not the selected teacher or phone model.

## Summary for reporting

Our goal is a model that labels Vehicle and Pedestrian objects in 2D and 3D
from one camera and runs near real time on an iPhone. We worked backward from
device speed, reduced the A2 architecture, and measured the resulting accuracy
loss. Label-based training recovered useful accuracy. Several ways of adding
teacher guidance improved individual measurements but did not preserve both
classes well enough to select a distilled checkpoint.

The current next candidate is the **full-data label-trained A2-Lite epoch-10
checkpoint**, not the most recent KD endpoint. Its moderate Vehicle/Pedestrian
3D AP_R40 is **3.9591 / 2.1411**, still well below the product targets. We will
check its ONNX accuracy and then measure those exact trained weights on the
phone. Earlier speed screens do not qualify this trained checkpoint.

The protected feature-KD experiment is closed. It obeyed its gradient rules
but substantially reduced Pedestrian 3D accuracy. No further weight sweep or
longer run of that recipe is planned.

## Models and data

| Property | Original A2 used as teacher | A2-Lite student |
| --- | --- | --- |
| Backbone | MobileNetV4 Conv Medium | Same backbone family and feature widths |
| Input | 1280 by 384 | 640 by 192 in the recovery/KD runs |
| Visual transformer encoder | 3 layers | First layer retained |
| Detection decoder | 3 layers | First layer retained with its corresponding heads |
| Queries and hidden width | 50 queries, width 256 | 50 queries, width 256 |
| Depth predictor | Original A2 depth module | Unchanged architecture |
| Outputs | Classification, 2D box/projected center, metric dimensions, depth and uncertainty, orientation | Same output types |
| Role | Offline training reference | Candidate edge model |

The Lite graph has 14,464,792 parameters. Pruning retains weights from A2 but
does not by itself make the resulting graph accurate. Both models emit three
classification logits per query; product decoding uses Vehicle and Pedestrian
and suppresses the unused channel.

KITTI uses the frozen Chen split: **3,712 TRAIN and 3,769 VAL images**. Vehicle
merges Car, Van, Truck and Tram. Pedestrian includes Pedestrian and
Person_sitting. Object counts are not balanced. The 512-frame pilots use 256
Pedestrian-containing frames plus 256 Vehicle-only frames selected from TRAIN.
That balances frame groups, not the number of annotated objects. All teacher
targets and gradient-audit samples come from TRAIN. VAL is used for evaluation,
not teacher-target selection or checkpoint selection in the Lite pilots.

## Benchmarks and definitions

AP_R40 measures ranked detection precision across 40 recall positions. These
values are AP percentage points, not the percentage of images reconstructed
perfectly. Product 3D/BEV overlap thresholds are 0.7 for Vehicle and 0.5 for
Pedestrian. Tables below report moderate difficulty on all 3,769 VAL images.

| Frozen product target | Value |
| --- | ---: |
| Vehicle 3D AP_R40 | 15.8713 |
| Pedestrian 3D AP_R40 | 5.1493 |
| Mean 3D AP_R40 | 10.5103 |
| Vehicle BEV AP_R40 | 21.3134 |
| Pedestrian BEV AP_R40 | 5.9365 |
| Nearby Vehicle recall | 0.85 |
| Nearby Pedestrian recall | 0.80 |
| Model inference p95 | At most 50 ms |
| Capture to decoded result p95 | At most 100 ms |
| Sustained processed rate | At least 10 FPS |

The accuracy thresholds retain approximately 90% of the frozen R0 reference,
as defined in [the accuracy-first contract](ACCURACY_FIRST_STUDENT_CONTRACT.md).
Nearby recall is a separate **2D** matching check: score at least 0.001 and 2D
IoU at least 0.5, within the evaluator's near-field distances (Vehicle below
40 m, Pedestrian below 30 m). Passing it does not establish correct depth or
3D boxes. ONNX CPU testing does not require Core ML or the Neural Engine.

## Earlier MonoDETR experiments

The detailed historical evidence remains in [the project tracker](PROJECT_TRACKER.md)
and the linked M63–M68 contracts. The main decisions are:

| Experiment family | Finding | Decision |
| --- | --- | --- |
| Original A2 GT training | Selected epoch 130: Vehicle/Pedestrian 3D AP 15.4573/7.5328; Vehicle BEV 21.3750, Pedestrian BEV 8.4892. Vehicle 3D missed its 15.8713 gate; nearby Pedestrian recall was about 0.692, below 0.80. | Retain as an accuracy reference, not a qualified edge model. |
| M63 depth KD and continuation | Both continuation arms lost original-A2 accuracy. KD ended at Vehicle/Pedestrian 3D 14.8930/6.3105. | Do not promote or extend that recipe. |
| M63b–e continuation and BatchNorm diagnosis | Changing normalization statistics explained part, but not all, of the loss. Frozen BatchNorm and lower learning rate did not preserve both classes. | Freeze normalization in subsequent bounded pilots; do not claim BatchNorm was the sole cause. |
| M63f–h class tradeoff and reproducibility | A later KD arm gained Vehicle over its matched control but lost Pedestrian performance. Runtime-recovery diagnostics encountered software/driver failures. | Preserve completed evidence; avoid making historical-runtime repair the main research task. |
| M64 teacher qualification | MonoPRIO scored higher AP than A2, but released prior-construction provenance remained unresolved. | Do not use that release for external-teacher KD until its construction is verified. This is not proof of data leakage. |
| M65 and M65c preservation | No external teacher was used. Continuation with an original-A2 preservation term still failed Pedestrian preservation. M65b measured small/conflicting local gradient components. | Reject the continuation recipes, not the concept of distillation. |
| M66 R0-to-A2 Vehicle feature KD | Vehicle 3D gain over matched control was only 0.000194 AP points, below the required gain. | No checkpoint promotion or longer run. |
| M67 and M68 A2 portability | M67 failed CPU-versus-CUDA raw parity. A separate ONNX CPU path and phone screens followed. | Keep failed M67 evidence visible. Do not describe successful export or an app launch as real-time qualification. |
| RTM3D/KM3D and compact CNN screens | Separate candidate/export investigations; the earlier compact CNN baseline in the supplied KD comparison had Vehicle/Pedestrian 3D AP 2.8903/0.7810. Its KD endpoint reached 2.6283/0.8799. | No qualified replacement selected. These models are not A2-Lite and are not the current teacher. |

Historical A2 re-evaluations differ slightly by run (for example 15.4475/7.5284
in M64). Do not attribute those small cross-run differences to a new method.

## A2-Lite experiments

### Architecture and resolution screens

We first kept A2's backbone and depth predictor while retaining one visual
encoder layer and one decoder layer. We exported the graph to ONNX and tested
lower resolutions. Earlier phone measurements suggested roughly 20 model-only
FPS at 640 by 192, but used the **pruned initialization**, not recovered weights
and not a live camera-to-label pipeline. This is historical speed-screen
context, not a qualified accuracy/speed operating point.

Full validation then found **zero moderate 3D and BEV AP** for the unrecovered
pruned graph at both original resolution and 640 by 192. At 640, nearby
Vehicle/Pedestrian 2D recall was 0.2626/0.0128. We therefore ran recovery rather
than presenting the fast initialization as a useful deployed detector.

### Recovery and teaching results

GT means ground-truth labels only. KD adds a teacher loss to the unchanged GT
loss. All rows evaluate the fixed endpoint on the same complete VAL split.
Five-epoch Lite pilots use 512 TRAIN frames and 2,560 optimizer updates per
arm. Full recovery uses all 3,712 TRAIN frames and 10 epochs, or 37,120 updates
per arm. Tiny-fit updates are diagnostics and never initialize a main arm.

| Experiment and arm | TRAIN frames / epochs | Vehicle 3D | Pedestrian 3D | Mean 3D | Vehicle BEV | Pedestrian BEV | Nearby V / P |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Recovery label-only control | 512 / 5 | 0.6644 | 0.9932 | 0.8288 | 2.1861 | 1.6267 | 0.7863 / 0.5855 |
| Selective all-head prediction KD | 512 / 5 | 1.1740 | 1.1842 | 1.1791 | 3.5660 | 1.8107 | 0.7879 / 0.5185 |
| Full-data label-only control | 3712 / 10 | **3.9591** | **2.1411** | **3.0501** | **8.0852** | **2.9356** | **0.8338 / 0.5604** |
| Full-data all-head prediction KD | 3712 / 10 | 4.0104 | 1.3816 | 2.6960 | 7.8790 | 1.7410 | 0.7968 / 0.5062 |
| Geometry-only prediction KD | 512 / 5 | 0.8352 | 0.2645 | 0.5499 | 2.5947 | 0.5435 | 0.8034 / 0.4894 |
| Geometry KD with angle multiplier 0.01 | 512 / 5 | 1.2557 | 0.5182 | 0.8870 | 3.8277 | 0.8765 | 0.8020 / 0.5511 |
| Vehicle-only geometry KD | 512 / 5 | 1.8102 | 0.7537 | 1.2820 | 4.7399 | 1.4508 | 0.7742 / 0.5260 |
| Fresh control for feature KD | 512 / 5 | 1.2775 | 0.9763 | 1.1269 | 3.4493 | 1.5122 | 0.7873 / 0.5697 |
| Foreground feature KD | 512 / 5 | 1.4203 | 0.5922 | 1.0063 | 3.6585 | 0.8564 | 0.8169 / 0.5564 |
| Fresh control for protected feature KD | 512 / 5 | 1.6711 | 1.5070 | 1.5890 | 4.6483 | 1.9487 | 0.7955 / 0.5194 |
| Protected foreground feature KD | 512 / 5 | 1.6660 | 0.0757 | 0.8708 | 4.0774 | 0.1056 | 0.8113 / 0.5613 |

The small all-head pilot improved AP but lost 0.0670 Pedestrian nearby recall
versus its control. Full-data KD lost Pedestrian 3D AP and both nearby recalls.
Geometry-only, angle-rebalanced and Vehicle-only guidance each failed the
preservation/benefit rules. Removing Pedestrian teacher targets did not isolate
the Pedestrian branch: shared parameters could still change it.

The geometry/angle/Vehicle-only screens reused reviewed historical controls,
with matching initial weights, subset, budget and recorded CPU environment.
They were not fresh simultaneous paired repeats. Feature and protected-feature
experiments each trained their own fresh control. **Compare methods within
their stated control pair**, not by treating different control runs as identical.
All Lite teaching results are single-seed evidence.

### Losses and training settings

The supervised objective remains:

`2 classification + 10 projected-center + 5 box-size + 2 GIoU + 1 depth + 1 dimensions + 1 orientation`.

Classification uses sigmoid focal loss (alpha 0.25, gamma 2). Projected center
and box extents use L1 losses. GIoU penalizes 2D box mismatch. Depth uses an
uncertainty-weighted absolute error plus its log-uncertainty term. Dimension
loss uses relative absolute error with detached scale compensation. Orientation
uses a 12-bin classification loss plus the selected bin's residual L1 loss.
Hungarian matching costs are classification 2, projected center 10, box 5 and
GIoU 2. See [the exact loss implementation](models/a2_lite_recovery_loss.py).

The pilots use batch size 1, seed 123, AdamW learning rates 0.0001 for the main
model and 0.00001 for the trainable backbone, weight decay 0.0001, gradient
clipping at norm 1, frozen BatchNorm, no dropout, no augmentation and no AMP.
They use the single deployed 50-query path, not training query groups or an
extra dense depth-map loss.

Prediction KD matches teacher and student objects through GT, never by assuming
their query numbers correspond. Teacher class confidence must be at least 0.1
and 2D overlap at least 0.5, with positive geometry. A component is taught only
when the teacher is at least 10% closer to GT than the current student on that
component. Present-class means receive equal weight; GT class weights do not
change. Classification and angle-bin teaching use temperature 2. The KD ceiling
is 0.25 with a 128-update warmup. Geometry-only removes teacher classification
and box guidance, but keeps their supervised label losses.

Feature KD instead matches internal visual-encoder features at four scales,
sampling a 3 by 3 grid inside each eligible GT box. Normalized cosine distance
is averaged across points, levels, objects and present classes. No adapter or
background feature loss is used. This is not direct teaching of depth or box
outputs. The measured scaling produced a fully ramped coefficient of 7.308938.

### Gradient audits and protected feature KD

The first 32-frame geometry-gradient audit performed zero updates. Angle
guidance was much larger than the other teacher components in several groups;
small or conflicting local gradients were also present. Scaling angle to 0.01
was a tested hypothesis, not proof that angle alone caused the AP loss.

The later feature audit inspected the same 32 TRAIN frames at initialization,
label-only endpoint and feature-KD endpoint: 96 frame-state probes, zero
updates. The encoder teacher/GT norm ratio, calibrated at about 5% initially,
was about 17.4% at the control endpoint and 12.6% at the feature endpoint.
Negative class/GT alignment existed. These local measurements motivated the
protected pilot but did not predict its final accuracy.

Protected KD measured total-GT, Pedestrian-GT, Vehicle-GT and teacher gradients
on every update. It rejected conflicting guidance and capped the teacher norm
at 5% of total GT within every checked parameter group before clipping/AdamW.
Of 2,560 updates, **651 accepted guidance, 1,909 rejected it for conflict, and
624 accepted updates were reduced by the cap**. There were **zero cap
violations**. This safeguard did not constrain the final AdamW update or
guarantee validation AP. Its accuracy gate failed sharply.

### Same-object Pedestrian diagnosis

The saved GT fingerprint was frame, class and exact GT depth. Fingerprints
had to be unique across matches plus misses in both runs, then agree on GT
yaw and size bucket. Fourteen ambiguous GT records were excluded. This yielded
948 Pedestrians detected by both models, including 947 near-field objects.
All 948 selected predictions were also recovered uniquely from raw serialized
outputs using score, yaw and depth-error agreement.

| Same 948 objects | Label-only | Protected KD |
| --- | ---: | ---: |
| Average 2D overlap | 0.6707 | 0.6737 |
| Average 3D overlap | 0.1867 | 0.1255 |
| Median absolute depth error | 0.56 m | 0.65 m |
| Average absolute depth error | 1.0628 m | 1.0033 m |
| Average dimension error | 0.1505 m | 0.1720 m |
| Selected boxes with 3D IoU at least 0.5 | 122 | 37 |

KD lost 116 formerly good 3D matches, retained 6 and gained 31. For those 116
lost matches, depth MAE rose **0.0776 to 0.4007 m**, mean predicted length fell
**0.9900 to 0.7872 m**, and signed depth bias rose **+0.0119 to +0.3809 m**.
Their average 2D overlap stayed almost unchanged (0.7377 to 0.7387).
Frame 005017 illustrates the failure: GT depth 8.43 m, control 8.47 m, KD
8.99 m; 2D overlap improved 0.8366 to 0.9080 while 3D overlap fell 0.7258 to zero.

Overall mean depth error alone hid damage to already accurate objects.
Orientation was not uniformly worse, and confidence ranking also weakened.
These are descriptive comparisons of selected detections, not a causal
per-head intervention or a re-computation of official AP. They establish a
real geometry regression, not just a different matched-object population.

## What we learned

1. A smaller graph can be faster and still be unusable before recovery.
2. More training data improved label-only recovery, but did not establish a
   successful full-data KD recipe.
3. Better 2D recall and lower average depth error do not guarantee better 3D AP.
4. Class-balanced sampling and class-averaged KD do not isolate shared features.
5. Technically enforced gradient safeguards are not accuracy guarantees.
6. A teacher's higher benchmark score does not mean every internal feature or
   output should be copied by a smaller student.
7. Inference speed, export preservation and product accuracy are separate gates.

## Current selection and next action

Select the full-data **GT-only control epoch 10** as the next deployment
feasibility candidate. This is a fixed completed endpoint, not a new validation
checkpoint sweep. Among reviewed full-data GT-only A2-Lite results it has the
strongest balanced accuracy; the KD arm's slightly higher Vehicle 3D AP does
not offset its Pedestrian losses.

The checkpoint is `control/last.pth` under the completed Drive run
`edge_candidates/a2_lite_full_recovery/a2_lite_640_full_recovery_kd_r1`.
Its SHA-256 is
`0cd8dd0d0c2e1e6acb5980806084264fa83eb588b1bde763746b14d47e176930`;
training signature is
`b5239955208c794cd3f0ab9bc5564b2d656a9fcec1397212a33cfb0a94890cab`.

Run [the trained ONNX notebook](notebooks/MonoDETR_A2_Lite_Trained_ONNX_Colab.ipynb)
in a CPU Colab session. It strictly loads those weights, restores a private
portable source checkout, exports FP32 ONNX and checks 16 fixed inputs. It
then stages only VAL with four I/O workers and resumes hash-bound per-image
inference. The five moderate AP values must stay within 0.05 AP points and
nearby recall within 0.005 of the reviewed native result, in either direction.
These are new export-preservation checks, not relaxed product gates or proof
of native-CUDA raw-tensor parity.

Return `a2_lite_trained_onnx_results.zip`. Host review precedes adapting the
existing phone harness to the new trained model hash. No phone connection is
needed until that package passes review. Measure model-only and capture-to-label
latency separately, memory and thermal behavior, without replacing the legacy
application or original model resources. No new training or automatic KD sweep
is authorized by this workflow.

## Evidence and reporting rules

The result archive inventory, exact hashes, full-precision metric rows and
selected checkpoint are recorded in
[`artifacts/a2_lite_experiment_review_20261015.json`](artifacts/a2_lite_experiment_review_20261015.json).
Original ZIPs remain read-only; checkpoints and feature tensors deliberately
remain on Drive. Reproduce a run with its notebook, signed manifest, pinned
source/implementation inventory, source-data hashes and declared budget.

Historical controls reused by later ablations are labelled as such. Cross-run
control differences have not been causally explained and do not prove a KD
gain. No A2-Lite checkpoint is currently claimed to meet both product accuracy
and end-to-end iPhone requirements. The new trained export and full validation
are prepared, not yet executed on the Drive checkpoint.

Local workflow verification passed 278 A2-Lite tests and compilation of all
five notebook code cells. A clean main checkout plus only the intended changes
also passed all 265 available A2-Lite tests, excluding 13 pre-existing local
phone tests outside this publication. A real portable E1/D1 graph with synthetic weights
also exported without custom operators and passed all 16 raw CPU comparisons,
with unchanged model state. That integration check used local Python 3.12 and
Torch 2.12, not the pinned Colab runtime or the selected trained weights. See
[`artifacts/a2_lite_trained_export_preflight_20261015.json`](artifacts/a2_lite_trained_export_preflight_20261015.json).
