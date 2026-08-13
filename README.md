# MobileADAS3D

Mobile monocular-3D detector for safety-relevant road objects. The locked
product target, class taxonomy, dataset isolation rules, candidate architecture,
and go/no-go metrics are defined in
[`PRODUCT_MODEL_CONTRACT.md`](PRODUCT_MODEL_CONTRACT.md). The legacy custom
model supports Car, Pedestrian, and Cyclist objects. The
model consumes a `1x3x384x1280` RGB `/255.0` tensor and predicts class, 2D box,
depth, camera-space location, 3D dimensions, yaw, center offset, and depth
uncertainty on a stride-16 feature map.

## Model lineages

- **Deployed reference:** v7 with MobileNetV3-Small. This is the model already
  validated in the iPhone benchmark app and its decode contract remains frozen.
- **Legacy custom student:** MobileNetV4 Conv Small with convolutional heads.
  It remains Core-ML-exportable but did not meet the 3D-accuracy target.
- **Locked product candidate:** MobileMonoDETR-VP1, using MobileNetV4 Conv Small
  with the unchanged MonoDETR depth-aware transformer and two production
  classes. Its highest-risk deformable-attention microkernel now passes native
  Core ML conversion at decoder and encoder scale, and the complete fixed-shape
  random-weight graph converts to a custom-op-free ML Program. Native package
  compilation/loading remains unresolved, so deployment approval is still
  conditional on that gate, trained-model parity, and physical-iPhone tests. See
  [`COREML_FEASIBILITY_REPORT.md`](COREML_FEASIBILITY_REPORT.md).

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

The same notebook now contains a separate **Teacher Task 2** section for that
cache. It runs MonoDETR with an isolated `monodetr_train_cache_clean` output
name and exposes the Chen train IDs through a `val`-named inference view. This
is required because MonoDETR automatically enables random augmentation for a
split literally named `train`. The cache validator rejects `train` or
`trainval` inference configurations. The notebook
constructs its train configuration directly after the common setup cells; it
does not require running the earlier validation-configuration cell. It then
uses `scripts/create_teacher_prediction_cache.py` to require an exact split
match, parse every scored KITTI file, copy predictions to Drive, verify a
deterministic prediction-tree checksum, and write
`teacher_cache_manifest.json` with `complete: true` only after validation.
The intended Drive destination is:

```text
/content/drive/MyDrive/mobile_adas3d_outputs/teachers/monodetr/chen_train_clean_20260804/
  teacher_cache_manifest.json
  runtime_config.yaml
  predictions/               # exactly 3,712 .txt files, including empty files
```

The earlier `chen_train_20260731` cache must not be used. Although it contains
3,712 files, its runtime configuration used `test_split: train`, causing
random inference augmentation and a collapsed Car 3D moderate AP_R40 of
`5.78%`. The corrected notebook marks that manifest incomplete and writes a
`DO_NOT_USE_AUGMENTED_CACHE.txt` warning.

The corrected `chen_train_clean_20260804` cache completed all 3,712 images with
augmentation disabled. Its prediction-tree SHA-256 is
`e0d155c8e93bc603cea7824260b4d59e26f96cd4c40bf96040f4785c99529608`,
and the official clean-train Car 3D AP_R40 was
`94.77 / 78.87 / 74.10`. This is an in-sample result, not a generalization
benchmark. Before adding a student distillation loss, the next task is to audit
teacher-score thresholds and one-to-one teacher/ground-truth matching; the
cache has 109,947 detections at the intentionally low `0.001` score floor.

Teacher Task 3 is implemented in the final notebook section and by
`scripts/audit_teacher_prediction_cache.py`. It verifies the clean cache
digest, sweeps scores `0.001` through `0.9`, and performs deterministic greedy
one-to-one Car matching by descending teacher score and 2D IoU ≥0.5. It writes
threshold precision/recall/F1, distance coverage, selected matched geometry,
and a JSON report containing a maximum-F1 recommendation plus an explicit 95%
recall target result. If no threshold reaches 95%, the report marks
`target_met: false` instead of mislabeling the maximum-recall fallback. The
selected-match CSV uses the maximum-F1 threshold. It does not modify student
training.

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

## Two-class R0 reference training

Run
[`notebooks/MonoDETR_R0_Two_Class_Reference_Colab.ipynb`](notebooks/MonoDETR_R0_Two_Class_Reference_Colab.ipynb)
from top to bottom in a GPU Colab runtime. Its frozen defaults are ResNet50
MonoDETR, 195 epochs, batch size 16, checkpointing every 5 epochs, the Chen
3,712/3,769 split, and the published MonoDETR checkpoint initialization. The
upstream adapter maps Car/Van/Truck/Tram to MonoDETR's native Car ID and maps
Pedestrian/Person_sitting to its native Pedestrian ID before filtering and
target encoding. The three-logit checkpoint-compatible classification head is
retained; Cyclist receives no training targets.

The durable output root is
`/content/drive/MyDrive/mobile_adas3d_outputs/references/monodetr_r0`. Do not
adopt upstream `checkpoint_best.pth` as the final R0 automatically because that
selection is Car-only. The next gate sweeps saved checkpoints with the frozen
Vehicle/Pedestrian product evaluator and selects mean moderate 3D AP_R40 as
defined in [`TWO_CLASS_REFERENCE_PROTOCOL.md`](TWO_CLASS_REFERENCE_PROTOCOL.md).
Before training, the notebook scans Drive for `checkpoint_epoch_*.pth`, rejects
partial or inconsistent files, and resumes the highest epoch containing both
model and optimizer state. Re-run the setup, resume-detection, and training
cells after an interruption; at most the work since the previous five-epoch
checkpoint is lost.
The training subprocess streams combined stdout/stderr to the cell and to a
timestamped durable file under `references/monodetr_r0/colab_logs`. On failure,
the cell prints its last 120 captured lines, return code, GPU state, disk state,
and discovered MonoDETR logs before raising an error that includes the Drive
log path.

After epoch 195, run the notebook's **Product-taxonomy checkpoint sweep** cell.
It evaluates all saved five-epoch checkpoints, caches each completed 3,769-image
result, and ranks checkpoints by mean Vehicle/Pedestrian moderate 3D AP_R40,
with Vehicle moderate 3D and mean moderate BEV as tie-breakers. Durable outputs
are `product_checkpoint_sweep/r0_product_checkpoint_sweep.csv` and
`r0_product_selection.json`; only the selected checkpoint is SHA-256 hashed.

## MonoDETR MobileNetV4 backbone ablation

[`notebooks/MonoDETR_MobileNetV4_Backbone_Ablation_Colab.ipynb`](notebooks/MonoDETR_MobileNetV4_Backbone_Ablation_Colab.ipynb)
is the next model-compression experiment. It pins the same validated MonoDETR
revision and changes only its ResNet50 backbone to timm MobileNetV4 Conv Small.
The replacement preserves the three downstream feature strides (8, 16, 32);
the feature channels become 64, 96, and 960, and MonoDETR's existing 1x1 input
projections adapt them to the unchanged 256-dimensional transformer.

The notebook applies two audited, idempotent patch scripts:

- `scripts/patch_monodetr_colab_compat.py` for current PyTorch/Colab APIs.
- `scripts/patch_monodetr_mobilenetv4.py` for the backbone option only.

`scripts/prepare_monodetr_mnv4_backbone_experiment.py` creates a strict-load
initialization checkpoint. MobileNetV4 starts from ImageNet weights; every
shape-compatible MonoDETR tensor outside the backbone and input projections is
copied from the validated official checkpoint. Unexpected downstream missing
or mismatched tensors stop the run. The first authorized run is a 20-epoch
gate with checkpoints every five epochs, not a full replacement training run.
Compare its official KITTI Car AP_R40, parameter count, memory, and inference
latency against the unmodified teacher before changing the depth predictor or
transformer.

After a successful epoch-20 gate, the notebook provides a stateful continuation
to epoch 50. `scripts/patch_monodetr_verbose_resume.py` lets MonoDETR resume
from an explicit checkpoint path and adds durable progress lines every 20
batches with current/average loss, learning rate, elapsed time, and CUDA memory.
It also logs epoch start, epoch-average loss, training duration, and validation
boundaries. The continuation restores model and optimizer state, epoch, best
AP, and scheduler position from `checkpoint_epoch_20.pth`; it does not reset
the run as pretrained fine-tuning.

`scripts/patch_monodetr_checkpoint_metadata.py` fixes the upstream save order
for future runs. It first writes a provisional resumable checkpoint for crash
recovery, runs validation, updates `best_result`/`best_epoch`, writes
`checkpoint_best.pth` when appropriate, and then overwrites the resumable epoch
checkpoint with finalized post-validation metadata. Existing epoch checkpoints
are not rewritten retroactively.

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
