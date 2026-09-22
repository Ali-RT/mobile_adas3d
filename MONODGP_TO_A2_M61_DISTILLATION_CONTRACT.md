# M61 — MonoDGP teacher to MobileNetV4 A2 student

Status (2026-09-22): **workflow prepared; Colab CUDA audit/smoke/pilot pending**.
Notebook revision: **M61-2026-09-22-r1**. No new training result is claimed.

## Decision and scope

Return to the teacher/student route explicitly requested by the user. The
deployed candidate is the student, not the teacher. M60 remains a completed
negative teacher-runtime feasibility result; its operator-profiling proposal
is deferred. This does not erase the successful MonoDGP Core ML parity work.

- Teacher: original M54 MonoDGP ResNet50 epoch 100, SHA-256
  `8e79f3921d96e1de70cbb4219245e3fcc3fa1fb67ae675468b4ebca90e579847`.
  Do not substitute M56d compressed weights or the M59 Core ML model.
- Student: A2 MonoDETR MobileNetV4 Conv Medium epoch 130, SHA-256
  `ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4`.
- Upstream commits: MonoDGP `aa059a18214aebf644510e7f0793971b403f9d14`;
  MonoDETR `6994b9f512400b258c6edb75f77423beb9c126f2`.
- Preserve student backbone, transformer, heads, resolution, taxonomy and GT
  criterion. Product Vehicle maps Car/Van/Truck/Tram to native **Car index 1**;
  Pedestrian/Person_sitting map to native **Pedestrian index 0**. Cyclist is
  excluded from targets; the native three-logit interface remains unchanged.
- Teacher and student upstream packages have the same `lib` namespace. Run
  them in separate processes, with separate repo-local CUDA extensions.

This is new supervision, not a repeat of R0-to-A1 distillation or A2g GT
geometry-loss weighting. There is no temperature sweep: the first experiment
does not distill class probabilities. The prior A1 temperature was 2.0 and
was not swept; it is not inherited here.

## Train-only cache and teacher-quality audit

Use the exact 3,712 Chen training IDs, disjoint from the 3,769 validation IDs.
Keep original labels as ground truth. Generate both frozen models' predictions
on the same **unaugmented** training images. Both continuation arms use this
same unaugmented image view. This is a deliberate paired pilot setting, not
the original A2 full-training augmentation recipe and not permission to reuse
cached targets on flipped/cropped/mixed images.

SHA-256-bind inputs (normalized image, calibration, original image size),
packed GT identity/order, checkpoints, code, patched source, configs and
runtime environment. Match each model to GT independently, then associate
through the common GT index. Never assume query numbers correspond. Check
input/GT fingerprints in every training batch, including the GT-only control.
Changing augmentation or data must stop the run.

Teacher object requirements:

- native winning class is Car/Vehicle; Car sigmoid confidence >=0.30;
- matched 2D IoU >=0.50, GT depth in [2,60) m;
- finite outputs, positive teacher depth and dimensions.

Audit depth absolute error (m), dimension mean H/W/L absolute error (m),
geometric-center Euclidean error (m, including camera translation), and
wrapped observation-alpha error (degrees). Each component is enabled only
if its teacher mean error over all quality-filtered paired objects is lower
than A2's, and at least 100 GT objects have teacher error strictly below
95% of A2 error. Only these per-component approved objects receive KD.
These are predeclared train-audit rules, not evidence of generalization.

If no component passes, emit `m61_teacher_audit.json` and stop before training.
No validation labels or validation metrics select KD objects/components.

## Loss and paired ten-epoch pilot

Preserve all native GT losses for both product classes. Additional KD is
Vehicle-only, final-output-only, with frozen overall weight 0.25:

```text
total = native_GT_loss + 0.25 * sum(enabled audited geometry losses)
depth:     |student_z - teacher_z| / max(GT_z, 1 m)
dimensions: mean(|student_HWL - teacher_HWL| / max(GT_HWL, 0.25 m))
center:    L1(student_XYZ - teacher_XYZ) / max(GT_z, 1 m)
angle:     CE(student_heading_bins, teacher_winning_bin)
           + SmoothL1(student_residual_at_teacher_bin, teacher_residual)
```

Camera-center decoding uses each GT calibration and original image size on
the fixed, unaugmented view. Component means normalize over their approved
student/GT assignments, including the native repeated training query groups.
Teacher tensors are detached. No teacher logit, 2D-box, Pedestrian, dense-depth,
feature, or temperature loss is added. Frozen audit approvals are not updated
from validation, or recomputed using the improving student.

Two arms: `control` and `vehicle_kd`, both from the exact A2 checkpoint. Fixed
10 additional epochs, batch size 4, seed 20268, native AdamW at constant 1e-5,
weight decay 1e-4 (native bias exemption retained), FP32, no AMP. Fresh optimizer
state in each arm; native GT loss/matching unchanged. Same epoch-scoped shuffle
and dropout seeds in both arms, with no stochastic data augmentation. Teacher
is absent from the training graph/optimizer; compact cached outputs are loaded.

First run a real CUDA forward/backward smoke with zero optimizer steps. Require
finite GT+KD loss/gradients, nonzero KD parameter gradients, and approved pairs.
Also test a native background-only GT batch when such a training image exists;
do not discard these images to make the test pass.
If it fails, do not train. CPU regression tests are not a substitute for this.

Print loss/progress every 20 batches and save every completed epoch atomically
to Drive, including optimizer state, history and experiment identity. Resume
at epoch boundaries. A crash mid-epoch replays that epoch with the same seeds.
An interrupted post-save checksum transaction can be recovered after checking
the restricted-load checkpoint identity; a checksum/provenance conflict stops.
CUDA arithmetic is not promised bitwise reproducible across hardware changes;
cache/run environment changes are rejected, not silently treated as paired.

## Frozen comparison and stop rule

Re-evaluate unmodified A2 in the same environment. Evaluate both arms at epochs
5 and 10 on all 3,769 validation IDs with the existing product evaluator,
score threshold 0.001, top-k 50, and nearby diagnostic 2D IoU 0.50.
**Epoch 10 is the decision point; epoch 5 is diagnostic only.**

The epoch-10 treatment must satisfy all of these:

| Gate | Requirement |
| --- | ---: |
| Vehicle moderate 3D AP_R40 | >=15.8713 |
| Pedestrian moderate 3D AP_R40 | >=5.1493 |
| Balanced moderate 3D mean | >=10.5103 |
| Vehicle moderate BEV AP_R40 | >=21.3134 |
| Pedestrian moderate BEV AP_R40 | >=5.9365 |
| Vehicle 3D gain over matched control | >=0.10 AP points |
| Pedestrian 3D/BEV AP and nearby recall | no drop vs either fresh A2 or control |
| Balanced moderate 3D mean | no drop vs either fresh A2 or control |
| Vehicle BEV loss vs control | <=0.15 AP points, while meeting absolute gate |
| Vehicle nearby-recall loss | <=0.01 absolute vs both fresh A2 and control |
| Prediction completeness | exact 3,769 IDs per checkpoint |

The additional preservation rules protect A2's Pedestrian strengths rather
than treating the lower historic R0-derived floor as sufficient. They are
frozen before the pilot. The 0.80 Pedestrian nearby-recall product target is
reported separately; the source A2 is only approximately 0.69224.

Always stop for review. A pass recommends independent confirmation; it does
not authorize a long run, checkpoint/temperature sweep, iPhone deployment or
safety claim. A negative result closes this particular treatment, not all KD.
Repeated Chen validation is development evidence, not an unseen test. Eventual
external validation and a student-specific Core ML/device feasibility probe
remain required before substantial deployment-oriented training.

## Run and return

Run `notebooks/MonoDGP_to_MonoDETR_M61_Vehicle_Distillation_Colab.ipynb`,
sections **1–12 in order**. Use CUDA; no phone connection or new dataset upload
is needed if the frozen Drive selections/checkpoints/KITTI data remain there.
The first code cell prints revision `M61-2026-09-22-r1`.

Default Drive output:
`mobile_adas3d_outputs/students/monodgp_to_monodetr_m61`.

After a Colab restart rerun from section 1. Existing hash-verified cache samples
and complete epochs are reused. No old transient runtime YAML, notebook helper
or stale local repository is assumed. Setup is idempotent, preserves existing
source changes, and never executes a destructive reset.

Return `m61_results.zip`: manifest, train audit, real CUDA smoke, fresh A2
baseline metrics, both training summaries, comparison JSON and CSV. If the
audit/smoke stops early, return that report and its durable log instead.

Local preparation verifies numerical logic, source patching, notebook syntax,
checkpoint recovery, and loader compatibility. It does not claim CUDA smoke,
pilot accuracy, new student weights, phone performance, or trained-model parity.
