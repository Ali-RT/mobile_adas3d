# M66 R0 to A2 Vehicle feature distillation

Revision: `M66-R0-A2-VEHICLE-FEATURE-KD-2026-10-06-r1`.

Status reviewed 2026-10-06: completed, gain gate failed. Original A2 remains
selected. The frozen protocol below is unchanged; do not rerun or extend it.
Evidence: `artifacts/m66_review_20261006.json`.

Run one paired experiment to test whether R0 can improve A2 Vehicle 3D accuracy
without sacrificing A2's Pedestrian performance. This is external-teacher
knowledge distillation, not another teacher-independent preservation control.
Original A2 epoch130 remains selected until the results are reviewed.

## Decision and evidence

M65c completed 928 updates but failed preservation: moderate Pedestrian 3D AP
fell 7.5284 to 7.0752 and Pedestrian BEV AP fell 8.4887 to 8.3088. Vehicle 3D
rose only 0.0600 points. No external teacher was used. The failed controls
show that continuation itself can lose accuracy; they do not show that KD
cannot help. Their historical results and authorizations remain unchanged.

The user approved a small real-distillation pilot on 2026-10-06. For M66 only,
a passing prior preservation control is no longer a prerequisite. Both arms
still require a fresh reproducing A2 baseline, exact model/data identity and
a real zero-update CUDA smoke. The final comparison must beat both original
A2 and its matched control. A failed control's accuracy gate does not prevent
measuring the KD arm; setup, numerical or incomplete-training failures do.

R0 is the existing ResNet50 MonoDETR Vehicle specialist. Its product moderate
Vehicle 3D AP is 17.6348 versus reproduced A2 15.4475, but its Pedestrian AP
5.7214 is below A2 7.5284. Do not transfer R0 Pedestrian/background targets.
MonoDGP remains parked. MonoPRIO's unresolved prior construction is not waived
by this decision and is not needed by M66.

## Frozen models and data

- Student and preservation anchor: original A2 epoch130, MobileNetV4 Conv
  Medium plus the full MonoDETR inference graph, checkpoint SHA-256
  `ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4`.
- External teacher: original R0 epoch185, checkpoint SHA-256
  `fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59`.
- Upstream source: `https://github.com/ZrrSkywalker/MonoDETR.git`, commit
  `6994b9f512400b258c6edb75f77423beb9c126f2`. Both models use the same fresh
  patched checkout, with their respective MobileNetV4 and ResNet50 configs.
  No model forward source change or inference adapter is added.
- The original M65c manifest, self-signature
  `853fc19cb6f7af250f8397bb688129efcc82a4c67eb7cee7a7daf7032ecbf724`,
  supplies original-A2 configuration and reviewed data identity only. Neither
  its trained control nor the earlier M65 endpoint initializes M66.
- Exact disjoint Chen train3712/val3769 split IDs, reviewed label/calibration
  hashes and unchanged product taxonomy: Vehicle maps Car/Van/Truck/Tram;
  Pedestrian maps Pedestrian/Person_sitting. Native IDs are Pedestrian0,
  Vehicle1, with the unchanged third Cyclist logit but no Cyclist GT targets.

Use a new `MonoDETR_M66_A2` checkout, output RUN_ID and private CUDA13 runtime.
The working M64 setup/build helpers are reused. Missing GPU/driver, wrong
source pin or changed package/GPU/data identity stops with an explicit error.
Do not rewrite manifests or reset an existing checkout. A new identity requires
a new RUN_ID and fresh full baseline. All checkpoint loads are restricted
tensor-state loads, followed by strict model-state loading.

## Paired treatment

Both arms start independently from original A2. Freeze all parameters except
the native `depth_embed.*` MLPs at the three decoder prediction levels. This
includes the point-depth regressor and native depth-log-uncertainty output.
Backbone, projections, dense-depth predictor, transformer, query embeddings,
classification, box, dimension and angle heads remain unchanged.

This narrower scope is a hypothesis intended to limit forgetting. It does not
guarantee fixed detection ranking: changing depth uncertainty can change native
scores, and depth affects 3D overlap for both classes. All original GT terms
and auxiliary weights remain present for Vehicle and Pedestrian; only terms
connected to the trainable depth MLPs can update their parameters.

The control loss is `native_GT + 0.10 * scaled_A2_preservation`.
The KD loss adds `0.10 * Vehicle_feature_KD`. Both coefficients are fixed
prospective design choices, not optimized settings or derived gradient ratios.
No calibration-endpoint dependency, coefficient grid or automatic adaptation
is included. A2 preservation uses the unchanged M65c reliable-object rules,
per-class means and fixed error scales, including depth log uncertainty.

Feature KD captures the final depth MLP's 256-channel hidden activation after
ReLU, immediately before its last linear output layer. This feature is inside
the trainable head. A frozen decoder feature alone would not give the student
head a KD gradient. The loss averages `1 - cosine_similarity` over eligible
Vehicle matches; there is no external depth/logit/angle output-copying term,
learned projector or softmax temperature. Direct channel alignment is a design
assumption, not evidence that the independently trained hidden bases coincide.
The held-out paired experiment measures whether it helps.

Student, R0 and frozen original-A2 anchor see the identical online transformed
image, calibration and size tensors. Each model is independently Hungarian
matched to transformed GT with one inference query group; shared GT identity,
never query index, associates their objects. R0 targets are detached.

Accept only training Vehicle GT with depth in `[2,60)` metres, positive model
depths, teacher winning Vehicle class, teacher class probability at least 0.30,
teacher 2D IoU at least 0.50 and relative teacher depth error at most 0.15.
Original A2 relative depth error must be at least 0.02, and the teacher error
must be at most 90% of original A2's error. The screen uses only current training
labels and the immutable anchor, not the moving student's error or validation
labels. It is a supervision mask, not a claim of higher AP on every matched
object. Record near and far Vehicle coverage separately.

Both arms execute the same frozen R0 forward and screening for diagnostics.
Only the KD arm includes its feature-loss gradient. Construct all models first,
then reset RNG for each arm so the data order and online augmentation match.
Record every batch's full transformed-input fingerprint; differing fingerprints
invalidate the paired comparison. Training hooks are removed before checkpoint
serialization and are absent from normal inference.

## Budget and preflight

Each arm runs exactly one epoch, 928 updates, seed444, batch4, fresh AdamW
LR1e-6, weight decay0, clip1 and no AMP. Freeze BN affine/buffers and disable
module/attention dropout. Keep native photometric augmentation, flip0.5,
crop0.5, scale/shift0.05 and mixup0. All 3712 training images are used; no class
oversampling, Vehicle-only fine-tuning or extra epochs are authorized.

Before updates, original A2 must reproduce all 3,769 validation images within
0.15 points on each of four moderate product AP metrics and 0.01 absolute
nearby recall on each class. Then run a zero-update CUDA smoke on the first
64 train images, with at least eight eligible Vehicle feature pairs. Require
finite nonzero native-GT and eligible-KD gradients into trainable student heads,
initial A2-preservation loss at most 1e-3, no Pedestrian external targets and
unchanged student/anchor/teacher state. Insufficient coverage stops for review;
do not silently loosen masks or start a different teacher search.

At completion, require unchanged frozen parameters, all buffers, original-A2
anchor and R0 teacher. Record learned parameter names, loss arithmetic, both
class preservation coverage, eligible KD coverage and all 928 input hashes.

## Full validation and decision

Evaluate original A2, control and KD on every one of the 3,769 validation images
with the unchanged product evaluator, score 0.001 and topk 50. The KD arm needs:

- At least **+0.15 Vehicle moderate 3D AP points** versus both original A2 and
  the matched control.
- No more than **0.15 AP-point loss** in each of Pedestrian3D, VehicleBEV and
  PedestrianBEV moderate metrics versus either comparator.
- No more than **0.005 absolute nearby-recall loss** for either class versus
  either comparator.

Report control stability separately; it is not a precondition for measuring KD.
Keep the historical product gates unchanged: Vehicle/Pedestrian3D
15.8713/5.1493, Vehicle/Pedestrian BEV 21.3134/5.9365, mean 3D 10.5103 and nearby
recall 0.85/0.80. AP values are product-taxonomy percentage points, not official
KITTI leaderboard values or perfect-image-reconstruction rates. These pilot
criteria are design thresholds, not a statistical significance claim.

Stop for review whether the pilot passes or fails. A promising result needs a
second seed before longer training. Failure does not authorize a weight/epoch
grid, another architecture or automatic teacher replacement. The repeatedly
used Chen-val is development data; later product qualification needs untouched
external evaluation. No checkpoint promotion, compression or phone run follows
automatically.

## Notebook and recovery

Run `notebooks/MonoDETR_A2_M66_R0_Vehicle_Feature_KD_Colab.ipynb` sections 1–6
in order. Section 1 supplies every import/helper/path and prints the revision;
2 builds the private runtime/source; 3 restores data and freezes models;
4 runs full baseline and actual KD smoke; 5 runs both arms; 6 evaluates and
bundles `m66_results.zip` for review.

After a runtime restart rerun 1–4, then 5–6. Completed atomic one-epoch checkpoints
recover missing sidecars/summaries without extra updates. Verified prediction
records are reused; incompatible ones are preserved and rejected. A partial
epoch restarts from original A2, not its last batch. No batch-level resume is
claimed. Never overwrite completed checkpoints.

Return the bundle containing the manifest, runtime/build logs, baseline and
signed smoke, both training summaries/checkpoint sidecars, full AP/nearby and
geometry diagnostics, paired gate and durable logs. Raw weights/prediction
caches stay on Drive. Section 6 can bundle partial failure diagnostics. Local
CPU tests do not establish CUDA behavior, memory use or an accuracy gain.

## Completed result

Both arms completed one epoch/928 updates on A100 40 GB. Original A2 reproduced
its baseline, the zero-update CUDA smoke passed with 23 eligible Vehicle pairs,
and training recorded 1,497 eligible Vehicle object presentations. All 928
transformed-input fingerprints match. Reports record unchanged frozen
parameters, buffers, anchor and teacher, with zero Pedestrian external targets.

| Moderate AP_R40 percentage points | Original A2 | Control | Feature KD |
| --- | ---: | ---: | ---: |
| Vehicle 3D | 15.447527 | 15.538300 | 15.538494 |
| Pedestrian 3D | 7.528416 | 7.394004 | 7.394002 |
| Vehicle BEV | 21.377606 | 21.295564 | 21.296920 |
| Pedestrian BEV | 8.488669 | 8.475375 | 8.475376 |

All three evaluations cover 3,769 images. KD gains only 0.090966 Vehicle 3D
points versus original and 0.000194 versus control, failing both +0.15 tests.
The control is stable within its limits and the other ten KD preservation
checks pass. Nearby Vehicle/Pedestrian recall is 0.883013/0.693122 for both
arms; Pedestrian still misses 0.80. Vehicle 3D and BEV also remain below the
historical 15.8713/21.3134 accuracy gates. This recipe supplies no useful
incremental KD gain and is not promoted or extended.

Teacher gradients were real, but their strength relative to GT is unmeasured.
Mean raw KD loss is 0.015620 (0.001562 weighted) versus GT 2.125265. Small loss
values do not prove small gradients. The separate M66b zero-update
component-gradient diagnosis is prepared before a different weight, scope or
transfer objective is chosen; see
`MONODETR_M66B_GRADIENT_DIAGNOSTIC_CONTRACT.md`. It does not extend this contract's
consumed training budget or change its frozen recipe.
One unsuccessful seed does not establish that R0 cannot teach A2 generally.
