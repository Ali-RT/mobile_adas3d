# A2 accuracy → distillation → iPhone plan

Decision updated: 2026-10-06. Keep A2 as the student and reassess both the
teacher and the transfer method. Do not extend the rejected scalar-depth pilot
or make historical-runtime recovery the main accuracy-development task.
MonoDGP and M61 are parked, not deleted.
M61's train-only audit rejected every component according to the user-provided
log. The full component report has not been independently reviewed. No M61
continuation-training result has been supplied. Do not overwrite its artifacts.

## Current action

M64 r2 completed on A100 80 GB: both models evaluated 3,769/3,769 images,
unchanged A2 reproduced its baseline, and MonoPRIO reproduced its published
seed 444 native results. No optimizer updates occurred. Product moderate 3D AP
was A2 Vehicle 15.4475/Pedestrian 7.5284 and teacher 19.4917/8.6633.
The teacher has higher AP but lower nearby Vehicle recall; no student improvement
or uniformly superior teacher geometry is established.

The exact released prior has 13 Pedestrian/12 Cyclist prototypes, exceeding the
pinned builder's default maximum 8 each, and contains no construction IDs.
This is an unresolved recipe/provenance gap, not proven validation leakage.
MonoPRIO external-teacher KD remains disabled. See `artifacts/m64_review_20261005.json`
and `MONODETR_M65_A2_PRESERVATION_CONTRACT.md`.

M65 completed the fixed one-epoch control on A100 40 GB. Both evaluations cover
3,769 images, but the control failed preservation: Pedestrian 3D AP fell
7.5284→7.0561 and Vehicle BEV AP fell 21.3776→21.0912. There were 928 updates,
with unchanged BN buffers and original-A2 anchor. No external teacher was used.
Keep original A2 epoch130; do not extend that failed control. Its old KD
authorization remains unchanged; the later M66 decision is separately scoped.

The completed archive diagnosis pairs 19,313 GT records and excludes 28 ambiguous
depth keys. Geometry changed slightly across several components; nearby Vehicle
depth improved while farther Vehicle depth worsened. Native scores also depend
on a depth-uncertainty output not explicitly preserved by M65. Neither scalar
loss means nor combined gradient norms establish which term caused regression.
See `artifacts/m65_regression_review_20261005.json`.

M65b r2 is complete; the uploaded `m65b_gradient_results.zip` was reviewed.
Both endpoints ran eight batches on the same 32 training images, with zero
updates and unchanged model/anchor/checkpoint state recorded. At the control
endpoint, median preservation/GT raw gradient-norm ratio is 0.605%, and native
depth has the largest median component norm. Classification dominates one
batch, so this is not a depth-only explanation. Depth-log-uncertainty changes
without an explicit preservation term. Only 12 Pedestrian pairs are eligible;
four batches have none. The measurement does not replay historical AdamW
updates, attribute AP loss, or validate a new model.
See `artifacts/m65b_gradient_review_20261006.json`.

M65c completed calibration, smoke and 928 updates on A100 40 GB, but failed
preservation. On full 3,769-image validation, Vehicle/Pedestrian moderate 3D AP
changed 15.4475/7.5284 to 15.5076/7.0752; BEV 21.3776/8.4887 to 21.2897/8.3088.
Both nearby recalls stayed within the preservation limits, but Pedestrian
recall 0.69092 remains below 0.80. The measured preservation coefficient was
0.15778017; the anchor and BN buffers remained unchanged according to the
reports. No external teacher was used. Evidence:
`artifacts/m65c_review_20261006.json`. Raw checkpoint/prediction bytes were
not included for an independent CUDA/AP rerun. Reject this recipe and retain
original A2. Do not repeat the completed M65b/M65c workflows.

The user approved a bounded actual-KD pilot on 2026-10-06. Requiring a passing
preservation-only control was an experiment-design guardrail, not a technical
prerequisite; that requirement is removed for M66, without changing historical
gates or authorizations. Prepare one matched no-KD control and one R0 Vehicle
feature-KD arm from original A2. Update only native depth MLPs; freeze the
backbone, transformer, dense-depth predictor and other heads. Retain both-class
native GT and identical original-A2 preservation in both arms. Teacher targets
are train-only Vehicle objects where frozen R0 has better depth than original
A2. Teach hidden depth-head features, not frozen decoder features or another
scalar-depth output loss. No MonoPRIO prior is used or newly qualified.

The executable recipe is `MONODETR_M66_VEHICLE_FEATURE_KD_CONTRACT.md` and
the six-section `notebooks/MonoDETR_A2_M66_R0_Vehicle_Feature_KD_Colab.ipynb`.
Each arm has 928 updates at LR 1e-6. KD needs at least 0.15 Vehicle 3D AP gain versus
both original A2 and control, with at most 0.15 AP loss in each other moderate
metric and 0.005 nearby-recall loss for either class versus either comparator.
The control's stability is reported, not required to run the KD arm. Actual
source/data identity, fresh baseline reproduction and finite nonzero KD CUDA
gradients remain prerequisites. Stop for review regardless of outcome; no M66
CUDA run or improvement is claimed by preparation.
The existing M63h notebook is archival, not the next accuracy experiment.
Preserve its files and hashes.

The latest M63h r5 log localizes SIGSEGV to Numba CUDA context initialization
while importing the KITTI AP evaluator through the dataset loader. Torch and
the deformable-attention extension import passed; that does not establish
attention forward/backward correctness. The underlying driver/binding conflict
is not yet proven. No M63h optimizer updates ran. A prospective training loader
must not initialize the AP evaluator at import time; run metric evaluation
separately and record a new runtime/source identity instead of changing M62.

M63f KD gained0.1528 Vehicle3D AP over its matched control, but lost0.1726
Pedestrian3D AP. Original A2 remains selected; the pilot failed.

M63g confirms both detection and geometry deterioration, not just depth.
On1,274 common moderate Pedestrian matches, depth MAE rose0.7389→0.7809m
and mean3D IoU fell0.2470→0.2377 from A2 to KD, with48 lost/24 gained
detections. Diagnostic matching is not an exact attribution of AP.
Repeat controls share recorded settings and unchanged BN buffers but have
different learned weights. This does not establish which operation caused it.

The evidence supports two separate problems: the chosen KD treatment did not
meet acceptance, and continuation training can lose A2 accuracy even without
KD. It does not prove that R0 is useless or that changing teachers alone will
solve the problem. M63 taught only final Vehicle depth, using a fixed
0.25-weight absolute error in metres, unaugmented views and all student
parameters trainable. It did not test object-aware feature distillation or a
loss preserving reliable original A2 predictions.

## Priority order

1. Keep the exact A2 epoch130 checkpoint and inference graph as the baseline.
2. Use the working prospective CUDA13 runtime and existing R0 Vehicle teacher.
   MonoPRIO stays unselected until its prior construction is resolved; do not
   make that unresolved candidate a blocker for this R0-specific experiment.
3. Re-evaluate unchanged A2 and test the real KD gradient with zero updates.
   Freeze one narrow head-update recipe and numerical limits before training.
4. Run the paired one-epoch M66 no-KD control and Vehicle feature-KD treatment.
   Keep both-class GT and identical A2-preservation supervision. A passing
   previous preservation control is not required; compare KD against both
   original A2 and the newly measured control instead.
5. Measure benefit against unchanged A2 and the matched control, including
   per-class AP, nearby recall and geometry tails. Confirm a promising result
   with a second seed before longer training; failure does not authorize a grid.
6. Freeze the improved student, convert with prediction-preservation checks,
   then measure actual iPhone model and end-to-end timing/memory/stability.
7. Optimize only a measured runtime bottleneck, one change at a time, retaining
   the uncompressed accurate checkpoint. Validate on untouched external data.

Accuracy is the immediate objective. KD is an experiment, not a guaranteed
improvement. Teacher runs on the training machine; student is the phone target.

## Frozen models and status

- Student A2: MobileNetV4 Conv Medium + MonoDETR, epoch130, GT-only training.
  SHA `ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4`.
  Moderate Vehicle/Pedestrian 3D AP_R40 15.4573/7.5328; BEV21.3750/8.4892.
  Nearby Vehicle<40m recall0.88293, Pedestrian<30m0.69224.
- Teacher R0: ResNet50 MonoDETR epoch185.
  SHA `fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59`.
  Moderate Vehicle/Pedestrian 3D AP_R40 17.6348/5.7214.
- A2 retains four of five historical 90%-of-R0 AP gates. Vehicle3D15.8713
  remains missed by0.4140; Pedestrian nearby target0.80 remains missed.
  These limits are not silently lowered. A2 iPhone performance is not measured.
- R0 is not a uniformly better teacher: restrict its external supervision to
  supported Vehicle strengths, not blind Pedestrian or all-output imitation.
  Rejected R0→A1 stays rejected.

## Teacher selection

Our own product-taxonomy moderate 3D AP_R40 results are:

| Model | Vehicle | Pedestrian | Current role |
| --- | ---: | ---: | --- |
| A2 MobileNetV4 Conv Medium MonoDETR | 15.4573 | 7.5328 | Selected student baseline |
| R0 ResNet50 MonoDETR | 17.6348 | 5.7214 | Available Vehicle specialist |
| M54 ResNet50 MonoDGP | 19.4519 | 6.1749 | Parked reference, not an approved KD teacher |

Neither existing teacher is uniformly stronger than A2. Higher aggregate AP
does not establish that every training target is better or transferable.
The M61 audit rejection remains part of the evidence; do not simply revive
that exact pipeline because M54 has higher Vehicle AP.

The single challenger is **MonoPRIO**, a May 2026 preprint with an
official implementation, unified Car/Pedestrian/Cyclist validation checkpoints,
logs and size-prior banks. Its reported median-of-five moderate 3D AP_R40 is
21.856 for Car and 9.361 for Pedestrian. These are published standard-KITTI
results, **not our Vehicle/Pedestrian benchmark**. M64 subsequently reproduced
the separate seed 444 native protocol and measured its product-taxonomy results;
the released prior's exact construction remains unverified.
See the [paper](https://arxiv.org/abs/2605.14781) and
[official implementation](https://github.com/Leon-Davies/MonoPRIO).

Qualification must check checkpoint/config/source identity, exact train/val
IDs, geometry conventions and prior-bank construction before inference.
Use a validation checkpoint, never a trainval/test checkpoint for Chen-val.
Prior banks must not use validation labels. Reproduce the published native
class protocol first, then evaluate with our unchanged product evaluator.
Do not equate native Car with Truck/Tram/Van or native Pedestrian with
Person_sitting without checking the training taxonomy. A native-class-only
teacher can supply class-supported targets, not claim full product coverage.

Select supervision by actual class/range strengths, uncertainty, coverage and
geometry errors as well as AP. Train-only target screening and development-set
teacher qualification serve different purposes; validation labels never become
KD targets. If the candidate cannot be reproduced or shows no useful advantage,
retain A2 and report that outcome. No automatic teacher-training campaign or
second challenger is authorized by a failed qualification.

## Distillation design

M66 implements **object-matched internal depth-head features**, not another
final-depth copying loss. Hungarian/GT associations align objects rather than
assume equal query indices. Its 256-channel cosine loss uses no projector;
direct feature-coordinate alignment is a hypothesis tested by the paired
comparison. The inference target remains A2, not MonoPRIO. This is not a validated
implementation of [DETRDistill](https://arxiv.org/abs/2211.10156), which studies
Hungarian-matched and target-aware feature KD for 2D DETR detectors.

Use three distinct sources of supervision:

- Original GT losses for Vehicle and Pedestrian.
- A soft preservation loss from frozen original A2 on reliable predictions,
  particularly Pedestrian classification, localization and geometry. It is
  intended to limit forgetting, not to make A2's errors immutable.
- One object-aware feature-KD loss from R0, masked to reliable Vehicle objects
  with a measured training-view depth advantage over original A2. R0 is not
  an assumed Pedestrian teacher. New geometry/logit KD is a later ablation,
  not added simultaneously to this first treatment.

The matched control receives the same GT and preservation losses, augmentations,
initial checkpoint, update budget and optimizer recipe; only stronger-teacher
KD differs. Preserve BN running buffers and document any affine-parameter
updates. Teachers run detached and in evaluation mode. Teacher and student
must see the same geometrically transformed image/calibration; restore useful
augmentation through online or transform-compatible teaching rather than reuse
unaugmented cached targets incorrectly. Address mixup separately before using it.

Class counts are about 7.2 Vehicle objects per Pedestrian object in train.
Report per-class contributions and normalize KD over objects within each class
so Vehicle count alone cannot set the loss scale. This is not an automatic
7.2-times Pedestrian GT weight: earlier sampling/focal-weight experiments did
not solve the gap. Freeze weights, gradient-scale checks and reliable-target
rules using training data before evaluating the pilot. Preservation is a
hypothesis to test, not a guarantee.

The current budget is one short paired M66 pilot and one confirmation seed
only if promising. Its epochs, fixed coefficients, masks and acceptance limits
are frozen in `MONODETR_M66_VEHICLE_FEATURE_KD_CONTRACT.md`. Do not continue
preservation-only iterations indefinitely or launch a teacher/weight grid.
Historical M65/M65c authorizations are unchanged; M66 authorizes only its new
paired treatment, not longer training, model promotion or deployment.
No temperature grid is justified for feature KD.

## M62 completed diagnostic

The following records the original M62 protocol, not a request to rerun it.
Its reviewed evidence remains available in `m62_results.zip`.

Notebook: `notebooks/MonoDETR_M62_R0_A2_Diagnostic_Colab.ipynb`.
Revision `M62-2026-09-28-r1`, sections1–8 in order on CUDA. No phone needed.
Uses only pinned MonoDETR6994b9f, original KITTI labels, the exact Chen train
3712 IDs, and timm1.0.20. All normal class/geometry encodings are preserved.
No validation images/metrics select teacher targets or components.

First, a native A2 forward checks finite outputs, lists custom attention modules,
counts parameters, and measures 5warmup/10 CUDA predictions on one train image.
This is an operator inventory/native timing check, **not successful Core ML
conversion, a phone speed estimate, or full export feasibility proof**. Custom
attention is recorded as a blocker requiring a later verified portable path.

Next, cache frozen R0 and A2 separately on all unaugmented train images. Check
exact normalized input/calibration/GT identity. Associate predictions through
independent Hungarian-to-GT matches, never by query index. Native VehicleID1,
PedestrianID0. Per-sample writes are atomic; resume verifies existing samples.
Bind source, code, checkpoint and environment hashes. GPU/version changes stop
with an explicit diff. Persistent I/O errors identify a file, never delete it.
Do not transplant or rewrite M61 caches.

Quality-filter pairs: teacher winning Vehicle class, score>=0.30, 2D IoU>=0.50,
GT depth[2,60)m, finite outputs and positive dimensions/depth. Report paired
means/medians and per-distance counts for depth, dimensions, geometric XYZ
center (including calibration translation), and wrapped observation alpha.

Before seeing results, freeze this screening rule:

- A teacher win is component error strictly below95% of A2 error.
- A component needs>=100 wins and>=25 wins in at least two of the fixed
  distance bins[2,20),[20,40),[40,60)m.
- Propose the eligible component with most wins. Ties: depth,center,dimensions,
  angle. Overall teacher mean superiority is reported but is not a requirement.
- These are candidate-supervision screening thresholds, not product gates.
  No candidate also produces a complete report; training stays blocked.

This is a new R0/A2 diagnostic policy, not relaxation of M61 after its result.
An aggregate training-error gate alone cannot establish whether KD improves
generalization; conversely, cherry-picked training wins do not establish it
either. Only a future controlled held-out comparison can measure that benefit.

## Historical review boundary

The delivered `m62_results.zip` contains the manifest, native compatibility
report, component summary and per-object geometry rows. M62 itself produced
no trainable model, selected KD weight or training command.

The historical M62/M63 pilot has already run and failed; its acceptance limits
and frozen artifacts remain unchanged. The redesigned pilot must separately
freeze numerical acceptance limits before training:
Vehicle3D gain vs unchanged A2 and matched control, Pedestrian AP/nearby-recall
preservation, Vehicle BEV/recall limits, exact split and decision epoch. Report
the original five AP gates separately. Keep GT supervision for both classes
and never reuse targets on incompatible transforms. A pass needs seed
confirmation; a failure does not automatically authorize extra epochs or more
variants. Repeatedly inspected Chen-val is development data, not an untouched
final test. Later qualification must include external, unseen evaluation.
