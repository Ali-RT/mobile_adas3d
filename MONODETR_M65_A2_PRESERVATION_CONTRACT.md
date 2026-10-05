# M65 A2 preservation control

Prepared 2026-10-05. Revision `M65-A2-PRESERVATION-2026-10-05-r1`.
This bounded experiment tests whether one continuation epoch can preserve A2's
existing detection and geometry performance. It uses GT plus a frozen copy of
original A2, not the external MonoPRIO teacher. It is a prerequisite control,
not an improved-student claim or a full distillation run.

## Evidence leading to this control

The reviewed `m64_results.zip` completed both models' 3,769-image validation,
reproduced unchanged A2, and reproduced MonoPRIO's published seed 444 native
results. Zero optimizer updates occurred. Product moderate 3D AP_R40 was
15.4475/7.5284 for A2 and 19.4917/8.6633 for MonoPRIO, Vehicle/Pedestrian.
The teacher's nearby Vehicle recall was lower than A2's, 0.85343 versus 0.88293;
Pedestrian recall was higher, 0.74691 versus 0.69268, but below the 0.80 target.
Higher AP therefore does not establish uniformly better supervision.

The prior-bank audit now identifies an unresolved release-recipe discrepancy.
Pinned `tools/build_priors.py` uses Pedestrian and Cyclist `k_geo=4,k_vis=2`,
so the default two-stage recipe cannot exceed 8 prototypes per class; subsequent
merging only reduces that number. The actual released unified bank contains
13 Pedestrian, 10 Car and 12 Cyclist prototypes. It has no construction sample
IDs, split digest or label digest. The paper states that priors use the detector
training split, and the builder defaults to `train`; neither fact independently
ties the downloaded bytes to the exact construction inputs.

This is **not evidence that validation leakage occurred**. It is evidence that
the default builder does not reproduce the released Pedestrian/Cyclist recipe.
External-teacher KD stays disabled pending the exact release construction
receipt or independently reproduced bank. The original-A2 control has no such
dependency and may proceed.

Sources: [official pinned builder](https://github.com/Leon-Davies/MonoPRIO/blob/884b7d8562e528031616c4773170bfc4fe211bf0/tools/build_priors.py),
[paper section 3.3](https://arxiv.org/html/2605.14781v1#S3.SS3),
and restricted inspection of the exact downloaded prior hash
`e2f519187b72f1f4c62d44cbef63f0d6d05f87c7c47fff160049b34d230b464c`.
These sources support different parts of the finding; the paper does not
independently verify the bank's construction IDs.

## Fixed model and data

Keep MobileNetV4 Conv Medium plus the full MonoDETR inference graph. Initialize
from original GT-trained A2 epoch 130, SHA256
`ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4`.
Never overwrite that checkpoint. Use a fresh `MonoDETR_M65_A2` checkout at
`6994b9f512400b258c6edb75f77423beb9c126f2`, with the same tested compatibility,
taxonomy, MobileNetV4 and M64 lazy-evaluator/CUDA-target patches.

Use exact Chen train 3,712/val 3,769 splits, original labels and calibration, and
the same two-class product mapping. Bind new runtime, source, binary and data
identities to a prospective M65 manifest. Do not rewrite M64/M62/M63 identities
or require their old `/content` checkouts to exist.

The exact reviewed M64 manifest/report file hashes are pinned in the workflow.
Reuse the successfully tested private PyTorch 2.10.0/torchvision 0.25.0/CUDA 13.0
recipe, not Colab's kernel packages. A GPU/runtime change needs a new RUN_ID.
The complete unchanged-A2 baseline is measured again in the prospective runtime
before optimization; no old AP table substitutes for this gate.

## Frozen one epoch recipe

- One epoch, 928 optimizer updates, batch 4, seed 444, num_workers=0.
- Fresh AdamW, LR 1e-6, weight_decay=0, global gradient-norm clip 1.0, no AMP.
- All normally trainable parameters may update except BatchNorm affine weights.
  Freeze all BatchNorm running buffers and affine weights; disable module and
  attention dropout. Record trainable parameter names and buffer hashes.
- Retain all native weighted GT losses and auxiliary losses for both classes.
  Do not add new GT class weights or train only Vehicle objects.
- Online photometric augmentation, horizontal flip 0.5 and crop 0.5 with native
  scale/shift 0.05; disable mixup. Student and original-A2 reference see identical
  transformed images, calibration, sizes and GT. No untransformed cache is used.
- Frozen original A2 is detached, in eval mode and never optimized. The student
  uses the native 11-group training forward for GT and a differentiable 50-query
  inference-group forward for preservation.

These choices are prospective hypotheses, not settings proven to prevent
forgetting. This control changes several continuation settings together; it
does not isolate the causal effect of preservation alone. A later feature-KD
treatment must use the identical control recipe.

## Object matched preservation loss

Independently Hungarian-match the two inference-group outputs to the same
transformed GT. A pair is associated through GT object index within the image,
never through equal query indices. Reject wrong-class references and require
class sigmoid score >= 0.30 and 2D IoU >= 0.50.

Preserve soft class probabilities with Bernoulli KL and normalized box outputs
with SmoothL1. Preserve depth only when original-A2 depth is positive and its
relative GT error <= 0.15; preserve physical H/W/L only when positive and mean
relative GT error <= 0.25. Preserve the angle distribution/residual only when
original A2 chooses the correct GT bin and its bin residual error <= 0.15 rad.
Unreliable source geometry is not made immutable. All GT terms remain active.

Within each component, average eligible objects separately for native
Pedestrian ID 0 and Vehicle/Car ID 1, then average classes with eligible objects.
Counts therefore do not automatically impose the roughly 7.2:1 Vehicle/Pedestrian
frequency ratio on preservation. This is not GT class reweighting.

The fixed loss is native GT plus preservation weight 1.0, with component weights
classification 2, box 5, depth 1, dimensions 1 and angle 1. Report every component's
per-class pair count. Exact equality should yield near-zero preservation loss;
the zero-update smoke uses a small output perturbation to verify a real finite,
nonzero preservation gradient through the actual student graph.

## Preflight and complete validation

Before training, unchanged A2 must stay within 0.15 AP points of each reviewed
M64 moderate 3D/BEV entry and within 0.01 absolute nearby recall per class.
A real CUDA smoke must pass GT/preservation gradient checks, unchanged model
parameters, unchanged running buffers and unchanged original-A2 reference,
with zero optimizer steps. CPU unit tests do not establish those GPU checks.

Evaluate the completed control on all 3,769 images with the unchanged product
taxonomy/evaluator, class threshold 0.001, top 50 and native two-decimal KITTI
export. Nearby diagnostic recall remains class-correct 2D IoU >= 0.5,
Vehicle <40 m/Pedestrian <30 m, exported score>=0.001; it is not 3D correctness.

The fixed **control stability gate** requires all four moderate AP entries to
lose no more than 0.15 AP points against the fresh unchanged-A2 baseline, and
each nearby recall to lose no more than 0.005 absolute. These are new control
stability criteria, not relaxed historical accuracy or product-safety targets.
Report the original five AP gates and 0.85/0.80 nearby targets separately.
Passing stability does not select this checkpoint or establish improvement.

## Restart and deliverables

Run `notebooks/MonoDETR_A2_M65_Preservation_Control_Colab.ipynb`, sections 1–6
in order. Prefer A100, especially the 80 GB model used successfully in M64.
No GPU execution or wall-time estimate is established by local CPU tests.
No iPhone connection is required.

Local preparation passed 34 M64 regression tests, 21 M65 tests, compilation
of all six code cells, and a CPU integration check using the exact pinned
native A2 Hungarian matcher. The notebook has no stored execution outputs.
CUDA smoke, one-epoch training and validation results remain pending.

Interrupted setup and per-image inference records are verified and reused.
An interruption before the one completed epoch restarts the epoch from original
A2 with the same seed and recipe; partial optimizer updates are not reused.
An atomic completed checkpoint is verified and reused with zero further
updates. Preserve incompatible or corrupt artifacts rather than deleting them.

Return `m65_results.zip` containing the prior audit, manifest, runtime receipt,
fresh baseline, CUDA smoke, one-epoch training summary, complete validation
metrics, geometry CSVs, stability decision and durable logs. Large weights and
per-image predictions remain on Drive. No artifact is modified on inspection.

If stable, review the control and resolve prior provenance before preparing one
matched object-aware feature-KD treatment. Obtain exact bank construction
source/parameters/seed, train IDs and data digests tied to the released bank hash,
or reproduce the bank without substituting different buffers into the qualified
checkpoint unnoticed. No publisher contact or new teacher training is authorized
by this notebook. Failure does not authorize extra epochs, a sweep or another
architecture. Chen-val is development data; later qualify on unseen data.
