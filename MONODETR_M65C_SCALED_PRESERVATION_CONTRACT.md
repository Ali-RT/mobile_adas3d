# M65c: scale-normalized A2 preservation control

Revision: `M65C-A2-SCALED-PRESERVATION-2026-10-06-r1`.
Prepared 2026-10-06. CUDA execution and accuracy results are pending.

## Decision and question

Keep original A2 epoch130 selected: MobileNetV4 Conv Medium backbone with the
full MonoDETR inference graph. Test one revised preservation control before
external-teacher distillation. No architecture changes, teacher downloads,
extra epochs, compression or phone deployment are included.

M65 failed preservation after one epoch without an external teacher. M65b
measured a median preservation/GT raw gradient-norm ratio of 0.605% at that
endpoint on 32 training images. The small sample also lacked eligible
Pedestrians in four batches. Depth log uncertainty, which affects native
ranking, had no explicit preservation term. These observations motivate a
new hypothesis; they do not prove the cause of AP regression or prescribe an
optimal coefficient. See `artifacts/m65b_gradient_review_20261006.json`.

M65c asks: can a scale-normalized, train-calibrated preservation term retain
A2 through the same bounded continuation? Success means control stability,
not that an external teacher has improved the student.

## Frozen provenance

- Original checkpoint SHA-256:
  `ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4`.
- Upstream source: `https://github.com/ZrrSkywalker/MonoDETR.git`, commit
  `6994b9f512400b258c6edb75f77423beb9c126f2`, with the same reviewed A2
  compatibility, taxonomy, backbone and inference-source repairs as M65.
- Exact reviewed M65 manifest self-signature:
  `d52fd78bd039615e932604ea8a51b7a1e9d5c4358b3bfb3c6f31301dd3c10d4b`.
- Calibration endpoint is the completed M65 checkpoint, SHA-256
  `e5c2b592977c68f79bc371196536c0263badaa7277e7fc17bd0ed6c7b64a6a18`.
  It is read only for calibration, never used as the new training parent.
- Reviewed gradient files are checked byte-for-byte against the local review
  artifact, whose SHA-256 is
  `a4a2f78793e292e793bd249380aaf4ae997ce8374bc259d3d38499bb0edc3c45`.
- Exact Chen train3712/val3769 splits, disjoint IDs and reviewed
  label/calibration identities. Full inference records input and prediction
  hashes. No historical manifests, sources or checkpoints are rewritten.

Create a fresh `MonoDETR_M65C` checkout, private CUDA13 environment and output
folder. The working M64/M65 CUDA13 helpers are reused; this is a new prospective
run, not historical-runtime replay. A changed package/GPU/source identity
requires a new RUN_ID, fresh baseline and calibration.

## Loss and reliable targets

Total loss is `native_GT + lambda * scaled_preservation`.
All original native GT loss weights and auxiliary terms remain active for both
product classes. The frozen original-A2 anchor and student inference queries
receive the same transformed image/calibration/size tensors. Queries are
associated through transformed GT identity, not matching query indices.

Preserve anchors with correct product class, class probability at least 0.30
and 2D IoU at least 0.50. Native IDs are Pedestrian0 and Car1 (product Vehicle).
Depth and uncertainty additionally require positive anchor/GT depth and
anchor relative depth error at most 0.15. Dimensions require positive H/W/L
and mean relative error at most 0.25. Angle requires the correct GT bin and
bin-residual difference at most 0.15 radians. All anchor values are detached.
Each component averages eligible objects within each class, then averages
available class means. This is not full-dataset class oversampling.

| Component | Error normalization before loss | Component weight |
| --- | --- | ---: |
| Classification | Bernoulli KL divided by 0.01 | 2 |
| Projected box | Smooth L1 of normalized-coordinate difference / 0.01 | 5 |
| Point depth | Smooth L1 of difference / (anchor depth, floored at 1 m, × 0.02) | 1 |
| Physical H/W/L | Smooth L1 of difference / (anchor dimension, floored at 0.05 m, × 0.02) | 1 |
| Angle | Bin KL / 0.01 plus Smooth L1 of anchor-bin residual difference / 0.05 rad | 1 |
| Depth log uncertainty | Smooth L1 of log-uncertainty difference / 0.10 | 1 |

These fixed normalization scales are experimental design choices, not accuracy
gates. Uncertainty preservation does not assert that original uncertainty is
perfectly calibrated. It is restricted to reliable anchor depths. Background
queries and intermediate features are not preserved in this trial.

## One training-only calibration

Select 64 unique train IDs using raw labels only and seed444: 32 images
containing Pedestrian or Person_sitting, and 32 containing a Vehicle source
class but neither Pedestrian class. Each of 16 batches contains two images
from each bucket. The selection uses no validation labels, predictions or
error ranking. Some Pedestrian-containing images can also contain Vehicles.

Load the fixed completed M65 endpoint for the differentiable student and
original A2 for the frozen anchor. Use the unchanged M65 online augmentation
and BN/dropout policy. Measure preclip GT and scaled-preservation gradients on
all 16 batches, with zero optimizer steps. Require finite, strictly positive
norms and at least eight eligible pairs from each class for classification,
depth and uncertainty over the calibration sample.

Set one scalar:

`lambda = 0.25 / median(per-batch preservation_norm / GT_norm)`.

The 25% raw-gradient target is a prospective hypothesis at that fixed endpoint.
It is not an exact AdamW update contribution, a guarantee of the ratio during
training, or the reciprocal of M65b's old 0.605% measurement. Require lambda in
`[0.01, 100]`; otherwise stop for review. Do not clamp it, change coverage
rules, search weights or select by validation accuracy.

The signed calibration receipt records IDs, every batch's norms, pair counts,
the scalar, unchanged model/anchor/buffers and zero updates. Training summaries
bind its signature. The endpoint process is discarded after calibration;
the training process loads original A2 again.

## Execution and acceptance

Before updates, reproduce original A2 on all3769 validation images within
0.15 AP points on each of four moderate product metrics and 0.01 absolute
nearby recall for each class. Then calibrate once. A real CUDA smoke requires
finite GT gradients and a nonzero uncertainty-preservation gradient through
the student graph after a small output perturbation, with zero updates and
unchanged parameters/buffers/anchor. Identical models' initial scaled
preservation loss must be at most `1e-3` (numerical sanity, not an AP gate).

Train exactly one epoch: seed444, batch4, 928 updates, fresh AdamW at `1e-6`,
zero weight decay, gradient clip1, no AMP. Freeze BatchNorm affine parameters
and running buffers; disable module/attention dropout. Keep photometric
augmentation, flip0.5, crop0.5, scale/shift0.05 and no mixup. Only calibration
selection is stratified; full training uses all3712 images with shuffled order.

Full validation must lose no more than 0.15 AP points on **each** of Vehicle
3D, Pedestrian3D, VehicleBEV and PedestrianBEV moderate AP_R40, and no more
than 0.005 absolute nearby recall on **either** class, relative to the fresh
baseline. Nearby recall uses the unchanged geometry evaluator definition.
Report historical gates separately: Vehicle/Pedestrian3D 15.8713/5.1493,
Vehicle/PedestrianBEV 21.3134/5.9365, mean3D 10.5103, nearby recall0.85/0.80.
All AP values are percentage points under the product taxonomy, not official
KITTI leaderboard results or percentages of perfectly reconstructed images.

Stop for review regardless of outcome. Failure rejects this recipe; success
does not automatically promote the control, select a teacher or authorize KD.
MonoPRIO prior-bank provenance remains unresolved separately. Original A2
remains selected until an explicit subsequent decision.

## Notebook, recovery and deliverables

Run `notebooks/MonoDETR_A2_M65c_Scaled_Preservation_Colab.ipynb`, six sections
top-to-bottom on a GPU. Section1 defines every helper/import and prints the
revision; section2 builds the isolated runtime and A2; section3 restores data
and freezes provenance; section4 runs full baseline, calibration and smoke;
section5 runs one epoch; section6 evaluates and bundles results for review.

After a runtime restart, rerun sections1–4. Full prediction caches and completed
calibration/checkpoint transactions are reused only with matching identities.
A completed atomic epoch checkpoint recovers a missing sidecar without further
updates. A partial epoch restarts deterministically from original A2: no
batch-level optimizer resume is claimed. Existing completed checkpoints are
never overwritten.

Return `m65c_results.zip`: manifest, runtime receipt, baseline gate, signed
calibration, smoke, training summary, full-evaluation summaries/diagnostic CSVs,
control gate and durable logs. Large weights/prediction caches remain on Drive.
Section6 also bundles partial diagnostics after failures. No CUDA execution,
selected lambda or accuracy improvement is claimed by local CPU tests.
