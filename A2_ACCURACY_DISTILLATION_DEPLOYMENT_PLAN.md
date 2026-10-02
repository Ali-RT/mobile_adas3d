# A2 accuracy → distillation → iPhone plan

Decision date: 2026-09-28. MonoDGP and M61 are parked, not deleted.
M61's train-only audit rejected every component according to the user-provided
log. The full component report has not been independently reviewed. No M61
continuation-training result has been supplied. Do not overwrite its artifacts.

## Current action — M63h, bounded reproducibility check

M63f KD gained0.1528 Vehicle3D AP over its matched control, but lost0.1726
Pedestrian3D AP. Original A2 remains selected; the pilot failed.

M63g confirms both detection and geometry deterioration, not just depth.
On1,274 common moderate Pedestrian matches, depth MAE rose0.7389→0.7809m
and mean3D IoU fell0.2470→0.2377 from A2 to KD, with48 lost/24 gained
detections. Diagnostic matching is not an exact attribution of AP.
Repeat controls share recorded settings and unchanged BN buffers but have
different learned weights. This does not establish which operation caused it.

M63h runs two fresh subprocesses, each with three GT-only updates from A2.
Keep seed20268, batch4, LR1e-5, fresh native optimizer, no augmentation,
FP32 and frozen BN running buffers. Compare input/GT identities, RNG states,
backend flags, losses, outputs, gradients and updated weights. Six updates
total; no KD, full epoch, validation sweep or promoted checkpoint. Temporary
local tensor snapshots are removed after comparison. Three instrumented
batches cannot prove full-run determinism or identify a specific faulty kernel.

Use notebook revision M63-NOTEBOOK-2026-10-02-r9, section17 in the configured
L4 runtime; after reset run setup1–3 then17, skip4–16. Return m63h_results.zip.
Review before choosing a preservation loss or restricting model updates.

## Priority order

1. Freeze A2 epoch130 as the working baseline, not a safety-qualified product.
2. Bound the initial deployment check to native inference and operator inventory.
   Record custom CUDA blockers; do not divert into another long export project.
3. Diagnose useful R0 teacher supervision on training data only (M62 below).
4. After review, freeze one matched 10-epoch GT-only / GT+KD pilot. Keep A2's
   graph and both classes' GT losses. No temperature/architecture sweep.
5. Measure gain against unchanged A2 and control; protect Pedestrian AP/recall.
   Confirm a successful pilot with another seed before longer training.
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
- R0 is not a uniformly better teacher: start with selective Vehicle geometry,
  not blind Pedestrian or all-output imitation. Rejected R0→A1 stays rejected.

## M62 — bounded diagnostic, zero training

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

## Review boundary and later pilot

Return `m62_results.zip`: manifest, native compatibility report, component
summary and per-object geometry rows. Review coverage, near-zero/noisy wins,
tail errors and operating ranges before freezing teacher targets and loss scale.
No trainable model, selected KD weight, or training command is produced here.

The later contract must freeze numerical acceptance limits before training:
Vehicle3D gain vs unchanged A2 and matched control, Pedestrian AP/nearby-recall
preservation, Vehicle BEV/recall limits, exact split and decision epoch. Report
the original five AP gates separately. Keep GT supervision for both classes;
one selected Vehicle component only. If unaugmented caches are used, both arms
must use that identical view; never reuse targets on incompatible transforms.
No class-logit KD means no temperature sweep. A pass needs seed confirmation;
a failure does not automatically authorize extra epochs or more variants.
