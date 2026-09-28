# M63 — R0 to A2 depth-distillation pilot

Revision M63-2026-09-28-r1. Status: prepared; CUDA smoke and paired training pending.
MonoDGP/M61 remain parked. No student architecture change, compression or phone run.

## Evidence and question

Reviewed M62 covers all 3,712 Chen training images. Among 12,127 quality-filtered
Vehicle pairs, R0 depth MAE was 0.197610 m versus A2 0.249763 m. Exactly 7,014
objects meet the predeclared per-object teacher-win rule. This is train-set
evidence of useful candidate supervision, not proof of validation improvement.
A2 is MobileNetV4 Conv Medium plus MonoDETR; teacher is R0 ResNet50 MonoDETR.
Preparation binds the exact reviewed M62 report hashes and regenerates the masks.

Question: can selective R0 depth teaching improve A2 Vehicle 3D performance
without sacrificing Pedestrian performance?

## Frozen experiment

- Same A2 epoch130 checkpoint initializes both arms independently.
- Control: original ground-truth losses only.
- Treatment: same GT losses plus 0.25 × mean absolute student/teacher final-depth
  difference in metres, restricted to the 7,014 approved Vehicle GT objects.
- Match student queries to GT through native Hungarian matching; never equate
  teacher and student query indices. Repeated training query groups contribute.
- All native GT supervision remains for Vehicle and Pedestrian.
- No uncertainty-channel, class-logit, 2D-box, dimension, yaw or Pedestrian KD.
  No temperature: this pilot uses regression targets, not softened logits.
- 10 epochs per arm; fresh native AdamW, learning rate 1e-5, weight decay 1e-4,
  batch4, seed20268, FP32, identical unaugmented inputs/order.
- All student parameters remain trainable. Depth supervision can propagate
  through multiple model components; this is not depth-head-only fine-tuning.
- Fixed weight is a conservative experimental choice, not a tuned optimum.
  Unlike parked M61, the depth difference is not divided by GT depth.

## Execution gates and recovery

Use the M62 NVIDIA L4/software environment. Frozen cache/source/checkpoint
identities cannot be bypassed after runtime changes. M62 files stay read-only.
Restore both original Chen splits, then prepare exact approved targets.
Re-evaluate source A2 on all 3,769 validation images before training; AP must be
within 0.15 points and nearby recall within 0.01 of its recorded baseline.
This new protocol-drift tolerance is not an accuracy target.
CUDA smoke must show finite GT/KD gradients and a nonzero teacher gradient,
with zero optimizer steps. Both runs require a successful matching smoke.

Atomic epoch checkpoints retain optimizer/history and bind manifest, audit,
environment and fresh baseline. A restarted run replays only an unfinished
epoch. Existing mismatched outputs stop instead of being overwritten.
Full training is performed in Colab, not validated on this Mac.

## Fixed epoch-10 decision

No checkpoint sweep or additional hyperparameter variants.
Treatment must satisfy all of:

- Vehicle moderate 3D AP gain >=0.10 percentage points over both fresh A2 and control.
- Pedestrian moderate 3D/BEV AP and nearby recall >= both baseline and control.
- Balanced moderate 3D AP >= both baseline and control.
- Vehicle moderate BEV AP >= the stronger comparator minus0.15 points.
- Vehicle nearby recall >= the stronger comparator minus0.01.

Historical product AP gates remain unchanged and are reported separately:
Vehicle3D15.8713, Pedestrian3D5.1493, mean3D10.5103,
VehicleBEV21.3134, PedestrianBEV5.9365.
Pedestrian nearby recall target remains0.80.
An incremental pilot pass does not imply any of these product gates passed,
statistical significance, cross-dataset robustness, safety qualification,
or iPhone deployability. Review results and consider seed confirmation;
never automatically start a longer run.

## Deliverable and priority

Run notebooks/MonoDETR_M63_R0_A2_Depth_Pilot_Colab.ipynb sections1–10.
Return m63_results.zip: manifest, approved-target audit, fresh A2 baseline,
CUDA smoke, both training summaries and full validation comparison.
Then: review KD benefit → confirm/freeze the chosen student → portable/export
parity → compression if needed → measured iPhone accuracy/latency/thermal gates.
No phone connection is needed for M63.
