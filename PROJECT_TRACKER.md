# MobileADAS3D project tracker

Last updated: 2026-08-14

This is the canonical status page. Update it whenever a task changes state,
an experiment finishes, a gate passes/fails, or the next action changes.
Detailed rationale remains in the linked contracts and handoff.

## Goal

Deploy a reliable monocular 3D road-object detector on iPhone with two product
classes, **Vehicle** and **Pedestrian**, while preserving a reproducible
accuracy reference, Core ML parity, real-time camera performance, external
generalization testing, and complete recording/export artifacts.

## Current position

- Current phase: **S1 supervised health gate**
- Active task: run and review the resumable 20-epoch GT-only
  MobileADAS3D-S1 health gate.
- Training teacher/reference: **R0 ResNet50 MonoDETR, epoch 185**
- Deployment candidate: **MobileADAS3D-S1 with MobileNetV4 Conv Small**
- Knowledge distillation: **not active yet**; it follows the GT-only S1
  baseline as a paired experiment.
- iPhone street recording: **not needed in the current phase**.

## Milestone tracker

| ID | Milestone | Status | Result/evidence |
| --- | --- | --- | --- |
| M1 | iPhone benchmark and recording application | Complete | KITTI parity, custom image, live camera, no-saving benchmark, full artifacts, clean ZIP, share sheet, and frame dropping validated. See `MobileADAS3D_Codex_Handoff_20260715.md`. |
| M2 | Live-camera preprocessing optimization | Complete | Replaced the slow per-pixel path; physical-device preprocessing moved far below the original 540–570 ms bottleneck. |
| M3 | Model-only and pipeline benchmarks | Complete | 5 warmups/100 predictions and 30-second no-saving/full-recording workflows implemented and device-tested. |
| M4 | Legacy MobileNetV4 convolutional model experiments | Complete | v1–v5 geometry, scoring, yaw, checkpoint, and AP experiments documented; accuracy remained below the product goal. |
| M5 | MonoDETR transfer-learning feasibility | Complete | Published Car teacher passed Chen-val with Car moderate 3D AP_R40 20.35. Clean train cache and distillation adapter validated. |
| M6 | Early legacy-student distillation gate | Complete—no authorization | Stable 100-step paired gate showed no meaningful AP benefit, so long training was rejected. |
| M7 | MonoDETR MobileNetV4 backbone-only ablation | Complete | Faster/smaller direction explored; accuracy loss ruled it out as the high-accuracy reference. |
| M8 | Product target and architecture contracts | Complete | Two-class taxonomy, dataset roles, S1 architecture, metrics, and runtime gates frozen in `PRODUCT_MODEL_CONTRACT.md` and `STUDENT_ARCHITECTURE_CONTRACT.md`. |
| M9 | S1 graph and random-weight edge gate | Recheck required | The original 10-head graph passed at 1.403M parameters, 2.155 GMAC, 2.73 MB FP16, iPhone p50 1.878 ms/p95 3.788 ms. Preflight exposed a missing 2D center-offset head; the corrected 11-head graph uses the same supported operators but requires refreshed parity/size/latency evidence before deployment. |
| M10 | Production taxonomy audit | Complete | Car/Van/Truck/Tram → Vehicle; Pedestrian/Person_sitting → Pedestrian. Chen train/val counts and hashes locked. |
| M11 | Two-class R0 protocol and evaluator | Complete | Separate KITTI-difficulty product-taxonomy AP_R40 implemented without changing official KITTI behavior. |
| M12 | R0 supervised training | Complete | 195 epochs completed in 5h23m with 39 durable Drive checkpoints. |
| M13 | R0 product checkpoint sweep | Complete | All 39 checkpoints evaluated on 3,769 Chen-val images; epoch 185 selected by balanced moderate 3D AP_R40. |
| M14 | GT-only S1 supervised baseline | In progress | Preflight exposed and fixed a missing 2D `center_offset` head required by the frozen target/loss/decoder geometry. The corrected 11-head graph must pass preflight before the 20-epoch GT-only health gate. |
| M15 | Paired S1 knowledge-distillation experiment | Pending | Start from the same S1 initialization and change only the approved R0 auxiliary teacher losses. |
| M16 | nuScenes zero-shot evaluation | Pending | Validate adapter on mini, then run the frozen LiDAR-supported external protocol. |
| M17 | Trained S1 Core ML parity | Pending | Export the selected trained checkpoint and enforce ≤1% relative AP degradation and depth parity. |
| M18 | Final physical-iPhone qualification | Pending | Runtime, sustained thermal, no-saving/full-recording, and artifact gates. |
| M19 | Deployment decision | Pending | Approve only if accuracy, generalization, parity, runtime, stability, and artifact gates all pass. |

## Frozen R0 reference

| Field | Value |
| --- | --- |
| Architecture | Original ResNet50 MonoDETR |
| Upstream commit | `6994b9f512400b258c6edb75f77423beb9c126f2` |
| Selected checkpoint | `checkpoint_epoch_185.pth` |
| Checkpoint SHA-256 | `fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59` |
| Evaluated checkpoints | 39, epochs 5–195 |
| Validation images/checkpoint | 3,769 |
| Score threshold / top-k | 0.001 / 50 |
| Vehicle moderate 3D AP_R40 | 17.6348 |
| Pedestrian moderate 3D AP_R40 | 5.7214 |
| Balanced moderate 3D mean | 11.6781 |
| Vehicle moderate BEV AP_R40 | 23.6816 |
| Pedestrian moderate BEV AP_R40 | 6.5961 |

R0 is an accuracy teacher/reference only. It is not deployed on iPhone.

## S1 acceptance denominators

The selected GT-only or distilled S1 checkpoint must be evaluated with the
same frozen product protocol. At minimum:

| Gate | Required |
| --- | ---: |
| Vehicle moderate 3D AP_R40 | ≥13.2261 (75% of R0) |
| Pedestrian moderate 3D AP_R40 | ≥4.2910 (75% of R0) |
| Vehicle moderate BEV AP_R40 | ≥20.0 |
| Complete Chen validation | 3,769/3,769 prediction files |

The full nearby-recall, nuScenes, Core ML, and iPhone gates remain mandatory;
passing these three AP values alone does not authorize deployment.

## Immediate execution plan

1. Record the R0 path, SHA-256, protocol parameters, and denominators in the S1
   run manifest; fail closed if any value changes.
2. Prepare a fresh GT-only S1 Colab notebook/config with Drive checkpoints,
   visible logs, automatic resume, and no teacher cache/losses.
3. Run a 20-epoch health gate and review losses, finite gradients, both classes,
   prediction completeness, and product AP.
4. Continue the same run only if healthy; save checkpoints at fixed epochs.
5. Sweep S1 checkpoints using the frozen R0 product evaluator and select using
   per-class moderate 3D AP plus nearby recall—not validation loss alone.
6. Freeze the GT-only S1 checkpoint and results.
7. Run one paired distillation experiment from the identical S1 initialization.
   Keep distillation only if it improves the frozen comparison without harming
   Pedestrian or nearby safety metrics.

## Decision rules

- Do not use validation data for optimization or target statistics.
- Do not call merged Vehicle AP an official KITTI leaderboard metric.
- Do not select a checkpoint from aggregate loss or Vehicle/Car AP alone.
- Do not change architecture, taxonomy, split, decoding, threshold, or
  selection rule inside a run; create a new versioned experiment.
- Do not deploy R0 or the older MobileMonoDETR graph on iPhone.
- Do not begin distillation until the GT-only S1 baseline is frozen.
- Every completed task updates this tracker, the handoff, and the plan before
  the next task begins.

## Canonical documents

- Product gates and priorities: `PRODUCT_MODEL_CONTRACT.md`
- S1 graph: `STUDENT_ARCHITECTURE_CONTRACT.md`
- R0 protocol: `TWO_CLASS_REFERENCE_PROTOCOL.md`
- Full chronological evidence: `MobileADAS3D_Codex_Handoff_20260715.md`
- Current status and next task: this file
