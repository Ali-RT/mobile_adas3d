# MobileADAS3D project tracker

Last updated: 2026-08-20

This is the canonical status page. Update it whenever a task changes state,
an experiment finishes, a gate passes/fails, or the next action changes.
Detailed rationale remains in the linked contracts and handoff.

## Goal

Deploy a reliable monocular 3D road-object detector on iPhone with two product
classes, **Vehicle** and **Pedestrian**, while preserving a reproducible
accuracy reference, Core ML parity, real-time camera performance, external
generalization testing, and complete recording/export artifacts.

## Current position

- Current phase: **MobileADAS3D-H1 GT-only 20-epoch health gate**
- Active task: run `notebooks/MobileADAS3D_H1_GT_Gate_Colab.ipynb` top-to-bottom
  on a Colab GPU and return the epoch-20 AP table and training summary.
- Training teacher/reference: **R0 ResNet50 MonoDETR, epoch 185**
- Deployment candidate: **MobileADAS3D-H1 teacher-shaped hybrid student**
- Knowledge distillation: **not active yet**; it follows the GT-only H1
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
| M14 | GT-only S1 supervised baseline | Failed—root cause isolated | Vehicle orientation axis was good (9.63° mean/4.71° p50), but the independent direction bit produced a 35.8% flip-candidate rate and 72.3° final yaw MAE. Do not resume this run. |
| M14b | S1-V2 continuous-yaw experiment | Failed—invalid loss | Epoch-20 Vehicle/Pedestrian moderate 3D AP_R40 was 0.133/0.256 and BEV 0.801/0.652. Direct dot-product cosine loss was not scale-invariant: yaw cosine loss fell to about -1.29 and total train loss became negative. Epochs 21–100 were run but not product-evaluated. Do not use any S1-V2 checkpoint. |
| M14c | S1-V2b bounded continuous-yaw experiment | Epoch-20 gate failed accuracy | Training was numerically healthy and stopped at 20. Vehicle/Pedestrian moderate 3D AP_R40 was 0.103/0.178 and BEV 0.581/0.454. Vehicle/Pedestrian mean yaw error was 40.20°/72.76°. This is far below R0 retention; continuation and distillation were denied. |
| M14d | Dense S1 family closure | Complete—rejected | User elected to move on without the optional 5/10/15/20 sweep. S1 is frozen as a speed-qualified but accuracy-inadequate baseline; no checkpoint is eligible for distillation or deployment. |
| M15 | Paired S1 knowledge-distillation experiment | Cancelled | S1 never produced a healthy accuracy baseline, so distillation was not authorized. |
| M16 | nuScenes zero-shot evaluation | Pending | Validate adapter on mini, then run the frozen LiDAR-supported external protocol. |
| M17 | Trained student Core ML parity | Pending | Export the selected H1 checkpoint and enforce ≤1% relative AP degradation and depth parity. |
| M18 | Final physical-iPhone qualification | Pending | Runtime, sustained thermal, no-saving/full-recording, and artifact gates. |
| M19 | Deployment decision | Pending | Approve only if accuracy, generalization, parity, runtime, stability, and artifact gates all pass. |
| M20 | H1 teacher-shaped hybrid contract | Complete | MobileNetV4 + Lite-FPN + fixed standard depth-aware encoder/query decoder is frozen in `HYBRID_STUDENT_ARCHITECTURE_CONTRACT.md`. |
| M21 | H1 random graph and edge preflight | Complete | 3.619M parameters, 4.907 GMAC, 10.35 MB FP16, no custom ops, FP16 raw delta 0.001941. iPhone 16 Pro Max CPU+NE, 5 warmups/100 runs: mean 5.042 ms, median 4.924 ms, p95 5.804 ms, max 7.137 ms. See `artifacts/h1_edge_preflight_20260821.json`. |
| M22 | H1 GT-only health-gate workflow | Training rerun pending | Query-native KITTI targets, Hungarian matching/set loss, H1 KITTI decoder, fail-closed provenance preparation, Drive checkpoints/logging, automatic resume, and complete product AP_R40 evaluation are implemented in `MobileADAS3D_H1_GT_Gate_Colab.ipynb`. Distillation is false. Real Colab execution exposed and fixed criterion device placement plus FP16/FP32 quality-target and Hungarian-cost handling. Resume the Drive-backed training run. |

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
   prediction completeness, and product AP. **Completed but failed accuracy:**
   stable epoch-20 train/val loss 1.2501/3.6891; moderate 3D AP_R40
   Vehicle/Pedestrian 0.024/0.519.
4. Sweep epochs 5/10/15/20. **Completed:** epoch 20 was best, but Vehicle
   moderate 3D/BEV was only 0.024/0.246; current continuation is denied.
5. Run the epoch-20 matched-geometry diagnostic and identify whether the first
   failure is 2D recall/localization, depth, dimensions, yaw, or 3D placement.
   **Completed:** 2D matches exist; yaw/shape/3D placement are the main errors.
6. Measure axis-aware yaw error and front/back flip rate from the saved matched
   CSV. **Completed:** Vehicle axis mean/p50 was 9.63°/4.71°, but flip rate was
   35.8%; reject the independent hard direction-bit representation.
7. S1-V2 completed epoch 20 and then continued to epoch 100, but reject the
   whole run: its unnormalized dot-product cosine objective was unbounded and
   drove yaw and total training loss negative. The epoch-20 AP/geometry result
   is diagnostic only, not a valid candidate.
8. **Completed but failed accuracy:** S1-V2b trained cleanly for 20 epochs and
   completed all 3,769 predictions, but moderate 3D AP_R40 was only
   `0.103/0.178` Vehicle/Pedestrian. Do not continue this run.
9. **Waived by move-on decision:** do not spend more evaluation time sweeping
   S1-V2b. Freeze the complete dense S1 family as rejected.
10. **Completed:** freeze MobileADAS3D-H1's teacher-shaped hybrid architecture.
11. **Local preflight complete:** H1 passes unit, trace, parameter, compute,
   package, operator, and FP16 parity checks (`0.001941` <= `0.002`).
12. **Physical edge gate complete:** iPhone 16 Pro Max CPU+NE p95 was
   `5.804 ms` over 100 timed predictions after 5 warmups (gate <=35 ms).
13. **Prepared:** run the fresh GT-only H1 20-epoch health gate with durable
   Drive checkpoints, visible logs, automatic resume, and full product AP.
   Distillation stays off.
14. Sweep a future healthy H1 run using the frozen R0 product evaluator and
   select using per-class moderate 3D AP plus nearby recall—not validation
   loss alone.
15. Freeze the GT-only H1 checkpoint and results.
16. Run one paired distillation experiment from the identical initialization.
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

- Reporting summary: `MODEL_MILESTONES_AND_ARCHITECTURE.md`
- Product gates and priorities: `PRODUCT_MODEL_CONTRACT.md`
- S1 graph: `STUDENT_ARCHITECTURE_CONTRACT.md`
- H1 active graph: `HYBRID_STUDENT_ARCHITECTURE_CONTRACT.md`
- R0 protocol: `TWO_CLASS_REFERENCE_PROTOCOL.md`
- Full chronological evidence: `MobileADAS3D_Codex_Handoff_20260715.md`
- Current status and next task: this file
