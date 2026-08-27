# MobileADAS3D project tracker

Last updated: 2026-08-27

This is the canonical status page. Update it whenever a task changes state,
an experiment finishes, a gate passes/fails, or the next action changes.
Detailed rationale remains in the linked contracts and handoff.

## Goal

Develop a reliable monocular 3D road-object detector with two product classes,
**Vehicle** and **Pedestrian**. First obtain accuracy comparable to the frozen
teacher; then compress and qualify the selected model for a deployment target.
The validated iPhone application remains available, but its hardware limits do
not constrain the current accuracy-development stage.

## Current position

- Current phase: **accuracy-first teacher-compatible student development**
- Active task: close rejected A2b and define one controlled A2c intervention
  that changes class-specific supervision rather than image frequency.
- Training teacher/reference: **R0 ResNet50 MonoDETR, epoch 185**
- Accuracy candidate: **MobileMonoDETR-Student-A2 epoch 130 (frozen diagnostic baseline)**
- S1/H1/H2 status: **frozen negative experiments; do not resume**.
- Knowledge distillation: **completed and rejected for A1**; it did not improve
  balanced accuracy and should not be retuned or resumed.
- iPhone model constraints: **suspended during accuracy development**; restore
  them after an accuracy-qualified student is frozen.
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
| M22 | H1 GT-only health-gate workflow | Complete—learning gate failed | Query-native training ran stably for 20 epochs with complete product AP evaluation. Runtime/device/AMP defects were fixed, but the model produced 0.00 AP_R40 and excessive background proposals. Do not resume it. |
| M23 | H1 v1 GT-only learning gate | Failed, diagnosed | The run produced 123,303 detections (32.72/image) with overwhelming false positives. Best validation loss was 7.443909 at epoch 9; latest epoch 20 was 7.742177. Query heads learned plausible geometry priors without reliable object presence/background ranking. |
| M24 | H1 v2 tiny-overfit workflow | Failed—partial separation only | The 16-image/400-step run completed. Matched score median was 0.172, unmatched p95 0.188, matched mean 2D IoU 0.258, and predictions averaged 15.63/image versus 3.94 GT. All four gates failed. Do not run full KITTI or distillation. |
| M25 | H1 v2 single-image capacity workflow | Complete—passed | Sample 000010 memorized all 9 objects: matched-score median 0.732, unmatched p95 <0.000001, matched mean 2D IoU 0.825, and predicted/GT count 9/9. Cross-image sensitivity also passed with zero repeat delta and substantial changes in every output head. See `artifacts/h1_v2_single_image_gate_20260824.json`. |
| M26 | H1 v2 staged Tiny16 optimization gate | Complete—failed | No milestone passed. From steps 400→2000, matched-score median changed 0.172→0.261, unmatched p95 worsened 0.188→0.367, mean IoU improved 0.258→0.425, and predictions/image changed 15.63→14.06 versus 3.94 GT. Step 1600 was the best compromise but still failed every gate. See `artifacts/h1_v2_tiny_2000step_gate_20260824.json`. |
| M27 | H1 v2 assignment and normalization diagnosis | Complete—matching instability isolated | Batch-statistics inference produced only small mixed changes, ruling out BatchNorm as the primary cause. Adjacent same-query rate was 7.14%, fully stable object rate 0%, and objects used 4.44 unique queries across five checkpoints on average. See `artifacts/h1_v2_assignment_normalization_20260824.json`. |
| M28 | H2 spatial-reference query graph and contract | Complete—local preflight passed | Preserves MobileNetV4, 3,619,457 parameters, transformer dimensions, 50 queries, and nine output shapes. Adds a fixed 10×5 reference grid, positional query encoding, and ±0.10 bounded box/projected-center offsets. All 125 tests passed with one expected CUDA skip. See `H2_SPATIAL_REFERENCE_QUERY_CONTRACT.md`. |
| M29 | H2 single-image capacity gate | Complete—failed localization | Confidence median 0.724, unmatched p95 0.000333, count 9/9, and image sensitivity passed, but matched mean IoU was 0.555 versus the 0.70 gate and H1's 0.825 on the same image. Median IoU 0.702 indicates a small set of severe localization outliers. See `artifacts/h2_single_image_gate_20260824.json`. |
| M30 | H2 Tiny16 capacity and assignment gate | Blocked by M29 | Do not run until the H2 single-image localization failure is resolved and the unchanged gate passes. |
| M31 | H2 reference-offset reachability diagnostic | Cancelled by strategy pivot | H2 remains reproducible, but further custom-query debugging is lower value than returning to the proven MonoDETR learning path. |
| M32 | Accuracy-first student contract | Complete | H1/H2 frozen; 90%-of-R0 comparable-performance gates and the MobileNetV4-MonoDETR A1 sequence are locked in `ACCURACY_FIRST_STUDENT_CONTRACT.md`. |
| M33 | Two-class A1 GT-only baseline workflow | Complete | `MonoDETR_A1_MobileNetV4_Two_Class_GT_Colab.ipynb` is Drive-backed, verbose, restartable, and fail-closed on the frozen R0 epoch/hash. The preparer transfers every compatible downstream R0 tensor, changes only backbone/projections, fixes the initialization seed, and disables distillation. The restartable sweep reports all five 90%-of-R0 gates. |
| M34 | Two-class A1 GT-only baseline run | Complete—healthy, below final gate | All 195 epochs and the product sweep completed. Epoch 140 won by balanced moderate 3D AP_R40: Vehicle 12.8604, Pedestrian 7.2669, mean 10.0636; BEV was 18.7852/8.5095. Pedestrian exceeded R0, but Vehicle 3D/BEV and balanced mean missed the 90% gates. Freeze epoch 140 as the GT-only comparison baseline, not an accuracy-qualified student. |
| M35 | Paired A1 distillation experiment | Complete—rejected | Vehicle moderate 3D AP_R40 changed `12.8604→12.9428` (`+0.0824`), but Pedestrian changed `7.2669→5.3299` (`-1.9370`), balanced mean changed `10.0636→9.1364` (`-0.9273`), and both BEV metrics regressed. The negligible Vehicle gain does not justify the material balanced/Pedestrian loss. Do not resume or retune this branch. |
| M36 | Accuracy-qualified student selection | Complete—none qualified | Neither GT-only A1 nor distilled A1 passed all five 90%-of-R0 gates. Epoch-140 GT-only A1 remains the stronger comparison baseline; no A1 checkpoint is eligible for compression or deployment. |
| M37 | Post-accuracy compression ladder | Blocked | Test FP16, INT8/QAT, structured pruning, depth/width/token reduction, and low-rank changes only after an accuracy-qualified student exists. |
| M38 | Deployment qualification | Blocked | Select hardware target and restore conversion/parity/runtime/stability gates only after accuracy qualification. |
| M39 | Higher-capacity A2 backbone experiment | Complete—strongest student, near gate | All 195 epochs and 39 complete checkpoint evaluations succeeded. Epoch 130 uniquely ranked first and was best for Vehicle 3D, Pedestrian 3D, balanced 3D mean, and Vehicle BEV. It passed four gates; Vehicle moderate 3D was `15.4573` versus `15.8713` (short `0.4140`, retaining `87.65%` of R0). Freeze SHA-256 `ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4`. |
| M40 | A2 epoch-130 nearby-recall and geometry diagnosis | Complete—Vehicle passes, Pedestrian fails | All 3,769 validation images were audited. Vehicle <40 m recall was `88.29%` (passes `85%`); Pedestrian <30 m recall was `69.22%` (fails `80%`). Vehicle matched 2D IoU was strong (`0.818` overall), but 3D IoU was `0.414`; yaw MAE was `42.64°` with p90 `177.80°`, exposing front/back flips. Vehicle depth MAE rose from `0.65 m` at 0–20 m to `1.72 m` at 20–40 m and `3.67 m` at 40–60 m. Pedestrian recall/objectness is the primary product failure. |
| M41 | A2b Pedestrian-balanced accuracy experiment | Complete—rejected | All 195 epochs and the complete product sweep finished. Epoch 150 was selected by the frozen balanced-3D rule: Vehicle/Pedestrian moderate 3D `15.2811/7.4720`, mean `11.3765`, BEV `21.0396/8.0988`. It passed only 3/5 gates and regressed versus A2 epoch 130 by `0.1762/0.0608/0.1185/0.3354/0.3904` across Vehicle 3D, Pedestrian 3D, mean 3D, Vehicle BEV, and Pedestrian BEV. Image-level repetition is rejected; do not resume or audit this checkpoint. |
| M42 | Temperature study | Deferred | A1 distillation used temperature `2.0`, but no temperature comparison or sweep was run. Temperature affects teacher/student soft targets and did not participate in GT-only A2/A2b. Reconsider only as a small paired gate if future teacher supervision is justified. |
| M43 | A2c class-specific supervision experiment | Define next | Preserve frozen A2 epoch 130 as the comparison baseline. Inspect pinned MonoDETR matching/classification losses and design one class-specific Pedestrian objectness/matching intervention; avoid repeating whole images, changing the backbone, adding yaw changes, or introducing temperature in the same run. |

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

## Accuracy-first A1 acceptance denominators

The selected GT-only or distilled A1 checkpoint must be evaluated with the
same frozen product protocol. Comparable performance means at least 90% of R0
for every frozen moderate 3D/BEV metric:

| Gate | Required |
| --- | ---: |
| Vehicle moderate 3D AP_R40 | ≥15.8713 |
| Pedestrian moderate 3D AP_R40 | ≥5.1493 |
| Balanced moderate 3D mean | ≥10.5103 |
| Vehicle moderate BEV AP_R40 | ≥21.3134 |
| Pedestrian moderate BEV AP_R40 | ≥5.9365 |
| Complete Chen validation | 3,769/3,769 prediction files |

Nearby recall and external validation remain accuracy-selection requirements.
Core ML and iPhone gates are deferred until after the accurate student is
frozen; passing AP does not by itself authorize deployment.

## Immediate execution plan

1. **Completed:** preserve S1/H1/H2 artifacts and mark those families rejected;
   cancel further H2 reachability and Tiny16 work.
2. **Completed:** freeze the accuracy-first A1 architecture, 90%-of-R0 gates,
   experiment discipline, and post-accuracy compression ladder.
3. **Completed:** prepare the two-class MobileNetV4-MonoDETR GT-only Colab
   workflow with exact provenance, deterministic initialization, durable resume,
   visible logs, and a restartable five-gate product sweep.
4. **Completed:** A1 trained through epoch 195 and all durable checkpoints were
   swept. Epoch 140 is frozen as the healthy GT-only baseline but missed three
   gates: Vehicle 3D, balanced 3D mean, and Vehicle BEV.
5. **Completed:** freeze the A1 initialization and epoch-140 GT-only result
   before adding teacher losses.
6. **Completed—rejected:** paired A1 distillation produced only `+0.0824`
   Vehicle moderate 3D AP_R40 while reducing Pedestrian by `1.9370`, balanced
   mean by `0.9273`, Vehicle BEV by `0.7449`, and Pedestrian BEV by `1.7519`.
7. **Completed:** A2 trained through epoch 195 and all 39 checkpoints were
   evaluated. Freeze epoch 130 as the strongest diagnostic baseline; no hidden
   checkpoint passes all five gates.
8. **Completed:** the A2 epoch-130 diagnostic found Vehicle <40 m recall
   `88.29%` (pass), Pedestrian <30 m recall `69.22%` (fail), strong Vehicle 2D
   IoU `0.818`, weak Vehicle 3D IoU `0.414`, and severe yaw front/back flips
   (`42.64°` mean, `177.80°` p90).
9. **Completed—rejected:** A2b 2× Pedestrian-image sampling regressed all
   five AP metrics and passed only three gates. Do not resume it or spend an
   additional nearby-recall inference run on its ineligible checkpoint.
10. **Next:** inspect pinned MonoDETR matching and classification supervision,
    then define one A2c class-specific intervention. Keep A2 epoch 130 frozen;
    do not mix sampling, yaw, temperature, architecture, or deployment changes.
11. Select and freeze a student only if every comparable-performance gate and
    nearby-recall review passes.
12. Run locked external validation, then compress the frozen student one change
    at a time and restore deployment-specific parity/runtime qualification.

## Decision rules

- Do not use validation data for optimization or target statistics.
- Do not call merged Vehicle AP an official KITTI leaderboard metric.
- Do not select a checkpoint from aggregate loss or Vehicle/Car AP alone.
- Do not change architecture, taxonomy, split, decoding, threshold, or
  selection rule inside a run; create a new versioned experiment.
- Do not resume S1, H1, or H2.
- Do not continue or retune the rejected A1 distillation branch.
- Do not reject A1 for current iPhone limits during accuracy development.
- Every completed task updates this tracker, the handoff, and the plan before
  the next task begins.

## Canonical documents

- Reporting summary: `MODEL_MILESTONES_AND_ARCHITECTURE.md`
- Product gates and priorities: `PRODUCT_MODEL_CONTRACT.md`
- S1 graph: `STUDENT_ARCHITECTURE_CONTRACT.md`
- H1 active graph: `HYBRID_STUDENT_ARCHITECTURE_CONTRACT.md`
- Accuracy-first A1 contract: `ACCURACY_FIRST_STUDENT_CONTRACT.md`
- R0 protocol: `TWO_CLASS_REFERENCE_PROTOCOL.md`
- Full chronological evidence: `MobileADAS3D_Codex_Handoff_20260715.md`
- Current status and next task: this file
