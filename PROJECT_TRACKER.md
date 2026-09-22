# MobileADAS3D project tracker

Last updated: 2026-09-22

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

- Current phase: **M60a complete—device parity passed; latency failed**.
  M59i remains complete; its conversion-preservation gates passed.
- All 3,769 delivered images/calibration/original labels and frozen hashes pass.
  Original trace restored from the prior archive and all five upstream decoder
  files match. No additional dataset upload is needed.
- The initial attempt stopped before inference: Mac preprocessing differs from
  the frozen Colab anchor at 11 image-channel values (one 8-bit intensity step).
  Across the 16 frozen inputs, 306 image-channel values differ; calibration and
  original dimensions match exactly. Pinning Pillow/OpenCV and a strict-FP
  Pillow build did not resolve it. Cause is not yet conclusively isolated.
- Docker startup approved. Linux/x86 with the frozen Colab library versions
  reproduces all 16 reference inputs **bit-for-bit** (zero changed values).
  All 3,769 source-bound tensors were collected and independently verified by
  the Mac evaluator. Paired inference and complete metric evaluation finished.
  No limits, transforms or weights changed; 57 regression tests pass.
- **All eight gated metrics are identical** in PyTorch and Core ML: moderate
  3D Vehicle19.380285/Pedestrian6.192034/mean12.786160; moderate BEV
  Vehicle25.737363/Pedestrian6.784285; nearby recall0.909847/0.723104;
  Pedestrian localization-failure rate0.238536. Both prediction sets contain
  the exact 3,769 IDs. All full-set raw checks and fixed16 numerical gates pass.
- Residuals are retained, not hidden: 93 images flag extended diagnostics
  outside fixed16 (68 confidence-only, eight box-coordinate-only, 17 order-only
  cases with unchanged candidate membership). Hard Pedestrian 3D AP differs by
  +0.000236 points; other AP entries are equal. Native .2f formatting unchanged.
  Evidence: `artifacts/m59i_full_validation_gate_20260921.json` and
  `MONODGP_M59I_FULL_VALIDATION_CONTRACT.md`. Historical M59g remains failed.
- User authorized the bounded physical-device feasibility step. M60a has an
  isolated iPhone application, exact model/fixed16 resources and independent
  host review. First protocol: parity, 5 warmups/100 model-only predictions,
  60-second stability loop. The complete physical-device run was retrieved and
  independently reviewed: all 16 cases pass both PyTorch and Mac parity.
  See `MONODGP_M60_DEVICE_FEASIBILITY_CONTRACT.md`. No live camera, training,
  quantization or deployment is launched; the legacy v7 app remains untouched.
- iPhone 16 Pro Max (iOS26.6.2) is wired, paired and Developer Mode is enabled.
  User restored Xcode sign-in; signed Release build and installation passed.
  Initial launch was denied; launch succeeded after user trusted the developer.
  Signed app resources retain the exact prepared hash. Benchmark completed:
  initial p95 233.73 ms; sustained p95 305.50 ms, both above the 50 ms target.
  Thermal nominal→fair, no crash/thermal stop in 60.262 seconds. Sampled peak
  RSS 491.67 MiB includes 90.00 MiB of cached inputs. No camera/FPS claim.
  Evidence: `artifacts/m60_device_installation_20260922.json`. The prior signing
  blocker is retained as historical evidence, not an active blocker.
- Result: `artifacts/m60_device_review_20260922.json`; raw device report:
  `artifacts/m60_device_report_20260922.json`. Evaluator exit1 is the expected
  latency-gate rejection, not an incomplete test. Next proposal is one bounded
  device operator/compute-placement profile before selecting an optimization.
  No automatic model change, quantization sweep or live-camera integration.
- Selected accuracy parent: **M54 MonoDGP epoch 100**, checkpoint SHA-256
  `8e79f3921d96e1de70cbb4219245e3fcc3fa1fb67ae675468b4ebca90e579847`.
- Legacy accuracy reference: **R0 ResNet50 MonoDETR, epoch 185**.
- Converted candidate: **MonoDGP M59f FP32 (M54/M56d weights)**; Mac preservation
  and fixed16 iPhone parity passed, but physical-device latency failed. MobileMonoDETR A2
  epoch130 remains a legacy diagnostic-only student.
- Open product gap: **latest paired Pedestrian nearby recall 0.723104 vs target
  0.80** (prior M54/native reports were approximately0.72487).
- S1/H1/H2 status: **frozen negative experiments; do not resume**.
- Knowledge distillation: **completed and rejected for A1**; it did not improve
  balanced accuracy and should not be retuned or resumed.
- iPhone model constraints: **now measured as M60 feasibility**, not used to
  retroactively change accuracy gates. The nearby-recall product gap remains.
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
| M43 | A2c class-specific supervision experiment | Complete - rejected | All four paired five-epoch branches completed. Weight `2.5` was strongest: Vehicle/Pedestrian moderate 3D `15.4745/7.5128`, mean `11.4937`, BEV `21.0777/8.6164`. Versus control it improved Pedestrian nearby recall only `0.00529` (required `0.02`) and reduced Vehicle BEV by `0.2462` AP (maximum allowed `0.15`). No branch was eligible; `selected=null` and `full_run_authorized=false`. Do not continue A2c. |
| M44 | A2 Pedestrian false-negative localization diagnosis | Complete - localization dominant | Corrected M44 exactly reconciled with frozen nearby recall: `1570/2268 = 69.224%`. Of 698 nearby misses, localization failure was `522` (`74.8%`), sub-threshold but well-localized query `77` (`11.0%`), missing query `84` (`12.0%`), assignment conflict `15` (`2.1%`), and wrong-class classification `0`. Localization worsened with difficulty, occlusion, and distance: detected/localization rates were `41.3%/48.1%` for hard, `38.5%/50.4%` at occlusion 2, and `40.3%/37.5%` at 20-30 m. |
| M45 | A2d Pedestrian localization-supervision gate | Complete—rejected | All four paired five-epoch branches completed. Weight `1.5` was least harmful: Pedestrian nearby recall improved only `+0.00176` (required `+0.02`) and localization-failure rate fell only `0.00397` (required `0.02`); Vehicle 3D changed `-0.0930` AP and BEV `+0.0230`. Weights `2.0`/`2.5` damaged Vehicle 3D and BEV beyond the allowed `0.15` AP, while `2.5` also reduced Pedestrian recall. No branch was eligible; `selected=null`, `full_run_authorized=false`. |
| M46 | A2e size-aware Pedestrian localization gate | Complete—rejected | All four paired five-epoch branches completed. Weight `2.0` produced the only positive recall/localization signal: nearby recall `+0.00353` and localization-failure reduction `0.00132`, far below both required `0.02` gates, while Vehicle 3D/BEV changed `-0.2902/-0.1901` AP. Weight `1.5` worsened recall/localization and Vehicle BEV; weight `2.5` reduced Pedestrian recall and BEV. No branch was eligible; `selected=null`, `full_run_authorized=false`. |
| M47 | A2f higher-resolution feature gate | Complete—rejected | A2f stride 4 reduced Vehicle/Pedestrian moderate 3D AP_R40 by `6.2316/4.0359`, reduced Pedestrian nearby recall by `0.01940`, and increased localization failure by `0.06305` versus its paired control. No full run is authorized. |
| M48 | R1 Pedestrian matcher-localization gate | Complete—rejected | The 2× Pedestrian assignment-cost branch improved nearby recall only `+0.00044` and reduced localization failures only `0.00176`, versus required `0.02/0.02`. It changed Vehicle 3D/BEV by `-0.15445/-0.06775` AP and regressed Pedestrian 3D/BEV by `-0.02964/-0.15296` AP. `selected=null`; no full R1 is authorized. |
| M49 | R2 stride-4 Pedestrian refinement gate | Complete—rejected, positive signal | Versus paired control, R2 improved Pedestrian nearby recall `+0.01190`, reduced localization failures `0.00529`, and improved Pedestrian 3D AP `+0.35580`; however, gains missed the `0.02/0.02` gates, Vehicle BEV regressed `0.18017` AP (limit `0.15`), and Pedestrian BEV regressed `0.02525`. `selected=null`; no full R2 run is authorized. |
| M50 | R2b frozen hard-gated refinement | Complete—rejected; refinement family closed | R2b preserved R0 almost exactly and slightly improved Pedestrian 3D/BEV AP (`+0.04435/+0.04960`), but nearby recall improved only `+0.00220` and localization failures fell only `0.00132`, versus required `0.02/0.02`. `selected=null`; no full run is authorized. R1, R2, and R2b establish that this local-refinement family does not close the safety gap. |
| M51 | Post-R2b model governance | Complete—contract frozen | `R0_COMPRESSION_CONTRACT.md` freezes R0 epoch 185/hash as the immutable parent. Candidates must retain 95% of every R0 AP metric, lose at most one absolute point of per-class nearby recall, add at most one point of Pedestrian localization failures, and evaluate all 3,769 images. Passing preserves R0 only; the unmet `0.80` Pedestrian target remains aspirational. |
| M52 | Selective mixed-precision R0 gate | Complete—all preservation gates passed | The expanded FP32 feature/depth/custom-attention policy passed its CUDA smoke and the complete 3,769-image Chen validation. Vehicle/Pedestrian moderate 3D AP_R40 was `17.6399/5.6952`, balanced mean `11.6676`, and BEV `23.6395/6.6603`. Nearby recall was `0.88215/0.68519`; Pedestrian localization-failure rate was `0.24559`. All nine frozen AP, recall, localization, and completeness gates passed with the exact R0 checkpoint hash. This authorizes the next compression rung but does not qualify product safety. The run did not include a comparable FP32 timing/memory baseline, so no speedup claim is authorized. |
| M53 | Official MonoDGP Car reference reproducibility | Complete—reference reproduced | The schema-v2 report confirms the pinned source/checkpoint/split, all 3,769 predictions, and every provenance/completeness/tolerance gate. MonoDGP's native official evaluator produced Car 3D AP_R40 `30.1185/22.6797/19.3705`, only `0.0129/0.0312/0.0273` below published `30.1314/22.7109/19.3978`. The MobileADAS3D independent evaluator produced `29.8114/22.1628/18.7457` and remains a diagnostic, not the published-result gate authority. `reference_reproduced=true`; `two_class_adaptation_authorized=true`; `product_safety_qualified=false`. |
| M54 | MonoDGP Vehicle/Pedestrian adaptation | Complete—new accuracy parent selected | All 12 frozen checkpoints were evaluated on all 3,769 Chen validation images. Epoch 100/hash `8e79f392…e579847` ranked first and passed all eight R0-comparability gates: moderate 3D Vehicle/Pedestrian/mean `19.4519/6.1749/12.8134`, moderate BEV `25.7751/6.7676`, nearby recall `0.90993/0.72487`, and Pedestrian localization-failure rate `0.23765`. Relative to R0, gains were `+1.8171/+0.4535/+1.1353` 3D AP, `+2.0935/+0.1715` BEV AP, and `+0.02746/+0.04145` nearby recall. M54 becomes the high-capacity accuracy parent. Pedestrian recall remains `0.07513` below the separate `0.80` product target, so `product_safety_qualified=false`. |
| M55 | M54 compression baseline and export feasibility | Complete—all feasibility gates passed | Exact M54 epoch-100/hash validation and all 19 checks passed. Untouched FP32 M54 has 42.164M parameters, 168.65 MB parameter bytes, a 495.99 MB checkpoint, and a partial 143.56 GFLOP lower bound. On RTX PRO 6000 Blackwell, batch-one 1280×384 model-only latency was 12.67 ms mean/12.72 ms p95 over 100 runs after five warmups; peak allocated/reserved CUDA memory was 492.97 MB/1.086 GB. Conv2d/Linear weights cover 156.33 MB (92.69%), authorizing M56. Direct Core ML remains unauthorized because nine custom MSDeformAttn modules require decomposition/replacement and raw-output parity. See the dated M55 artifacts. |
| M56 | M54 FP16 parameter-storage sensitivity | Complete—rejected at CUDA smoke | Every provenance, storage, size, structure, finite-output, timing, and safety-claim gate passed except raw depth parity. Final `pred_depth` max absolute delta was `0.710739` versus the frozen `0.500000` limit (mean delta `0.012378`). All other output families passed. Same-environment mean/p95 latency was `12.9778/13.0003 ms` versus M55 `12.6725/12.7235 ms` (`+2.41%/+2.18%`), confirming no speed benefit. Full validation is unauthorized; do not change the M56 threshold post hoc. |
| M56b | Selective FP16 storage with FP32 depth geometry | Complete—rejected at CUDA smoke | Exact source/evidence, storage, alias, size, structure, finite-output, and all non-depth parity checks passed. Final `pred_depth` max absolute delta was `0.709734` versus the unchanged `0.500000` limit; decoded depth caused the failure while the log-variance channel delta was only `0.007577`. Mean/p95 latency was `12.6467/12.6813 ms`, effectively unchanged from M55. Full validation was correctly blocked. |
| M56c | Grouped FP16 parameter-sensitivity diagnosis | Complete—2D transformer isolated | All 16 policies and every diagnostic-integrity gate passed. `det2d_transformer` was the only singleton failure (`pred_depth=0.568808`). Holding it in FP32 while rounding every other eligible group passed all output limits (`pred_depth=0.030006`) at projected parameter-size ratio `0.564627`. Backbone/input-projection holds also rescued parity but at worse ratios `0.814675/0.603382`. No candidate or full evaluation was produced. |
| M56d | FP16 storage with complete 2D transformer in FP32 | Complete—offline compressed checkpoint selected | The candidate is 98,048,696 bytes versus 173,610,189 bytes for the paired FP32 model-only artifact (`0.564763` ratio; 75,561,493 bytes saved). Every smoke gate and all nine complete-validation preservation gates passed on 3,769/3,769 images. Moderate 3D Vehicle/Pedestrian/mean was `19.4666/6.1848/12.8257`; BEV was `25.7349/6.7651`; nearby recall was `0.90993/0.72399`; Pedestrian localization-failure rate was `0.23854`. This selects an offline storage artifact only. |
| M57 | Deformable-attention export replacement | Complete—portable operator selected | Exact manifest/smoke hashes `7c475779…313`/`2a96cffd…3b0` passed nine module checks, three traces, and raw-output parity with zero native-extension calls. Complete validation produced 3,769/3,769 files, all nine preservation gates passed, and every metric exactly matched M56d at reported precision. A100 portable CUDA inference was ~1.24× slower and used ~52% more peak allocated memory. The next fixed-shape Core ML parity experiment is authorized; direct conversion, deployment, and product-safety qualification are not. |
| M58 | Fixed-shape FP32 Core ML conversion | Complete—Stop point 1 passed | The corrected workflow produced the iOS 17 FP32 ML Program, TorchScript, reference I/O, and ZIP. All export gates passed: exact M56d/M57 provenance, ten-output interface, trace parity, `grid_sampler`, MIL `resample`, and zero custom MIL operators. The report sets `macos_coreml_prediction_parity_authorized=true`; physical-device testing, FP16/quantization, deployment, and product safety remain false. |
| M59 | macOS Core ML raw and decoded-candidate parity | Complete—rejected | The package executed on macOS 26.6.2 with Core ML Tools 9.0 and ALL compute units in `8.50 s`, but the gate failed: logits/boxes/dimensions/depth/angle exceeded unchanged M56 limits (`0.2009/0.0262/0.6534/9.4795/1.0098` max deltas), and decoded candidates reached `39.6842` max delta with 7/50 class-rank changes. Depth-map and all four region outputs passed. CPU_ONLY compilation did not finish within the bounded retry. No iPhone, FP16, quantization, or product-safety work is authorized. |
| M59b | Core ML intermediate-tensor diagnostic | Complete—followed by M59c/M59d | Exported the frozen diagnostic package. The strict backbone mismatch required a scale-aware check in M59c before localizing transformer drift. |
| M59c | Backbone and projection audit | Complete—passed scale-aware audit | The deepest backbone feature had a large absolute delta but normalized max error of ~1.43e-6; projections passed the strict limit. See `artifacts/m59c_backbone_audit_20260918.json`. |
| M59d | 2D-transformer layer diagnostic | Complete—runtime parity failed, first stage located | Colab export passed with zero trace deltas for all 13 outputs. On macOS, six region/depth outputs passed; first failure was `det2d_encoder_layer_0` (max delta 1.2977898 vs 0.001). Encoder layers 1/2 reached 1.4707522/1.9415662; final 2D query delta was 0.8606860. ALL and CPU_AND_GPU reports were identical. Package/reference hashes verified. One frozen sample only; no AP or steady-state latency claim. See `artifacts/m59d_macos_2d_transformer_all_20260918.json` and `artifacts/m59d_macos_2d_transformer_cpu_gpu_20260918.json`. |
| M59e | First encoder internal and positional-layout diagnosis | Complete—first input mismatch explained | Original-output preservation and 18-tap trace parity passed. macOS first fails at `enc0_pos`; only fourth (6×20) scale differs. Grouped sine/cosine ordering explains the failure to max residual `4.917383e-7` after removing learned level bias. No weights changed or model repaired. CPU-only replay notebook is available; no Colab rerun is needed for the reviewed result. See `MONODGP_M59E_ENCODER_INTERNALS_CONTRACT.md`. |
| M59f | Export-only positional-interleaving repair | Complete—bug repaired; final numerical gate fails | Explicit concatenation/channel selection preserves CPU PyTorch outputs exactly. All 18 internal, 13 layer, and ten full raw checks pass. Decoded depth/dimensions fail the unchanged 0.0001 limit (max 0.000415802); all 50 query/class selections remain identical. No full-validation or device approval. See `MONODGP_M59F_POSITION_INTERLEAVE_CONTRACT.md`. |
| M59g | Residual numerical precision audit | Complete—strict decoded gate failed | 16 real inputs verified/executed. All 160 raw checks pass; CPU rewrite/repeats and Core ML repeats are bit-exact; all 800 ranked query/class selections unchanged. Decoded gate fails on 14/16 inputs; worst selected-depth delta 0.762 mm. CPU_ONLY timeout at 45 seconds; no tolerance change or deployment approval. See `MONODGP_M59G_PRECISION_AUDIT_CONTRACT.md`. |
| M59h | Final-geometry impact diagnostic | Complete—measurement, not acceptance | 16 inputs/800 candidates; decoder port bit-exact to native on both backends. Max continuous differences: 0.00351 px, depth 0.762 mm, corner 0.898 mm, yaw 0.000426 degrees. No identity/bin/filter changes. Native .2f rounding changes 40 rows (up to 1 cm); AP unmeasured. Unit-aware criteria proposed for review only. See `MONODGP_M59H_GEOMETRY_CONTRACT.md`. |
| M59i | Approved policy and full KITTI preservation | Complete—passed frozen preservation gates | All 3,769 paired predictions evaluated; all eight gated metrics exactly equal across PyTorch/Core ML. Full raw and fixed16 checks pass. Extra diagnostics flag 93 non-fixed images; 17 are order-only, not candidate-set changes. Hard Pedestrian 3D AP +0.000236 points; no automatic device/deployment approval. See `MONODGP_M59I_FULL_VALIDATION_CONTRACT.md`. |
| M60a | Trained MonoDGP physical-device feasibility | Complete—parity passed, latency failed | iPhone16 Pro Max: all16 fixed-input raw/decoded parity checks pass vs PyTorch/Mac. p95 233.73 ms initially/305.50 ms sustained vs50 ms. 60.262-second loop completed, thermal nominal→fair, sampled peak RSS491.67 MiB including fixtures. No camera or deployment qualification. See `MONODGP_M60_DEVICE_FEASIBILITY_CONTRACT.md`. |

**M53 completion note:** the model and official evaluator passed after the public-API import correction. The prior JSON failed only because it compared an independent reimplementation directly with published native-evaluator values. The corrected schema-v2 finalizer reused the complete prediction set and native log, and all frozen M53 gates passed.

**M54 smoke note:** `m54_training_smoke.json` passed every frozen preflight
requirement on 2026-09-11. The reported `71.7326` loss is only a finite
forward/backward sanity value, not a validation metric or accuracy baseline.
The controlled 100-epoch training and subsequent frozen sweep are authorized.
**M54 result note:** epoch 100 is the only supplied checkpoint that both ranks
first by the frozen balanced selection rule and passes every R0-comparability
gate. It is the selected high-capacity accuracy parent. The separate offline
product gate still fails because Pedestrian nearby recall is `0.72487 < 0.80`;
external-domain, calibration, runtime, and device qualification remain pending.

**M55 result note:** M55 changed no weights and performed no training. All 19
feasibility checks passed against exact M54 hash `8e79f392…e579847`. The
frozen compression floors remain moderate 3D AP_R40 `18.4793/5.8661` for
Vehicle/Pedestrian, balanced 3D `12.1727`, moderate BEV `24.4863/6.4292`,
nearby recall `0.89993/0.71487`, and Pedestrian localization-failure ceiling
`0.24765`. Offline Conv2d/Linear weight compression is authorized for M56;
direct Core ML conversion and product safety are not.

**M56 preparation note:** M56 stores every floating-point parameter directly
owned by a Conv2d or Linear module as FP16 while leaving other state tensors
unchanged. Loading the candidate into the unchanged MonoDGP graph restores
FP32 runtime parameters; this is a storage-sensitivity experiment, not mixed
precision execution. The paired FP32 model-only control prevents optimizer
removal from being counted as compression. Stop point 1 requires an exact
provenance manifest, candidate/control hashes and sizes, a size ratio no larger
than `0.60`, finite outputs, and bounded raw-tensor deltas on real CUDA. Only
after review may the candidate run the frozen AP, nearby-recall, localization,
and 3,769-file completeness gates.

**M56 result note:** candidate hash `8da64f18…1ea4404` stored all 327
eligible parameters in FP16 and loaded them into FP32 runtime parameters.
Output structure and finiteness passed, as did logits, boxes, dimensions,
angle, depth-map, and region-probability parity. Only final depth failed:
maximum absolute delta `0.710739 > 0.500000`; the mean was `0.012378`.
The same-environment candidate was also about 2.4% slower in mean latency.
Accordingly, `all_smoke_gates_passed=false` and
`full_evaluation_authorized=false`. The complete 3,769-image evaluation
must not run. M56b will preserve only the directly depth-sensitive geometry
heads in FP32; the M56 parity limits remain unchanged.

**M56b preparation note:** M56b is a single mechanistic follow-up, not a
threshold retune. It verifies the exact rejected M56 manifest and isolated
`pred_depth` failure, then retains every alias of `bbox_embed`,
`dim_embed_3d`, and `depth_embed` in FP32. Every other eligible Conv2d/Linear
parameter alias is stored in FP16. This is important because MonoDGP shares
some decoder prediction modules; applying inconsistent dtypes to duplicate
state-dict aliases could silently undo compression during load. The runtime
graph and execution remain FP32. Stop point 1 freezes paired artifact hashes
and sizes, explicit policy-name lists, exact FP32-head equality, output
structure/finiteness, unchanged M56 raw-output limits, separate final-depth
and uncertainty diagnostics, and same-environment 5/100 CUDA timing. Return
`m56b_compression_manifest.json` and
`m56b_selective_fp16_storage_smoke.json` before complete evaluation.

**M56b result note:** candidate hash `ea2d32db…a190f4` preserved all aliases
of `bbox_embed`, `dim_embed_3d`, and `depth_embed` in FP32 and stored 271
remaining eligible parameter identities in FP16. All integrity, paired-size,
structure, finiteness, logits, boxes, dimensions, angle, depth-map, and region
probability gates passed. Final `pred_depth` still failed at
`0.709734 > 0.500000`; its decoded-depth channel carried the failure while
the uncertainty/log-variance channel delta was `0.007577`. Mean/p95 latency
was `12.6467/12.6813 ms` versus M55 `12.6725/12.7235 ms`, so there is no
runtime-speed claim. `full_evaluation_authorized=false`; no complete
validation may be run for M56b.

**M56c preparation note:** M56c is a diagnostic matrix, not another guessed
compression policy. It partitions eligible Conv2d/Linear parameters by shared
parameter identity into `backbone`, `input_projection`, `region_head`,
`depth_predictor`, `det2d_transformer`, `det3d_transformer`,
`prediction_heads`, and `other_eligible`. It evaluates all-eligible and exact
M56b references, every nonempty one-group-only policy, and every corresponding
all-except-one policy on the frozen sample with unchanged M56 thresholds.
States are rounded and loaded transiently in memory; no candidate checkpoint,
timing claim, full-split evaluation, model selection, or deployment claim is
allowed. The result must reproduce M56b's isolated depth failure within
`0.01 m` and report singleton failures plus complement rescues.

**M56c result note:** exact manifest/JSON/CSV hashes are respectively
`0c501fa2…417796`, `7f1ca25e…321ba1`, and `8d9eea64…2c5abe`.
All 16 rows ran and all diagnostic-integrity gates passed. Rounding only the
2D transformer failed final-depth parity at `0.568808 > 0.50`; holding the
complete group in FP32 while rounding all other eligible groups passed every
raw-output limit with final-depth delta `0.030006` and projected size ratio
`0.564627`. This is both the strongest rescue and the smallest passing
complement. M56c performed no training, retained no candidate checkpoint, and
did not authorize complete validation.

**M56d preparation note:** M56d freezes that single evidence-selected policy.
All 80 aliases/unique parameters owned by `det2d_transformer` remain FP32;
295 aliases representing 247 unique parameters and 146,855,348 original FP32
bytes outside it are stored as FP16. Shared aliases are classified by unique
parameter identity. Stop point 1 regenerates paired model-only artifacts,
independently reproduces the policy, enforces every unchanged M56 parity limit
and the `≤0.60` size gate, and records 5/100 CUDA timing. Complete 3,769-image
validation remains locked until the exact manifest and smoke report are
reviewed.

**M56d result note:** manifest and smoke SHA-256 values are respectively
`ce2cf03c37530f924f75f0f44ea94fe233d8aac916c5c51fb26cd8bfa84d0bc0`
and `56678d529fda787441f842453e5f198c3d5017b4a0d8cb82373110713f737f87`.
The full gate JSON and comparison CSV SHA-256 values are
`4012d800c3972e15922ca0d3dd530cad8437f575f433d93b3708182a062227ad`
and `d64529315414d841f2658f69442722c2548e783a375f5bfcb9a46bdb2937c95f`.
Every preparation and smoke check passed. Complete evaluation produced
3,769/3,769 prediction files
and all nine preservation checks passed. Candidate-minus-parent changes were
`+0.01475/+0.00990/+0.01233` for Vehicle/Pedestrian/mean moderate 3D AP,
`-0.04017/-0.00256` for Vehicle/Pedestrian moderate BEV AP, `0/-0.000882`
for nearby recall, and `+0.000882` for Pedestrian localization-failure rate.
The checkpoint-size reduction is real, while the cross-environment latency
comparison is informational only. Direct Core ML conversion and product
safety remain unauthorized.


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
10. **Completed - rejected:** A2c positive Pedestrian focal weights `1.5`,
    `2.0`, and `2.5` produced no eligible branch. The best recall change was
    only `+0.00529` versus control and its Vehicle BEV loss was `-0.2462` AP.
    Do not launch a full A2c run.
11. **Completed:** corrected M44 found localization in `522/698 = 74.8%`
    of nearby Pedestrian misses, with zero wrong-class failures.
12. **Completed—rejected:** A2d global matched-Pedestrian box/GIoU scaling
    produced at most `+0.00176` nearby recall and `0.00485` localization-rate
    reduction, far below both `0.02` gates. No full run is authorized.
13. **Completed—rejected:** A2e size-aware box weighting produced only
    `+0.00353` nearby recall and `0.00132` localization reduction at its best
    signal, with unacceptable Vehicle AP loss. No full run is authorized.
14. **Completed—rejected:** A2f stride 4 reduced Vehicle/Pedestrian moderate
    3D AP_R40 by `6.2316/4.0359` points versus control, reduced Pedestrian nearby
    recall by `0.01940`, and increased localization failure by `0.06305`. No full
    A2f run is authorized.
15. **Completed:** local A2 loss, sampling, and feature-path tuning is closed.
    A2 epoch 130 remains the strongest current student diagnostic.
16. **Completed—rejected:** A3 Conv Large selected epoch 140 at Vehicle/
    Pedestrian moderate 3D AP_R40 `14.9492/7.6859` and BEV `20.7324/8.8273`.
    It misses Vehicle 3D by `0.9221` and Vehicle BEV by `0.5810`; the 15
    strongest balanced checkpoints shown all miss the Vehicle 3D gate.
17. **Completed:** end MobileNetV4 capacity escalation. Frozen R0 epoch 185 is
    the only current model meeting the accuracy denominator and becomes the
    accuracy parent; A2 epoch 130 remains the best MobileNetV4 diagnostic.
18. **Completed—nearby gate failed:** frozen R0 epoch 185 regenerated all
    3,769 predictions. Vehicle nearby recall is `0.88246` versus the `0.85` gate;
    Pedestrian nearby recall is `0.68342` versus the `0.80` gate.
19. **Completed diagnosis:** among 718 missed nearby Pedestrians, `558` (`77.7%`)
    are localization failures, `82` missing queries, `57` well-localized but
    subthreshold, and `21` assignment conflicts. R0 is the AP parent but is not
    yet a fully qualified safety parent.
20. **Completed—rejected:** R1 matcher weighting produced only `+0.00044`
    Pedestrian nearby recall and `0.00176` localization-failure reduction, while
    Vehicle 3D fell `0.15445` AP and Pedestrian 3D/BEV also regressed. Do not
    launch a full R1 or continue scalar matcher/loss-weight tuning.
21. **Completed—rejected, positive signal:** R2 delivered `+0.01190` nearby
    Pedestrian recall, `0.00529` localization reduction, and `+0.35580`
    Pedestrian 3D AP versus control, but failed both gain thresholds and the
    Vehicle/Pedestrian BEV preservation rules. Do not launch a full R2 run.
22. **Completed—rejected:** R2b preserved Vehicle AP/recall and slightly
    improved Pedestrian AP, but delivered only `+0.00220` nearby recall and
    `0.00132` localization-failure reduction. No full run is authorized;
    structural local refinement is closed.
23. **Completed:** `R0_COMPRESSION_CONTRACT.md` freezes R0 epoch 185 as the
    immutable parent and separates relative compression preservation from the
    unmet `0.80` Pedestrian nearby-recall product target.
24. **Completed—passed:** M52 passed all nine frozen preservation gates over all
    3,769 validation images using the exact R0 checkpoint. This authorizes only
    the next compression rung; it does not prove safety or a runtime speedup.
25. **Completed—passed:** M53 reproduced official Car 3D AP_R40 within `0.032`
    AP at every difficulty. The schema-v2 report records all 3,769 predictions,
    immutable provenance, `reference_reproduced=true`, and
    `two_class_adaptation_authorized=true`.
26. **Completed—prepared:** froze the M54 taxonomy, exact-checkpoint
    initialization, unchanged MonoDGP architecture, GT-only 100-epoch schedule,
    balanced checkpoint-selection rule, R0-comparability gates, offline nearby
    targets, fail-closed patching, resumability, and complete evaluation path.
27. **Completed—passed:** the M54 real-CUDA mixed-class smoke used 3 Vehicle
    and 6 Pedestrian targets, produced finite outputs/loss/gradients, and
    completed one in-memory optimizer step from the exact M53 checkpoint.
28. **Completed—passed:** M54 epoch 100 ranked first across 12 checkpoints,
    evaluated all 3,769 images, and passed all eight R0-comparability gates.
    Freeze checkpoint hash `8e79f392…e579847` as the accuracy parent; retain
    Pedestrian nearby recall `0.80` as an unmet product target.
29. **Completed—prepared:** `MONODGP_M55_COMPRESSION_CONTRACT.md` and the
    self-contained M55 Colab workflow freeze exact M54 provenance, relative
    preservation gates, a 5-warmup/100-run CUDA baseline, model size/memory,
    ordinary-weight coverage, and fail-closed export/operator requirements.
30. **Completed—passed:** M55 validated exact M54 provenance, finite outputs,
    5/100 CUDA timing, memory/size/partial-FLOP accounting, and the operator
    audit. All 19 feasibility checks passed; 92.69% of parameter bytes are in
    ordinary Conv2d/Linear weights. Direct Core ML remains blocked by nine
    custom deformable-attention modules.
31. **Completed—prepared:** M56 freezes one FP16 parameter-storage policy,
    exact M54/M55 evidence, paired model-only artifacts, a `≤0.60` size gate,
    real-CUDA raw-output parity, and complete frozen preservation evaluation.
32. **Completed—rejected:** M56 passed all checks except final-depth raw parity:
    max absolute delta `0.710739` exceeded the frozen `0.500000` limit.
    Full validation was correctly blocked.
33. **Completed—prepared:** M56b froze alias-consistent FP32 storage for
    `bbox_embed`, `dim_embed_3d`, and `depth_embed`, FP16 storage elsewhere,
    exact M55/M56 evidence, paired size accounting, and unchanged thresholds.
34. **Completed—rejected:** M56b passed every gate except final-depth parity:
    max absolute delta `0.709734` exceeded the unchanged `0.500000` limit.
    Full validation was correctly blocked.
35. **Completed—prepared:** M56c freezes the exact alias-consistent eight-group
    sensitivity matrix, exact M55/M56/M56b evidence, the same CUDA sample, and
    unchanged M56 parity thresholds without retaining candidate checkpoints.
36. **Completed—isolated:** all 16 M56c rows completed. The 2D transformer was
    the only singleton failure, while holding it in FP32 passed all raw limits
    at projected ratio `0.564627`.
37. **Completed—selected:** M56d passed its CUDA smoke and every frozen
    preservation gate over 3,769 images at model-only size ratio `0.564763`.
38. **Completed—operator selected:** M57 adds an opt-in rank-five `grid_sample`
    decomposition while native CUDA remains the unchanged default path. Both
    stop points passed; the complete 3,769-image run preserved all nine metrics
    exactly at reported precision.
39. **Complete:** M59b/M59c checked backbone and projection parity; M59d
    Colab export passed with all 13 reference/trace deltas equal to zero.
40. **Complete:** M59d macOS ALL and CPU_AND_GPU runs both failed first at
    `det2d_encoder_layer_0`; six preceding outputs passed. Reports are in
    `artifacts/m59d_macos_2d_transformer_*_20260918.json`.
41. **Complete:** M59e exposed 18 first-encoder tensors. Runtime drift starts
    before attention, at the fourth scale's positional channels. A grouped
    sine/cosine permutation explains the observed mismatch to `4.917383e-7`.
42. **Complete:** M59f repaired positional interleaving. CPU rewrite equivalence
    is exact; intermediate and full raw checks pass. The strict decoded gate
    remains failed on depth/dimensions; selected queries/classes are unchanged.
43. **Complete—failed strict gate:** M59g verified/executed all 16 delivered
    real inputs. All raw checks, repeatability, CPU rewrite equivalence, and
    query selection pass. Strict decoded fails 14/16; max depth delta 0.762 mm.
    CPU_ONLY produced no result within 45 seconds. No tolerance was relaxed.
44. **Complete—measurement only:** M59h measured final geometry on the same
    16 inputs, preserving all decoding and formatting. Sub-mm continuous
    geometry differences; no bin/filter changes. Native rounding changes 40
    rows, so full AP preservation still needs measurement.
45. **Complete—preservation passed:** M59i received the
    complete ZIP. Approved local Linux/x86 preprocessing reproduces all 16
    references exactly. Full tensor preparation/verification is complete;
    the unchanged paired accuracy-preservation run finished on the Mac. All
    eight gated metrics are identical and all frozen gates pass. Review the
    93 additional diagnostic flags before deciding a physical-device benchmark.
    No automatic follow-on experiment or deployment authorization.

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
- M54-parent compression contract: `MONODGP_M55_COMPRESSION_CONTRACT.md`
- M56 FP16 storage contract: `MONODGP_M56_FP16_STORAGE_CONTRACT.md`
- M56b selective FP16 contract: `MONODGP_M56B_SELECTIVE_FP16_STORAGE_CONTRACT.md`
- M56c grouped sensitivity contract: `MONODGP_M56C_GROUP_SENSITIVITY_CONTRACT.md`
- M56d det2d-FP32 storage contract: `MONODGP_M56D_DET2D_FP32_STORAGE_CONTRACT.md`
- M57 portable deformable-attention contract: `MONODGP_M57_DEFORMABLE_ATTENTION_CONTRACT.md`
- M58 fixed-shape Core ML contract: `MONODGP_M58_COREML_CONVERSION_CONTRACT.md`
- M59b intermediate-tensor diagnostic contract: `MONODGP_M59B_COREML_DIAGNOSTIC_CONTRACT.md`
- M59d 2D-transformer diagnostic contract: `MONODGP_M59D_2D_TRANSFORMER_CONTRACT.md`
- M59e encoder-internal diagnosis and replay: `MONODGP_M59E_ENCODER_INTERNALS_CONTRACT.md`
- M59f positional repair and remaining numerical gate: `MONODGP_M59F_POSITION_INTERLEAVE_CONTRACT.md`
- M59g precision audit and input collection: `MONODGP_M59G_PRECISION_AUDIT_CONTRACT.md`
- M59h geometry impact and proposed numerical-policy review: `MONODGP_M59H_GEOMETRY_CONTRACT.md`
- M59i approved policy, data notebook and full validation: `MONODGP_M59I_FULL_VALIDATION_CONTRACT.md`
- M60 device feasibility, isolated app and test procedure: `MONODGP_M60_DEVICE_FEASIBILITY_CONTRACT.md`
- R0 protocol: `TWO_CLASS_REFERENCE_PROTOCOL.md`
- Full chronological evidence: `MobileADAS3D_Codex_Handoff_20260715.md`
- Current status and next task: this file
