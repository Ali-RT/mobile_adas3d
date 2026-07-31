# MobileADAS3D v7 iPhone/Core ML Engineering Handoff

Date: 2026-07-15
Training repository: `/Users/ral3ply/Documents/GitHub/mobile_adas3d`
iPhone application repository: `/Users/ral3ply/Documents/GitHub/MobileADAS3DBenchmarkFresh`

This is the durable repository copy of the handoff supplied at
`/Users/ral3ply/Downloads/MobileADAS3D_Codex_Handoff_20260715.md`. It combines
the supplied constraints, the measured live-run summary, and the implementation
plan revised after inspecting the current iPhone working tree.

Source integrity:

- Supplied handoff SHA-256:
  `b7f691404d31bafe09e09ae83961e650b2ab3121d7bfddf9d839581c740261fd`
- Supplied live-run CSV SHA-256:
  `ff414c08f582414385b4fcbc8cb399f9997dc5e1b256fbab1b0e7c53c63fc1f9`

## Current verified state

- The v7 Core ML model runs on a physical iPhone.
- KITTI Golden Parity passes for image `000054`.
- Golden resources:
  - `MobileADAS3D_v7_cuboid_fp16.mlpackage`
  - `python_golden_topk_v7_000054.json`
  - `kitti_000054_v7_sha104fee.png`
  - `kitti_000054_v7_rgb8_1224x370.bin`
- Python golden top-1 is `Pedestrian`, grid `(52,14)`, logit approximately
  `3.4843`, and score approximately `0.9702`.
- The current uncommitted iPhone working tree compiles successfully for a
  generic iOS device with code signing disabled.

The model is a custom lightweight one-stage anchor-free dense detector, not
YOLO and not DETR. It uses a MobileNetV3-small backbone, a stride-16/FPN-like
feature map, a `[24,80]` output grid, and the heads `cls_logits`, `box2d`,
`log_depth`, `dim`, `yaw`, `center_offset`, `depth_uncertainty`, and `loc_xy`.

## Decode contract: do not change

Use the v7 repository decode exactly:

```text
score = sigmoid(cls_logit)

center_x = (grid_x + 0.5 + raw_center_offset_x) * 16
center_y = (grid_y + 0.5 + raw_center_offset_y) * 16

left_px   = raw_l * 1280
top_px    = raw_t * 384
right_px  = raw_r * 1280
bottom_px = raw_b * 384

bbox_model = [
  center_x - left_px,
  center_y - top_px,
  center_x + right_px,
  center_y + bottom_px
]

z = exp(raw_log_depth)
x = loc_xy[0] * z
y = loc_xy[1] * z
location_xyz_camera_m = [x, y, z]

dims_hwl = class_mean_hwl[class_id] * exp(raw_dim)
rotation_y = atan2(raw_yaw_sin, raw_yaw_cos)
```

Forbidden legacy behavior:

- No ImageNet mean/std normalization.
- No sigmoid on `center_offset`.
- No `exp(raw_ltrb) * stride` box decode.
- No cuboid backprojection from the 2D bounding-box center.

## User-facing modes that must remain available

1. KITTI Golden Parity
2. Custom Image Inference
3. Live Camera Inference / Pipeline No-Saving Benchmark
4. Recording + Benchmark Session

The current SwiftUI navigation has three top-level tabs and places the two live
workflows inside Live Camera. Implementation may retain that organization, but
the four workflows above must remain independently selectable and testable.

## Verified 2026-07-15 performance baseline

Source: `/Users/ral3ply/Downloads/mobileadas3d_live_runs_20260715_summary.csv`

The CSV contains four runs and 76 measured frames after excluding each run's
first warm-up frame.

| Compute mode | Frames | Mean total ms | Mean preprocess ms | Mean inference ms | Preprocess share |
|---|---:|---:|---:|---:|---:|
| `.cpuOnly` | 34 | 584.442 | 549.250 | 14.974 | 93.98% |
| `.cpuAndNeuralEngine` | 42 | 589.059 | 562.402 | 5.313 | 95.47% |
| Combined | 76 | 586.993 | 556.518 | 9.635 | 94.80% |

Decode top-K averages about 2.9 ms, NMS about 0.01 ms, overlay rendering about
2.3 ms, and capture conversion about 12-13 ms. Preprocessing is therefore the
dominant problem. Reaching 20 ms requires about a 28x preprocessing speedup;
reaching 10 ms requires about a 56x speedup.

## Current iPhone implementation assessment

### Preprocessing

The slow live path currently performs:

```text
CVPixelBuffer -> CIImage -> CGImage -> UIImage -> RGBA bytes -> RGB bytes
-> nested Swift bilinear sampling -> Float32 NCHW MLMultiArray
```

It also performs the expensive image conversion before rejecting a frame when
inference is already busy. KITTI Golden Parity has a separate RGB-sidecar path;
that reference path must remain isolated from the live optimization.

### Recording

The current working tree partially implements session metadata, benchmark CSV,
JSONL, KITTI-like labels, preview JPEGs, and ZIP creation. Remaining gaps:

- Pipeline No-Saving still performs filesystem writes.
- Required JSONL and KITTI-like labels can be disabled in the UI.
- Optional raw and overlay video settings exist, but no `AVAssetWriter`
  implementation exists.
- Preview-frame limits are declared but not enforced.
- Recorder finalization and export need explicit ordering and error handling.
- A legacy `V7LiveRunRecorder` overlaps with the newer `V7SessionRecorder` and
  should be removed or excluded to avoid duplicated recording logic.

### ZIP export

The archive must not contain absolute or malformed entries such as
`/privatebenchmark_frames.csv`. It must contain exactly one clean session root:

```text
MobileADAS3D_Run_YYYYMMDD_HHMMSS/
  session_metadata.json
  benchmark_frames.csv
  detections.jsonl
  labels_kitti_like/
  preview_overlays/                 # when enabled
  optional_raw_frames/              # when enabled
  optional_overlay_video.mov        # when enabled
  optional_raw_video.mov            # when enabled
```

## Required preprocessing contract

Live camera preprocessing must:

- Accept the camera `CVPixelBuffer` directly.
- Center-crop to an exact `1280 / 384 = 3.333333...` aspect ratio.
- Resize to `1280x384` with bilinear interpolation.
- Convert camera BGRA to RGB.
- Scale each channel by exactly `1 / 255.0` with no mean/std normalization.
- Produce Float32 NCHW `[1,3,384,1280]` input.
- Avoid Swift per-pixel loops and reuse working buffers.

Primary implementation choice: Accelerate/vImage. Build a cropped view over
the locked BGRA pixel-buffer memory, use `vImageScale_ARGB8888`, and convert
the scaled BGRA buffer directly into the R/G/B Float32 planes of the model
input. If numeric or latency validation shows vImage cannot satisfy the
contract, use a Metal compute kernel with PyTorch `align_corners=False`
half-pixel bilinear coordinates.

## Artifact contracts

Every full recording session always writes:

- `session_metadata.json`
- `benchmark_frames.csv`
- `detections.jsonl`
- `labels_kitti_like/frame_NNNNNN.txt`, including an empty file when a frame
  has no detections

Optional artifacts:

- `preview_overlays/` when enabled, default every 15 or 30 frames
- `optional_raw_frames/` when enabled, off by default
- `optional_overlay_video.mov` when enabled, off by default
- `optional_raw_video.mov` when enabled, off by default

`benchmark_frames.csv` must use this exact header:

```text
frame_index,timestamp_ms,capture_ms,preprocess_ms,inference_ms,tensor_read_ms,decode_topk_ms,nms_ms,overlay_ms,json_write_ms,label_write_ms,jpeg_write_ms,total_ms,fps,dropped_frames,raw_topk_count,detections_before_nms,detections_after_nms,num_car,num_pedestrian,num_cyclist,max_score,mean_score,threshold,nms_iou,topk,compute_units
```

`detections.jsonl` contains one JSON object per processed frame with frame and
image metadata, crop/resize transform, stage timings, and all decoded detection
fields.

KITTI-like lines use:

```text
Class 0.00 0 -10.00 x1 y1 x2 y2 h w l x y z ry score
```

## Clarified Pipeline No-Saving behavior

The supplied requirements say both that the measured pipeline must have no
file I/O and that `benchmark_frames.csv` must exist after a 30-second run. The
implementation will satisfy both:

1. During the measured loop, retain benchmark rows in memory only.
2. Perform no JSON, label, JPEG, video, metadata, or CSV writes per frame.
3. After capture stops and timing is complete, create the benchmark folder and
   write `benchmark_frames.csv` once, outside all measured frame timings.

The measured path remains:

```text
camera -> crop/resize/normalize -> Core ML -> tensor read -> decode -> NMS
-> overlay display
```

## Implementation progress

- 2026-07-16 — Task 1 complete: added focused iPhone-app regression tests for
  v7 constants and the benchmark CSV schema, aspect-crop box mapping, and the
  raw-offset/raw-LTRB/3D decode contract. Generic iOS `build-for-testing`
  succeeded, and all three tests passed on the iPhone 17 Pro simulator.
- 2026-07-16 — Task 2 complete: added `V7PixelBufferPreprocessor`, a reusable
  Accelerate/vImage path that consumes 32BGRA `CVPixelBuffer` data directly,
  applies an exact centered 10:3 crop, performs bilinear resize to 1280x384,
  and writes RGB `/255.0` Float32 NCHW planes into a reused `MLMultiArray`.
  Deterministic crop/channel/normalization/resize-reference tests pass; all
  five app tests pass on the iPhone 17 Pro simulator, and generic physical-iOS
  `build-for-testing` succeeds. Camera integration and on-device timing remain
  Task 3.
- 2026-07-16 — Task 3 implementation complete: live capture now retains the
  accepted `CVPixelBuffer` and runs `V7PixelBufferPreprocessor` directly on the
  serial inference queue. The busy admission gate runs before preprocessing or
  image conversion, every busy rejection increments the protected drop count,
  `alwaysDiscardsLateVideoFrames` remains enabled, and the fixed 0.5-second
  camera throttle was removed. UI image/log publishing is limited to 5 Hz;
  `CGImage`/`UIImage` and overlays are created after decode only when required
  for UI or enabled recording outputs. Capture rotation now uses
  `AVCaptureDevice.RotationCoordinator` once before `startRunning`, shutdown
  closes admission before draining/finalizing the inference queue, and a
  nonisolated AVFoundation delegate proxy plus retained-pixel-buffer wrapper
  remove the camera-path Swift concurrency warnings. The five regression tests
  passed twice after the functional integration, and the final generic
  physical-iOS build succeeded after the delegate/concurrency cleanup. No
  physical iPhone was connected, so live `preprocess_ms`, rotation appearance,
  and the 30-second no-freeze acceptance run still require on-device validation.
- 2026-07-17 — Task 4 complete and validated on an iPhone 16 Pro Max running
  iOS 26.5.2. Pipeline No-Saving now reserves only a future run URL during
  startup, buffers every benchmark CSV row in memory, automatically closes
  camera admission after 30 seconds, drains the accepted in-flight inference,
  and only then creates the run folder and writes `benchmark_frames.csv` once
  plus final `session_metadata.json`. The stop gate is idempotent, so a manual
  stop racing the timer cannot double-finalize. A focused physical-device UI
  test completed without freeze and re-enabled the start control after the
  automatic stop. The captured run contained 869 frames with a 29.730-second
  row timestamp span and 30.322-second metadata elapsed time. Preprocessing was
  0.650 ms mean, 0.647 ms p50, 0.742 ms p95, and 2.246 ms max; CPU-only Core ML
  inference averaged 15.771 ms, total pipeline time averaged 23.698 ms, and 31
  camera frames were dropped rather than blocking. The finalized no-saving
  folder contained only the expected CSV and metadata. Six unit tests passed
  on the same physical iPhone, including a new test proving the run directory
  does not exist before finalization.
- 2026-07-17 — Task 5 implementation and artifact validation complete. The
  model-only contract is now fixed in code at exactly 5 warmup predictions and
  100 timed predictions. The input tensor is allocated and zero-filled before
  warmup, the selected Core ML model is loaded during warmup, timing uses the
  monotonic dispatch uptime clock, and there is no camera, overlay, output
  decoding, or filesystem work in the timed loop. The output directory is not
  created until all 100 predictions have completed; it then receives exactly
  `model_only_benchmark.csv` (header plus 100 one-based run rows) and
  `model_only_benchmark_summary.json` (5/100 counts, compute units, input shape,
  mean/median/p90/p95/p99/min/max). Compute mode and millisecond precision are
  included in the folder name to prevent collisions. A deterministic injected
  predictor test ran on the physical iPhone and proved 105 total calls, absence
  of the output directory during every call, the exact two-file layout, all 100
  CSV rows, and the required JSON fields. All seven unit tests passed on the
  device. Per the user's request, real CPU-only and CPU+Neural Engine timing
  runs are deferred; they do not require street footage because this benchmark
  intentionally uses a prepared zero tensor. The temporary UI-test Runner app
  was uninstalled, leaving only the normal MobileADAS3DBenchmarkFresh app.
- 2026-07-17 — Task 6 implementation and physical-device artifact validation
  complete. Full Recording now forces `benchmark_frames.csv`,
  `detections.jsonl`, and one `labels_kitti_like/frame_*.txt` per processed
  frame even if stale caller settings attempt to disable them. Overlay-preview
  and raw-frame directories are created only when enabled; preview stride and
  the maximum-preview cap are enforced. The UI exposes independent overlay and
  raw video toggles while clearly marking CSV/JSONL/labels as mandatory.
  `V7VideoWriter` uses `AVAssetWriter` with H.264 and a BGRA pixel-buffer
  adaptor. Raw video appends the retained, physically rotated camera
  `CVPixelBuffer` directly, without UIImage conversion; overlay video renders
  into adaptor-pool buffers. Both use original capture presentation timestamps.
  Encoder backpressure drops only the current video frame rather than blocking
  capture/inference, and appended/dropped video counts are recorded in final
  metadata. Shutdown first closes camera admission and drains inference, then
  closes CSV/JSONL handles, finalizes both movie trailers, writes metadata with
  `status: finished`, and only afterward enables export. A synthetic two-frame
  full-recording test passed on the iPhone 16 Pro Max and verified nonempty
  metadata/CSV/JSONL, two label files, two preview JPEGs, two raw JPEGs, both
  optional `.mov` files, and a valid video track in each movie. All eight unit
  tests passed on the physical phone. Only the normal app remains installed and
  is launched; a later street run is needed only for real-world detection and
  visual-orientation validation, not artifact correctness.
- 2026-07-17 — Task 6 real-street post-validation complete from two user-
  exported runs. The readable 09:08 Full Recording archive contains 338 CSV
  rows, 338 JSONL rows, a continuous sequence of 338 KITTI-like label files,
  23 preview JPEGs matching frame 1 plus the configured stride of 15, and final
  metadata with `status: finished`. Optional raw frames and both videos were
  disabled and correctly absent. All 941 decoded detections have boxes inside
  the 1920x1080 source image and positive depth. Sampled preview overlays are
  upright, landscape, and spatially aligned with visible vehicles. CPU-only
  preprocessing measured 0.971 ms mean, 0.885 ms p50, 1.041 ms p95, 1.246 ms
  p99, and 23.484 ms max; the first-frame model-load spike accounts for that
  maximum and a simultaneous 282.264 ms inference outlier. Steady-state Core
  ML inference was 14.473 ms p50 and 15.481 ms p95. The run processed 338
  frames across a 12.489-second timestamp span (26.98 processed fps) while
  dropping 49 camera frames instead of blocking.

  A second 09:11 run proves the requested 30-second real recording did not
  freeze: its locally recoverable content has a continuous 462-frame CSV,
  JSONL, and label sequence over 30.790 seconds (14.97 processed fps) with 465
  dropped frames. This run saved/rendered overlays every frame, so total time
  rose to 42.586 ms mean while preprocessing remained 0.970 ms mean and 1.049
  ms p95. Its shared ZIP is truncated before the central directory, metadata,
  and preview entries, so it is recovery evidence rather than a valid final
  export.

  Both archives expose the remaining Task 7 defect: ZIP entry names begin with
  absolute-looking `/private...` paths and have no clean session root. The root
  cause is `V7ZipExporter.relativePath`: on iOS the enumerated file path uses
  the `/private/var/...` spelling while the session root can use `/var/...`;
  substring replacement leaves the `/private` prefix attached to every entry.
  The synchronous STORE writer also exposes a partially valid archive if a
  generated ZIP is copied/shared before its central directory has reached
  durable storage. Task 7 must normalize resolved URLs, stage to a temporary
  filename, validate entry paths and the completed archive, atomically publish
  the final `.zip`, and only then expose the share action.
- 2026-07-17 — Task 7 ZIP/export implementation and acceptance validation
  complete.
  `V7ZipExporter` now resolves the session root and enumerated URLs before
  comparing path components, rejects absolute paths, traversal components,
  backslashes, NULs, symlinks, and items outside the session root, and prefixes
  every entry with one session-folder name. Directory records are emitted, so
  enabled empty subdirectories are preserved. Files—including optional large
  videos—are streamed in 1 MiB chunks with data descriptors instead of being
  loaded fully into memory. The writer synchronizes and closes a sibling
  staging file, validates its EOCD, central-directory bounds, entry list,
  unique safe names, and matching local headers, then atomically publishes or
  replaces the final `.zip` and validates it again. The UI clears any stale ZIP
  URL, disables conflicting actions, shows export progress, and exposes the
  ZIP ShareLink only after validation succeeds.

  The final physical-iPhone suite passed all nine tests after the repeated-
  export assertion and legacy-source cleanup. The ZIP test reproduces the iOS
  `/var` versus `/private/var` spelling difference, verifies the exact clean
  session-root layout and empty directories, rejects unsafe paths, and exports
  the same session twice to verify atomic replacement without stale or
  duplicated names. The full-recording test still verifies all mandatory and
  enabled optional artifacts. An independent macOS harness exercised both
  first export and replacement through the same exporter, and `zip -T`
  reported `OK`. Its listing contained exactly one
  `MobileADAS3D_Run_HostValidation/` root, nested label and preview directories,
  and no absolute or `/private...` paths; the replacement archive contained
  the updated CSV content.

  Six unreferenced legacy implementations were removed so there is one active
  implementation for each concern: `V7LiveRunRecorder.swift`,
  `V7Preprocessor.swift`, `V7Decoder.swift`, `V7CleanTypes.swift`,
  `V7RawPNG.swift`, and `V7ResourceVerifier.swift`. The preserved Golden and
  Custom Image path uses `V7ImagePreprocessor`, `V7DetectionDecoder`,
  `V7Resources`, and `V7RGBSidecar`; live camera uses
  `V7PixelBufferPreprocessor`. A final clean, signed one-target build succeeded,
  contained no test-runner plug-in, installed on the physical iPhone, and
  launched as `com.ali.MobileADAS3DBenchmarkFresh`. The other visible `ART`
  icon is the separate pre-existing `com.ali.xdetector` app, not a duplicate or
  test runner.
- 2026-07-17 — App identity/signing follow-up. The original personal-team
  provisioning profile expired after its seven-day lifetime, causing iOS to
  report that `MobileADAS3DBenchmarkFresh` was no longer available even though
  the installed bundle had previously launched. The Home Screen identity is
  now `MobileADAS 3D`, the bundle identifier is `com.ali.MobileADAS3D`, and the
  app has a new opaque 1024x1024 automotive depth-perception icon. Xcode created
  a fresh personal-team profile valid from 2026-07-17T20:52:59Z through
  2026-07-24T20:52:59Z; the new bundle passed strict local code-signature
  verification, installed, and was subsequently used for the field run below.
  Personal-team installs will continue to require re-signing every seven days;
  TestFlight or another paid-program distribution method is required for a
  longer-lived installation.
- 2026-07-17 — Final fixed-ZIP Full Recording field inspection complete for
  `MobileADAS3D_Run_20260717_141225_614.zip` (47 MiB, SHA-256
  `022f002746ef695e457edaa887ccbbb0095ae4ed5c2b08b8ab1644c582e71caa`).
  `unzip -t` passed every entry. The archive has 808 unique entries under the
  single clean `MobileADAS3D_Run_20260717_141225_614/` root, with no absolute,
  traversal, backslash, symlink, duplicate, or `/private...` entry names.

  Metadata is final (`status: finished`) and records CPU-only inference at
  1920x1080 with the required 1280x384 RGB `/255.0` NCHW preprocessing. The
  archive contains 751 continuous CSV frames, 751 valid JSONL records, 751
  continuous KITTI-like label files, and exactly 51 1920x1080 overlay previews
  for frame 1 and stride 15 through frame 750. All 974 post-NMS detections
  reconcile across CSV, JSONL, and label lines; all label rows have 16 fields,
  and all decoded boxes, depths, and dimensions are finite, positive, and
  inside the source image. Disabled raw-frame and video outputs are correctly
  absent.

  Capture timestamps span 27.919 seconds and metadata reports 28.313 seconds,
  so this run is 1.687 seconds short of a strict 30-second acceptance run. It
  processed 751 frames at 26.863 fps while dropping 90 of approximately 841
  admitted/captured frames (89.3% processed), confirming nonblocking frame
  dropping. Preprocessing is 0.887 ms mean, 0.872 ms p50, 0.983 ms p95, 1.041
  ms p99, and 3.134 ms max; excluding first-frame warmup, its max is 1.304 ms.
  Inference is 15.143 ms mean and 16.286 ms p95, with the 56.455 ms first-frame
  warmup as the maximum. Full saved-pipeline time is 25.538 ms mean, 23.339 ms
  p50, and 41.700 ms p95. The CSV `capture_ms` field remains zero on every row,
  so it is currently a placeholder rather than useful capture-stage timing.

  The 974 predictions consist of 964 Cars, 5 Pedestrians, and 5 Cyclists.
  Sampled previews are upright and landscape; car boxes are spatially aligned
  on visible vehicles. The low-score Pedestrian prediction at frame 570 lands
  on landscaping, and the frame-720 Cyclist prediction is clipped at the right
  image edge with no visible cyclist, indicating model-quality false positives
  near the 0.25 threshold rather than an export or coordinate-transform defect.

## Revised implementation plan

### Phase 1: Regression guardrails

- **Completed 2026-07-16:** preserve the KITTI RGB-sidecar preprocessing and
  v7 decoder unchanged.
- **Completed 2026-07-16:** add tests for v7 decode invariants, transform
  mapping, constants, and the benchmark CSV schema.
- **Completed 2026-07-16:** add reference-vs-fast preprocessing tests using
  deterministic BGRA buffers as part of Phase 2.
- Require a successful iOS build before and after each major phase.

### Phase 2: Direct pixel-buffer preprocessing

- **Completed 2026-07-16:** add a reusable live preprocessor backed by
  Accelerate/vImage.
- **Completed 2026-07-16:** allocate and reuse the scaled BGRA scratch buffer,
  conversion scratch storage, vImage temporary storage, and model
  `MLMultiArray`.
- **Completed 2026-07-16:** validate channel order, `/255.0`, NCHW strides,
  exact crop coordinates, and resize error against a deterministic scalar
  reference implementation.
- Keep UIImage preprocessing for custom images and the sidecar path for Golden
  Parity until separate regression evidence supports changing them.

### Phase 3: Nonblocking camera pipeline

- **Completed 2026-07-16:** test and set the inference-busy flag before any
  image conversion or preprocessing.
- **Completed 2026-07-16:** retain the selected `CVPixelBuffer` only for the
  serial inference task.
- **Completed 2026-07-16:** keep `alwaysDiscardsLateVideoFrames = true` and
  count every busy rejection.
- **Completed 2026-07-16:** remove the fixed 0.5-second inference throttle and
  throttle UI publishing to 5 Hz rather than camera processing.
- **Completed 2026-07-16:** modernize physical-buffer rotation with
  `AVCaptureDevice.RotationCoordinator` and isolate the AVFoundation delegate
  conformance from the observable UI manager.
- **Completed 2026-07-16:** create UIImage/overlay representations only after
  inference and decoding, and only when needed for display or recording.
- **Completed 2026-07-17:** physical-device 30-second no-freeze validation,
  live preprocessing percentiles, and a real-street visual-orientation smoke
  check using exported preview overlays.

### Phase 4: Benchmarks

- **Completed 2026-07-17:** Model-only mode prepares input before timing,
  performs exactly 5 warmups and 100 timed predictions, then writes
  `model_only_benchmark.csv` and `model_only_benchmark_summary.json` after
  timing. Physical accelerator timing samples remain deferred by user request.
- **Completed 2026-07-17:** Pipeline No-Saving runs automatically for 30
  seconds, buffers benchmark rows in memory with no persistence in the measured
  loop, drains capture/inference, and writes the CSV once after timing.
- Record percentile summaries, processed/dropped frames, compute units, and
  input shape.

### Phase 5: Session recording

- **Completed 2026-07-17:** consolidate full-session persistence under
  `V7SessionRecorder`.
- **Completed 2026-07-17:** make metadata, benchmark CSV, JSONL, and one label
  file per processed frame mandatory for full recordings.
- **Completed 2026-07-17:** make preview JPEGs, raw JPEGs, overlay video, and
  raw video independently optional.
- **Completed 2026-07-17:** add H.264 `AVAssetWriter`/pixel-buffer-adaptor
  writers using capture timestamps and nonblocking backpressure drops.
- **Completed 2026-07-17:** finalize file handles and video writers before
  metadata status becomes `finished` and before export becomes available.

### Phase 6: ZIP and share sheet

- **Completed 2026-07-17:** generate ZIP entries from resolved, component-based
  relative paths only.
- **Completed 2026-07-17:** reject absolute paths, `..`, backslashes, NULs,
  symlinks, duplicate entries, and paths outside the session root.
- **Completed 2026-07-17:** prefix every entry with the session folder name.
- **Completed 2026-07-17:** preserve all enabled subdirectories, including
  empty directories.
- **Completed 2026-07-17:** stream to and synchronize a staging archive,
  validate it, atomically publish it, and only then present the URL through the
  iOS share sheet.

### Phase 7: Acceptance validation

- **Completed 2026-07-17:** retained the on-device KITTI Golden Parity PASS and
  preserved the Custom Image and live-camera paths.
- **Completed 2026-07-17:** model-only benchmark produces both required files
  with 5 warmup and 100 timed runs.
- **Completed 2026-07-17:** Pipeline No-Saving ran for 30 seconds without a
  freeze and publishes its CSV only after measurement.
- **Completed 2026-07-17:** Full Recording produced every mandatory artifact
  and every enabled optional artifact in physical-device tests; the street run
  additionally validated real detections and overlay orientation.
- **Completed 2026-07-17:** ZIP validation and host `zip -T` inspection show
  one clean session root, safe relative entries, and correct repeated export.
- **Completed 2026-07-17:** real-street mean preprocessing is approximately
  0.97 ms (1.05 ms p95), well below both the 20 ms target and 10 ms stretch
  goal.
- **Completed 2026-07-17:** removed stale resource/recording/preprocessing
  implementations; final source scan found no references to the six removed
  legacy types.
- **Completed 2026-07-17:** final physical-device XCTest result is 9 tests,
  0 failures; final signed app build, install, and launch all succeeded.

## Acceptance evidence to retain

For the final implementation, retain:

- Generic-device build log.
- KITTI Golden Parity log/screenshot.
- Model-only CSV and JSON summary for both compute modes.
- Thirty-second Pipeline No-Saving benchmark CSV.
- Thirty-second Full Recording session folder and exported ZIP listing.
- Before/after latency comparison against the 2026-07-15 baseline above.

## Training-code handoff — updated 2026-07-19

There are now two deliberately separate model lineages:

1. **Deployed reference:** v7 is the custom anchor-free MobileNetV3-Small
   monocular-3D CNN already validated in the iPhone app. Preserve its Core ML
   model, Swift preprocessing, decoder, and golden-parity contract.
2. **Active training baseline:** train a fresh model using
   `mobilenetv4_conv_small.e2400_r224_in1k`. There is no reusable
   MobileADAS3D checkpoint. Do not initialize it from unrelated `best.pt`
   files or report v6/v7 metrics as MobileNetV4 results.

The fresh MobileNetV4 lineage retains the stride-16/32 FPN, `1280x384`
resolution, and RGB `/255.0` external input contract. v0/v1 preserve the
original eight-output path. v2 adds an optional projected-center output for
calibration-aware geometry while preserving the old `loc_xy` fallback. ImageNet
mean/std normalization is embedded in the model so the external preprocessing
contract does not change.

### Canonical training entry point

Run this notebook from top to bottom in a Google Colab GPU runtime:

```text
notebooks/MobileADAS3D_MobileNetV4_Colab_Baseline.ipynb
```

It now defaults to `configs/kitti_mnv4_calibrated_geometry_v2.yaml` and
`requirements-colab.txt`, mounts Drive, optionally stages KITTI on the Colab
SSD using a fast archive path from `datasets/kitti/zips` when available, then
falls back to folder `rsync`. The staging cell includes explicit notebook
diagnostics, a per-folder progress bar, resumable copy behavior, and a local
completion manifest, installs and verifies the canonical Chen 3,712/3,769
split, performs a strict preflight, trains, resumes safely after interruption,
and evaluates the best checkpoint on all 3,769 validation images.
The train/resume cell streams stdout/stderr live, saves the same output under
`mobile_adas3d_outputs/mnv4_conv_small_baseline/colab_logs/`, and prints the
log tail automatically on failure.

Run comparison IDs:

```text
v0_earlystop_val_loss:
  run_id: 20260720_212816_baseline_mnv4_conv_small_stride16
  config: configs/kitti_mnv4_conv_small_baseline.yaml
  stopped: epoch 27 by val-loss early stopping

v1_long80_no_earlystop:
  run_name: mnv4_v1_long80_no_earlystop
  config: configs/kitti_mnv4_conv_small_ap_v1.yaml
  policy: no early stopping, save checkpoints every 5 epochs

v2_calibrated_geometry_quality:
  run_name: mnv4_v2_calibrated_geometry_quality
  config: configs/kitti_mnv4_calibrated_geometry_v2.yaml
  policy: no early stopping, save checkpoints every 5 epochs
  completed_run_id: 20260721_142002_mnv4_v2_calibrated_geometry_quality
  car_focused_checkpoint: epoch_040.pt
  balanced_all_class_checkpoint: latest.pt / epoch_080.pt

v3_quality_scoring:
  run_name: mnv4_v3_quality_scoring
  config: configs/kitti_mnv4_quality_scoring_v3.yaml
  policy: v2 geometry + quality head, no early stopping, save every 5 epochs
  status: next fresh Colab run
```

### 2026-07-21 modeling handoff: calibrated geometry v2

The v1 long-80 run proved that the MobileNetV4 student is learning useful
signals but not ranking accurate 3D boxes well enough for KITTI AP_R40. The
best checkpoint sweep reached only about `0.596` Car BEV moderate AP_R40 and
about `0.094` Car 3D moderate AP_R40. Matched-object diagnostics were much
healthier than AP: latest v1 showed roughly `1.83m` all-class depth MAE,
`0.085` depth relative error, and `0.757` mean 2D IoU, but yaw remained around
`33.8°` MAE and 3D center/corner errors remained too high. That pattern points
to geometry/ranking failure rather than a completely broken detector.

The follow-up run `mnv4_v2_calibrated_geometry_quality` keeps the
MobileNetV4 Conv Small backbone and external RGB `/255.0`, `1280x384` input
contract, but adds a lightweight `projected_center_offset` head. Training
builds a target by projecting each KITTI 3D bottom-center through the resized
`P2` calibration. Evaluation decodes camera-frame X/Y by back-projecting the
predicted projected center with `P2` and predicted depth. The legacy `loc_xy`
head remains enabled with a lower auxiliary weight, so old configs and old
decode behavior are preserved.

The completed v2 run
`20260721_142002_mnv4_v2_calibrated_geometry_quality` confirmed the diagnosis.
`latest.pt` beat `best.pt` by validation loss:

```text
latest.pt KITTI AP_R40:
  Car BEV moderate:        6.30
  Car 3D moderate:         2.87
  Pedestrian 3D moderate:  1.07
  Cyclist 3D moderate:     1.04

latest.pt matched 3D diagnostics:
  ALL depth_mae:       1.799m
  ALL loc_xyz_mae:     (0.503, 0.145, 1.799)m
  ALL center3d_mae:    1.958m
  ALL corner3d_mae:    2.335m
  ALL yaw_mae:         34.05deg
  ALL corner2d_mae:    33.2px
```

The full checkpoint AP sweep then showed that the best checkpoint depends on
the selection objective:

```text
checkpoint_ap_sweep_val:
  best Car 3D moderate:   epoch_040.pt = 3.02
  best mean 3D moderate:  epoch_080.pt/latest.pt = 1.661
  best Car BEV moderate:  epoch_045.pt = 6.392
  best mean BEV moderate: epoch_040.pt = 3.155

epoch_040.pt KITTI AP_R40:
  Car BEV moderate:        6.33
  Car 3D moderate:         3.02
  Pedestrian 3D moderate:  1.26
  Cyclist 3D moderate:     0.58

epoch_040.pt matched 3D diagnostics:
  ALL depth_mae:       1.798m
  ALL loc_xyz_mae:     (0.508, 0.146, 1.798)m
  ALL center3d_mae:    1.959m
  ALL corner3d_mae:    2.330m
  ALL yaw_mae:         34.23deg
  ALL corner2d_mae:    33.3px
  Car yaw_mae:         30.72deg
  Pedestrian yaw_mae:  63.64deg
  Cyclist yaw_mae:     46.07deg

epoch_080.pt/latest.pt KITTI AP_R40:
  Car BEV moderate:        6.30
  Car 3D moderate:         2.87
  Pedestrian 3D moderate:  1.07
  Cyclist 3D moderate:     1.04
```

Use `epoch_040.pt` when optimizing/reporting Car-focused KITTI AP. Use
`epoch_080.pt`/`latest.pt` when optimizing balanced all-class 3D moderate AP,
because it preserves much stronger Cyclist AP.

Compared with v1 latest, v2 substantially reduced X/Y localization error,
center/corner error, and projected-corner error. Yaw remains the next major
geometry bottleneck, and AP checkpoint selection is still misaligned with
validation loss.

### 2026-07-27 modeling handoff: quality scoring v3

The v2 AP sweep showed that `epoch_040.pt` and `epoch_080.pt/latest.pt` have
nearly identical matched 3D geometry but different AP tradeoffs. That means the
next controlled experiment should improve detection ranking rather than change
the projected-center geometry again.

`mnv4_v3_quality_scoring` keeps the v2 projected-center location path and adds
a lightweight one-channel `quality` head. The target is a soft center-quality
score for positive cells, with background at zero. Decode uses
`class_prob * quality_prob` when `score_mode: class_quality` is enabled. This
is intentionally isolated from yaw-bin/residual work so the AP effect of
quality-aware ranking is measurable.

Fresh-run checklist:

```text
notebook:
  notebooks/MobileADAS3D_MobileNetV4_Colab_Baseline.ipynb

experiment:
  EXPERIMENT_ID = mnv4_v3_quality_scoring
  CONFIG = configs/kitti_mnv4_quality_scoring_v3.yaml
  RUN_NAME = mnv4_v3_quality_scoring

expected artifacts:
  runs/<timestamp>_mnv4_v3_quality_scoring/checkpoints/latest.pt
  runs/<timestamp>_mnv4_v3_quality_scoring/checkpoints/best.pt
  runs/<timestamp>_mnv4_v3_quality_scoring/kitti_r40_latest/kitti_r40_summary.json
  runs/<timestamp>_mnv4_v3_quality_scoring/checkpoint_ap_sweep_val/checkpoint_ap_summary.csv
  mobile_adas3d_outputs/mnv4_conv_small_baseline/colab_logs/train_mnv4_v3_quality_scoring_<timestamp>.log
```

After training, compare checkpoints using the same full validation split, same
`score-threshold 0.001`, `topk 300`, and `nms-iou-threshold 0.5`. Use
`scripts/sweep_kitti_r40_checkpoints.py` to evaluate `epoch_*.pt`, `best.pt`,
and `latest.pt`; the key selection numbers are Car 3D moderate AP_R40 and mean
3D moderate AP_R40. Keep reporting all-class 3D center/corner MAE, Car yaw MAE,
and Car depth MAE for the selected AP checkpoint.

#### 2026-07-28 v3 result and selected checkpoint

The completed run is:

```text
20260727_184204_mnv4_v3_quality_scoring
```

A full validation sweep compared `quality_score_power` values `0.0`, `0.25`,
`0.5`, `0.75`, and `1.0` for `epoch_045.pt`, `epoch_065.pt`, and
`latest.pt`. The selected candidate is:

```text
checkpoint: epoch_065.pt
score_mode: class_quality
quality_score_power: 0.0
Car 3D AP_R40 easy/moderate/hard: 5.229 / 3.107 / 2.397
Cyclist 3D AP_R40 easy/moderate/hard: 3.955 / 1.795 / 1.569
Pedestrian 3D AP_R40 easy/moderate/hard: 1.428 / 1.020 / 0.744
Car BEV AP_R40 easy/moderate/hard: 9.281 / 5.986 / 5.027
mean all-class 3D moderate AP_R40: 1.974
```

The selected checkpoint's matched-object geometry summary is:

```text
matched: 14566
2D IoU mean: 0.766
depth MAE: 1.870 m
depth relative error: 0.083
yaw MAE: 32.54 degrees
dimension MAE: 0.174 m
projected-corner 2D MAE: 32.81 px

Car:
  matched: 12780
  2D IoU mean: 0.778
  depth MAE: 1.898 m
  depth relative error: 0.079
  yaw MAE: 29.20 degrees
  dimension MAE: 0.180 m
```

This is a small Car 3D moderate improvement over the v2 `epoch_040.pt`
candidate (`3.107` versus `3.020`) and improves the balanced moderate 3D
mean. Geometry remains broadly stable, with better overall and Car yaw than
the v2 reference. However, every positive quality-score power reduced Car 3D
AP in this sweep. Therefore the quality head remains in the trained model, but
the checked-in inference default neutralizes it with
`quality_score_power: 0.0`. Do not select `best.pt` by validation loss for
deployment; use `epoch_065.pt` for this v3 run.

### 2026-07-28 modeling handoff: angular yaw v4

The v3 selected checkpoint has low median yaw error but a very large tail:
overall yaw MAE is `32.54` degrees while p50 is only `5.54` degrees and p90 is
`152.34` degrees. Car shows the same pattern (`29.20` mean, `4.64` p50,
`149.93` p90). This tail is consistent with front/back orientation flips
dominating the mean and should be confirmed with the axis-aware yaw diagnostic.

`mnv4_v4_angular_yaw` is a controlled response:

- Preserve the two-channel `[sin(yaw), cos(yaw)]` output, decoder, export
  contract, and projected-center geometry.
- Retain the existing normalized smooth-L1 yaw loss.
- Add `yaw_cosine_weight: 1.0`. Cosine distance is zero for aligned yaw, one
  at 90 degrees, and two for a 180-degree front/back flip.
- Remove the quality head from this fresh experiment and use class-only
  scoring, because every positive quality power reduced Car AP in v3.
- Keep all other optimization, target, loss, scheduler, and checkpoint
  settings aligned with v3 so the yaw change remains measurable.

Fresh-run identifiers:

```text
EXPERIMENT_ID = mnv4_v4_angular_yaw
CONFIG = configs/kitti_mnv4_angular_yaw_v4.yaml
RUN_NAME = mnv4_v4_angular_yaw
```

Evaluate every saved checkpoint with the canonical full Chen validation split.
In addition to AP_R40, run `evaluate_3d_metrics.py` and
`evaluate_yaw_diagnostics.py` for the selected checkpoint. The v4 success
criteria are:

```text
Car 3D moderate AP_R40 > 3.107
mean all-class 3D moderate AP_R40 > 1.974
overall yaw p90 < 152.34 degrees
Car yaw p90 < 149.93 degrees
```

Do not resume a v3 checkpoint into v4: the v4 model intentionally removes the
quality head, and the new loss changes the training objective. Start a fresh
run with the v4 run name.

### 2026-07-29 modeling handoff: axis plus direction v5

V4 improved balanced and BEV AP but did not improve Car 3D AP or yaw:

```text
selected balanced checkpoint: epoch_050.pt
Car 3D moderate AP_R40: 2.885
mean all-class 3D moderate AP_R40: 2.055
Car BEV moderate AP_R40: 6.235
mean all-class BEV moderate AP_R40: 3.497

overall yaw mean/p50/p90: 35.26 / 5.93 / 163.20 degrees
Car yaw mean/p50/p90: 32.26 / 5.01 / 164.22 degrees
overall axis-aware yaw mean/p50/p90: 12.70 / 4.78 / 37.95 degrees
Car axis-aware yaw mean/p50/p90: 10.65 / 4.18 / 28.95 degrees
```

The large gap between standard and axis-aware yaw error confirms that the
orientation axis is substantially better than the final front/back choice.
V5 therefore replaces direct full-yaw supervision with:

1. `yaw_axis`: `[sin(2*yaw), cos(2*yaw)]`, invariant under a 180-degree flip.
2. `yaw_direction`: a binary logit selecting one of the two directions along
   that axis.
3. `yaw`: the reconstructed `[sin(yaw), cos(yaw)]` tensor. This remains the
   exported output, preserving the existing Core ML/iPhone decoder contract.

The v5 model exposes `yaw_axis` and `yaw_direction` only as auxiliary training
outputs. The fixed export wrapper continues to select the original output
list, including the reconstructed two-channel `yaw`.

Fresh-run identifiers:

```text
EXPERIMENT_ID = mnv4_v5_axis_direction
CONFIG = configs/kitti_mnv4_axis_direction_v5.yaml
RUN_NAME = mnv4_v5_axis_direction
```

Start v5 from pretrained MobileNetV4 weights, not from a v3/v4 detector
checkpoint. Its yaw heads and objective are structurally different. Training
logs include `yaw`, `yaw_cos`, and `yaw_dir`. Sweep every five-epoch
checkpoint, then run both geometry and yaw diagnostics on the best balanced
and best Car checkpoints.

V5 success criteria:

```text
Car 3D moderate AP_R40 > 3.107
mean all-class 3D moderate AP_R40 > 2.055
overall standard yaw p90 < 152.34 degrees
Car standard yaw p90 < 149.93 degrees
axis-aware yaw must not regress materially from v4
```

#### V5 early-training NaN fix

The first v5 attempt reached epoch 4 before reporting a non-finite loss. The
dedicated yaw losses were finite, but the reconstructed yaw also entered the
cuboid corner loss. Backpropagating through the hard direction decision and
`atan2` is undefined when the raw axis approaches `[0, 0]`, which can poison a
later batch with NaN parameters.

For v5 only, the corner loss now treats reconstructed yaw as detached. The
axis head remains supervised by smooth-L1 plus cosine loss, and the direction
head remains supervised by binary cross entropy. Corner loss continues to
train depth, location, and dimensions. A regression test starts with an exact
zero axis, runs the complete loss backward, and requires finite axis and
direction gradients.

It is safe to resume from the last checkpoint saved before the failed epoch.
Do not resume from a checkpoint containing non-finite parameters.

### 2026-07-29 transfer-learning pivot: Task 1 teacher gate

V1 through v5 showed that the compact student can learn useful depth,
projected-center geometry, and orientation axes, but repeatedly training all
heads from ImageNet initialization is not producing competitive detection AP.
The next program keeps MobileADAS3D as the deployable student and evaluates a
pretrained monocular-3D teacher before implementing distillation.

Task 1 is isolated to teacher feasibility:

```text
notebook: notebooks/MonoDETR_Teacher_Feasibility_Colab.ipynb
teacher: official MonoDETR
official source commit: 6994b9f512400b258c6edb75f77423beb9c126f2
official checkpoint Google Drive ID: 1d8fbAt-CQF-IN8UEHuw3NimmfONhH6iA
split: Chen val, 3769 images
evaluated class: Car
gate: Car 3D moderate AP_R40 >= 15.0
```

MonoDETR writes standard scored KITTI files. The new
`scripts/evaluate_kitti_prediction_dir.py` loads those files and uses the same
local `data.kitti_r40` evaluator as MobileADAS3D. It requires all expected
split files unless `--allow-missing` is explicitly supplied; incomplete
results are marked with `complete_split: false`.

The notebook creates a symlinked MonoDETR dataset view rather than copying
KITTI again. It prefers an already complete `/content/kitti` stage, otherwise
reads `/content/drive/MyDrive/datasets/kitti` directly for this one-pass
teacher evaluation. Both canonical `training/image_2` and `training/label_2`
and Drive aliases `training/image_02` and `training/label_02` are resolved
into canonical MonoDETR symlinks. The notebook compiles the official
deformable-attention CUDA extension,
downloads the checkpoint once to Drive, runs inference at threshold `0.001`,
and saves the comparison under:

```text
/content/drive/MyDrive/mobile_adas3d_outputs/teachers/monodetr/
  checkpoint_best.pth
  chen_val_20260729/
    kitti_r40_metrics.csv
    kitti_r40_summary.json
```

Do not implement teacher caching or student distillation unless this gate
passes. A failure caused by installation, checkpoint loading, split mismatch,
or incomplete predictions is an infrastructure failure and must be fixed
before judging the teacher.

#### 2026-07-31 teacher feasibility result: PASS

The official MonoDETR checkpoint completed the exact 3,769-image Chen
validation split. Both MonoDETR's official evaluator and this repository's
independent KITTI prediction-directory evaluator produced consistent results.

```text
prediction files: 3769 / 3769
complete_split: true
Car BEV AP_R40 easy/moderate/hard: 38.02 / 27.35 / 23.51
Car 3D  AP_R40 easy/moderate/hard: 28.27 / 20.35 / 17.11
required Car 3D moderate AP_R40: >= 15.00
gate result: PASS
```

Canonical artifacts:

```text
/content/drive/MyDrive/mobile_adas3d_outputs/teachers/monodetr/chen_val_20260729/
  kitti_r40_metrics.csv
  kitti_r40_summary.json
```

Teacher Task 1 is complete. The next isolated task is **Teacher Task 2:
reproducible train-split prediction cache**. It should run the same pinned
teacher over all 3,712 Chen training images and save one scored KITTI file per
image plus a manifest containing the teacher source commit, checkpoint
SHA-256, runtime configuration, split-file digest, image count, and completion
state. Do not alter student losses or start a new student run until that cache
is complete and validated.

Teacher Task 2 is implemented in the final section of
`notebooks/MonoDETR_Teacher_Feasibility_Colab.ipynb`. It uses the isolated
MonoDETR model/output name `monodetr_train_cache`, leaving the successful
validation predictions untouched. After inference,
`scripts/create_teacher_prediction_cache.py` requires exactly the Chen train
IDs with no missing or extra files, parses all KITTI predictions, copies them
to Drive, revalidates the copy, and records source/config/split/checkpoint and
prediction-tree SHA-256 values. The manifest is initially written with
`complete: false` and is changed to `true` only after the Drive copy passes.

Expected Task 2 artifact:

```text
/content/drive/MyDrive/mobile_adas3d_outputs/teachers/monodetr/chen_train_20260731/
  teacher_cache_manifest.json
  runtime_config.yaml
  predictions/  # exactly 3712 files
```

Task 2 remains in progress until Colab prints the complete manifest. On a
Colab interruption, rerun the train inference cell and then the cache cell;
an interrupted Drive copy remains explicitly incomplete.

#### 2026-07-30 MonoDETR CUDA build compatibility

Current Colab PyTorch rejects the pinned MonoDETR extension's deprecated
`AT_DISPATCH_FLOATING_TYPES(value.type(), ...)` calls. The teacher notebook
now checks for exactly two such calls and replaces them with
`value.scalar_type()` before compilation. It installs Ninja, clears only the
generated extension `build/` directory so a failed object is not reused, and
replaces the removed private PyTorch `_LinearWithBias` import with the
state-dict-compatible public `torch.nn.Linear`, and replaces the removed
`torch._overrides` fallback with public `torch.overrides`. It requires
successful `MultiScaleDeformableAttention` and full MonoDETR model imports
before proceeding to checkpoint download or inference. Rerun the existing
clone/compile cell; the pinned `git checkout` and guarded patches make the
cell safe to rerun.

PyTorch 2.6 also changed `torch.load` to default to `weights_only=True`. The
official MonoDETR checkpoint includes pickled NumPy training metadata, so the
notebook explicitly applies `weights_only=False` only inside the pinned
MonoDETR checkpoint loader. Before inference it records
`checkpoint_best.sha256` beside the Drive checkpoint, loads it once on CPU,
and requires the expected `model_state` dictionary. Do not apply this setting
globally or use it for an untrusted checkpoint.

The Drive dataset may use either canonical KITTI object-folder names
`training/image_2` and `training/label_2` or raw aliases `training/image_02`
and `training/label_02`. The notebook stages either form into canonical
`/content/kitti/training/image_2` and `/content/kitti/training/label_2`.

Expected Drive outputs:

```text
/content/drive/MyDrive/mobile_adas3d_outputs/mnv4_conv_small_baseline/runs/<run>/checkpoints/best.pt
/content/drive/MyDrive/mobile_adas3d_outputs/mnv4_conv_small_baseline/runs/<run>/checkpoints/latest.pt
/content/drive/MyDrive/mobile_adas3d_outputs/mnv4_conv_small_baseline/runs/<run>/kitti_r40_val/kitti_r40_summary.json
```

The run is reportable only when `complete_split` is `true`. The evaluator uses
KITTI difficulty filtering, Car IoU 0.7, Pedestrian/Cyclist IoU 0.5, and BEV
and 3D AP_R40. Legacy random 70/15/15 splits and threshold-sweep F1 are not the
canonical comparison.

### Verified locally before Colab

- v0/v1 output contracts remain unchanged;
- v2 emits the extra `projected_center_offset` output with the expected `24x80`
  spatial shape;
- real-sample losses are finite;
- the exact pretrained timm backbone loads;
- TorchScript/export checks should be rerun for any selected deployment
  checkpoint because v2 intentionally changes the training output set;
- the repository tests cover split integrity, geometry/AP_R40, model output,
  projected-center back-projection, and checkpoint resume.

### Work after the untouched baseline

1. Diagnose class-, distance-, yaw-, depth-, and localization-specific errors.
2. Add geometry-safe augmentation, calibration propagation, EMA, and
   validation-AP3D checkpoint selection as controlled experiments.
3. Evaluate stronger depth/orientation supervision and lightweight feature
   fusion one change at a time.
4. Consider teacher distillation only after a reproducible student baseline.
5. Select deployment candidates using AP3D/BEV R40, Core ML parity, model size,
   and physical-iPhone latency together.

Teacher Task 1 is complete. Teacher Task 2 is implemented and awaiting its
Colab cache manifest; student distillation has not started.
