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
MonoDETR model/output name `monodetr_train_cache_clean`, leaving the successful
validation predictions untouched. Its first cell builds the train runtime
configuration and checkpoint digest directly, without depending on the prior
validation-configuration cell. It exposes the 3,712 Chen train IDs through an
isolated dataset view whose inference split is named `val`; MonoDETR enables
random data augmentation whenever the split is literally `train` or
`trainval`. After inference,
`scripts/create_teacher_prediction_cache.py` requires exactly the Chen train
IDs with no missing or extra files, parses all KITTI predictions, copies them
to Drive, revalidates the copy, and records source/config/split/checkpoint and
prediction-tree SHA-256 values. The manifest is initially written with
`complete: false` and is changed to `true` only after the Drive copy passes.

Expected Task 2 artifact:

```text
/content/drive/MyDrive/mobile_adas3d_outputs/teachers/monodetr/chen_train_clean_20260804/
  teacher_cache_manifest.json
  runtime_config.yaml
  predictions/  # exactly 3712 files
```

Task 2 remains in progress until Colab prints the complete manifest. On a
Colab interruption, rerun the train inference cell and then the cache cell;
an interrupted Drive copy remains explicitly incomplete.

#### 2026-08-04 clean train cache result: COMPLETE

The corrected non-augmented pass completed successfully. The inference view
used `dataset.test_split: val`, but its split file was an exact byte-equivalent
copy of the Chen `train.txt`; both SHA-256 values are
`e85ce0142be11c7e4196fd7b79a8bc8c2cefdd6fe754ac61fef8d421e37aba5c`.

```text
cache: chen_train_clean_20260804
prediction files: 3712 / 3712
detections at score >= 0.001: 109947
empty prediction files: 0
inference data augmentation: false
Car 3D AP_R40 easy/moderate/hard: 94.77 / 78.87 / 74.10
checkpoint SHA-256: b618f6da895c1aabafb6a155ab22c312de374133044cc80dfe751280de441ea6
prediction-tree SHA-256: e0d155c8e93bc603cea7824260b4d59e26f96cd4c40bf96040f4785c99529608
manifest complete: true
```

The high AP is an in-sample teacher result because the official MonoDETR
checkpoint was trained on these images. It validates clean deterministic cache
generation, but it is not an unbiased benchmark and must not be compared to
the Chen validation AP as a generalization result.

Teacher Task 2 is complete. The next isolated task is **Teacher Task 3:
teacher/ground-truth matching audit**. The cache contains approximately 29.6
detections per image because it deliberately used a `0.001` score floor. Before
changing student losses, audit score thresholds and one-to-one matching against
Chen train ground truth, reporting precision/recall, unmatched teacher boxes,
per-distance coverage, and 2D/BEV/3D geometry errors. Teacher predictions must
not replace KITTI ground truth; the audit will define which soft teacher signals
are safe as auxiliary supervision.

Teacher Task 3 is implemented by
`scripts/audit_teacher_prediction_cache.py` and the final section of the
MonoDETR Colab notebook. The association rule is deliberately independent of
predicted 3D geometry: process teacher predictions in descending score order
and greedily claim the unmatched Car ground truth with maximum 2D IoU when it
is at least 0.5. BEV/3D IoU, depth, yaw, and dimensions are measured only after
association. The audit sweeps scores `0.001, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3,
0.5, 0.7, 0.9` and reports both maximum-F1 and a 95%-recall target result. The
target result explicitly records whether 95% was achieved; otherwise it labels
the maximum-recall row as a fallback. The selected-match CSV uses maximum F1.

Expected artifacts:

```text
chen_train_clean_20260804/matching_audit_v1/
  teacher_matching_audit.json
  teacher_threshold_sweep.csv
  teacher_distance_coverage.csv
  teacher_selected_matches.csv
```

Task 3 remains in progress until the Colab report is reviewed. Do not select a
distillation threshold merely from train AP or replace KITTI ground truth with
teacher boxes.

#### 2026-08-04 teacher matching audit result

Task 3 completed against the clean cache and verified both the checkpoint and
prediction-tree digests. Score `0.30` maximized F1 and is the selected policy:

```text
association: greedy one-to-one Car match, 2D IoU >= 0.50
score threshold: 0.30
GT Cars: 14357
teacher predictions: 11958
matched: 11324
precision / recall / F1: 0.9470 / 0.7887 / 0.8606
mean 2D / BEV / 3D IoU: 0.9165 / 0.7950 / 0.7666
depth MAE / relative error: 0.252 m / 0.0092
yaw MAE: 1.279 degrees
dimension MAE: 0.045 m
```

No score reached the requested 95% recall. The maximum observed recall was
`0.8718` at score `0.001`, but precision fell to `0.3508`; this row is a
fallback diagnostic, not the distillation policy.

Distance coverage at score `0.30`:

```text
00-20 m: 81.44% recall, 0.849 mean 3D IoU, 0.098 m depth MAE
20-40 m: 97.25% recall, 0.765 mean 3D IoU, 0.246 m depth MAE
40-60 m: 55.24% recall, 0.572 mean 3D IoU, 0.653 m depth MAE
60 m+:    0.92% recall, 0.232 mean 3D IoU, 2.191 m depth MAE
```

Teacher Task 3 is complete. The approved initial distillation policy is:

- retain KITTI ground truth as the primary target for every object;
- add auxiliary teacher geometry only for one-to-one matched Car predictions
  with score >=0.30, 2D IoU >=0.50, and GT depth below 60 m;
- ignore unmatched teacher predictions rather than creating pseudo-labels;
- keep unmatched Car GT, all Pedestrian/Cyclist GT, and all 60 m+ GT on normal
  supervised losses only.

#### 2026-08-04 Teacher Task 4: distillation target adapter

Teacher Task 4 is implemented. `data/teacher_target_adapter.py` now:

- verifies manifest completion, schema, clean/no-augmentation provenance,
  split count and SHA-256, checkpoint SHA-256, and prediction-tree SHA-256;
- reproduces the approved greedy matching policy: Car score >=0.30, 2D IoU
  >=0.50, and GT depth below 60 m;
- emits object-aligned tensors for validity, score, match IoU, 2D box, 3D
  dimensions, 3D location, and yaw while preserving the existing KITTI GT
  object order;
- leaves unmatched Car, all Pedestrian/Cyclist, and 60 m+ objects masked out;
- provides deterministic batch padding without changing the training collate.

`scripts/validate_teacher_target_adapter.py` validates the full Chen train
split and writes `teacher_target_adapter_validation.json`. The final notebook
cell pins the approved checkpoint/tree digests and requires exactly `11,317`
approved teacher matches after the below-60 m mask. The Task 3 audit's `11,324`
count includes seven matched 60 m+ Cars and is therefore not the distillation
target count. The validation report records all three values explicitly:
`11,324` associations, `7` distance-masked associations, and `11,317` approved
teacher targets. This task does not connect teacher tensors to loss terms and
does not start training.

#### 2026-08-04 Teacher Task 5: auxiliary loss integration

Teacher Task 5 is implemented without starting training. The integration:

- keeps KITTI classification, 2D boxes, object assignment, and all existing 3D
  losses unchanged as primary supervision;
- maps approved object-aligned teacher targets only onto the stride-16 cells
  already owned by the corresponding KITTI GT object;
- adds confidence-weighted auxiliary log-depth, dimensions, normalized X/Z and
  Y/Z location, and yaw-cosine losses;
- is controlled by `distillation.enabled`, which is explicitly `false` in
  `configs/kitti_mnv4_quality_scoring_v3.yaml`;
- verifies cache/split/checkpoint/prediction-tree provenance when enabled;
- applies teacher targets to training only, leaving validation supervised-only;
- fails immediately if training is enabled without teacher targets.

`tests/test_teacher_distillation_integration.py` proves exact tensor equality
between the legacy loss and the explicitly disabled path, checks finite enabled
losses/gradients, and checks object-to-dense target mapping.
`scripts/smoke_test_teacher_distillation.py` performs one real cache-backed
forward/backward pass with zero optimizer steps and writes
`teacher_distillation_smoke.json`. Run the final teacher-notebook cell before
creating an enabled experiment configuration or starting a full run.
The notebook invokes the focused tests through `unittest discover` because
Colab can resolve an unrelated installed `tests` package when given a dotted
module name.

Task 5 passed its Colab gate on 2026-08-04:

```text
focused integration tests: 3/3 passed
sample: 000003
device: cuda
approved teacher cells: 9
teacher depth/dim/location/yaw losses: all finite and positive
model gradients: finite
optimizer steps: 0
report: chen_train_clean_20260804/teacher_distillation_smoke.json
```

The smoke model intentionally used `pretrained: false`, so its total loss
(`144.2423`) is not a quality metric and must not be compared with trained
validation loss. The gate verifies target/loss/gradient plumbing only.

The next isolated task is **Teacher Task 6: distillation experiment gate**.
First require the real-data Task 5 smoke report to pass. Then create a distinct
enabled run configuration and run a short, fixed-seed comparison against the
same supervised checkpoint before authorizing an 80-epoch experiment.

Teacher Task 6 is implemented and awaiting its Colab result. The distinct
`configs/kitti_mnv4_teacher_distillation_v6.yaml` inherits V3, explicitly
enables distillation, and lowers fine-tuning LR to `1e-5`. Config inheritance is
deep-merged and cycle-checked by `tools/config.py`.

`scripts/run_distillation_experiment_gate.py` creates two branches from the
same V3 epoch-80 state dict:

```text
control:       100 normal supervised updates
distillation:  100 supervised + teacher auxiliary updates
batch size:    2
sample order:  identical, fixed Chen-train order
evaluation:    same first 64 val batches (512 images), supervised losses only
optimizer:     fresh AdamW for each branch, LR 1e-5
```

It writes `distillation_experiment_gate.json` plus two compatible checkpoints.
This loss comparison is a bounded health gate, not evidence of AP improvement.
Do not authorize 80 epochs until both short-run checkpoints have also been
compared with the same KITTI AP_R40 command.

The 100-step paired loss gate completed on 2026-08-05:

```text
teacher-positive cells: 5009
control supervised val loss:       2.05702383
distillation supervised val loss:  2.05698905
relative distillation change:     -0.00169%
finite run: yes
full training started: no
```

This is a stability pass but a quality-neutral result: the loss difference is
too small to support an improvement claim. The final Task 6 notebook cell now
evaluates the original V3 epoch-80 checkpoint, 100-step control, and 100-step
distillation checkpoints on all 3,769 Chen-val images with identical threshold,
TopK, NMS, config, and AP_R40 implementation. It writes
`kitti_ap_r40_comparison.csv`; that table is the long-run authorization gate.

#### 2026-08-05 Teacher Task 6 result: stable but no long-run authorization

All three checkpoints were evaluated on the complete 3,769-image Chen val split
with score threshold `0.001`, TopK `300`, NMS IoU `0.5`, and the same V3 config.
Moderate AP_R40 results:

```text
                         3D Car  3D Ped  3D Cyc  BEV Car  BEV Ped  BEV Cyc
source V3 epoch 80        2.763   1.107   1.550    5.890    1.577    2.391
control after 100 steps   2.706   1.003   1.712    5.775    1.787    2.147
distill after 100 steps   2.718   1.001   1.710    5.744    1.786    2.171
```

Distillation versus the matched control changed Car moderate AP by only
`+0.0118` 3D and `-0.0308` BEV. Both 100-step branches remained below the
source checkpoint on the primary Car metrics. Therefore the current V6 recipe
is stable but **does not pass the gate for an 80-epoch run**.

The likely limitation is experimental placement rather than a plumbing bug:
the epoch-80 student is already converged and its teacher losses were tiny
(`0.00028` depth, `0.00154` dimensions, `0.00007` location, `0.00223` yaw).
Teacher supervision has little residual signal at that point. Do not simply
increase weights and launch 80 epochs. The next isolated experiment should test
learning acceleration from an early V3 checkpoint (preferably epoch 5) with a
bounded paired run and identical AP evaluation. Only that result can determine
whether teacher guidance is useful during early optimization.

The Task 6 notebook cell is self-contained: `STUDENT_OUTPUT_ROOT` resolves to
`/content/drive/MyDrive/mobile_adas3d_outputs/mnv4_conv_small_baseline`. It
prefers the known V3 run `20260727_184204_mnv4_v3_quality_scoring`, then searches
other matching V3 runs for `epoch_080.pt` if that exact path is absent.

#### 2026-08-04 rejected augmented train cache

The first train-cache run completed 3,712 files but used
`dataset.test_split: train`. MonoDETR's `KITTI_Dataset` sets
`data_augmentation=True` for `train` and `trainval`, so those were predictions
from randomly augmented inputs rather than a deterministic clean-image teacher
pass. The unusually low official metrics exposed the problem:

```text
invalid cache: chen_train_20260731
Car 3D AP_R40 easy/moderate/hard: 6.80 / 5.78 / 5.47
prediction files: 3712
manifest complete: true at file level, semantically invalid
```

Do not use prediction-tree SHA-256
`da4bdb184a1eda3b948913307d2f8d00885fed0b0191db35b5ec3a569cf5e51d`.
The corrected notebook invalidates that manifest when it sees the saved
`test_split: train` configuration. The cache script now rejects augmented
inference splits and verifies that the runtime inference-view IDs exactly equal
the requested Chen train IDs. A fresh clean inference run is required.

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

Teacher Tasks 1 through 6 are complete. The 100-step paired distillation gate
was stable but did not improve AP enough to authorize a long student run.

#### 2026-08-06 MonoDETR MobileNetV4 backbone-only ablation

The next controlled experiment is direct transfer learning inside MonoDETR,
not another custom-student distillation run. The first ablation replaces only
the official ResNet50 backbone with
`mobilenetv4_conv_small.e2400_r224_in1k`. It preserves feature strides
8/16/32, the depth predictor, deformable transformer, queries, prediction
heads, losses, Chen split, and official KITTI evaluation. Backbone feature
channels change from 512/1024/2048 to 64/96/960; MonoDETR's existing 1x1 input
projections absorb that interface change.

Implementation artifacts:

```text
notebooks/MonoDETR_MobileNetV4_Backbone_Ablation_Colab.ipynb
scripts/patch_monodetr_colab_compat.py
scripts/patch_monodetr_mobilenetv4.py
scripts/prepare_monodetr_mnv4_backbone_experiment.py
third_party/monodetr/mobilenetv4_backbone.patch
```

The MonoDETR source revision remains pinned to
`6994b9f512400b258c6edb75f77423beb9c126f2`. The backbone patch checks the
exact original and patched source SHA-256 values and is idempotent. The
experiment initializer uses ImageNet MobileNetV4 weights and copies every
compatible downstream tensor from the validated official MonoDETR checkpoint;
unexpected downstream missing keys or shape mismatches are fatal. Backbone
and input-projection tensors are intentionally initialized for the new
interface.

Run the dedicated notebook with `BATCH_SIZE=16`, `GATE_EPOCHS=20`, and
`LEARNING_RATE=1e-4`. If batch 16 OOMs, restart and use 8, documenting the
change. The gate checkpoints every five epochs to:

```text
/content/drive/MyDrive/mobile_adas3d_outputs/monodetr_backbone_ablation/mnv4_conv_small_gate20/
```

Do not replace the depth predictor or transformer yet. First compare official
KITTI Car AP_R40, total/trainable parameters, peak memory, and inference
latency with the unmodified teacher. Only proceed to a full schedule if the
20-epoch curve is stable and the accuracy/efficiency tradeoff is credible.

#### 2026-08-06 MobileNetV4 epoch-20 gate result and continuation

The gate completed all 3,769 Chen-val predictions. Epoch 20 was best:

```text
epoch   Car 3D moderate AP_R40   Car BEV moderate AP_R40
5       6.618                    11.565
10      9.602                    16.008
15      10.893                   16.823
20      12.399                   18.871
teacher 20.328                   27.097
```

The curve improved at every checkpoint, so the same architecture is authorized
to continue to epoch 50. The notebook now applies
`scripts/patch_monodetr_verbose_resume.py`, resumes explicitly from
`checkpoint_epoch_20.pth`, and restores optimizer, epoch, best-result, and
scheduler position. It logs every 20 batches with current/average loss, LR,
elapsed time, and CUDA allocated/reserved memory, plus an average loss and
duration for every epoch. Do not use the initialization checkpoint for this
continuation because that would reset optimizer and schedule state.

#### 2026-08-11 hardened product contract and architecture decision

`PRODUCT_MODEL_CONTRACT.md` is now the authoritative product decision. Priority
order is nearby Vehicle/Pedestrian detection, distance/location, edge runtime,
2D/3D localization, then dimensions/yaw. The first product taxonomy contains
exactly `Vehicle` and `Pedestrian`; official KITTI labels remain separately
reportable for benchmark compliance.

The epoch-50 continuation completed. Epoch 40 remains the Car-only backbone
ablation winner:

```text
epoch 40 Car 3D moderate AP_R40: 14.065
epoch 40 Car BEV moderate AP_R40: 19.656
teacher Car 3D moderate AP_R40:   20.328
retained 3D performance:         69.2%
```

Epochs 45 and 50 regressed slightly, so the Car-only checkpoint is a research
candidate, not a product acceptance result. A fresh two-class run is required.
The contract fixes dataset roles: KITTI Chen train for optimization, Chen val
for model selection, nuScenes mini for adapter smoke tests, locked nuScenes
`CAM_FRONT` validation for one zero-shot LiDAR-ground-truth evaluation, and a
future locked Waymo subset for independent confirmation.

The locked architecture candidate is MobileMonoDETR-VP1: 1280x384 RGB input,
MobileNetV4 Conv Small stride-8/16/32 features, 256-channel projections,
unchanged MonoDETR LID depth predictor and 3+3-layer depth-aware deformable
transformer, 50 queries, and Vehicle/Pedestrian heads. This is conditional,
not deployment-final: MonoDETR's custom multi-scale deformable-attention path
must first export to Core ML without unsupported/host fallback, pass numerical
and decoded parity, and meet the physical-iPhone latency/stability gates. If it
fails, retain MonoDETR as teacher and return to a Core-ML-native student rather
than weakening deployment requirements.

Next work order:

1. fix post-validation resumable-checkpoint metadata ordering;
2. run the MobileMonoDETR-VP1 Core ML feasibility gate;
3. implement and test the two-class KITTI mapping;
4. train the fresh Vehicle + Pedestrian model;
5. freeze it, run locked nuScenes zero-shot evaluation, then physical-iPhone
   parity and runtime acceptance.

#### 2026-08-11 checkpoint metadata ordering complete

The first hardened-contract task is complete for future runs. The pinned
MonoDETR trainer now receives an exact-SHA, idempotent
`scripts/patch_monodetr_checkpoint_metadata.py` patch after the existing
verbose/resume patch. At every validation checkpoint it:

1. writes the model/optimizer state before validation for crash recovery;
2. evaluates and updates `best_result` and `best_epoch`;
3. writes `checkpoint_best.pth` through a distinct variable when improved;
4. overwrites the resumable epoch checkpoint with finalized post-validation
   metadata.

This prevents the observed epoch-20 mismatch where the model/optimizer were at
epoch 20 but resumable metadata still named epoch 15 as best. Existing saved
checkpoints are intentionally not rewritten. The dedicated MobileMonoDETR
notebook applies the new patch during setup, and focused tests enforce save
ordering and distinct best/resume checkpoint paths. The next contract task is
the MobileMonoDETR-VP1 Core ML graph feasibility gate.

#### 2026-08-11 Core ML deformable-attention microkernel gate complete

The highest-risk Core ML operator gate passed. A fixed-geometry probe models
the locked 1280x384 feature maps (`48x160`, `24x80`, `12x40`, `6x20`), eight
heads, 32 channels per head, and four points per level. It passed for both the
50-query decoder and the 10,200-query encoder case using coremltools 9.0 ML
Program conversion with no custom MIL operations.

MonoDETR's rank-six `[N,Q,H,L,P,2]` sampling layout exceeded Core ML's rank
limit. Flattening `L*P` into a rank-five export layout is mathematically
equivalent and lowered the four levels to native `resample` operations. The
decoder and encoder maximum FP32 deltas were `2.22e-6` and `1.11e-5`, below the
probe's `2e-5` tolerance. Development-Mac CPU microkernel time was 24.7 ms for
Q=50 and 170.7 ms for Q=10,200; these are not iPhone or full-model timings.

The reproducible probe is `scripts/probe_coreml_ms_deform_attn.py`; the complete
decision record is `COREML_FEASIBILITY_REPORT.md`. Status is a microkernel pass,
not deployment approval. Next implement the rank-five export branch in the
pinned MonoDETR patch and convert the complete fixed-shape random-weight graph.

#### 2026-08-11 complete Core ML graph conversion

The pinned MonoDETR source now has a guarded export-only patch covering all
whole-graph blockers: optional CUDA-extension import, rank-five deformable
attention, native fixed-batch multi-head attention, and functional replacements
for in-place box updates. Every `coreml_export` flag defaults to false, so the
training path and checkpoint parameter contract are unchanged.

The complete two-class MobileMonoDETR-VP1 random-weight graph traced with zero
delta and converted all 2,322 frontend operations to a 55.8 MB Float32 iOS 17
ML Program. The graph has 27 `resample`, 14 `matmul`, and zero custom
operations. Outputs are logits `[1,50,2]`, boxes `[1,50,6]`, dimensions
`[1,50,3]`, depth `[1,50,2]`, and angle `[1,50,24]`.

At this stage, Core ML package generation had passed while native compile/load
was still unresolved. A development-Mac coremltools load attempt did not return
within ten minutes, and the active command-line-tools selection did not expose
`coremlcompiler`/`coremlc`. The next gate below resolved this using full Xcode
and a physical iPhone. Reproduce package generation with
`scripts/probe_coreml_full_monodetr.py`; do not commit generated model packages.

#### 2026-08-11 native iPhone graph gate complete: compatibility pass, latency fail

The 55.8 MB random-weight MobileMonoDETR-VP1 ML Program compiled with full
Xcode, embedded in a signed app, loaded with `MLComputeUnits.all`, and returned
all five outputs on an iPhone 16 Pro Max running iOS 26.6. Model load took
566.0 ms; the first warmup took 5,422.2 ms; three steady predictions took
176.6, 161.3, and 161.2 ms.

This proves the complete export graph is natively executable, but rejects it as
the production architecture under the locked `Core ML inference p95 <= 50 ms`
contract. The probe is too small to estimate p95, yet every steady sample is
more than 3x over budget. Random weights are valid for this compute decision
because weights do not change the graph cost. MobileMonoDETR remains the
accuracy teacher/reference; the next task is to lock a Core-ML-native student
architecture before implementing the two-class dataset mapping and training.
Raw evidence is in
`reports/coreml/mobilemonodetr_vp1_random_iphone16promax.json`.

#### 2026-08-11 Core-ML-native student contract complete

`STUDENT_ARCHITECTURE_CONTRACT.md` locks MobileADAS3D-S1 for the next graph
implementation. S1 retains the proven fixed 1280x384 RGB `/255.0` input and
dense Swift-decode approach, but uses MobileNetV4 Conv Small stride-8/16/32
features, a 96-channel Lite-FPN, one shared depthwise-separable prediction
tower, and inexpensive 1x1 output projections. It has exactly two classes,
Vehicle and Pedestrian, plus quality, 2D box, projected-center, depth,
uncertainty, dimensions, axis/direction yaw, and auxiliary location heads.

S1 explicitly excludes transformers, deformable attention, dynamic shapes,
custom Core ML operations, and in-model NMS. Before training, its random graph
must be <=10M parameters, <=15 GMAC, <=25 MB FP16, pass raw Core ML parity, and
achieve <=35 ms p95 across 100 predictions after five warmups on the iPhone 16
Pro Max. The tighter pre-training target reserves margin beneath the 50 ms
product gate. Next implement this graph under a new architecture name and run
the random-weight compile/parity/device gate; do not modify old checkpoint
semantics.

#### 2026-08-11 MobileADAS3D-S1 implementation and device gate complete

S1 is implemented as a separate `MobileADAS3D-S1` build target in
`models/mobile_adas3d_s1.py`; legacy `MobileADAS3D` dispatch and checkpoints are
unchanged. The original graph probe exposed ten learned stride-8 heads. S1
preflight later found that this omitted the distinct 2D `center_offset` head
required by the frozen target, loss, and decoder geometry; the corrected graph
therefore exposes eleven learned heads. PyTorch also derives
`yaw` for existing training/decode code, while Core ML exports only axis and
direction so an FP16 value near the direction threshold cannot create a
spurious pi-flip inside the graph.

The original ten-head random graph has 1.403M parameters, 2.155 GMAC, and a
2.73 MB FP16 Core ML
package. TorchScript delta was zero; maximum learned-head Core ML delta was
0.001506; conversion produced 64 convolutions, two bilinear upsample operations,
and no custom operation. Four focused forward/export tests passed.

The signed model then passed the iPhone 16 Pro Max 5-warmup/100-prediction gate
with `MLComputeUnits.all`: 1.878 ms p50, 2.903 ms p90, 3.788 ms p95, 4.570 ms
p99, 2.204 ms mean, and 13.560 ms max. The 35 ms pre-training gate therefore
passes with substantial margin. Temporary probe code/model were removed from
the app repository after capture. Next implement and audit the two-class KITTI
Vehicle/Pedestrian mapping before any training run.

#### 2026-08-11 production taxonomy implementation complete; Drive audit pending

The dataset boundary now applies an explicit source-to-production map:
Car/Van/Truck/Tram become Vehicle; Pedestrian/Person_sitting become Pedestrian;
Cyclist, Misc, DontCare, and all unlisted labels are excluded. Every retained
object stores both `class_name` and `source_class_name`, so subtype composition
remains auditable. Legacy configs without `class_mapping` retain their original
behavior.

`scripts/create_kitti_taxonomy_manifest.py` scans the exact Chen train and val
split files, reports raw source counts, mapped counts, excluded counts,
source-within-production counts, samples containing mapped objects, and mean
H/W/L dimensions. The manifest binds the policy, split files, and every label
file with SHA-256. Training and preflight fail closed when the manifest is
missing, incomplete, or stale. Four taxonomy tests pass, including mapping the
repository's real sample to Vehicle/Pedestrian IDs while excluding DontCare.

The local repository contains only one KITTI sample, so the reportable 3,712
train / 3,769 val count manifest must be generated where the full dataset lives
in Google Drive. Until that command passes and its counts/dimensions are
reviewed, training remains unauthorized.

#### 2026-08-11 full production-taxonomy Drive audit complete

The Colab audit completed all 3,712 train and 3,769 validation samples, with no
duplicate IDs and no frame lacking a mapped production object. The mapping and
both split/label-tree hashes passed, so the taxonomy gate is complete.

```text
train: Vehicle 16,366; Pedestrian 2,263; total 18,629
val:   Vehicle 16,895; Pedestrian 2,446; total 19,341

train Vehicle sources: Car 14,357; Van 1,297; Truck 488; Tram 224
train Pedestrian sources: Pedestrian 2,207; Person_sitting 56
```

Vehicle is 7.23x more frequent than Pedestrian in train. The initial
Pedestrian loss weight remains 2.5 rather than adding another uncontrolled
change; per-class nearby recall will decide whether a later weighting ablation
is justified. Only train statistics were used to replace the provisional class
mean dimensions:

```text
Vehicle H/W/L:    1.663947 / 1.681781 / 4.309594 m
Pedestrian H/W/L: 1.755652 / 0.625749 / 0.824048 m
```

Validation dimensions were recorded for drift analysis but were not copied
into training configuration. The audit summary is preserved at
`reports/data/kitti_s1_taxonomy_manifest_summary.json`. Next establish the
two-class teacher/reference evaluation protocol before starting S1 training.

#### 2026-08-11 two-class reference protocol complete

`TWO_CLASS_REFERENCE_PROTOCOL.md` freezes R0 as the original ResNet50 MonoDETR
at upstream commit `6994b9f512400b258c6edb75f77423beb9c126f2`, initialized
from the published checkpoint and fine-tuned on the exact production object
mapping. The existing Car-only checkpoint is explicitly rejected as a
Pedestrian denominator.

The evaluator now keeps official KITTI semantics unchanged while supporting a
separate, explicitly named KITTI-difficulty product-taxonomy AP_R40 protocol.
Vehicle uses IoU 0.70, Pedestrian uses 0.50, and mapped Van/Truck/Tram and
Person_sitting objects are positives rather than official neighboring ignores.
Both native S1 output and external KITTI-format R0 predictions pass through the
same mapping. This metric is not represented as an official KITTI leaderboard
claim. The next isolated task is to prepare and run the R0 Colab training job;
GT-only S1 training must wait until the denominator exists.

#### 2026-08-11 R0 Colab preparation complete

`notebooks/MonoDETR_R0_Two_Class_Reference_Colab.ipynb` is the self-contained
GPU workflow for the R0 reference run. It resolves either staged or Drive KITTI
aliases, pins MonoDETR commit `6994b9f512400b258c6edb75f77423beb9c126f2`,
applies the current-PyTorch compatibility and verbose checkpoint patches, and
prepares a hash-tracked 195-epoch ResNet50 run at batch size 16 with checkpoints
every five epochs.

`scripts/patch_monodetr_product_taxonomy.py` maps source labels before filtering
and target encoding while retaining MonoDETR's native three-logit head. This
allows the published checkpoint to load exactly: Car/Van/Truck/Tram train the
native Car logit and Pedestrian/Person_sitting train the native Pedestrian
logit; Cyclist receives no target. `scripts/prepare_monodetr_r0_reference.py`
writes the resolved upstream config, clean initialization checkpoint, and
experiment manifest under the durable Drive run directory.

The preparation task is complete but R0 training has not run. The user should
open the R0 notebook, choose a GPU runtime, retain the frozen parameters, and
run through the real training cell. Upstream `checkpoint_best.pth` is not the
final selection because MonoDETR selects it using Car AP only. After training,
the next implementation task is the product-taxonomy checkpoint sweep and R0
selection using mean Vehicle/Pedestrian moderate 3D AP_R40.

The R0 notebook is interruption-safe. Immediately before the training cell it
scans the durable Drive run directory for `checkpoint_epoch_*.pth`, loads each
candidate on CPU, verifies that filename and payload epochs agree, and requires
both model and optimizer state. It skips corrupt/partial files, resumes from the
highest valid epoch through MonoDETR's patched explicit `resume_model` path, and
does not restart a run that already reached epoch 195. With five-epoch saves,
an interruption loses at most the current checkpoint interval.

After an initial training attempt returned only `Exit 1` without an apparent
child traceback, the notebook training wrapper was hardened to merge and stream
stdout/stderr while writing a timestamped Drive log under
`references/monodetr_r0/colab_logs`. A failure now prints the last 120 captured
lines, subprocess return code, GPU and disk state, and recent upstream log paths
before reporting the durable log location.

The first captured R0 failure was a deterministic adapter defect at epoch 1,
batch 0: the target class used `mapped_class`, but the later mean-dimension
lookup still indexed `cls2id` with source label `Van`. The taxonomy patch now
uses `mapped_class` for both class target and mean-size lookup and its regression
test rejects any remaining `self.cls2id[objects[i].cls_type]` expression. No
checkpoint was produced, so the corrected run should restart cleanly at epoch 0.

#### 2026-08-13 R0 training complete; product sweep prepared

The corrected R0 run completed 195/195 epochs in 5h23m and finalized durable
Drive checkpoints. Epoch 195 official AP_R40 moderate was Car 3D 19.79 / BEV
26.70 and Pedestrian 3D 6.24 / BEV 6.77. Upstream recorded its Car-only best at
epoch 175 (`20.6309`), but that is not the frozen product selection rule.

`scripts/sweep_monodetr_r0_product_checkpoints.py` and the final R0 notebook
cells now perform the required restartable product sweep. Each five-epoch
checkpoint is loaded by pinned MonoDETR, all 3,769 validation predictions are
required, native Car output is mapped to Vehicle, and the shared product AP_R40
evaluator scores Vehicle/Pedestrian. Completed checkpoint evaluations are
cached. Selection maximizes mean Vehicle/Pedestrian moderate 3D AP_R40, then
Vehicle moderate 3D, then mean moderate BEV. The selected checkpoint receives a
SHA-256 and the complete ranking is written to Drive as CSV/JSON. No retraining
is required for this task.

#### 2026-08-13 canonical tracker and frozen R0 result

`PROJECT_TRACKER.md` is now the canonical concise status page and must be
updated after every completed task or gate. The completed product sweep selected
R0 epoch 185 from all 39 checkpoints. Its SHA-256 is
`fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59`;
Vehicle/Pedestrian moderate 3D AP_R40 is 17.6348/5.7214 and moderate BEV is
23.6816/6.5961. This freezes S1's 75%-retention minimums at 13.2261 Vehicle and
4.2910 Pedestrian moderate 3D AP_R40. The active task is the fresh GT-only S1
baseline; knowledge distillation remains intentionally pending until that
baseline is selected and frozen.

#### 2026-08-14 GT-only S1 baseline preparation complete

`notebooks/MobileADAS3D_S1_GT_Baseline_Colab.ipynb` is the dedicated supervised
student workflow. `configs/kitti_mobileadas3d_s1_gt_baseline.yaml` locks the
MobileADAS3D-S1 MobileNetV4 graph, Vehicle/Pedestrian taxonomy, 100-epoch upper
schedule, dedicated Drive output, and `distillation.enabled: false`.

`scripts/prepare_s1_gt_baseline.py` fails closed unless the complete 39-model R0
sweep selects epoch 185 with the frozen metrics and SHA-256. In Colab it also
hashes the referenced R0 checkpoint bytes, writes separate epoch-20 gate and
epoch-100 continuation configs, and records the denominators in
`s1_gt_baseline_manifest.json`. The notebook uses a complete local KITTI stage
when present or a canonical zero-copy Drive alias view, validates split and
taxonomy hashes, requires CUDA, and runs one real finite S1 loss before
training. `latest.pt` is atomic and saved every epoch; rerunning selects only a
matching S1 run. The continuation cell defaults to
`AUTHORIZE_CONTINUATION = False` until the complete 3,769-image epoch-20 product
AP table is reviewed. Knowledge distillation remains inactive.

#### 2026-08-14 S1 GT-only epoch-20 health gate failed accuracy

The corrected eleven-head MobileADAS3D-S1 completed 20 epochs without NaNs or
resume failures. Epoch-20 train/validation total loss was `1.250115/3.689127`
and the best validation loss was `3.552878`. Full Chen validation completed,
but moderate 3D AP_R40 was only `0.024` Vehicle and `0.519` Pedestrian;
Vehicle moderate BEV AP_R40 was `0.246`. Continuation to epoch 100 is not
authorized. The next diagnostic is a fixed-protocol sweep of epochs
5/10/15/20 followed by geometry-error inspection. The inherited
`quality_score_power: 0.0` makes current ranking class-only in effect, so the
dormant quality loss does not explain this failure.

The subsequent full-validation sweep of epochs 5/10/15/20 showed no hidden
early peak. Epoch 20 was best with mean moderate 3D/BEV AP_R40 `0.281/0.645`.
Pedestrian moderate 3D improved from `0.06` at epoch 5 to `0.54` at epoch 20,
while Vehicle remained effectively collapsed (`0.00` through epoch 15 and
`0.02` at epoch 20). The next no-retraining task is an epoch-20 class-aware
2D-match/3D-geometry diagnostic; the notebook now contains a durable cell for
that evaluation before the locked continuation section.

The epoch-20 matched-geometry diagnostic found `12,105` Vehicle and `744`
Pedestrian class-aware matches at 2D IoU >=0.5. Vehicle mean 2D IoU was
`0.675`, showing that the first failure is not absence of 2D detections.
Vehicle depth relative error was `0.111`, but yaw MAE was `72.284` degrees,
dimension MAE `0.405 m`, center MAE `2.958 m`, and corner MAE `3.849 m`.
Pedestrian yaw was similarly poor at `68.188` degrees. Before changing the
model or schedule, the next CPU-only diagnostic measures the 180-degree-axis
error and front/back flip rate from the already saved matched CSV.

The yaw diagnostic isolated the representation failure. Vehicle standard yaw
mean/p50/p90 was `72.284/13.608/178.142` degrees, while its 180-degree-invariant
axis mean/p50/p90 was only `9.632/4.705/23.361` degrees. Vehicle front/back
flip-candidate rate was `0.358`, and `0.391` of matched predictions exceeded
90 degrees standard error. Pedestrian axis mean was weaker at `27.649` degrees
and its flip-candidate rate was `0.179`. The axis feature is therefore learned
for Vehicle, but the separately classified direction bit and hard branch make
final yaw unreliable. This run must not continue. The next controlled S1-V2
experiment holds every other variable fixed and replaces only axis+direction
with direct continuous `[sin(yaw), cos(yaw)]` regression and decoder
normalization.

#### 2026-08-19 reporting summary added

`MODEL_MILESTONES_AND_ARCHITECTURE.md` is the concise reporting document for
the product targets, class mapping, model-selection milestones, published
MonoDETR reproduction versus frozen R0 results, AP_R40 interpretation, S1
architecture, Core ML/iPhone evidence, teacher/student status, and the S1-V2
next step. Keep it synchronized with `PROJECT_TRACKER.md` when a model gate or
deployment milestone changes.

#### 2026-08-19 S1-V2 continuous-yaw gate prepared

S1-V2 is implemented as a controlled yaw-only change: the separate
double-angle axis and hard direction heads are replaced by one head regressed
directly against unit `[sin(yaw), cos(yaw)]` targets. Normalization occurs in
the decoder rather than the FP16 graph. The remaining MobileNetV4/Lite-FPN graph, two-class
taxonomy, input, targets, seed, schedule, and evaluation protocol are held
constant. `configs/kitti_mobileadas3d_s1_v2_continuous_yaw.yaml` isolates the
run and disables distillation. A numerical guard detaches yaw from the cuboid
corner loss so a near-zero raw vector cannot create extreme `atan2` gradients;
direct yaw regression and cosine losses remain active.

`notebooks/MobileADAS3D_S1_V2_Continuous_Yaw_Colab.ipynb` provides CUDA
preflight, fresh-run/resume validation, durable Drive checkpoints, the
20-epoch gate, complete 3,769-image product AP evaluation, geometry diagnostics,
and a fail-closed continuation cell. Focused V1/V2 graph, export-contract,
configuration, notebook, resume, and finite bounded-gradient tests pass. The
next user action is to run that notebook on a Colab GPU through diagnostics and
return the AP/geometry summaries; full continuation and distillation remain
unauthorized.

The final random-weight S1-V2 Core ML probe also passed: 1.403M parameters,
2.155 GMAC, 2.73 MB, no custom operations, and maximum raw-output delta
`0.001508` versus the `0.002` gate. An initial in-graph L2-normalized yaw export
failed parity at `0.009114`; moving scale-invariant normalization to the
decoder removed the reduction/tile operations and restored parity.

#### 2026-08-19 S1-V2 gate rejected due to unbounded yaw objective

Run `20260819_174023_mobileadas3d_s1_v2_continuous_yaw` completed the full
3,769-image epoch-20 product evaluation. Moderate 3D AP_R40 was `0.133`
Vehicle and `0.256` Pedestrian; moderate BEV was `0.801/0.652`. Relative to
S1-V1, Vehicle improved from `0.024` 3D AP and its matched yaw mean/flip rate
improved from `72.28°/35.8%` to `37.77°/17.45%`, but Pedestrian regressed from
`0.519` 3D AP and retained `74.25°` mean yaw error. The run remained far below
the frozen R0-retention gates (`13.226/4.291` moderate 3D and `20.0` Vehicle
BEV).

The training log exposed a correctness defect: treating the raw direct yaw
vector as already normalized made cosine loss `1 - dot(pred,target)` unbounded
with respect to vector magnitude. The yaw cosine term fell to roughly `-1.29`,
and total training loss became negative after continued training. Although the
run proceeded through epoch 100, only the epoch-20 checkpoint was product
evaluated and no checkpoint from this run is valid for selection, deployment,
or distillation.

Next task: create S1-V2b with scale-invariant normalization inside the yaw loss
and a norm floor that bounds gradients near zero, while retaining raw yaw
export plus decoder normalization for Core ML parity. Add explicit zero, small,
unit, and large-vector loss/gradient tests, isolate the output directory, and
rerun a fresh 20-epoch gate without resuming S1-V1 or S1-V2.

#### 2026-08-19 S1-V2b bounded-yaw gate prepared

S1-V2b implements the correction without changing the exported model graph.
For direct sine/cosine supervision, raw yaw is divided by
`max(L2_norm, 0.1)`. The angular loss is therefore bounded to `[0,2]`, the
near-zero normalization Jacobian is capped at 10, and vectors at or above the
floor are scale-invariant. Yaw remains detached from corner loss; raw Core ML
output is normalized by the decoder.

Tests cover zero, `1e-6`, `0.1`, unit, and magnitude-100 yaw vectors, finite
bounded gradients, legacy S1/V5 behavior, teacher-distillation integration,
notebook parsing, isolated resume policy, and the absence of any continuation
cell. All 43 focused tests pass. The unchanged random graph also passes Core ML
again at 1.403M parameters, 2.155 GMAC, 2.73 MB, no custom ops, and `0.001508`
maximum output delta.

Run only `notebooks/MobileADAS3D_S1_V2b_Bounded_Yaw_Colab.ipynb` on a Colab
GPU, top-to-bottom. It writes to the dedicated
`mobileadas3d_s1_v2b_bounded_yaw` Drive directory and stops after complete
epoch-20 AP and geometry diagnostics. The previous S1-V2 notebook is marked
retired. Return the AP table, final training summary, and per-class geometry
table before any continuation or distillation decision.

#### 2026-08-20 S1-V2b epoch-20 gate failed accuracy

Run `20260820_174415_mobileadas3d_s1_v2b_bounded_yaw` used the correct isolated
configuration (`yaw_norm_floor: 0.1`, 20 epochs, distillation disabled), stopped
at epoch 20, produced all 3,769 validation files, and completed product AP plus
geometry diagnostics. The corrected objective remained healthy: epoch-20
train/validation total loss was `0.852646/3.246247`, train yaw regression and
cosine losses were `0.016036/0.034656`, and no negative/unbounded loss recurred.

Accuracy still failed decisively. Moderate 3D AP_R40 was `0.103` Vehicle and
`0.178` Pedestrian; moderate BEV was `0.581/0.454`. These miss the frozen gates
of `13.226/4.291` moderate 3D and `20.0` Vehicle BEV by orders of magnitude.
The matched diagnostic contained 13,511 Vehicle and 900 Pedestrian matches.
Vehicle mean/p50/p90 yaw error was `40.20°/10.59°/161.10°`, axis mean `16.72°`,
and >90° flip rate `18.73%`. Pedestrian mean yaw remained `72.76°` with a
`39.22%` flip rate. Vehicle/Pedestrian depth relative error was approximately
`0.107/0.157`, so yaw and full 3D scoring remain much weaker than basic matched
depth/2D geometry.

Do not continue S1-V2b and do not start distillation. The final no-retraining
audit is to sweep its saved epochs 5/10/15/20 with the frozen product evaluator.
If no hidden AP peak exists, close the dense MobileADAS3D-S1 family as a
speed-qualified but accuracy-inadequate baseline and move to Plan B: a
fixed-shape MobileNetV4 student with a small depth-aware attention encoder and
query decoder, closer to MonoDETR for weight/feature transfer while still
targeting Core ML and iPhone limits.

#### 2026-08-20 dense S1 closed and H1 structure frozen

The user elected to move on without spending another evaluation cycle on the
optional S1-V2b epoch 5/10/15/20 sweep. This closes MobileADAS3D-S1 as a
speed-qualified but accuracy-inadequate baseline. S1-V1, invalid S1-V2, and
corrected S1-V2b are not eligible for continuation, distillation, conversion,
or deployment.

`HYBRID_STUDENT_ARCHITECTURE_CONTRACT.md` is now the active student contract
and provides the requested side-by-side teacher/student structure. The frozen
teacher remains ResNet-50 MonoDETR R0 with an 80-bin depth branch, three-layer
depth-aware deformable encoder, three-layer decoder, and epoch-185 weights.
The new MobileADAS3D-H1 student uses MobileNetV4 Conv Small, a 128-channel
stride-8/16/32 Lite-FPN, a 40-bin stride-16 depth context, a standard two-layer
encoder over 480 stride-32 tokens, and a standard two-layer decoder with 50
queries at width 192. The decoder cross-attends to fixed multi-scale memory so
stride-8 pedestrian evidence is retained.

H1 is closer to the teacher for future query/output/feature distillation but
does not copy the teacher's expensive deformable-attention graph. It excludes
custom ops, dynamic shapes, ROI/grid sampling, recurrent state, and in-model
NMS. The next task is random-graph implementation and edge preflight only:
<=10M parameters, <=15 GMAC, <=25 MB FP16, no custom/fallback ops, <=0.002 raw
Core ML delta, and physical-iPhone p95 <=35 ms. Training and distillation remain
unauthorized until these gates pass.

#### 2026-08-20 H1 random graph implemented; FP16 parity remains blocked

The frozen H1 architecture is implemented in `models/mobile_adas3d_h1.py` and
wired through the standard model builder and a dedicated configuration. Its
fixed attention uses explicit projections, reshapes, matrix multiplication,
softmax, and layer normalization rather than PyTorch's dynamic multi-head
attention export path. The nine teacher-compatible query outputs have the
locked shapes, and all focused H1/S1 forward, export-wrapper, finite-value, and
backward tests pass (12 tests).

Local edge evidence at 1280x384 is 3,619,457 parameters, 4.9068 GMAC, a
10.35 MB FP16 Core ML package, zero trace delta, and no custom Core ML
operation. The strict FP16 raw-output parity gate does not yet pass: maximum
absolute delta is `0.0713265` versus the frozen `0.002` limit, with the largest
differences in class and depth logits. A full-FP32 diagnostic conversion gives
`0.0000361` maximum delta and a 20.56 MB package. This demonstrates faithful
graph conversion and isolates the blocker to reduced-precision accumulation;
the FP32 artifact is not being substituted for the required FP16 artifact.

Do not start H1 training. Next resolve/version the FP16 numerical policy, rerun
the local gate, and then run 5 warmups plus 100 timed predictions on the
physical iPhone. Mac prediction timing is diagnostic only. Distillation stays
disabled until a healthy GT-only H1 checkpoint is trained and frozen.

#### 2026-08-20 H1 FP16 parity gate passed

The parity blocker was traced to default random query-head weights amplifying
insignificant FP16 feature-rounding noise before the model had learned any
signal. H1 now uses neutral query-head initialization: weight standard
deviation `0.001` and zero biases across all nine heads. This does not clamp
outputs, alter the graph contract, introduce mixed precision, or relax the
frozen tolerance.

The all-FP16 Core ML graph now passes with maximum raw-output delta `0.001941`
against the `0.002` gate. It remains 3,619,457 parameters, 4.9068 GMAC, and
10.35 MB with no custom Core ML operations. A trained checkpoint must still
pass its own export parity gate because this result qualifies only the
random-weight architecture.

The remaining pre-training gate is physical iPhone 16 Pro Max model-only
latency: 5 warmups, 100 timed predictions, and p95 <=35 ms. Xcode currently
reports the registered phone as unavailable; reconnect, unlock, and trust it
before continuing. Do not start H1 training yet.

#### 2026-08-21 H1 physical edge gate passed

The validated random H1 FP16 package was added only to an isolated iOS unit
test target; the app's working trained model and runtime paths were not
replaced. The test ran on the paired iPhone 16 Pro Max (`iPhone17,2`, iOS 26.6,
build `23G71`) with `cpuAndNeuralEngine`, 5 warmups, and 100 timed predictions.

Measured latency was mean `5.0417 ms`, median `4.9243 ms`, p95 `5.8043 ms`,
minimum `4.8522 ms`, and maximum `7.1365 ms`. The frozen p95 ceiling is
`35 ms`, so the physical gate passes with substantial margin. Xcode recorded
one passing test in `/tmp/H1EdgeGate-20260821.xcresult`; durable summarized
evidence is committed as `artifacts/h1_edge_preflight_20260821.json`. Temporary
test/model resources were removed from the iPhone app repository afterward.

M21 is complete. The next task is to prepare the fresh GT-only H1 20-epoch
health-gate configuration and Colab notebook with Google Drive checkpoints,
visible epoch/batch logs, and automatic resume. Keep teacher distillation off.

#### 2026-08-21 H1 GT-only health-gate workflow prepared

H1 now has a separate query-native supervised path rather than reusing the
retired S1 dense-grid loss. KITTI objects are encoded as normalized boxes,
projected bottom centers, 40-bin log-depth plus residual, log-dimension
residuals, continuous sine/cosine yaw, and X/Z/Y/Z geometry. Fifty predictions
are assigned one-to-one with Hungarian matching using class, box L1, GIoU, and
projected-center costs. Unmatched queries remain negative focal/quality
targets. The H1 decoder reconstructs KITTI-format 2D/3D predictions for the
unchanged product AP_R40 evaluator.

Run `notebooks/MobileADAS3D_H1_GT_Gate_Colab.ipynb` top-to-bottom on a Colab
GPU. It validates frozen R0 and H1 edge evidence, refreshes the taxonomy audit,
runs focused tests and a real CUDA loss preflight, saves every epoch and the
streamed log to Google Drive, resumes only the exact H1 run, stops at epoch 20,
and evaluates all 3,769 Chen-val images. Distillation is fail-closed false and
there is no continuation cell. Return the final training summary and AP table
before changing the schedule or enabling teacher losses.

#### 2026-08-23 H1 v1 learning gate failed; v2 tiny-overfit gate prepared

The H1 GT-only run completed all 20 epochs and the full 3,769-image evaluation,
but product BEV and 3D AP_R40 rounded to 0.00. It emitted 123,303 detections
(32.72/image); most were plausible road-region geometry priors rather than
image-conditioned objects. Best validation loss occurred at epoch 9
(`7.443909`) and epoch 20 ended at `7.742177`. Do not resume this run and do not
enable distillation.

H1-v2 preserves the exported two-class tensor and the validated Core ML graph.
It appends a fixed zero no-object logit only inside training/decoding for
DETR-style softmax supervision, uses ordinary matched 2D IoU for quality, and
normalizes matched/background quality terms separately. Run
`notebooks/MobileADAS3D_H1_V2_Tiny_Overfit_Colab.ipynb` next. The 16-image,
400-step memorization gate must pass matched confidence, unmatched p95,
matched IoU, and object-count checks before another full KITTI experiment.

#### 2026-08-23 H1-v2 tiny16 gate failed; single-image gate prepared

The tiny16 run completed 100 epochs/400 optimizer steps but failed every hard
gate: matched-score median `0.172`, unmatched-score p95 `0.188`, matched mean
2D IoU `0.258`, and `15.63` predictions/image versus `3.94` ground truth. The
implicit background objective improved separation but did not establish
memorization or localization. Do not start full KITTI training or distillation.

Run `notebooks/MobileADAS3D_H1_V2_Single_Image_Overfit_Colab.ipynb` next. It
selects one Chen-train image containing Vehicle and Pedestrian, performs exactly
1,000 optimizer steps with atomic Drive checkpoints every 100 steps, repeats
the query gate, and compares raw outputs against a different image. Failure to
memorize one image means H1 requires structural/loss revision; successful
memorization plus image sensitivity points instead to schedule/matching issues.

#### 2026-08-24 H1-v2 single-image capacity and sensitivity gates passed

The 1,000-step single-image run memorized sample `000010` containing nine
objects. Matched confidence mean/median/p95 was `0.693/0.732/0.853`, unmatched
confidence mean/p95 was below `0.000001`, matched 2D IoU mean/median/p95 was
`0.825/0.854/0.916`, and decoded prediction count exactly matched ground truth
(`9/9`). All four query gates passed.

The image-conditioning check also passed. Repeating the same image produced
zero maximum output delta. Comparing sample `000010` with `000000` changed
every output head; representative mean absolute deltas were `9.197` for class
logits, `0.336` for boxes, `4.471` for depth logits, `0.433` for yaw, and
`4.827` for quality. H1-v2 therefore has sufficient local capacity, receives
image information, can localize objects, and can suppress unmatched queries.

This does not overturn the failed 16-image/400-step result; it narrows its root
cause to multi-image optimization, stable assignment, or schedule rather than
a disconnected or incapable graph. The next controlled gate is a fresh
Tiny16 run extended to 2,000 optimizer steps, with diagnostics at steps
400/800/1200/1600/2000. Keep the model, loss, matcher, data, and thresholds
fixed, and do not initialize from the one-image memorization checkpoint. Full
KITTI training and distillation remain unauthorized until that gate passes.

#### 2026-08-24 H1-v2 Tiny16 2,000-step gate prepared

Run `notebooks/MobileADAS3D_H1_V2_Tiny_2000Step_Colab.ipynb` top-to-bottom on
a Colab GPU. It reproduces the exact prior Tiny16 split and inherits the same
H1-v2 graph, Hungarian matcher, loss weights, learning rate, batch size, and
inference thresholds. The only experimental change is training duration: 500
four-batch epochs equal exactly 2,000 optimizer steps.

The run uses a new Google Drive output directory so it cannot accidentally
resume the failed 400-step run or the single-image checkpoint. It saves atomic
epoch checkpoints and durable logs, automatically resumes only its own run,
and preserves milestone checkpoints at steps 400/800/1200/1600/2000. The final
cell applies the unchanged query confidence, background, IoU, and count gates
to every milestone and writes `h1_v2_tiny_2000step_diagnostics.json`. Return
that file before authorizing full KITTI training or distillation.

#### 2026-08-24 H1-v2 Tiny16 2,000-step gate failed

The fresh run completed all 2,000 optimizer steps and evaluated checkpoints at
steps 400/800/1200/1600/2000. No checkpoint passed any of the four hard gates.
Matched-score median was `0.172/0.180/0.237/0.287/0.261`; unmatched-score p95
was `0.188/0.229/0.309/0.291/0.367`; mean matched IoU was
`0.258/0.321/0.294/0.397/0.425`; and predictions/image was
`15.63/12.25/9.31/10.25/14.06` versus `3.94` ground truth.

The longer schedule is therefore ruled out as a sufficient fix. Localization
did improve, proving continued learning, but confidence/background separation
remained weak and the false-positive tail worsened late in training. Do not
continue this run, start full KITTI, or enable distillation.

Before revising the loss or model, use the saved five checkpoints for a
bounded diagnostic that measures GT-to-query assignment churn and per-object
localization across milestones. Also compare eval-mode outputs with controlled
BatchNorm behavior on the identical Tiny16 data. This will distinguish unstable
Hungarian ownership from small-batch running-statistics mismatch and prevents
combining multiple speculative changes in the next experiment.

#### 2026-08-24 H1-v2 assignment/normalization diagnostic prepared

Run
`notebooks/MobileADAS3D_H1_V2_Assignment_Normalization_Diagnostic_Colab.ipynb`
top-to-bottom on a Colab GPU. It performs no training and requires the one
completed Tiny16 2,000-step run with milestone checkpoints at epochs
100/200/300/400/500.

For every milestone, the diagnostic computes matched/unmatched score
distributions, matched 2D IoU, and prediction-count error in normal eval mode
and with BatchNorm layers using current batch statistics. In eval mode it also
records the Hungarian query owner and IoU for every one of the 63 Tiny16
objects, then reports adjacent query retention, fully stable object rate, and
unique queries used per object across milestones. The durable output is
`h1_v2_assignment_normalization_diagnostic.json`. Return that file before any
new training, loss changes, graph changes, full KITTI run, or distillation.

#### 2026-08-24 H1-v2 assignment/normalization diagnosis completed

BatchNorm running statistics are not the primary failure. Across steps
400/800/1200/1600/2000, switching 52 BatchNorm layers to current batch
statistics produced only small, mixed changes. At step 1600, for example,
eval/batch-stat matched-score median was `0.287/0.275`, unmatched p95 was
`0.291/0.258`, mean IoU was `0.397/0.372`, and count error was `6.31/5.81`.
Neither mode approached the gates.

Hungarian ownership was extremely unstable. Only `7.14%` of adjacent
checkpoint assignments retained the same query, no object kept one query
through all five checkpoints, and the 63 objects used `4.44` unique query IDs
on average (median and p95 both `5`). Query permutation across widely separated
checkpoints is not independently an error, but combined with weak IoU,
confidence separation, and count control it is strong evidence that H1's
spatially indistinguishable learned queries never establish stable ownership.

Freeze H1-v2 as rejected; do not tune BatchNorm, extend its schedule, run full
KITTI, or enable distillation. The next bounded architecture revision is H2:
keep MobileNetV4, transformer size, 50 queries, and all nine exported tensor
shapes, but assign fixed 2D reference points and predict bounded box-center and
projected-center offsets from those references. Repeat the single-image and
Tiny16 capacity gates before rerunning Core ML or full KITTI qualification.

#### 2026-08-24 H2 spatial-reference graph implemented and locally qualified

H2 is a versioned subclass; the frozen H1 graph and its old checkpoints remain
reproducible. H2 keeps MobileNetV4 Conv Small, Lite-FPN, depth context,
transformer width/layers/heads, 50 queries, 40 depth bins, 3,619,457 trainable
parameters, and the exact nine exported output names and shapes.

The 50 queries now receive fixed row-major 10×5 normalized reference points
and the corresponding 2D sine/cosine encoding. Box center and projected center
are decoded as `reference + 0.10*tanh(raw_offset)` and clamped to `[0,1]`;
width/height and all other heads retain their H1 behavior. Thus neutral initial
heads begin at 50 spatially distinct centers rather than all near `(0.5,0.5)`.

Local forward/backward, finite-output, fixed-grid, wrapper, output-contract,
parameter, criterion-routing, and H1 regression checks pass. The full suite is
125 passing tests with one expected CUDA-only skip. No Core ML or physical
iPhone claim is made for H2 yet. Next prepare the fresh 1,000-step H2
single-image capacity gate with unchanged H1-v2 loss and thresholds; Tiny16 is
authorized only if that first gate passes.

#### 2026-08-24 H2 single-image capacity workflow prepared

Run `notebooks/MobileADAS3D_H2_Single_Image_Overfit_Colab.ipynb`
top-to-bottom on a Colab GPU. It deterministically selects the same Chen-train
image containing Vehicle and Pedestrian as the H1 capacity test, but uses a
new split/output directory and refuses cross-architecture checkpoint resume.

The workflow validates the H2 graph and unchanged H1-v2 implicit-background
criterion on CUDA, performs exactly 1,000 optimizer steps, atomically saves and
resumes every 100 steps, then runs the unchanged query confidence/background,
2D IoU, object-count, deterministic-repeat, and cross-image sensitivity gates.
Return `single_image_query_diagnostics.json` and
`single_image_sensitivity.json`. A failure blocks Tiny16, Core ML, full KITTI,
and distillation. The full local suite passes 128 tests with one expected
CUDA-only skip.

#### 2026-08-24 H2 single-image gate failed localization

After 1,000 steps, H2 passed matched confidence (`0.724` median), background
suppression (`0.000333` unmatched p95), exact count (`9/9`), deterministic
repeat, and cross-image sensitivity. Every output head changed materially for
the comparison image. However, matched 2D IoU mean was only `0.555` against
the `0.70` gate, although median/p95 was `0.702/0.817`. The overall query gate
therefore failed and Tiny16 is not authorized.

Compared with H1-v2 on the same image (`0.825` mean IoU), H2 preserved object
presence/background learning but degraded localization. The median/mean gap
suggests a few severe outliers rather than uniform failure. Do not relax the
gate, extend training, or run Tiny16 yet. First reuse the saved checkpoint for
a per-object reachability diagnostic: record matched query reference points,
GT and predicted box/projected centers, whether each target is representable
within the ±`0.10` bounds, boundary saturation, size error, and IoU. Use that
evidence to choose between changing the offset scale, anchoring only box
centers, or revising the reference layout.

#### 2026-08-24 accuracy-first strategy pivot

The iPhone constraint is suspended during model development. S1, H1, and H2
are frozen as reproducible negative experiments: they demonstrated excellent
edge speed, but not adequate supervised two-class learning. The planned H2
reachability diagnostic and Tiny16 continuation are cancelled.

The active candidate is now MobileMonoDETR-Student-A1: the proven two-class
MonoDETR pipeline with only ResNet50 replaced by MobileNetV4 Conv Small. The
first run is GT-only; a paired distilled run may follow only from the identical
initialization and schedule. The existing Car-only teacher cache is not valid
for this two-class experiment.

Comparable performance is locked at 90% of frozen R0 epoch 185 for all five
moderate metrics: Vehicle/Pedestrian 3D AP_R40 `15.8713/5.1493`, balanced 3D
mean `10.5103`, and Vehicle/Pedestrian BEV AP_R40 `21.3134/5.9365`. After an
accurate student is frozen, compression proceeds one controlled variable at a
time before Core ML and physical-device qualification are restored. See
`ACCURACY_FIRST_STUDENT_CONTRACT.md` and `PROJECT_TRACKER.md`.

#### 2026-08-24 A1 GT-only workflow prepared

Run `notebooks/MonoDETR_A1_MobileNetV4_Two_Class_GT_Colab.ipynb`
top-to-bottom on a Colab GPU. It pins MonoDETR commit
`6994b9f512400b258c6edb75f77423beb9c126f2`, reconstructs the Chen 3,712/3,769
view, and refuses to prepare A1 unless the R0 product selection identifies
epoch 185 with checkpoint SHA-256
`fc0eba200e44b88921af76b0a5c94279872fd5c4838ab4d8936838447debfa59`.

The preparer fixes initialization seed `20260824`, loads ImageNet MobileNetV4
Conv Small plus every shape-compatible downstream tensor from the frozen
two-class R0 checkpoint, and treats only the backbone and required feature
projections as new. Taxonomy, resolution, transformer, queries, heads, decoder,
training data, and product evaluator remain unchanged. Distillation is false.

Training output is unbuffered and duplicated to timestamped Google Drive logs.
Every five epochs is durably saved; restart scans only the exact A1 run and
skips incomplete checkpoints. The final restartable product sweep evaluates
all 3,769 images and writes `a1_product_selection.json` with the five frozen
90%-of-R0 gates. Return the experiment manifest, final training summary, and
selection report. Do not start paired distillation until this result is frozen.
