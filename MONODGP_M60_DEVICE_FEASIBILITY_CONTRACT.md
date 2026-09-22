# M60 — physical-iPhone feasibility of the M59i model

Status (2026-09-22): **complete—fixed16 parity passed; device latency failed**.
Revision: **M60-2026-09-21-r1**. This is a bounded runtime experiment, not training.

## Physical-device result (2026-09-22)

The complete run was retrieved from the iPhone 16 Pro Max (iOS 26.6.2).
All 16 inputs and all ten outputs per input passed independent host review
against both frozen PyTorch and Mac references, including native decoding,
query/class ordering, heading bins, score-filter decisions and final geometry.
Maximum differences versus PyTorch were 0.003043 px for box coordinates,
0.666 mm depth, 0.783 mm corner displacement and 0.000331 degrees yaw.
No model, input, acceptance limit or compute-unit setting was changed.

| Measurement | Initial 100 predictions | 60-second loop |
| --- | ---: | ---: |
| Predictions | 100 | 235 |
| Median inference | 231.78 ms | 240.22 ms |
| p95 inference | 233.73 ms | 305.50 ms |
| Maximum inference | 236.07 ms | 350.80 ms |
| Mean inference | 231.72 ms | 255.76 ms |
| p95 <= 50 ms | **Fail** | **Fail** |

Five warmups completed. The stability loop lasted 60.262 seconds and finished
without a recorded exception or thermal stop. Thermal state changed from
`nominal` to `fair`; no `serious` or `critical` observation occurred. This does
not establish that thermal throttling caused the slowdown. Sampled peak process
RSS was 491.67 MiB, including 90.00 MiB of cached input fixtures; final sampled
RSS was 285.89 MiB. This is not model-only or exact peak allocation. Observed
compilation/load durations were 334.4 ms/1,149.4 ms; not a cold-start guarantee.

Evidence (host review byte-exact; device JSON gains only a terminal newline):

- `artifacts/m60_device_review_20260922.json`, SHA-256
  `a77bdb10367db74e4b1f2409ef8313db1706e3eda87ce46c38587fb2db4b528d`.
- `artifacts/m60_device_report_20260922.json`, SHA-256
  `c76898a6448cc696fd52a972d042bfa1e3441011cddc3b2d392f6019e09798f4`.
  Original downloaded device report, before the terminal newline: SHA-256
  `5432f6956bd1680b21edfb7919140d46a2e2ab13fa694ded3296740ed645ce29`.
- Full raw tensors remain in `outputs/m60_device_run_20260922`, tree SHA-256
  `0b18f053c6c7cba81fe1058a20e4e37cbf47a0cd330b222535df7a13d3fe5dc9`.

The evaluator exits 1 intentionally because latency fails, despite
`complete=true` and `device_parity_passed=true`. This is a completed negative
feasibility result, not an execution crash. No full KITTI AP was run on iPhone;
device parity is limited to the frozen16 set. No camera integration or new
optimization experiment is authorized by this result.

Next proposal: one bounded device operator/compute-placement profile of this
unchanged graph, to identify the dominant cost before choosing an optimization.
ALL does not identify actual CPU/GPU/Neural Engine placement. Do not begin a
blind quantization sweep or integrate a ~6.1x-over-budget sustained p95 model
into the live camera pipeline. The separate Pedestrian nearby-recall gap remains.

## Preparation and installation history

Preparation verified on this Mac: unsigned Release iPhoneOS build passed with
Xcode27.0; fresh Mac/PyTorch checks passed for all16 fixtures; the actual Swift
reader preserved every input byte and correctly serialized a padded output.
Seven new Python tests and57 M59 regressions pass. The built app resources match
the prepared bundle byte-for-byte. Evidence:
`artifacts/m60_preparation_20260921.json`. Signing, installation and app launch
and the completed fixed-input device run are now verified (results above).

Connection check: physical iPhone16 Pro Max, iOS26.6.2, wired/paired and Developer
Mode enabled. Signed Release build exited65: Xcode reports invalid saved account
credentials (`missing Xcode-Username`), `No Accounts`, and no provisioning
profile for `com.ali.MonoDGPM60`. User sign-in in Xcode Settings > Accounts
was required before retrying. Historical evidence:
`artifacts/m60_signing_blocker_20260921.json`. Do not change bundle ID to overwrite
the legacy app or weaken signing to work around account authentication.

On2026-09-22 the user signed back in. Signed Release build and installation
passed with matching app/team/device provisioning and byte-exact resources.
First launch was denied by iOS; after the user trusted the developer profile,
`devicectl` successfully launched the app. A Mac `codesign --verify` check
reported `CSSMERR_TP_NOT_TRUSTED`; this is retained in the installation record,
not described as a successful host signature check. Profile expires
2026-09-29T13:24:19UTC. No benchmark report existed at launch.
Evidence: `artifacts/m60_device_installation_20260922.json`.

## Frozen candidate

- Exact M59f FP32 ML Program, M54/M56d weights, validated by M59i on all
  3,769 Chen validation images. Package tree SHA-256:
  `90a5146acc25eb96d8bdafd27c3434898f737082ef5c1c8140d3ba97aaba1459`.
- M59i final report SHA-256:
  `2aff5f2603d44d30dea1edce5cf6104375b1550eb560f0497296b2f6b21c4ad2`.
- ALL compute units, unchanged model precision and outputs. This requests the
  OS scheduler; it does **not** prove all computation uses the Neural Engine.
- Exact frozen16 FP32 inputs from the hash-verified M59g archive. No new Swift
  image preprocessing or camera-calibration assumptions are introduced.
- No Core ML re-export, new checkpoint, distillation or quantization.

## M60a: first bounded device run

The standalone `ios/M60Benchmark` application has bundle ID
`com.ali.MonoDGPM60`. It does not replace or edit the dirty legacy v7 app
repository and does not use its incompatible dense decoder.

1. Verify resource hashes and compile/load the unchanged package on the phone.
   Record compilation and model-loading durations separately.
2. Predict all16 inputs, save all10 raw tensors in row-major FP32, respecting
   Core ML output strides. Compare raw values with fresh verified Mac outputs.
3. Stop timing if raw checks fail. Otherwise perform5 extra warmups and100
   timed synchronous predictions, cycling the same16 inputs.
4. Run an additional60-second fixed-input loop. Save latency samples, thermal
   state and sampled process resident memory. Stop on serious/critical thermal
   state; preserve a partial report. Core ML calls cannot be interrupted mid-call.
5. Download the complete run directory. The host repeats native candidate
   selection and decoded geometry comparisons versus **both** frozen PyTorch
   and Mac references, using the unchanged M59i fixed16 limits. Changed query
   rank/identity is a failed gate, not silently re-aligned.

Numerical gate: all16 pass raw, geometry, class/rank/bin/filter checks.
Model-only speed target: p95≤50ms for both100 predictions and60-second loop,
inherited from `PRODUCT_MODEL_CONTRACT.md`. Report p50/p90/p95/p99/max and mean.
There is no newly invented memory pass threshold. Sampled RSS includes the app,
model and ~94MB cached input fixtures; it is not model-only memory or an exact
allocator peak. A60-second loop does not certify30-minute thermal stability.
All results retain device hardware, iOS version, compute policy and manifest hash.

Model-only latency excludes input-file reading, camera capture, preprocessing,
decoding, comparisons and artifact writes. Sampling/report writes can affect
loop throughput, so do not infer capture FPS from this test. The same compiled
model exposes depth/region diagnostics; no inference-output pruning is assumed.

## Decision and stop conditions

- Compile/load failure: retain the error, inspect memory/operator support once;
  do not start a training or quantization sweep.
- Numerical failure: preserve outputs and localize the device-specific delta.
  Do not relax M59i limits to obtain a pass.
- Numerical pass but slow: measure bottlenecks and propose a targeted runtime
  change or a different hardware target. Preserve the validated baseline.
- Numerical/speed pass: propose M60b live-camera integration with the correct
  MonoDGP calibration/preprocessing/decoder contract. Only then measure
  preprocessing≤20ms, capture-to-decoded p95≤100ms, ≥10 processedFPS,
  frame dropping, battery use, and30-minute no-saving/full-artifact stability.

None of M60a approves deployment, external generalization or safety. Product
Pedestrian nearby recall remains0.723104 vs0.80. The legacy app's runtime and
camera evidence cannot be transferred to this new model.

## Reproduce preparation on this Mac

Use the existing frozen artifacts; no Colab or further dataset upload needed:

```sh
.venv/bin/python -u scripts/prepare_monodgp_m60_device_bundle.py \
  --package outputs/monodgp_m59f_position_interleave/full_diagnostic_only/MonoDGP_M59f_fp32.mlpackage \
  --trace outputs/m59i_restored_m58_20260921/monodgp_m58_coreml_conversion/MonoDGP_M58_fixed_fp32.pt \
  --reviewed-input-archive /Users/ral3ply/Downloads/monodgp_m59g_precision_audit-20260918T222148Z-1-001.zip \
  --upstream-repo outputs/m59i_restored_MonoDGP_source_20260921 \
  --output-dir outputs/monodgp_m60_device_bundle/M60

DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer xcodebuild \
  -project ios/M60Benchmark/M60Benchmark.xcodeproj -scheme M60Benchmark \
  -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath outputs/m60_xcode_build CODE_SIGNING_ALLOWED=NO build
```

Preparation refuses an existing output directory; reuse its verified manifest
or choose a new directory and update the Xcode folder reference deliberately.
Large model/input resources and generated builds remain ignored under `outputs/`.
The source application, evaluator, tests and contract are in main.

## Phone handoff

After unsigned build and fixture checks pass, connect the iPhone by USB, unlock
it, tap **Trust** if prompted, and leave it connected. Developer Mode may need
enabling if Xcode requests it. No driving, street recordings or camera access
are needed. Install a signed Release build after discovering the actual device;
open **MonoDGP M60** and press **Start device benchmark** while cool and idle.
Keep it foreground and unlocked. Do not interpret a saved run as a parity pass
until the host has checked it.

Fetch `Documents/M60-<UUID>` (all binary files plus `report.json`) with Xcode/
devicectl or Files sharing. Run:

```sh
.venv/bin/python scripts/evaluate_monodgp_m60_device.py \
  --bundle outputs/monodgp_m60_device_bundle/M60 \
  --device-run /path/to/downloaded/M60-UUID \
  --upstream-repo outputs/m59i_restored_MonoDGP_source_20260921 \
  --output outputs/m60_device_review.json
```

The existing signing team is reused from the user's app project; account/device
signing availability is checked only after connection. A successful generic
unsigned build is not proof of installation or physical-device execution.
