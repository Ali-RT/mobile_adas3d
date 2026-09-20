# M59h — final decoded geometry diagnostic

Status at measurement (2026-09-19): **measurement complete, no new acceptance policy adopted**.
Subsequent decision: the user explicitly approved the proposed policy.
It is frozen separately for M59i in `MONODGP_M59I_FULL_VALIDATION_CONTRACT.md`.
The measurement report and historical M59g failure remain unchanged.
M59g's strict decoded gate remains failed. One planned experiment remains:
complete KITTI preservation evaluation, conditional on explicit numerical-policy
review. No new training or architecture experiment is proposed.

## Frozen scope and provenance

Use the same 16 hash-verified real Chen-val inputs, original M58 CPU TorchScript,
and repaired M59f FP32 Core ML package (ALL) as the completed M59g audit.
No weights, configuration, score threshold, taxonomy, NMS, or top-k changes.
There is no ground truth or AP computation in this diagnostic.

- Input manifest: `artifacts/m59g_input_manifest_20260918.json`.
- Parent: `artifacts/m59g_full16_precision_audit_20260918.json`.
- Runtime config: `configs/monodgp_m56d_runtime_frozen.yaml`, SHA-256
  `4ad6d50241e5a6dd552e5d8b9c043241a11377c26c59b84ab3d2f7f58c7a42af`.
- Result: `artifacts/m59h_geometry_diagnostic_20260919.json`.
- Paired raw outputs and inputs are retained locally in
  `outputs/monodgp_m59h_geometry/full16_20260919/*_paired_raw.npz`
  (Git-ignored); every pair's SHA-256 is recorded in the result.

The script checks the pinned upstream decoder, box utilities, angle utility,
calibration functions, and serialization source hashes. It executes their
pure decoding functions without importing the upstream CUDA model. On every
input and both implementations, the independent geometry port agrees
**bit-for-bit** with the actual native decoder. Tests cover angular wrapping,
bottom-center coordinates, calibration translation, width/length rotation,
score filtering order, heading-bin changes, and text-rounding boundaries.
All **39 M59 regression tests pass**, including nine new M59h tests.

Native extraction uses PyTorch top-k and box conversion on both sets of raw
outputs. Its rankings also match the existing M59g NumPy rankings on all 800
candidates. The old M59g comparisons remain unchanged and separately recorded.

## Decoder semantics retained

- Native classes are `[Pedestrian, Car, Cyclist]`; Car maps to product Vehicle.
  Class 2 is not silently remapped. All 800 selected candidates in this
  particular set belong to the two product classes.
- `meanshape: false`: dimensions are absolute h/w/l outputs, without adding a
  class prior.
- Use original image size and P2 calibration. Unproject the predicted 3D image
  center, then add half the height to obtain KITTI bottom-center location.
- Select one of 12 heading bins and its residual. Convert alpha to yaw using
  the **2D bounding-box center**, matching upstream behavior.
- Filter at native class score `>=0.001`, then multiply by depth confidence.
  Keep top-k 50; apply no new NMS.
- Native text uses **two decimal places** for all 13 numeric output fields.
  Compare continuous geometry and text-rounded geometry separately.
- Derive geometric centers and corners in float64 from the decoded boxes.
  Corner calculations measure implementation differences, not overlap with GT.

## Measured results — 16 inputs, 800 candidates

| Continuous quantity | Maximum absolute difference |
| --- | --- |
| 2D box coordinate | 0.00350654 pixels |
| Forward depth | 0.000761986 m = **0.762 mm** |
| Dimension | 0.000147820 m = **0.148 mm** |
| Bottom-center position, Euclidean | 0.000862420 m = **0.862 mm** |
| Radial distance to geometric center | 0.000861271 m = **0.861 mm** |
| Yaw, circular difference | **0.000426141 degrees** |
| Furthest 3D corner, Euclidean | 0.000898120 m = **0.898 mm** |
| Final confidence | 0.000041008 |

Discrete results:

- Zero selected query/class identity or rank changes.
- Zero heading-bin changes.
- Zero native score-filter or product-taxonomy decisions changed.
- Zero nearby-score decisions changed before or after native text formatting.
- Zero invalid depth/dimension outputs in either implementation.
- The independent decoder port is exactly equal to native decoding on every
  sample/backend, including thresholded output.
- Existing raw checks still pass; the old strict decoded gate still fails.

### Native file formatting matters

**40/800 serialized rows differ**, despite the small continuous deltas.
The maximum stored 2D-coordinate difference is 0.01 pixels; stored depth,
dimensions, position, and corners can differ by **0.01 m (1 cm)** after crossing
a rounding boundary. Stored yaw, alpha, and confidence are unchanged here.
This is why unchanged AP cannot be inferred from small raw deltas alone.

All 800 candidates pass the initial native class-score threshold. Applying the
downstream 0.001 threshold to unrounded final confidence retains 772; applying
it to the actual two-decimal native confidence retains 522. These counts are
identical on both backends. The pre-existing confidence-rounding policy is
preserved, **not fixed or tuned in M59h**, and is not evidence by itself about
the cause of the product pedestrian-recall gap. Changing output precision
would be a separate protocol change, not a conversion fix.

## Decision and proposed review

The evidence supports moving toward the final full-split preservation check,
not changing the model architecture or training. However, this measurement
does **not** override the failed blanket 0.0001 criterion, approve deployment,
or establish AP/nearby-recall preservation.

Proposed **continuous decoded-geometry** limits for explicit review only:

| Quantity | Proposed limit | Rationale |
| --- | --- | --- |
| 2D box coordinate | 0.01 pixel | One native text-format coordinate increment |
| Depth, dimensions, radial distance, center and corner displacement | 0.01 m | One native metric text-format increment; numerical budget, not a safety accuracy target |
| Circular yaw/alpha difference | 0.01 degree | Fine angular budget, far below the native 0.01-radian serialization increment |
| Final confidence | 0.0001 | Retain the existing score-scale numerical budget |

Retain the existing raw-output limits and require unchanged query/class
selection, heading bins, and score-filter decisions on the fixed diagnostic
set. These are **proposed values, not active gates or values inferred to be
safe from 16 images**. The metric/pixel budgets are tied to existing output
resolution; the angular budget is an explicit engineering choice requiring
approval. Do not use this review to silently rewrite M59g's historical result.

If approved, freeze the new policy before the one remaining planned experiment:
paired complete KITTI Chen-val (3,769 images) using the exact native class
mapping, decoding, two-decimal serialization, and frozen accuracy-preservation
gates. Compare actual AP and nearby recall, with all required classes and
complete prediction-file counts. No tuning on that comparison. If preservation
fails, stop and review the conversion approach rather than automatically
starting further small experiments. The full dataset is not in the local
16-input bundle; collection/inference logistics still need preparation.

No iPhone, latency, external-generalization, product-safety, or quantization
qualification is implied by this result. Torch 2.12.0 remains outside Core ML
Tools 9.0's advertised tested Torch range; the warning is retained in evidence.

## Reproduction

```sh
python scripts/diagnose_monodgp_m59h_geometry.py \
  --artifact-dir /absolute/path/to/m59f/full_diagnostic_only \
  --m58-dir /absolute/path/to/monodgp_m58_coreml_conversion \
  --bundle-dir /absolute/path/to/m59g_inputs_TIMESTAMP \
  --m59g-report artifacts/m59g_full16_precision_audit_20260918.json \
  --upstream-repo /absolute/path/to/reviewed_MonoDGP \
  --output-dir /absolute/path/to/new_m59h_diagnostic
```

The reviewed upstream checkout must have the exact decoder/source hashes
listed in the report. New output directories only. `complete: true` means
measurement completed; `all_parity_gates_passed: false` deliberately preserves
the unapproved status. No new notebook run is needed for the reported result.
