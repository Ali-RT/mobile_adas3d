# M59g — bounded numerical precision audit

Status (2026-09-18): **16-input audit complete; strict decoded gate failed**.
Raw limits, repeatability, rewrite equivalence, and query selection pass on
all 16 inputs. No acceptance threshold has changed.

## Question and fixed scope

M59f corrected the positional-channel defect, but its decoded gate still
fails. M59g checks whether the smaller remaining residual is repeatable,
whether query/class selection changes, and whether it persists across a
fixed set of validation inputs. This is a numerical diagnostic of the same
model, not a training experiment, AP evaluation, or architecture change.

- Use the unchanged M59f FP32 full package and the exact original M58 trace.
- Original trace SHA-256:
  `5f4cf0992dd39fc521793a66f53d754b200ef97d0b243f218f9410e1635d3fca`.
- Original input/output anchor SHA-256:
  `6d018439a909678546cdb6060ca7ee5ad6fe08296e22462155e9314b243406f2`.
- Select 16 inputs using `linspace(0,3768,16,dtype=int)` on the hash-verified
  Chen validation list. No selection based on model results or labels.
- Reproduce upstream validation preprocessing: centered width-scaled affine
  sampling to 1280×384, RGB/ImageNet normalization, original P2 calibration,
  original image size. Do not substitute independent width/height resizing.
- Require bit-exact agreement with the original `000001` input before
  collecting the remaining inputs. Stop if preprocessing differs.
- Preserve the existing raw limits and the strict decoded absolute limit
  `0.0001`. The old frozen anchor remains a separate mandatory check; local
  CPU reference generation cannot replace or erase its failure.

For new inputs the reference is the unmodified CPU TorchScript. This tests
TorchScript-to-Core-ML preservation, not eager-model generalization or AP.
Record software versions and input/source hashes. Repeat original CPU inference
twice and Core ML ALL inference three times per input; compare original and
position-repaired CPU outputs once per input. Distinguish rank swaps from
added/removed query-class identities. Run one supplementary CPU_ONLY anchor
control with a 45-second process budget; a timeout is an unavailable result,
not a failed numerical comparison or a latency benchmark.

## Complete 16-input results

The delivered archive passed its prescribed split, source, sample, and input
hash checks. The collected 000001 input is bit-exact against the original
frozen anchor. The audit ran the unchanged M59f full package on all 16 real
inputs using Core ML ALL, with unmodified local CPU TorchScript references.

| Check | Actual result |
| --- | --- |
| Complete selected input set | 16/16 |
| Original CPU repeatability, two calls/input | 16/16 bit-exact |
| Repaired vs original CPU | 16/16 bit-exact |
| Core ML repeatability, three calls/input | 16/16 bit-exact |
| Ten raw-output limits per input | 160/160 checks pass |
| Top-50 query/class identities and rank order | 800/800 unchanged |
| Strict decoded candidate gate | **2/16 pass; 14/16 fail** |
| Supplementary CPU_ONLY anchor control | Timeout at 45 seconds; no parity result |

The two complete decoded passes are 003010 and 003515. The frozen 000001
anchor is additionally compared against its original Colab reference, with
the unchanged result: ten raw passes but decoded max 0.000415802 (fail).
The complete audit therefore records `complete: true`,
`multi_input_complete: true`, and **`all_parity_gates_passed: false`**.

Per-field decoded differences against local CPU (unchanged 0.0001 limit):

| Field | Largest absolute difference | Worst input | Inputs failing |
| --- | --- | --- | --- |
| Class | 0 | — | 0/16 |
| Score | 0.000027746 | 001514 | 0/16 |
| Normalized 2D center | 0.000004292 | 002014 | 0/16 |
| Normalized 2D size | 0.000010371 | 002014 | 0/16 |
| Depth, metres | **0.000761986** | 002014 | **14/16** |
| Angle-code vector | **0.000163436** | 002014 | **7/16** |
| Dimension output | **0.000147820** | 000001 | **2/16** |
| Normalized projected 3D center | 0.000003934 | 002014 | 0/16 |
| Depth confidence | 0.000056863 | 006039 | 0/16 |

The largest selected-depth difference is **0.762 mm**. This is an
implementation-to-implementation difference, not distance error against
ground truth. The angle-code vector contains encoded outputs; its delta is
**not a yaw error in degrees**. The normalized centers are not full
camera-space 3D positions. KITTI conversion and full AP evaluation have not
run in this audit.

These results support a focused downstream-geometry/numerical-policy review;
they do not establish that the model meets a safety or accuracy target.
Repeatability and unchanged top-k selection do not by themselves prove AP
preservation. Do not relabel the failed old gate as passed.

Execution environment: macOS 26.6.2 arm64, Python 3.12.11, Torch 2.12.0,
Core ML Tools 9.0, NumPy 2.4.6. This was runtime execution, not conversion.
The Torch-version compatibility warning remains a recorded caveat.

Evidence:

- `artifacts/m59g_full16_precision_audit_20260918.json`
- `artifacts/m59g_full16_cpu_only_anchor_20260918.txt`
- `artifacts/m59g_input_manifest_20260918.json`
- Delivered archive SHA-256:
  `7eefd4dc780f4f76c46686d57c1ebf61be6cedbc40fcc60690328078a1e6d3d1`
- Input manifest SHA-256:
  `b98049bfd1a6df04d97ecb1ad0afc7702327b86a21b7e602f6f0f45a52015647`

## Initial one-input control (historical)

Before input delivery only one verified validation input was available.
On that input:

| Check | Actual result |
| --- | --- |
| Original CPU, two calls | Bit-exact |
| Repaired vs original CPU | Bit-exact |
| Core ML ALL, three calls | Bit-exact |
| Top-50 identities and order vs CPU | No changes |
| Core ML vs frozen anchor, raw outputs | 10/10 pass unchanged limits |
| Core ML vs frozen anchor, decoded | **Fail**, max 0.000415802 (depth) |
| Core ML vs local CPU, decoded | **Fail**, max 0.000373840 |
| Core ML CPU_ONLY control | Timed out at 45 seconds, no parity result |

The depth residual against the frozen anchor is approximately **0.416 mm**;
it is a difference between implementations, not error against KITTI labels.
The dimension residual is 0.000161171. One-frame repeatability does not prove
multi-input correctness, accuracy preservation, or a particular root cause
for these residuals. Torch 2.12.0 is outside Core ML Tools 9.0's advertised
tested Torch range; the evidence records that environment caveat.

Evidence: `artifacts/m59g_anchor_precision_audit_20260918.json` and
`artifacts/m59g_cpu_only_anchor_20260918.txt`. `complete: true` means the
one-input diagnostic finished; `multi_input_complete: false` and
`all_parity_gates_passed: false` remain explicit.

## Input collection replay (already completed)

The user delivered the complete real-input bundle; no notebook rerun is
needed for the results above. The following instructions are for reproduction.

Open `notebooks/MonoDGP_M59g_Precision_Inputs_Colab.ipynb`, revision
**M59g-2026-09-18-r1**, from **main**. Use a **CPU runtime** and run all three
code cells in order. No other notebook is a prerequisite in the session.

1. Mount Drive and define every path/helper.
2. Sync the clean project main checkout and install input-only dependencies.
3. Verify source artifacts and preprocessing, collect the predetermined
   frames, and write a uniquely named ZIP on Drive.

Required existing files: M58 export gate, `.pt`, reference `.npz`, canonical
`kitti_chen/val.txt`, and original KITTI images/calibrations. The notebook
checks local data first, then `/content/drive/MyDrive/datasets/kitti`.
Edit `DRIVE_KITTI` in the first cell only if your dataset is elsewhere. It
does not clone MonoDGP, rebuild CUDA, download weights, or run training.

Return the printed `m59g_inputs_<timestamp>.zip`, containing the manifest,
validation list, and 16 input NPZ files. The archive has no model weights.
If collection fails, return the printed durable log; do not bypass the
anchor or provenance checks. The actual delivered capture passed these checks.

## Mac execution after input delivery

Extract the bundle to a fresh directory and run:

```sh
python scripts/audit_monodgp_m59g_precision.py \
  --artifact-dir /absolute/path/to/m59f/full_diagnostic_only \
  --m58-dir /absolute/path/to/monodgp_m58_coreml_conversion \
  --bundle-dir /absolute/path/to/m59g_inputs_TIMESTAMP \
  --output-dir /absolute/path/to/new_m59g_audit
```

Use `--anchor-only` instead of `--bundle-dir` only to reproduce the partial
control. Output directories are never silently reused. Exit code 1 with a
complete report means the audit is partial or a parity check failed.

## Recommended next task — M59h, measurement only

Compare final camera-space boxes, distances, dimensions, heading-bin choices,
and decoded yaw on these same 16 inputs, retaining the paired raw outputs.
This separates encoded-tensor differences from actual geometric changes.
Do not retrain, change the model, or change thresholds in that diagnostic.

Then propose unit-aware numerical acceptance criteria for explicit review,
with provenance and justification separate from the failed M59g criterion.
Do not pick a larger tolerance merely to pass these 16 observed samples.
If a new criterion is approved, run the frozen complete 3,769-image
preservation evaluation before any device qualification. This is a proposed
follow-up, **not yet implemented or executed**.

Any
proposal to change numerical acceptance criteria requires a separate,
explicitly reviewed version; this audit never relaxes them automatically.
No physical-device, quantization, full-validation, or deployment approval is
implied. Product accuracy/nearby-recall gaps remain separate open work.

Implementation verification: 30 M59 regression tests pass, including an
end-to-end **synthetic fixture** test of the 16-file collector/ZIP path.
The preprocessing matrix also matches the reviewed upstream implementation
bit-for-bit on six image sizes. Those tests are separate from the completed
real-input macOS experiment above.
