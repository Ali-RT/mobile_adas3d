# M59i — approved numerical policy and complete KITTI preservation

Status (2026-09-21): **Linux input preflight passed; complete tensor collection running**.
The user approved the M59h proposal before any complete-split run. This is the
one remaining planned conversion experiment, not another model/training sweep.

## Delivered dataset and execution attempt (2026-09-21)

Received `m59i_chen_val3769.zip`, archive SHA-256
`9a34ac90e33d3d487ec60129edbcc2d8e76a372c6bb043fb0650358711cebcd0`.
All 11,309 archive entries were safely extracted, and the complete 3,769-sample
bundle, original labels, split, and reviewed image/calibration hashes verified.
The temporary M58/source folders were missing; restored the exact original
trace from the delivered M58 archive and the five unmodified decoder source
files from pinned MonoDGP commit `aa059a18214aebf644510e7f0793971b403f9d14`.
All expected hashes match. Restored files are under project `outputs/` rather
than relying on temporary folders.

The full runner exits **before inference** at the existing bit-exact image
anchor guard. Mac preprocessing changes 11 channel values in the first image,
each by one source-intensity increment (normalized max delta 0.017506957).
Calibration and image size are exact. Across all 16 previously frozen inputs,
306 image-channel values differ. Restoring Pillow11.3/OpenCV5, disabling OpenCV
optimizations, and building isolated Pillow11.3 with `-ffp-contract=off` did
not resolve it. This does not establish which underlying operation causes the
platform difference; no model-accuracy or conversion-preservation result exists.

Evidence: `artifacts/m59i_preprocessing_preflight_20260921.json`.
The delivered dataset is valid. The user approved starting Docker. Linux/x86
with Python3.13.15, NumPy2.1.3, Pillow11.3.0 and OpenCV5.0.0 reproduces **all 16
frozen image/calibration/size tensors bit-for-bit**, including the M58 anchor.
Evidence: `artifacts/m59i_linux_preprocessing_probe_20260921.json`. The probe
also records affine coefficients that differ from Mac coefficients; the full
underlying implementation cause is not isolated. This is not an accuracy claim.

The exact same preprocessing code now runs in the pinned container and saves
source-bound FP32 tensors for both Mac backends. No transform, weights, limits,
or anchor checks change. No new Colab run or dataset upload is needed.

## Frozen decision

`configs/monodgp_m59i_validation_policy.json`, policy
`m59i-approved-20260919-v1`, SHA-256
`8306ec063a55b81d4db9b852242144fd42d078e7ca4644d3a6bcf18e0d7c340f`:

- Fixed 16-input diagnostic set: 0.01 px box coordinates; 0.01 m depth,
  dimensions, radial distance, centers and corners; 0.01 degree circular
  yaw/alpha; 0.0001 final confidence.
- Retain all ten existing raw-output checks and unchanged query/class order,
  heading bins and score-filter decisions on that fixed set. Require native
  decoder parity and valid geometry.
- The policy passes all 16 saved M59h samples. Separate evidence:
  `artifacts/m59i_approved_diagnostic_gate.json`. This is application of the
  approved policy to existing evidence, **not new inference or full AP**.
- M59g's historical strict gate remains failed; no old report is overwritten.
- Complete validation also checks raw-output limits on every image. Unit-aware
  and discrete differences outside the fixed set are reported per image, not
  silently treated as same-rank geometry when query selection changes. Their
  effect is measured by complete AP/nearby evaluation. They require review even
  if the frozen accuracy gates pass; they do not authorize automatic deployment.

These are conversion-error budgets, not ground-truth model-accuracy targets.

## Unchanged accuracy-preservation gates

Both paired PyTorch and Core ML results must meet the pre-existing M55/M56/M57
thresholds. These are **not** new stricter product qualification targets, nor a
claim that passing means bit-identical predictions or zero AP loss. Report the
actual paired differences as well as these absolute frozen floors.

| Metric | Required |
| --- | ---: |
| Vehicle moderate 3D AP_R40 | ≥18.479261 |
| Pedestrian moderate 3D AP_R40 | ≥5.866135 |
| Balanced moderate 3D AP_R40 | ≥12.172698 |
| Vehicle moderate BEV AP_R40 | ≥24.486333 |
| Pedestrian moderate BEV AP_R40 | ≥6.429235 |
| Vehicle nearby recall (<40 m) | ≥0.899925 |
| Pedestrian nearby recall (<30 m) | ≥0.714868 |
| Pedestrian nearby localization-failure rate | ≤0.247654 |
| Prediction files, each backend | Exact 3,769 Chen validation IDs |

The independent product Pedestrian nearby-recall target remains **0.80**.
Preservation must not be presented as meeting that target or safety approval.

## Data handoff — one CPU Colab notebook

Open `notebooks/MonoDGP_M59i_Full_Validation_Bundle_Colab.ipynb` from **main**.
Check the visible revision **M59i-2026-09-19-r1**. Run **all three code cells**
top to bottom. A GPU is unnecessary.

1. Mount Drive, define every path/import and the durable logging helper.
   Set `DRIVE_KITTI` if the original dataset is not `MyDrive/datasets/kitti`.
2. Safely update a clean main checkout and install CPU collection dependencies.
   No upstream clone, destructive reset, CUDA compilation or model setup.
3. Find all 3,769 files, check original label and split hashes, check the
   reviewed 16 image/calibration hashes and exact M58 anchor preprocessing,
   then package original images/calibration/labels and manifest.

Required in Drive: the canonical Chen `val.txt`, original KITTI PNGs/calibration
and **original, unmodified labels**, plus the existing
`monodgp_m58_coreml_conversion/m58_reference_io.npz`. The original KITTI
label-tree hash is
`adf36ff179f3cc66a6b9432cd135a6afa4bf52588b7742584b1870e76f3f9dc7`.
Relabeled MonoDGP training labels are rejected to avoid double-remapping GT.

Return this single ZIP:

```text
/content/drive/MyDrive/mobile_adas3d_outputs/compression/monodgp_m59i_full_validation/m59i_chen_val3769.zip
```

The archive may be several GB. Leave room for both the subset copy and its ZIP.
After a Colab restart, rerun the same three cells: matching files are reused;
different files stop collection and are not deleted. The archive contains no
weights, training state or already-computed accuracy claims.

## Mac paired inference and evaluation

### Verified Linux tensor handoff

The received ZIP is already verified. The raw-Mac preprocessing path remains
strict and rejects its non-bit-exact anchor; do not bypass that failure.
`tools/m59i_preprocessing/Dockerfile` pins the reviewed Linux/amd64 image and
libraries. `scripts/prepare_monodgp_m59i_tensor_inputs.py` runs there, checks all
raw source hashes plus the original 16 NPZ hashes, and requires exact equality
to all reference inputs before collecting the complete 3,769 tensors.

The tensor manifest binds the Docker image content ID, software, preprocessing
and producer source hashes, original dataset manifest, each image/calibration
hash, NPZ hash, and canonical tensor digest. It supports restart only with the
same binding. The consumer independently verifies all tensor files and repeats
the original M58/fixed16 comparisons. Missing, changed or extra inputs fail.
No synthetic, approximate or Mac-regenerated images are substituted.

Current local tensor preparation output:
`outputs/m59i_linux_tensor_inputs_20260921/m59i_tensor_manifest.json`.
Runtime image: `sha256:525a3e09319f9ea93c919396e6b6ffd69a53a48d0ef2026a01d3dcd0b6351604`.
The runtime has no network access, read-only source/data mounts, and one writable
tensor-output mount. Model inference is not performed in Docker.

### Paired inference

Core ML executes on the Mac using the existing reviewed trace, package and
decoder. Wait for the tensor manifest to be complete, then run:

```sh
.venv/bin/python -u scripts/evaluate_monodgp_m59i_coreml.py \
  --dataset-bundle /absolute/path/to/m59i_chen_val3769 \
  --artifact-dir outputs/monodgp_m59f_position_interleave/full_diagnostic_only \
  --m58-dir outputs/m59i_restored_m58_20260921/monodgp_m58_coreml_conversion \
  --upstream-repo outputs/m59i_restored_MonoDGP_source_20260921 \
  --tensor-input-dir outputs/m59i_linux_tensor_inputs_20260921 \
  --reviewed-input-archive /absolute/path/to/monodgp_m59g_precision_audit-20260918T222148Z-1-001.zip \
  --output-dir outputs/monodgp_m59i_full_validation_20260921
```

The restored project-output paths contain the hash-verified original artifacts.
Do not substitute another model/checkpoint if they are missing.
This runner uses original CPU TorchScript versus M59f FP32 Core ML (`ALL`).
It pins the trace, Core ML package, approved policy, runtime config, upstream
decoder functions, data manifest, software versions and local evaluator code.
Only this output directory is written. No source checkpoint/model is modified.

Preserve top-k50, score threshold0.001, no new NMS, absolute dimensions,
original P2/image size, native class order and **two-decimal native text**.
Prediction files retain native names; existing evaluators perform the same
product mapping for both predictions and original GT. Do not improve score
precision within this conversion experiment.

Per-image completion is committed only after both prediction files and paired
candidate tensors have been saved and hashed. Rerunning the exact command
checks bindings/file hashes and resumes completed images. Software, code,
model or policy changes reject reuse. Do not pull new code in the middle of a
run. Progress prints every 25 images; downstream evaluator output is streamed
into durable logs. No inference timing here is a device benchmark.

Both prediction sets are evaluated with the existing product AP, nearby
geometry and pedestrian false-negative scripts. Their legacy `--checkpoint`
provenance field points to the actual hash-pinned TorchScript inference file,
not a fabricated training checkpoint. The parent model/weights remain identified
by the M58/M56d chain.

Deliverables: `m59i_full_validation_gate.json`, paired AP and nearby summaries,
paired metric differences, per-image numerical diagnostics, prediction trees,
candidate tensors and logs. `complete: true` means evaluation completed; consult
`all_validation_gates_passed` separately. No missing-file bypass or threshold
override is exposed.

## Decision after the run

Review the full result once. If it fails, stop and reconsider the conversion
approach; do not automatically start another round of tolerance or architecture
tweaks. If it passes, accuracy preservation is established only under this
frozen protocol. Device latency, thermal stability, external generalization,
quantization and deployment require separate decisions. The known Torch2.12 /
Core ML Tools9.0 tested-version warning remains disclosed.

Local verification: 57 M59 regression tests (including 18 M59i tests) pass.
The collector's copy/ZIP/restart path was exercised on a small synthetic fixture;
notebook code cells compile. The real complete dataset has **not** been run.
