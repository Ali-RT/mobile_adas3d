# M66b: zero-update Vehicle-KD gradient diagnostic

Revision: `M66B-ZERO-UPDATE-KD-GRADIENTS-2026-10-06-r1`.
Prepared: 2026-10-06. Native CUDA execution and the diagnostic result remain pending.

## Decision and question

Original A2 epoch130 remains the selected student. M66's real R0-to-A2 Vehicle
feature distillation added only 0.000194 moderate Vehicle 3D AP points over its
matched no-KD control. This does not establish that R0 cannot teach A2. It also
does not establish that a small scalar KD loss caused the negligible gain.

M66b measures whether the **actual weighted KD gradient** is small relative to
GT/preservation, locally opposed to them, sparsely activated by eligible
Vehicle targets, or limited to a narrow part of the depth heads. These are
diagnostic observations, not causal attribution of the historical AP result.
No coefficient or trainable-scope change is selected automatically.

## Frozen inputs

Use the reviewed M66 manifest, signature
`68c2247a9ab28603e10f7f9748fc17a43e5bdefeedc57eebe87e2c62f430d968`,
and its unchanged implementation, smoke, paired gate, checkpoint receipts and
training summaries. The probe validates all four weight-file SHA-256 values:

- Original A2 epoch130: MobileNetV4 Conv Medium and the full MonoDETR graph.
- R0 epoch185: ResNet50 MonoDETR; frozen Vehicle specialist, not a new teacher.
- Completed M66 no-KD control: epoch1, 928 historical updates.
- Completed M66 KD endpoint: epoch1, 928 historical updates.

Raw checkpoints must still be on Drive. `m66_results.zip` contains reports,
not weights. Historical manifests, sources, receipts, summaries and checkpoints
are read-only; the probe never calls checkpoint recovery or rewrites lineage.
Endpoint embedded training summaries must agree with their reviewed sidecars.

## Prospective runtime and data

Notebook: `notebooks/MonoDETR_A2_M66b_Gradient_Diagnostic_Colab.ipynb`.
Run its three sections in order on a Colab GPU; A100 is recommended. No phone
is needed. The notebook defines every helper/path/import in section 1.

Reuse the working private Torch 2.10/CUDA 13 runtime and local attention-build
helpers from M64–M66. Use a fresh `/content/MonoDETR_M66B_A2` checkout of the
public `ZrrSkywalker/MonoDETR` repository, pinned to commit
`6994b9f512400b258c6edb75f77423beb9c126f2`. Its patched source must match M66.
Do not reconstruct M62/M63 or mix Colab's globally installed CUDA packages.

Each model role runs in a fresh subprocess. The new diagnostic identity records
its runtime, GPU, build receipt/binary, probe source, and frozen input hashes.
It does not claim historical runtime equivalence. A changed runtime/GPU/input
identity requires a fresh diagnostic `RUN_ID`; preserve earlier reports.

Use the exact Chen train3712/val3769 split files, labels and calibration. Only
the **first 64 training images in saved order** are forwarded: 16 batches of 4,
seed 444, shuffle off, zero loader workers. Use the unchanged M66 photometric,
crop/flip and scale/shift augmentation. Reset the seed after constructing all
models and loading the selected endpoint. Verify all 16 transformed-image,
calibration and target fingerprints match across the three roles. Validation
labels may be checked for dataset identity but never forwarded or used to
choose samples, targets, weights or scope.

## Measurement, not training

All twelve native `depth_embed.*` parameter tensors are the measurement
coordinates; the backbone, transformer, other heads, BN buffers and affine
parameters remain frozen. Preserve the original M66 training/inference modes
and independent GT-identity Hungarian matching. The original A2 anchor and R0
teacher are detached and unchanged. Preserve the exact Vehicle masks; no
Pedestrian or background external targets are introduced.

Differentiate these three components separately with `torch.autograd.grad`:

1. Native GT loss, including its original native loss weights.
2. Scaled both-class A2 preservation, multiplied by its M66 coefficient 0.10.
3. Vehicle hidden-feature cosine KD, multiplied by its M66 coefficient 0.10.

Do not instantiate an optimizer or take a backward/update/clipping step. Do
not change model state, populate parameter `.grad` buffers, save weights, or
rerun the full validation split. Endpoint probes use the same KD expression
even at the historical no-KD control, solely to measure the hypothetical local
teacher direction there; this does not retroactively make that arm distilled.

For each batch, aggregate tensor dot products in float64 to report vector
norms, not an average of tensor-wise cosines. Report:

- Weighted GT, preservation and KD norms; preservation/GT, KD/GT and
  KD/(GT+preservation) norm ratios.
- KD cosine with GT, preservation and their sum.
- The norm and direction change between GT+preservation and
  GT+preservation+KD, without clipping or optimizer momentum.
- Per-decoder-head hidden/output-layer reach: connected, nonzero tensors and
  nonzero elements; distinguish a disconnected gradient from a connected zero.
- Eligible Vehicle count, near/far coverage, transformed GT counts, and
  preservation masks/counts. External Pedestrian target count must stay zero.

The KD tap is immediately before the final depth-head output linear. Thus its
gradient can reach the final head's hidden linear but not its output linear
or earlier decoder depth heads. GT/preservation can have different reach.
The report measures this structural distinction explicitly.

Undefined ratios/cosines are `null`. At original A2, zero preservation gradients
are expected at anchor equality and are not a failure. Summary gradient
statistics use batches with eligible KD targets, with active and total batch
counts shown separately. Fewer than eight eligible pairs is marked insufficient
diagnostic coverage, not silently treated as proof of weak/conflicting KD.
No strength/conflict threshold is an accuracy gate or an automatic new recipe.

## Integrity and return files

Before and after each probe, all student/anchor/teacher parameters and buffers
must match, `.grad` buffers must remain empty, and frozen input files must keep
their exact bytes. Successful endpoint reports are signed and written once.
Matching completed reports can be verified/reused; conflicting or incomplete
reports are rejected without overwriting them. After a reset, run sections 1–3.

Return `m66b_gradient_results.zip`, containing:

- `original_gradients.json`, `control_gradients.json`, `kd_gradients.json`.
- `m66b_gradient_comparison.json`, with paired-input verification, active-batch
  statistics, coverage, per-layer reach and input/report signatures.
- The new runtime receipt, durable setup/build/test/probe logs and small
  historical M66 input reports. No weights or prediction caches are bundled.

Section 3 also bundles partial diagnostics after a probe failure. All outputs
go to a new M66b diagnostic directory, not into historical model artifacts.

## Stop and interpretation

Review the measurements before proposing one further bounded paired recipe.
This prospective endpoint snapshot does not replay 928 AdamW updates, estimate
their cumulative contribution, decompose AP, guarantee that larger KD weights
help, or establish the optimal teacher/objective.

Training, coefficient sweeps, changed eligibility masks, teacher replacement,
checkpoint promotion, full evaluation and deployment remain unauthorized.
Original A2 stays selected. Local CPU tests verify diagnostic math, unchanged
M66 loss integration, provenance guards and notebook structure; they cannot
establish native CUDA behavior or student accuracy.

Preparation verification: all 25 new M66b CPU tests and 115 existing related
M64–M66 tests passed. Notebook JSON/code-cell syntax, Python compilation and
`git diff --check` passed. The full CPU toy probe measured 16 batches without
changing any model state and reused its completed report without recomputation.
These checks do not substitute for the three-role native CUDA run in Colab.
