# M59f — export-only positional-interleaving repair

Status (2026-09-18): **channel-order defect repaired; final decoded numerical gate remains open**.

## Scope and implementation

Reuse the trusted, hash-verified M59d and M58 TorchScript artifacts. Replace
only the four traced `PositionEmbeddingSine` methods' two sine/cosine pairs
(eight pairs total):

```
stack([sin(even), cos(odd)], dim=4).flatten(3)
→ cat([sin(even), cos(odd)], dim=3).index_select(3, [0,64,1,65,...,63,127])
```

This explicitly retains the intended interleaving without the problematic
rank-five stack/flatten path. It is not a post-hoc permutation of detector
outputs. Weights, architecture, source inputs, FP32 compute precision, and all
acceptance limits stay unchanged. The script rejects an unexpected positional
graph instead of guessing. Original artifacts are never overwritten.

The full-model unmodified/repaired CPU PyTorch outputs are **bit-for-bit
identical** on the frozen sample. The equivalent rewrite also passes tests
with four different spatial shapes and nontrivial masks. All three exports
pass PyTorch preservation against their frozen reference; there are no custom
MIL operators. Export success is separate from runtime acceptance.

## Measured macOS results

| Check | Result |
| --- | --- |
| Four positional scales | All pass; fourth-scale error drops from 1.999016 to 4.768372e-7 |
| First encoder internals | 18/18 pass at unchanged 0.001 limit |
| Region/depth and 2D transformer layers | 13/13 pass at unchanged 0.001 limit |
| Full-model raw outputs | 10/10 pass existing M56-family limits |
| Top-50 selected query/class indices | Zero changes |
| Decoded candidate gate | **Fail:** max delta 0.000415802 vs 0.0001 |

Final raw max absolute deltas: logits 4.863739e-5; boxes 1.564622e-6;
dimensions 1.611710e-4; depth 5.836487e-4; angle 9.202957e-5.

The strict decoded failure is in depth (0.000415802 m, approximately 0.416 mm)
and dimensions (0.000161171). Other decoded fields pass 0.0001. These are
**numerical differences against the same model's reference**, not detection
errors against KITTI ground truth. All 50 selected indices and class ranks are
unchanged. The unmodified CPU PyTorch control passes the decoded gate with
max delta 6.866455e-5; repaired vs unmodified CPU outputs have zero delta.
Thus the rewrite itself is equivalent, but the smaller Core ML residual is
still a separate unresolved acceptance issue. Do not mark the complete gate
as passed or relax its threshold automatically.

ALL and CPU_AND_GPU produced identical raw and decoded comparison results.

Scope: one frozen KITTI input (`000001`), not a complete validation run or
latency benchmark. macOS 26.6.2, Core ML Tools 9.0, Torch 2.12.0; export uses
NumPy 2.3.5. Torch is newer than Core ML Tools' advertised tested range.
No inference-speed, AP preservation across the full split, or iPhone readiness
claim follows from these results.

## Evidence and reproducibility

- `artifacts/m59f_internals_export_gate_20260918.json`
- `artifacts/m59f_layers_export_gate_20260918.json`
- `artifacts/m59f_full_export_gate_20260918.json`
- `artifacts/m59f_internals_macos_all_20260918.json`
- `artifacts/m59f_layers_macos_all_20260918.json`
- `artifacts/m59f_full_macos_all_20260918.json` (includes CPU control and field breakdown)
- `artifacts/m59f_full_macos_cpu_gpu_20260918.json`

The repaired full package and reference/gate are retained locally under
`outputs/monodgp_m59f_position_interleave/full_diagnostic_only/` (ignored by Git).
The scripts and small evidence JSONs are versioned; large packages are not.

Optional Colab replay: `notebooks/MonoDGP_M59f_Position_Interleave_Colab.ipynb`,
revision `M59f-2026-09-18-r1`, three cells top to bottom on CPU. It needs the
original M59d and M58 export gate, `.pt`, and reference `.npz` files in Drive.
It does not need KITTI data or an upstream repository. The local experiment
has already run; a new Colab session is not required for this result.

From the project checkout, in an environment with Torch, Core ML Tools 9.0,
and `numpy>=2.0,<2.4`:

```sh
python scripts/export_monodgp_m59f_position_interleave.py \
  --stage internals --source-dir /absolute/path/to/m59d \
  --output-dir /absolute/path/to/new_m59f_internals
python scripts/export_monodgp_m59f_position_interleave.py \
  --stage layers --source-dir /absolute/path/to/m59d \
  --output-dir /absolute/path/to/new_m59f_layers
python scripts/export_monodgp_m59f_position_interleave.py \
  --stage full --source-dir /absolute/path/to/m58 \
  --output-dir /absolute/path/to/new_m59f_full
```

Run the validator for each artifact directory on macOS. Add the optional CPU
control only for the full stage:

```sh
python scripts/validate_monodgp_m59f_macos.py \
  --artifact-dir /absolute/path/to/new_m59f_full --compute-units ALL \
  --source-dir /absolute/path/to/m58 --output /absolute/path/to/full_report.json
```

An exit code of 1 after a complete report means parity failed, not that
inference crashed. Inspect `all_parity_gates_passed` and the field breakdown.

## Next step

M59g: a bounded **numerical precision audit**, not retraining or a new
architecture. Capture several fixed validation inputs with exact provenance,
compare PyTorch/CPU and repaired Core ML per field, and check repeatability
and unchanged query selection. Assess the depth/dimension residual and a
bounded CPU-only control before proposing a numerical policy for review.
Keep the present gate failed until it passes or an explicitly reviewed,
versioned criterion replaces it. No physical-device, quantization, or
deployment qualification is authorized by M59f.
