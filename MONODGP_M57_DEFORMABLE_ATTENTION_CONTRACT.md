# MonoDGP M57 Portable Deformable-Attention Contract

Status: complete; portable operator candidate selected.

## Purpose

M57 addresses the export blocker identified by M55 after M56d selected the
offline compressed checkpoint. It adds one opt-in implementation of
`MSDeformAttn` using ordinary PyTorch tensor operations and bilinear
`grid_sample`. Native CUDA remains the default. M57 performs no training,
changes no parameter values, and does not change decoding, taxonomy, input
geometry, or the validation protocol.

M57 is an operator-replacement parity experiment, not a product deployment.
Passing it cannot qualify Core ML, iPhone performance, or safety.

## Immutable evidence

- MonoDGP commit: `aa059a18214aebf644510e7f0793971b403f9d14`.
- Selected M56d checkpoint SHA-256:
  `7d18883d6f998e7616beaa45b92d348b22728f3fad87a891618b5ce5a1cde17c`.
- M56d manifest SHA-256:
  `ce2cf03c37530f924f75f0f44ea94fe233d8aac916c5c51fb26cd8bfa84d0bc0`.
- M56d smoke SHA-256:
  `56678d529fda787441f842453e5f198c3d5017b4a0d8cb82373110713f737f87`.
- M56d gate SHA-256:
  `4012d800c3972e15922ca0d3dd530cad8437f575f433d93b3708182a062227ad`.
- M56d comparison CSV SHA-256:
  `d64529315414d841f2658f69442722c2548e783a375f5bfcb9a46bdb2937c95f`.
- M56d runtime-config SHA-256:
  `4ad6d50241e5a6dd552e5d8b9c043241a11377c26c59b84ab3d2f7f58c7a42af`.
- Validation sample: first Chen validation image, `000001`.
- Static feature geometry: `48x160`, `24x80`, `12x40`, and `6x20`.

Preparation fails closed if any artifact, hash, source file, module inventory,
checkpoint, runtime config, split, or default execution mode changes.
The durable checked-in runtime YAML is allowed to replace the ephemeral
absolute path recorded by M56d only when its SHA-256 is exactly identical.

## Replacement boundary

The exact nine `MSDeformAttn` modules are:

- three 2D-transformer encoder self-attention modules;
- three 2D-transformer decoder cross-attention modules;
- three 3D-transformer decoder cross-attention modules.

The portable branch preserves the learned value projection, sampling-offset
projection, attention-weight projection, and output projection. It changes only
the sampling/weighted-sum kernel:

```text
native:   rank-6 sampling tensor -> custom CUDA autograd extension
portable: rank-5 [N,Q,H,L*P,2] -> four bilinear grid_sample calls -> weighted sum
```

The environment variable `MONODGP_PORTABLE_DEFORM_ATTN=1` opts into the
portable path. With the variable absent, all modules must use native CUDA.

## Stop point 1: CUDA parity and trace audit

The smoke uses the same checkpoint, input, device, and software for both paths.
It must prove:

1. all nine expected modules and nine native CUDA calls are observed;
2. every portable module output, evaluated on its exact captured native input,
   has maximum absolute delta no greater than `1e-3`;
3. all three observed signatures (`Q=10200/ref=2`, `Q=50/ref=2`, and
   `Q=50/ref=6`) trace through `aten::grid_sampler` with no custom extension or
   `MSDeformAttnFunction` node;
4. the full portable model makes zero native-extension calls;
5. full-model output paths and shapes are unchanged and finite;
6. full-model raw outputs pass the unchanged M56-family limits;
7. all runtime parameters remain FP32;
8. both paths complete five warmups and 100 timed CUDA predictions;
9. training, weight changes, direct Core ML, and product-safety claims remain
   false.

The unchanged full-output limits are:

| Output | Maximum absolute delta |
| --- | ---: |
| `pred_logits` | 0.10 |
| `pred_boxes` | 0.01 |
| `pred_3d_dim` | 0.10 |
| `pred_depth` | 0.50 m |
| `pred_angle` | 0.10 |
| `pred_depth_map_logits` | 0.10 |
| `pred_region_prob` | 0.01 |

Native-versus-portable timing is comparable because it is collected in one
process with identical model, sample, device, software, warmups, and event
timing. It is diagnostic at this rung and does not determine parity acceptance.

Stop and return:

- `m57_deformable_attention_manifest.json`
- `m57_deformable_attention_smoke.json`

## Reviewed stop-point evidence

The exact returned manifest and smoke report passed review:

- manifest SHA-256:
  `7c4757798dfff4d95053912730faff4c5d7a4626ff41a539ad87b2f9193eb313`;
- smoke SHA-256:
  `2a96cffd8e1e6b77f2c547c2b94dca4bde2e72bf4185d853ee7bca71ed28b3b0`.

All nine module comparisons, all three trace signatures, every raw-output
family, output finiteness/structure, FP32 runtime parameters, and the zero
portable native-call requirement passed. The largest local module delta was
`3.11e-6` against `1e-3`; final depth delta was `9.16e-5 m` against
`0.50 m`.

On the same A100 process, the native/portable mean latency was
`37.92/47.00 ms` and p95 was `38.53/48.30 ms`. The portable path was
about `1.24x` slower and peak allocated CUDA memory increased from
`653,408,256` to `991,836,160` bytes. These are diagnostic CUDA results,
not Core ML or iPhone projections.

## Continuation barrier

The exact manifest/smoke pair above passed review and authorized a complete 3,769-image
portable-path validation against all M54/M56d preservation gates
using `MonoDGP_M57_Complete_Validation_Colab.ipynb`. The smoke field
`coreml_microkernel_conversion_authorized=true` records operator
traceability only; it does not authorize full-model conversion. Only after the
complete validation passes may a separate fixed-shape Core ML conversion gate
begin.

## Stop point 2: complete validation result

The reviewed gate JSON and comparison CSV SHA-256 values are
`63f2b71d1f5be69bad31907db229f79e39fb17c033ba43999c6fdfe1df3b97a7`
and `687b15cd4a044981f4e00fa038f2fc5e7053cee692e13bfdb85fe2f90656412d`.
The portable prediction-tree SHA-256 is
`b550362a6f0ba77667b69cf2a15a457c419c1e0db4751e2704228a762841ca05`.
All 3,769 prediction files were present and all nine preservation gates
passed. Every portable result equaled M56d at the reported precision: moderate
3D Vehicle/Pedestrian/mean AP_R40 was `19.4666/6.1848/12.8257`, moderate BEV
was `25.7349/6.7651`, nearby Vehicle/Pedestrian recall was
`0.90993/0.72399`, and Pedestrian localization-failure rate was `0.23854`.
M57 therefore selects the opt-in rank-five `grid_sample` operator candidate
and authorizes a separate fixed-shape Core ML conversion/parity experiment.
It performed no training and changed no weights.

This M57 pass does not authorize direct conversion or deployment.
Physical-device latency, memory, thermal stability, artifact integrity,
decoded parity, and external validation remain later gates. Pedestrian nearby
recall also remains below the separate `0.80` product target.
