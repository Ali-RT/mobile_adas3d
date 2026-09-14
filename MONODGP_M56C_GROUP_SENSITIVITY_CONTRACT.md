# MonoDGP M56c Grouped FP16-Storage Sensitivity Contract

Status: frozen before M56c execution.

## Purpose

M56c is a no-training diagnostic following the rejected M56 and M56b
FP16-storage candidates. M56 compressed all eligible Conv2d/Linear parameters;
M56b retained the directly geometric `bbox_embed`, `dim_embed_3d`, and
`depth_embed` heads in FP32. Both failed only the frozen final-depth raw-output
limit, at `0.710739` and `0.709734` metres respectively versus `0.50`.

M56c asks which upstream architectural stage introduces that depth drift. It
does not create or select a deployable checkpoint, perform complete validation,
change a threshold, modify the graph, or claim Core ML readiness.

## Immutable evidence

- Parent: selected M54 MonoDGP epoch 100.
- Parent checkpoint SHA-256:
  `8e79f3921d96e1de70cbb4219245e3fcc3fa1fb67ae675468b4ebca90e579847`.
- Upstream MonoDGP commit:
  `aa059a18214aebf644510e7f0793971b403f9d14`.
- M56b manifest SHA-256:
  `6251abd0ca2eb39b4d1376b9efb836221565a38b4359801efb5e2318f1a2f4ef`.
- M56b smoke SHA-256:
  `c9510f67588a4998c1f84e2c309c98c3def3931627874200662fac92690fd8d8`.
- M56b candidate SHA-256:
  `ea2d32db0836dc35d226ec17072f445368d79d4d36dbb864b4759e89c8a190f4`.
- Fixed sample: first Chen validation image, `000001`.
- Runtime: CUDA only; stored FP16 values are loaded into the unchanged FP32
  model.

Preparation fails closed if the M55 or M56b evidence, source checkpoint,
runtime configuration, architecture, or parameter inventory changes.

## Fixed group taxonomy

Every unique parameter directly owned by an ordinary Conv2d or Linear module
belongs to exactly one alias-consistent group:

1. `backbone`
2. `input_projection`
3. `region_head`
4. `depth_predictor`
5. `det2d_transformer`
6. `det3d_transformer`
7. `prediction_heads`
8. `other_eligible`, a fail-visible catchall that is tested only if nonempty

Prediction-head ownership has priority for shared parameters. It covers
`class_embed`, `bbox_embed`, `dim_embed_3d`, `depth_embed`, and `angle_embed`
under any state-dict alias. Every alias of one shared parameter must remain in
the same group.

The grouped unique-byte total must reproduce the M55 eligible total of exactly
`156,331,452` bytes and its Conv2d/Linear family split.

## Fixed diagnostic matrix

M56c runs two controls plus two tests for every nonempty group:

- an alias-consistent all-eligible FP16 reference;
- the exact M56b policy reference;
- `only_<group>`: round just that group to FP16;
- `all_except_<group>`: round every eligible group except that group.

Each candidate is constructed transiently in CPU memory from the exact FP32
parent, loaded into the unchanged FP32 CUDA model, evaluated on image `000001`,
and discarded. Candidate checkpoints are not retained on Google Drive. This
avoids writing roughly one full model per diagnostic row.

The singleton tests reveal a stage that fails by itself. The complement tests
reveal whether preserving one complete stage restores the combined policy. If
all singletons pass and no complement passes, the drift is cumulative and the
next experiment must use a separately frozen progressive-addition search.

## Unchanged raw-output limits

Maximum absolute differences from the untouched FP32 parent remain:

| Output | Limit |
| --- | ---: |
| `pred_logits` | 0.10 |
| `pred_boxes` | 0.01 |
| `pred_3d_dim` | 0.10 |
| `pred_depth` | 0.50 m |
| `pred_angle` | 0.10 |
| `pred_depth_map_logits` | 0.10 |
| `pred_region_prob` | 0.01 |

Every row also records separate decoded-depth and log-variance differences.
The exact M56b reference must again fail only `pred_depth` and reproduce its
maximum depth difference within `0.01` metres. This tolerance covers ordinary
same-environment kernel variability; it does not relax the candidate limit.

## Integrity gates

The diagnostic completes only if:

- all provenance hashes and epoch bindings pass;
- the generated group aliases and candidate specifications exactly reproduce
  the manifest;
- all expected rows execute;
- every row has exact source/candidate state keys;
- every intended FP16 tensor is actually FP16 before loading;
- every held-FP32 and noneligible tensor is bitwise unchanged;
- all runtime parameters are FP32;
- output paths and shapes are unchanged and all outputs are finite;
- the M56b failure is reproduced;
- no training, graph change, full evaluation, Core ML claim, or product-safety
  claim occurs.

Candidate parity failures are measurements, not diagnostic-integrity failures.
The runner exits successfully once all rows and integrity checks complete.

## Decision barrier

M56c never authorizes the 3,769-image validation and never selects an offline
compression artifact. Its only authorized output is evidence for freezing the
next combined-candidate experiment. Direct Core ML conversion and product
safety remain false.

## Required artifacts

- `m56c_group_sensitivity_manifest.json`
- `m56c_group_sensitivity.json`
- `m56c_group_sensitivity.csv`
- `colab_logs/m56c_prepare.log`
- `colab_logs/m56c_group_sensitivity.log`
