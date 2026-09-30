# M63d — one-epoch frozen-BatchNorm control

Prepared: 2026-09-29. Status: local tests passed; CUDA run pending.

## Question and evidence

Does freezing A2's BatchNorm running statistics during a GT-only continuation
prevent the early accuracy regression seen in M63?

M63c restored only means/variances after training and recovered 72.16% of the
Vehicle 3D AP loss, 92.53% of Vehicle BEV loss, and 10.06% of Pedestrian 3D
loss. Nearby recall worsened. That establishes an inference-time contribution,
not that frozen-statistics training will succeed. Original A2 stays selected.

Reviewed M63c comparison SHA256:
`1342fa4d8839476a65457239df679c7a225d0686d9f68af94fbd325c56939526`.

## Frozen design

- Start from original A2 MobileNetV4 Conv Medium MonoDETR epoch130.
- Exactly one epoch, 3712 training images, batch4, 928 optimizer steps.
- Seed20268, LR1e-5, fresh native AdamW, FP32, no augmentation.
- Original GT losses for both Vehicle and Pedestrian; no KD.
- Only change: after model.train(), put all76 BatchNorm modules in eval mode.
  Their means, variances and counters must remain bitwise unchanged.
- Preserve each parameter's existing requires_grad, including BN affine
  parameters. Do not put the whole model in eval mode during training.
- Full Chen val3769 evaluation at fixed epoch1. Compare with original A2,
  untouched M63 control epoch1, and M63c post-hoc statistics-restored epoch1.
- Report both classes' moderate 3D/BEV AP and nearby recall. No automatic
  promotion, longer training or distillation; stop for review.

This single-seed diagnostic does not establish generalization or safety.
The old failed M63 acceptance decision remains unchanged.

## Run instructions

Use the same NVIDIA L4 and pinned environment. Reopen the updated
`notebooks/MonoDETR_M63_R0_A2_Depth_Pilot_Colab.ipynb`; confirm
`M63-NOTEBOOK-2026-09-29-r5`.

Run setup sections1–3, then section13's three code cells:
13A tests and zero-update CUDA smoke; 13B one GT-only epoch; 13C complete
validation and result ZIP. Skip sections4–12. Return `m63d_results.zip`.

All new outputs stay under the existing M63 output's `m63d_frozen_bn`
directory. Previous caches, manifests, reports and checkpoints remain intact.
The reviewed M63c report must already exist in its original Drive location.

## Recovery and verification

A completed epoch1 checkpoint is checksum/lineage-checked and never trained
again. An interrupted partial epoch restarts from original A2; there is no
mid-epoch optimizer resume. Evaluation reuses only complete, hash-verified
predictions. Identity/environment differences stop rather than mixing runs.

The smoke tests finite loss/gradients without optimizer steps or model-state
changes. Training checks frozen BN state repeatedly; inference checks it
against original A2 and verifies no model-state mutation.

Local validation: 64 M61–M63 regression tests passed plus CLI import/help.
CUDA training and full validation have not been run locally.
