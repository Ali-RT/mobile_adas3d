# M63b continuation-regression diagnostic

Prepared 2026-09-29. No training or model promotion authorized.
Use the existing M63 notebook revision M63-NOTEBOOK-2026-09-29-r3:
sections1–3 for setup, then section11 directly. L4/CUDA required for inference.
Do not rerun sections4–10 or recreate targets/checkpoints.

## Reviewed M63 outcome

Both ten-epoch arms completed with correct provenance. M63 failed.
Moderate 3D AP: original A2 Vehicle15.450529 / Pedestrian7.528523;
control14.783510 /6.640430; KD14.893036 /6.310502.
KD Vehicle gain versus control0.109526 does not recover the0.557494 loss
versus original A2. Pedestrian KD loss versus control0.329928.
Original A2 epoch130 remains the working model. All historical product targets
remain unchanged. The supplied archive contains reports, not raw prediction
files or checkpoint binaries; reported bindings and gate calculations were verified.

## Question and bounded work

The GT-only continuation also regressed, so inspect the continuation recipe
before attributing everything to KD or buying another training run.
Compare original preparation defaults/available experiment manifest and saved
optimizer learning rates with M63. Explicitly label reconstructed defaults
rather than claiming they are verified saved runtime settings.

Known recipe changes include disabled augmentation, batch16 to4 (original
preparer default; original run manifest checked when available), fresh optimizer,
and a constant1e-5 learning rate. Inspect saved normalization running means,
variances and counters against source A2, without altering them.

Evaluate saved epochs1,3,5 for both arms on the complete Chen validation split:
six inference passes. Include already reviewed baseline/epoch10 rows.
All new reports/predictions are isolated under diagnostics_m63b.
Cache identities/checkpoint checksums and provenance are enforced.
No weights, BN buffers, source manifest, historical gate or predictions are edited.

## Interpretation boundary

This is post-hoc exploration, not an amended M63 selection rule.
No best checkpoint is selected or promoted. Earlier recovery cannot make the
fixed epoch10 pilot pass. Trends can identify when degradation appears;
buffer drift/recipe differences alone cannot prove causality.
Review results before proposing a separately controlled follow-up.

Return m63b_results.zip with identity, recipe/normalization audit and diagnostic
comparison. No phone needed. No new training should start automatically.
