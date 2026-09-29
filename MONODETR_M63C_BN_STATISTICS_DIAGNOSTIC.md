# M63c — inference-only BatchNorm statistics intervention

Prepared 2026-09-29. Revision M63c-2026-09-29-r1.
No optimizer steps, checkpoint saves, teacher retraining or checkpoint promotion.

## Evidence and fixed question

M63b showed that GT-only continuation already reduced Vehicle/Pedestrian
moderate 3D AP from15.450529/7.528523 to14.849458/6.832163 after one epoch.
Neither class recovered to original A2 at any inspected epoch1,3,5,10.
The source optimizer learning rate was already1e-5, matching continuation;
the issue is not an increase from the source checkpoint's saved learning rate.
Batch size changed16→4, optimizer moments were reset, and augmentation was
disabled. Original augmentation is reconstructed from pinned code, not a
verified runtime YAML. Normalization counters updated928 times per epoch
in76 modules. These observations alone do not establish causality.

Question: how much of the GT-only epoch1 regression is attributable to its
inference-time BatchNorm running statistics, conditional on its learned weights?

## Exact intervention

- Candidate: M63 GT-only control epoch1, matched to the reviewed M63b SHA.
- Donor: original frozen A2 epoch130, exact existing checkpoint hash.
- Copy only running_mean and running_var of76 tracked BatchNorm modules.
- Preserve all learned parameters (including BN affine weights/biases), all
  other buffers, and num_batches_tracked. Copy in memory only.
- Verify state fingerprints before/after; changed entries must be a subset
  of the152 allowed statistic buffers and donor values must match exactly.
- Set evaluation mode; no_grad inference on all3,769 Chen validation images.
- Confirm full model state unchanged during inference and original checkpoint
  files unchanged on disk.
- Reuse the exact reviewed baseline and untouched control metrics; perform
  one new full validation pass using identical preprocessing/decoding/metrics.
- Isolate outputs under diagnostics_m63c. Do not edit M63 or M63b reports.

Report per-metric deltas against untouched control and original A2.
For metrics where the control regressed, report the fraction recovered
without clamping (negative recovery and overshoot remain visible).
This is a diagnostic comparison, not a significance or deployment test.

## Interpretation and next boundary

Recovery demonstrates an inference-time contribution of the replaced statistics
for this checkpoint. It does not prove frozen-BN training will work; learned
weights and statistics may have co-adapted. No recovery does not rule out BN's
effect during training. Optimizer reset, augmentation and batch-size effects
remain separate hypotheses. Do not automatically launch a follow-up run.
M63 remains failed; original A2 remains the working model.

## Run and return

Open the existing M63 notebook revision M63-NOTEBOOK-2026-09-29-r4.
On the same L4/software environment, run sections1–3 then12 only; skip4–11.
Return m63c_results.zip containing identity, metric comparison, intervention
audit and complete prediction-file manifest. No phone needed.
