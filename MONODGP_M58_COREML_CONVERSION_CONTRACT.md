# MonoDGP M58 Fixed-Shape Core ML Conversion Contract

Status: prepared; experimental conversion Stop point 1 pending.

## Purpose

M58 asks one question: can the selected M56d checkpoint, running through the
M57 portable deformable-attention implementation, be represented as a
fixed-shape FP32 Core ML Program without changing its PyTorch outputs?

M58 performs no training and changes no weights. It is a conversion and parity
experiment, not a deployment. The initial FP32 artifact deliberately
prioritizes numerical diagnosis over model size or speed.

## Frozen source evidence

- MonoDGP commit:
  `aa059a18214aebf644510e7f0793971b403f9d14`.
- Selected checkpoint SHA-256:
  `7d18883d6f998e7616beaa45b92d348b22728f3fad87a891618b5ce5a1cde17c`.
- M57 manifest SHA-256:
  `7c4757798dfff4d95053912730faff4c5d7a4626ff41a539ad87b2f9193eb313`.
- M57 smoke SHA-256:
  `2a96cffd8e1e6b77f2c547c2b94dca4bde2e72bf4185d853ee7bca71ed28b3b0`.
- M57 complete gate SHA-256:
  `63f2b71d1f5be69bad31907db229f79e39fb17c033ba43999c6fdfe1df3b97a7`.
- M57 comparison CSV SHA-256:
  `687b15cd4a044981f4e00fa038f2fc5e7053cee692e13bfdb85fe2f90656412d`.
- M57 prediction-tree SHA-256:
  `b550362a6f0ba77667b69cf2a15a457c419c1e0db4751e2704228a762841ca05`.

Preparation fails closed if these artifacts, the source tree, checkpoint,
runtime YAML, Chen validation split, or M57 decision fields change.

## Frozen interface

The traced graph has three FP32 tensor inputs:

| Input | Fixed shape |
| --- | --- |
| `image` | `1 x 3 x 384 x 1280` |
| `calibration` | `1 x 3 x 4` |
| `image_size` | `1 x 2` |

The graph preserves seven semantic MonoDGP output families: `pred_logits`,
`pred_boxes`, `pred_3d_dim`, `pred_depth`, `pred_angle`,
`pred_depth_map_logits`, and `pred_region_prob`. The first six are individual
tensors. `pred_region_prob` is the model's four-level feature pyramid and is
therefore exported losslessly as `pred_region_prob_0` through
`pred_region_prob_3`. The Core ML signature has ten tensors in total;
post-processing stays outside the graph.

The conversion target is an FP32 ML Program with minimum deployment target
iOS 17, using `coremltools==9.0`. The conversion uses a traced TorchScript
graph because Apple documents that route as the stable PyTorch conversion
path. Colab uses `skip_model_load=True`: it may create and inspect the model
package, but it cannot execute Core ML predictions.

## Stop point 1: experimental conversion

Run `notebooks/MonoDGP_M58_CoreML_Conversion_Colab.ipynb` on a Colab GPU.
The gate requires:

1. exact reviewed M57 evidence and exact M56d checkpoint;
2. all nine portable attention modules and zero native CUDA calls;
3. the exact fixed input and ten-tensor output signatures above;
4. finite reference outputs and traced-output parity within the unchanged
   M56-family raw-output limits;
5. a trace containing `aten::grid_sampler` with no custom attention node;
6. successful FP32 ML Program conversion;
7. at least one MIL `resample` operation and zero MIL `custom` operations;
8. saved TorchScript, reference I/O, `.mlpackage`, and zipped package with
   hashes; and
9. no training, weight-change, deployment, or safety claim.

Stop and return `m58_coreml_export_gate.json`. If conversion fails, return
the same JSON and the durable log; the report records the exception.

## Later barriers

A Stop point 1 pass authorizes only a macOS Core ML prediction-parity run using
the exact package and reference I/O hashes. That run must compare all ten raw
tensors across the seven semantic output families, plus decoded detections on
the same samples.

Only after macOS parity passes may physical-iPhone latency, peak memory,
sustained thermal behavior, stability, and artifact-integrity testing begin.
FP16 Core ML conversion or quantization is a later controlled experiment.
M58 does not qualify product safety; Pedestrian nearby recall remains below
the separate `0.80` target.
