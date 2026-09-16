# MonoDGP M59 macOS Core ML Prediction-Parity Contract

## Purpose

M59 executes the exact M58 FP32 `.mlpackage` on macOS. It compares all ten
exported tensors with the frozen `m58_reference_io.npz` sample and repeats
MonoDGP's deterministic top-50 candidate decode. It is a runtime-parity gate,
not a KITTI re-evaluation and not a physical-device test.

## Inputs

Use only the artifacts from the passed M58 export gate:

- `MonoDGP_M58_fixed_fp32.mlpackage`
- `m58_reference_io.npz`
- `m58_coreml_export_gate.json`

The validator checks package/reference paths, shapes, dtypes, and hashes in
its output report. Core ML must execute on macOS; Colab cannot pass this gate.

## Frozen checks

1. Inputs remain `image [1,3,384,1280]`, `calibration [1,3,4]`, and
   `image_size [1,2]`, all FP32.
2. All ten output names and shapes remain unchanged.
3. Every output is finite and within the unchanged M56-family maximum absolute
   limits: logits `0.10`, boxes `0.01`, dimensions `0.10`, depth `0.50`,
   angle `0.10`, depth-map logits `0.10`, and each region level `0.01`.
4. The deterministic top-50 candidate tensor produced from Core ML outputs
   matches the reference candidate tensor within `1e-4` maximum absolute
   error. This covers class, score, 2D box, depth, angle, dimensions, 3D
   center ratios, and uncertainty before calibration-dependent KITTI writing.

The validator writes `m58_macos_coreml_parity.json`. A complete report is
required before any physical-iPhone latency, memory, thermal, stability,
FP16, quantization, or product-safety experiment. Those remain unauthorized
even when M59 passes.
