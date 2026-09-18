# MonoDGP M59d 2D-Transformer Diagnostic Contract

M59c showed that the backbone and input projections are numerically consistent
with Core ML after accounting for the large scale of `backbone_feat_2`. M59d
isolates the remaining 2D-transformer mismatch by exporting the region/depth
context plus every 2D encoder and decoder layer.

M59d reuses the exact checkpoint and source commit recorded in the passed M59b
export gate. It performs no training, changes no weights, and does not
authorize FP16, quantization, iPhone execution, or product deployment.

## Colab export

```bash
python3 -u scripts/export_monodgp_m59d_2d_transformer.py \
  --monodgp-repo /content/MonoDGP_M59b \
  --m59b-export-gate /content/drive/MyDrive/mobile_adas3d_outputs/compression/monodgp_m59b_coreml_diagnostic/m59b_coreml_diagnostic_export_gate.json \
  --dataset-root /content/monodgp_kitti_m56d \
  --split-dir /content/drive/MyDrive/mobile_adas3d_splits/kitti_chen \
  --output-dir /content/drive/MyDrive/mobile_adas3d_outputs/compression/monodgp_m59d_2d_transformer
```

The exporter must report `all_export_gates_passed: true`. Copy the generated
`.mlpackage`, `m59d_2d_transformer_reference_io.npz`, and
`m59d_2d_transformer_export_gate.json` to macOS.

## macOS validation

```bash
python3 scripts/validate_monodgp_m59d_macos.py \
  --mlpackage MonoDGP_M59d_2d_transformer_fp32.mlpackage \
  --reference-io m59d_2d_transformer_reference_io.npz \
  --export-gate m59d_2d_transformer_export_gate.json \
  --compute-units ALL \
  --output m59d_macos_2d_transformer.json
```

The first failed `det2d_encoder_layer_*` or `det2d_decoder_layer_*` identifies
the first transformer stage that diverges. Do not proceed to precision
compression or physical-device testing until that layer is understood.
