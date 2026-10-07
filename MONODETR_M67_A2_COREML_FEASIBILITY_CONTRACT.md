# M67 — exact A2 Core ML feasibility

Status: **M67 executed; frozen CPU/CUDA parity gate failed; Core ML conversion not reached**
Decision date: 2026-10-07
Notebook: `notebooks/MonoDETR_A2_M67_CoreML_Feasibility_Colab.ipynb`

## Purpose

The user set the priority as deployment feasibility first: establish whether
the **trained A2 student itself** can run fast enough on iPhone before spending
further effort on distillation or model redesign. M67 is the first, bounded prerequisite: export the exact selected A2
checkpoint to Core ML and verify that export preserves its predictions. It is
not an iPhone speed result, accuracy evaluation, or deployment approval.

Prior phone timing is not a proxy for A2. MobileMonoDETR-VP1 used random weights
and a Small backbone; M60 measured MonoDGP, not A2. Neither result may be
reported as A2 latency.

## Frozen model and source

- Model: original A2 epoch 130, MobileNetV4 Conv Medium plus the full MonoDETR
  graph; no M65/M66 continued-training endpoint.
- Checkpoint SHA-256:
  `ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4`.
- Source: `https://github.com/ZrrSkywalker/MonoDETR.git`, commit
  `6994b9f512400b258c6edb75f77423beb9c126f2`.
- Original M66 manifest SHA-256 identity:
  `68c2247a9ab28603e10f7f9748fc17a43e5bdefeedc57eebe87e2c62f430d968`.
- Native inference source SHA-256:
  `f1ce39fa8877e76c3fc1c456de98ec57a8a9db288fdd5a0d68d9ba34f92dfb82`.
- Exact Chen validation split; first 16 samples in its fixed order. This is a
  numerical/export smoke set, not the complete 3,769-image product evaluation.

The export patch is a prospective fixed-shape implementation of deformable
attention, with source and runtime provenance recorded. It does not change
learned weights or architecture. The cloud-GPU latency in the report is only a
CUDA diagnostic and must not be compared to iPhone targets.

## Reviewed result (2026-10-07)

The supplied `m67_a2_export_results (1).zip` and
`m67_cpu_trace_diagnostic_results.zip` bind to the frozen epoch-130 checkpoint
above. The portable CUDA path matched native CUDA on all 16 fixed samples;
traced CPU matched eager portable CPU bit-for-bit on all 16. CPU-versus-CUDA
raw-output parity failed on all 16. The first divergent **top-level** component
was `backbone`; this report does not isolate a specific backbone operation.
Maximum CPU-versus-CUDA deltas were logits `0.0866718`, boxes `0.00369291`,
dimensions `0.0646670`, depth `0.474918 m`, and angle `0.359506` (frozen limits
remain unchanged).

M67 stopped before Core ML conversion. There is no M67 `.mlpackage`, no Mac
parity result, and no iPhone performance measurement. The diagnostic performed
zero optimizer steps, did not quantize, and verified the model state unchanged.
This is a backend-parity failure, not evidence that the trained A2 weights are
bad or that the model is too slow. Do not rerun the same fixed-input audit or
claim Mac/phone authorization. A new diagnostic route must keep the failed M67
result visible and define its own evidence and acceptance criteria; it cannot
retroactively mark M67 passed or silently replace its reference backend.

## Stages and decision boundaries

1. **Colab export gate (M67):** isolated CUDA 13 runtime; local A2 attention
   build; native versus portable-export CUDA outputs; CPU and TorchScript raw
   output parity; FP32 Core ML conversion; no custom MIL operators. The audit
   uses fixed prospective raw-output limits; the depth bound is 0.01 m (1 cm).
   Those limits are numerical smoke limits, not KITTI AP or deployment
   acceptance criteria.
2. **Mac parity:** only after M67 passes, execute the same FP32 package on macOS
   against the saved 16 PyTorch references. Record parity and Mac latency
   separately. A Mac pass is still not an iPhone result.
3. **Physical iPhone feasibility:** only after Mac parity passes, measure the
   exact A2 package using the existing benchmark harness or the product-app
   source if available. Do not create another app. The only iOS project currently
   tracked in this repository is `ios/M60Benchmark`, a standalone MonoDGP test
   harness rather than the MobileADAS3D product app; its label/resources must
   not be mistaken for A2 or the product app. Decide whether to extend that
   harness or use the product-app source after M67/Mac review. Report device/iOS/build/package identity, warmups,
   per-prediction latency distribution, sustained throughput, thermal state,
   memory and prediction parity. Keep model-only timing distinct from
   preprocessing and capture-to-decoded-result timing.

The frozen product targets remain: model inference p95 <= 50 ms,
capture-to-decoded-result p95 <= 100 ms, sustained processed rate >= 10 FPS,
and preprocessing p95 <= 20 ms. An M67 or Mac pass does not establish any of
these physical-device targets. Passing speed also does not qualify accuracy:
A2 retains its existing product-accuracy gaps and Chen-val remains development
data.

If exact A2 fails export or iPhone speed, preserve the evidence. Do not loosen
parity limits or quantize in this milestone. First determine whether the
transformer, backbone, or other operations dominate. A smaller detector such as
a YOLO-derived graph with explicitly trained metric-depth/3D heads is a
separate architecture experiment, not a drop-in replacement, and requires its
own accuracy and runtime gates. A MobileNet backbone swap alone is not assumed
to resolve transformer cost.

## Execution and outputs

The frozen M67 notebook has already run; its reviewed evidence is in the two
ZIPs named above. Preserve those outputs. Do not rerun sections 1–5 under the
same identity or move on to Mac/phone. The next experiment must be separately
specified and use a new run ID. It should either localize the backbone
CPU/CUDA drift, or explicitly label any exploratory Core ML conversion and
full-val accuracy check as diagnostic-only while leaving the M67 gate failed.

The notebook performs zero optimizer updates and no training, full-set AP,
quantization, checkpoint selection, phone test or deployment. **Do not connect
the phone for M67.** After the returned ZIP passes review and Mac parity, the
next step is physical-device measurement using an existing harness/product-app
source, without creating a new app. M66b gradient diagnosis and further KD are
deferred until this feasibility question is answered.
