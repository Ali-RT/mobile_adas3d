# MobileADAS3D Intermediate Presentation — Speaker Scripts

Use this as a presenter script alongside the interactive HTML deck.

This deck preserves the historical v6/MobileNetV3 milestone. As of 2026-07-19,
the iPhone deployment work described as “next” in the original deck is complete;
the active next step is the fresh MobileNetV4 Conv Small Colab baseline.

## Slide 1: Title

Open with the motivation: this project is an end-to-end exploration of building a compact monocular 3D detector for ADAS, then pushing it through training, evaluation, optimization, export, and edge-readiness. Emphasize that this is a learning and prototyping project built on public KITTI-style data, not a Bosch production system.

## Slide 2: Problem

State the core problem: 2D boxes are useful, but ADAS needs distance, size, and orientation. The project asks whether a lightweight single-camera model can produce usable 3D-like outputs while staying deployable.

## Slide 3: Task

Explain the task definition. The model input is one RGB image and the outputs cover class, 2D box, depth, dimensions, yaw, center offset, and uncertainty. The classes are intentionally constrained to ADAS-relevant objects: cars, pedestrians, cyclists.

## Slide 4: Baseline

Show the baseline architecture. It started with MobileNetV3-Small and dense heads at stride 32. This was a good MVP, but the feature grid was too coarse for smaller road users.

## Slide 5: Diagnosis

Use the diagnostic findings to explain why the model struggled. At stride 32, many pedestrians and cyclists occupy roughly one cell or less in width, and collisions are more common. This turned architecture tuning from guesswork into an evidence-driven decision.

## Slide 6: v6 Architecture

Introduce the v6 changes. Stride-16 FPN improved spatial resolution; l/t/r/b made box regression local; center sampling gave more positives; class-balanced loss countered KITTI imbalance.

## Slide 7: Accuracy

Summarize the accuracy gain. The key story is that the model went from a weak localization baseline to a much stronger detector, especially for pedestrians and cyclists.

## Slide 8: 3D Metrics

Explain 3D metrics only on matched detections: once a prediction matches the right object by 2D IoU, we evaluate depth, yaw, and dimensions. This separates detection quality from 3D regression quality.

## Slide 9: Yaw Diagnostics

Yaw diagnostics showed that 180-degree front/back flips were not the main issue. Car and cyclist yaw were acceptable; pedestrian yaw remained the weakest, which is expected because pedestrian orientation is visually ambiguous.

## Slide 10: Complexity

Transition from model accuracy to deployability. Static complexity shows the model is small in weights but moderate in compute because the input resolution is large and the stride-16 heads cover 1920 cells.

## Slide 11: Benchmarking

Explain why end-to-end benchmarking matters. The first CUDA benchmark showed Python decode/NMS was slower than the neural network itself. That changed the optimization priority.

## Slide 12: Runtime Evolution

Show the topK tradeoff. topK=50 gives near-identical accuracy with lower latency, so it became the deployment setting. Formal evaluation can still use topK=100.

## Slide 13: topK Sweep

The vectorized decoder was a major engineering win. It reduced repeated CUDA-to-CPU synchronization and moved the decoder from being the bottleneck to being a small part of the pipeline.

## Slide 14: Decoder Optimization

TorchScript provided a second major speedup and a stable export artifact. Parity passed at 1e-4, so we can trust the exported graph.

## Slide 15: TorchScript

Core ML export is the bridge to iPhone. FP16 reduced the package to about 12 MB and produced an iOS 15+ mlpackage with all seven output tensors preserved.

## Slide 16: Core ML

Summarize the current milestone: accuracy, model size, latency, FPS, memory, and Core ML package. This slide is the headline result of the project.

## Slide 17: Current Milestone

Share lessons learned. Architecture changes helped accuracy, but benchmarking showed the most important deployment optimization was actually postprocessing. Export parity is essential before trusting runtime numbers.

## Slide 18: Next Steps

Close with the updated next step: the real-iPhone benchmark, Swift decode/NMS,
camera integration, recording artifacts, and ZIP/share export are complete for
v7. Now run the untouched MobileNetV4 Conv Small baseline on the canonical
KITTI Chen split, report complete-split AP_R40, and only then consider controlled
accuracy/latency ablations and Core ML export.
