# Parking detector and depth candidate screen

The next comparison tests a compact YOLO detector with a separate depth model,
plus a small cuboid detector as an optional alternative. The goal is Vehicle
and Pedestrian labels and useful distances in the critical 2 to 6 metre parking
range. A2 remains the existing reference. Neither candidate is promoted by
this first export and runtime screen.

## Run the detector and depth notebook first

Open
[`Parking_YOLO26n_Metric_Depth_Screen_Colab.ipynb`](notebooks/Parking_YOLO26n_Metric_Depth_Screen_Colab.ipynb)
on a fresh Colab CPU runtime. It runs YOLO26n at 640 pixels and YOLO26n-depth
at 768 pixels. It uses the released depth scale unchanged: no training,
per-image scale alignment, calibration fitting, or teacher is involved.

The first stage deliberately selects 64 validation images with nearby examples
from both product classes and each one-metre band. Selection uses ground-truth
labels, not model predictions. Labels from all 3,769 Chen validation frames
are indexed with four workers, but only the 64 selected images and labels
are copied. A reusable archive avoids repeating small-file Drive reads on
later sessions. This enriched subset is not representative full-validation AP.

The screen produces fixed FP32 ONNX graphs, graph input and output metadata,
tensor fixtures, per-frame PyTorch and ONNX predictions, 16 overlays, export
parity checks and CPU timings. Whole-frame pipeline timing includes detector,
depth and box-ROI aggregation but excludes image read, overlay rendering and
camera capture. Raw neural timings use two ONNX Runtime CPU threads, five
warm-ups and 20 measured repetitions. These are Colab or Mac timings, not
iPhone measurements.

The result is `yolo_depth_parking_edge_results.zip`. Return it for review
before installing a new model in the phone app. A failed export still produces
a diagnostic bundle; do not loosen the frozen parity tolerances to obtain a
pass. Separate output folders preserve failed and completed screens.

## Interpret the depth output correctly

The depth model returns a numeric depth map, not a colored visualization.
For each detection, this first screen reports median depth in the central
half of the box. This simple region can contain background or another object;
it is not an instance mask, nearest surface or guaranteed ultrasonic reflection
point. The output is nominal camera-axis surface depth using the checkpoint's
released scale, not sensor-frame range. Its absolute accuracy on the mounted
camera must be measured independently.

KITTI annotated 3D box centers identify which objects are nearby for the
diagnostic detection-recall table. They are not used as reference depths for
surface-depth error. The screen therefore reports no centimetre accuracy
score, parking pass or ultrasonic qualification. The existing
[`parking validation contract`](PARKING_RANGE_VALIDATION_CONTRACT.md) and
historical A2 gates remain unchanged.

COCO person maps to Pedestrian; car, bus and truck map to Vehicle. This is not
an exact replacement for KITTI's merged taxonomy: Tram and Person_sitting
have no dedicated COCO output class. Category and truncation limitations stay
visible in the per-frame records.

## Run the optional cuboid screen separately

Open
[`Parking_UrbanOmniDetect2_Nano_Screen_Colab.ipynb`](notebooks/Parking_UrbanOmniDetect2_Nano_Screen_Colab.ipynb)
in a separate fresh CPU session. It reuses the same input archive, but pins
Ultralytics 8.4.61 and the upstream source commit for the custom checkpoint
loader. The primary detector and depth screen pins Ultralytics 8.4.183.

UrbanOmniDetect-2 Nano predicts detections and eight projected cuboid corners.
The screen checks those image coordinates after ONNX export and draws them
without a ground-plane solve or offline refinement. It does not invent metric
distance from the corners. Its result is
`urban_nano_parking_edge_results.zip`. Published GPU speed and custom 3D
evaluation results are not substitutes for this phone and distance validation.

Both workflows verify official checkpoint SHA-256 values before loading Python
checkpoint objects. Only those trusted upstream releases should be used.
The checkpoints and source list AGPL licensing; review product licensing before
integration. GUPNet is reserved as a later calibrated metric-3D reference,
not a third integration in this first screen. Depth Anything V2 Metric Outdoor
Small is a possible later depth comparison, not loaded by these notebooks.

## Next decision

First inspect detections, export preservation and combined runtime cost.
Advance a technically sound candidate to a separate iPhone test, then compare
distance to independently referenced surfaces with the camera mounted on the
vehicle. Record intrinsics, camera-to-sensor transform, timestamps and vehicle
motion. The ultrasonic sensor under test must not provide its own ground truth.
Model training or distillation comes after establishing a useful baseline.

## Sources and verification

The fixed configuration is
[`parking_edge_candidates_v1.json`](configs/parking_edge_candidates_v1.json),
revision `PARKING-EDGE-SCREEN-2026-10-21-r1`. The two pinned checkpoints in the
primary screen are official Ultralytics release assets; the Urban checkpoint
is pinned to its Hugging Face release commit.

- [YOLO26 architecture and detection](https://docs.ultralytics.com/models/yolo26/)
- [YOLO depth outputs and scale calibration](https://docs.ultralytics.com/tasks/depth/)
- [UrbanOmniDetect-2 model card](https://huggingface.co/mehmetkeremturkcan/UrbanOmniDetect-2)
- [Urban source and geometry solver](https://github.com/mkturkcan/urbanomnidetect)
- [GUPNet official implementation](https://github.com/SuperMHP/GUPNet)

Local verification uses unit tests for selection, caching, immutable outputs,
coordinate handling, parity and notebook execution order. All 31 parking tests
passed, including a mocked 64-frame orchestration and resume test. Real
checkpoint and ONNX checks also passed on one supplied KITTI frame for all
three neural graphs. The YOLO depth conversion's maximum difference was
0.00044632 m; this is conversion agreement, not distance accuracy. The
[`local verification record`](artifacts/parking_edge_local_verification_20261021.json)
keeps the scope and graph hashes explicit. Phone performance and independent
distance accuracy remain separate unmeasured gates.
