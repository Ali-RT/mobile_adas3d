# M68 A2 ONNX Runtime CPU iPhone pilot

The M68 Colab notebook exports the unchanged A2 epoch-130 checkpoint to FP32 ONNX and runs the complete Chen validation split with ONNX Runtime CPU. Do not stage the phone package until its full-validation AP and nearby-recall gates pass.

After the M68 quality gate passes:

1. Download `m68_a2_onnx_phone_bundle.zip` from the M68 Drive output and extract/replace the `A2_ONNX` directory at `ios/M60Benchmark/A2_ONNX` in this repository.
2. In Terminal, run `cd ios/M60Benchmark && pod install`.
3. Open `ios/M60Benchmark/M60Benchmark.xcworkspace` in Xcode (not the `.xcodeproj`).
4. Select the connected iPhone and run `M60Benchmark`.
5. In the existing app, choose **Run A2 MonoDETR — ONNX Runtime CPU**. The separate M60 Core ML benchmark remains available.
6. Keep the phone unlocked and the app foregrounded. The run checks 16 outputs against the Linux ONNX Runtime CPU references, then measures 100 predictions and a 60-second sustained run. The report is saved in the app's Documents folder and is accessible through Files.

This is a fixed-preprocessed-image model benchmark, not a live camera pipeline. It excludes image capture, resize/normalization, 3D decoding, and rendering. The app does not register the Core ML execution provider; ONNX Runtime uses CPU only. A good model-only timing result is not by itself a real-time or deployment qualification.

If Xcode does not show the iPhone, reconnect/unlock it and trust the computer. Verify the selected Xcode installation before changing any project/device settings.
