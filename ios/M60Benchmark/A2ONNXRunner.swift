import Foundation
import CryptoKit
import Darwin

private struct A2TensorFile: Decodable {
    let file: String
    let shape: [Int]
    let dtype: String
    let sha256: String
    let bytes: Int
}

private struct A2Fixture: Decodable {
    let sample_id: String
    let inputs: [String: A2TensorFile]
    let cpu_reference_outputs: [String: A2TensorFile]
}

private struct A2ModelFile: Decodable {
    let file: String
    let sha256: String
    let size_bytes: Int
    let opset: Int
    let custom_operator_domains: [String]
}

private struct A2Manifest: Decodable {
    let revision: String
    let complete: Bool
    let checkpoint_sha256: String
    let model: A2ModelFile
    let providers: [String]
    let precision: String
    let onnxruntime_version: String
    let sample_ids: [String]
    let fixtures: [A2Fixture]
    let warmups: Int
    let timed_predictions: Int
    let sustain_seconds: Int
    let raw_limits: [String: Double]
}

private struct A2LoadedFixture {
    let id: String
    let inputs: [String: ORTValue]
    // Keep caller-owned NSMutableData alive for the full ORTValue lifetime.
    let inputBuffers: [String: NSMutableData]
    let referenceOutputs: [String: Data]
}

/// Fixed-input ONNX Runtime CPU benchmark for A2; intentionally does not use Core ML.
/// This is a model-only latency measurement, not camera-to-label end-to-end performance.
final class A2ONNXRunner: @unchecked Sendable {
    let progress: @Sendable (String) -> Void
    init(progress: @escaping @Sendable (String) -> Void) { self.progress = progress }

    private let inputShapes: [String: [Int]] = [
        "image": [1, 3, 384, 1280], "calibration": [1, 3, 4], "image_size": [1, 2]
    ]
    private let outputShapes: [String: [Int]] = [
        "logits": [1, 50, 3], "boxes": [1, 50, 6], "dimensions": [1, 50, 3],
        "depth": [1, 50, 2], "angle": [1, 50, 24]
    ]
    private let rawLimits: [String: Double] = [
        "logits": 0.001, "boxes": 0.0001, "dimensions": 0.001,
        "depth": 0.01, "angle": 0.001
    ]

    private func fail(_ message: String) -> NSError {
        NSError(domain: "A2ONNXRunner", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }

    private func sha256(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty { digest.update(data: data) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func checkedData(_ root: URL, _ tensor: A2TensorFile) throws -> Data {
        guard tensor.dtype == "float32", tensor.shape.allSatisfy({ $0 > 0 }) else {
            throw fail("Unexpected tensor metadata: \(tensor.file)")
        }
        let url = root.appendingPathComponent(tensor.file).standardizedFileURL
        guard url.path.hasPrefix(root.standardizedFileURL.path + "/"),
              try sha256(url) == tensor.sha256 else { throw fail("Tensor path/checksum mismatch: \(tensor.file)") }
        let data = try Data(contentsOf: url)
        guard data.count == tensor.shape.reduce(1, *) * 4, data.count == tensor.bytes else {
            throw fail("Tensor byte count mismatch: \(tensor.file)")
        }
        return data
    }

    private func percentiles(_ values: [Double]) throws -> [String: Double] {
        guard !values.isEmpty, values.allSatisfy({ $0.isFinite && $0 > 0 }) else {
            throw fail("Invalid latency samples")
        }
        let sorted = values.sorted()
        func percentile(_ fraction: Double) -> Double {
            let position = fraction * Double(sorted.count - 1)
            let lower = Int(position.rounded(.down)), upper = Int(position.rounded(.up))
            if lower == upper { return sorted[lower] }
            let weight = position - Double(lower)
            return sorted[lower] * (1 - weight) + sorted[upper] * weight
        }
        return ["count": Double(values.count), "mean_ms": values.reduce(0, +) / Double(values.count),
                "p50_ms": percentile(0.50), "p95_ms": percentile(0.95),
                "min_ms": sorted[0], "max_ms": sorted[sorted.count - 1]]
    }

    private func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    private func machine() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var value = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &value, &size, nil, 0)
        return String(cString: value)
    }

    private func residentBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? info.resident_size : 0
    }

    private func floats(_ data: Data) throws -> [Float] {
        guard data.count.isMultiple(of: 4) else { throw fail("Tensor data is not FP32-aligned") }
        return try stride(from: 0, to: data.count, by: 4).map { offset in
            let raw = data.withUnsafeBytes { $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self) }
            let value = Float(bitPattern: UInt32(littleEndian: raw))
            guard value.isFinite else { throw fail("Non-finite output tensor") }
            return value
        }
    }

    func run() throws -> URL {
        guard let root = Bundle.main.url(forResource: "A2_ONNX", withExtension: nil) else {
            throw fail("A2_ONNX bundle missing. Complete the M68 Colab export and copy its A2_ONNX folder into ios/M60Benchmark/A2_ONNX/.")
        }
        let manifestURL = root.appendingPathComponent("manifest.json")
        let manifest = try JSONDecoder().decode(A2Manifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.revision == "M68-A2-ONNX-ORT-CPU-IPHONE-FEASIBILITY-2026-10-08-r1",
              manifest.complete,
              manifest.checkpoint_sha256 == "ed2134a98acbf1ab2fc61f7c8749b38fdfd2418e7f7932593e5e37a8d9ef33f4",
              manifest.model.opset == 17, manifest.model.custom_operator_domains.isEmpty,
              manifest.precision == "FP32", manifest.providers == ["CPUExecutionProvider"],
              manifest.onnxruntime_version == "1.30.0",
              manifest.sample_ids.count == 16, manifest.fixtures.count == 16,
              manifest.fixtures.map({ $0.sample_id }) == manifest.sample_ids,
              manifest.warmups == 5, manifest.timed_predictions == 100, manifest.sustain_seconds == 60,
              manifest.raw_limits == rawLimits else {
            throw fail("A2 ONNX manifest/protocol does not match the reviewed M68 CPU-only run")
        }
        let modelURL = root.appendingPathComponent(manifest.model.file).standardizedFileURL
        guard modelURL.path.hasPrefix(root.standardizedFileURL.path + "/"),
              modelURL.standardizedFileURL.lastPathComponent == "A2_M68_FP32.onnx",
              modelURL.fileSize == manifest.model.size_bytes,
              try sha256(modelURL) == manifest.model.sha256 else { throw fail("A2 ONNX model checksum/size mismatch") }

        let directory = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask,
                                                      appropriateFor: nil, create: true)
            .appendingPathComponent("A2-ONNX-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        #if targetEnvironment(simulator)
        let physicalDevice = false
        #else
        let physicalDevice = true
        #endif
        var report: [String: Any] = [
            "schema_version": 1, "revision": manifest.revision, "complete": false,
            "manifest_sha256": try sha256(manifestURL), "model_sha256": manifest.model.sha256,
            "checkpoint_sha256": manifest.checkpoint_sha256,
            "runtime": "ONNX Runtime Objective-C API", "runtime_version": manifest.onnxruntime_version,
            "provider": "CPUExecutionProvider", "intra_op_threads": 2,
            "graph_optimization_level": "all",
            "coreml_execution_provider_enabled": false, "physical_device": physicalDevice,
            "hardware": machine(), "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
            "started_at": ISO8601DateFormatter().string(from: Date()), "warmups_completed": 0,
            "parity_samples": [], "prediction_ms": [], "sustain_prediction_ms": [],
            "resource_observations": [],
            "camera_pipeline_included": false, "preprocessing_included": false,
            "decode_included": false, "deployment_qualified": false,
            "latency_scope": "ORTSession.run only, fixed preprocessed tensors; excludes camera capture, image resize/normalization, 3D decode and rendering"
        ]
        let reportURL = directory.appendingPathComponent("report.json")
        func save() throws {
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: reportURL, options: .atomic)
        }
        var peakSampledResidentBytes: UInt64 = 0
        var observations: [[String: Any]] = []
        func observe(_ phase: String) throws {
            let memory = residentBytes()
            peakSampledResidentBytes = max(peakSampledResidentBytes, memory)
            let state = thermal()
            observations.append(["phase": phase, "uptime_seconds": ProcessInfo.processInfo.systemUptime,
                                 "thermal": state, "resident_bytes": memory])
            report["resource_observations"] = observations
            report["peak_sampled_resident_bytes"] = peakSampledResidentBytes
            try save()
            if state == "serious" || state == "critical" {
                throw fail("Thermal stop: \(state). Let the phone cool before another run.")
            }
        }
        try save()
        do {
            progress("Loading ONNX Runtime CPU session…")
            try observe("start")
            let env = try ORTEnv(loggingLevel: .warning)
            let options = try ORTSessionOptions()
            try options.setIntraOpNumThreads(2)
            try options.setGraphOptimizationLevel(.all)
            let loadStart = ProcessInfo.processInfo.systemUptime
            let session = try ORTSession(env: env, modelPath: modelURL.path, sessionOptions: options)
            report["model_load_ms"] = (ProcessInfo.processInfo.systemUptime - loadStart) * 1000
            try observe("session_loaded")
            guard Set(try session.inputNames()) == Set(inputShapes.keys),
                  Set(try session.outputNames()) == Set(outputShapes.keys) else {
                throw fail("ONNX session names differ from the fixed M68 tensor contract")
            }

            var loaded: [A2LoadedFixture] = []
            for fixture in manifest.fixtures {
                var values: [String: ORTValue] = [:]
                var buffers: [String: NSMutableData] = [:]
                var refs: [String: Data] = [:]
                for name in inputShapes.keys {
                    guard let tensor = fixture.inputs[name], tensor.shape == inputShapes[name] else {
                        throw fail("Input contract changed for \(fixture.sample_id)/\(name)")
                    }
                    let data = try checkedData(root, tensor)
                    let buffer = NSMutableData(data: data)
                    buffers[name] = buffer
                    values[name] = try ORTValue(tensorData: buffer, elementType: .float,
                                                shape: tensor.shape.map { NSNumber(value: $0) })
                }
                for name in outputShapes.keys {
                    guard let tensor = fixture.cpu_reference_outputs[name], tensor.shape == outputShapes[name] else {
                        throw fail("Output reference contract changed for \(fixture.sample_id)/\(name)")
                    }
                    refs[name] = try checkedData(root, tensor)
                }
                loaded.append(A2LoadedFixture(id: fixture.sample_id, inputs: values,
                                              inputBuffers: buffers, referenceOutputs: refs))
            }
            report["fixture_input_count"] = loaded.count
            report["fixture_input_bytes"] = loaded.reduce(0) { $0 + $1.inputBuffers.values.reduce(0) { $0 + $1.length } }
            try observe("fixtures_loaded")

            var parityRows: [[String: Any]] = []
            var allParityPassed = true
            for (index, fixture) in loaded.enumerated() {
                progress("ONNX Runtime fixed-input parity \(index + 1)/\(loaded.count): \(fixture.id)")
                let outputs = try session.run(withInputs: fixture.inputs,
                                              outputNames: Set(outputShapes.keys), runOptions: nil)
                var deltas: [String: Double] = [:]
                var outputFiles: [String: [String: Any]] = [:]
                var passed = true
                for name in outputShapes.keys {
                    guard let output = outputs[name] else { throw fail("Missing ONNX output: \(name)") }
                    let shapeInfo = try output.tensorTypeAndShapeInfo()
                    guard shapeInfo.elementType == .float,
                          shapeInfo.shape.map({ $0.intValue }) == outputShapes[name] else {
                        throw fail("Unexpected ONNX output shape/type: \(name)")
                    }
                    let actualData = Data(referencing: try output.tensorData())
                    let outputFilename = "\(fixture.id)_\(name).bin"
                    let outputURL = directory.appendingPathComponent(outputFilename)
                    try actualData.write(to: outputURL, options: .atomic)
                    outputFiles[name] = ["file": outputFilename, "sha256": try sha256(outputURL),
                                         "shape": outputShapes[name]!, "dtype": "float32"]
                    let actual = try floats(actualData)
                    let expected = try floats(fixture.referenceOutputs[name]!)
                    guard actual.count == expected.count else { throw fail("Output length mismatch: \(name)") }
                    let delta = zip(actual, expected).map { abs(Double($0 - $1)) }.max() ?? 0
                    let limit = manifest.raw_limits[name]!
                    deltas[name] = delta
                    passed = passed && delta <= limit
                }
                allParityPassed = allParityPassed && passed
                parityRows.append(["sample_id": fixture.id, "passed": passed,
                                   "max_abs_deltas": deltas, "outputs": outputFiles])
                report["parity_samples"] = parityRows
                report["device_raw_parity_passed"] = allParityPassed
                try observe("parity_\(fixture.id)")
            }
            report["device_raw_parity_passed"] = allParityPassed
            report["raw_parity_note"] = allParityPassed
                ? "Device output matched the fixed ONNX Runtime CPU references within the frozen numerical limits."
                : "Device output differed from Linux ONNX Runtime references; timing is still reported, but this is not a deployment pass."
            try save()

            func predict(_ index: Int) throws -> Double {
                let fixture = loaded[index % loaded.count]
                return try autoreleasepool {
                    let start = ProcessInfo.processInfo.systemUptime
                    let outputs = try session.run(withInputs: fixture.inputs,
                                                  outputNames: ["logits", "boxes", "dimensions", "depth", "angle"],
                                                  runOptions: nil)
                    guard outputs["depth"] != nil else { throw fail("Timed inference did not return depth") }
                    return (ProcessInfo.processInfo.systemUptime - start) * 1000
                }
            }
            for index in 0..<manifest.warmups {
                _ = try predict(index)
                report["warmups_completed"] = index + 1
            }
            var timings: [Double] = []
            for index in 0..<manifest.timed_predictions {
                timings.append(try predict(index))
                if (index + 1).isMultiple(of: 10) {
                    progress("Timing \(index + 1)/\(manifest.timed_predictions)")
                    report["prediction_ms"] = timings
                    try save()
                }
            }
            let sustainStart = ProcessInfo.processInfo.systemUptime
            var sustained: [Double] = []
            while ProcessInfo.processInfo.systemUptime - sustainStart < Double(manifest.sustain_seconds) {
                sustained.append(try predict(sustained.count))
                if sustained.count.isMultiple(of: 25) { try observe("sustain_\(sustained.count)") }
            }
            let sustainElapsed = ProcessInfo.processInfo.systemUptime - sustainStart
            let regularStats = try percentiles(timings)
            let sustainStats = try percentiles(sustained)
            let fps = Double(sustained.count) / sustainElapsed
            report["prediction_ms"] = timings
            report["prediction_summary"] = regularStats
            report["sustain_prediction_ms"] = sustained
            report["sustain_summary"] = sustainStats
            report["sustain_elapsed_seconds"] = sustainElapsed
            report["sustained_processed_fps"] = fps
            report["phone_latency_targets"] = ["model_p95_ms": 50.0, "sustained_processed_fps": 10.0]
            report["model_speed_gate_passed"] = regularStats["p95_ms"]! <= 50.0 && fps >= 10.0
            try observe("finished")
            report["deployment_qualified"] = false
            report["complete"] = true
            report["finished_at"] = ISO8601DateFormatter().string(from: Date())
            try save()
            progress("Benchmark saved. Review the report before changing the model.")
            return directory
        } catch {
            report["error"] = error.localizedDescription
            try? save()
            throw error
        }
    }
}

private extension URL {
    var fileSize: Int? {
        (try? resourceValues(forKeys: [.fileSizeKey]))?.fileSize
    }
}
