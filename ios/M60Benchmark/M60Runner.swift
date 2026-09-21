import Foundation
import CoreML
import CryptoKit
import Darwin

struct TensorFile: Codable {
    let file: String
    let shape: [Int]
    let sha256: String
}

struct Fixture: Decodable {
    let sample_id: String
    let inputs: [String: TensorFile]
    let mac_outputs: [String: TensorFile]
}

struct Manifest: Decodable {
    let complete: Bool
    let revision: String
    let model_directory: String
    let model_files: [String: String]
    let compute_units: String
    let warmups: Int
    let timed_predictions: Int
    let sustain_seconds: Int
    let sample_ids: [String]
    let samples: [Fixture]
    let policy: NumericalPolicy
}

struct NumericalPolicy: Decodable { let raw_limits: [String: Float] }

enum M60Error: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? {
        switch self { case .invalid(let text): return text }
    }
}

/// No camera, old dense decoder, GPU fallback selection or model mutation.
/// The host independently checks all saved outputs with the native decoder.
final class M60Runner: @unchecked Sendable {
    let progress: @Sendable (String) -> Void
    init(progress: @escaping @Sendable (String) -> Void) { self.progress = progress }

    func hash(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let data = try handle.read(upToCount: 1_048_576), !data.isEmpty { digest.update(data: data) }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    func tensorData(_ root: URL, _ tensor: TensorFile) throws -> Data {
        let url = root.appendingPathComponent(tensor.file).standardizedFileURL
        guard url.path.hasPrefix(root.standardizedFileURL.path + "/"),
              try hash(url) == tensor.sha256 else { throw M60Error.invalid("Fixture checksum/path: \(tensor.file)") }
        let data = try Data(contentsOf: url)
        guard data.count == tensor.shape.reduce(1, *) * 4 else { throw M60Error.invalid("Fixture size: \(tensor.file)") }
        return data
    }

    func provider(_ root: URL, _ fixture: Fixture) throws -> MLDictionaryFeatureProvider {
        var values: [String: MLFeatureValue] = [:]
        for (name, tensor) in fixture.inputs {
            let data = try tensorData(root, tensor)
            let array = try MLMultiArray(shape: tensor.shape.map { NSNumber(value: $0) }, dataType: .float32)
            // MLMultiArray(shape:) owns contiguous row-major storage. Inputs are
            // explicit little-endian FP32; supported Apple devices are LE.
            _ = data.withUnsafeBytes { memcpy(array.dataPointer, $0.baseAddress!, data.count) }
            values[name] = MLFeatureValue(multiArray: array)
        }
        return try MLDictionaryFeatureProvider(dictionary: values)
    }

    func floats(_ array: MLMultiArray, shape: [Int]) throws -> [Float] {
        guard array.shape.map({ $0.intValue }) == shape, array.dataType == .float32 else {
            throw M60Error.invalid("Unexpected Core ML output shape/type")
        }
        // Core ML outputs may be padded/strided; never memcpy them blindly.
        let strides = array.strides.map { $0.intValue }
        let source = array.dataPointer.assumingMemoryBound(to: Float.self)
        return try (0..<array.count).map { flat in
            var rest = flat, offset = 0
            for dimension in shape.indices.reversed() {
                offset += (rest % shape[dimension]) * strides[dimension]
                rest /= shape[dimension]
            }
            let value = source[offset]
            guard value.isFinite else { throw M60Error.invalid("Nonfinite Core ML output") }
            return value
        }
    }

    func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: return "nominal"
        case .fair: return "fair"
        case .serious: return "serious"
        case .critical: return "critical"
        @unknown default: return "unknown"
        }
    }

    func residentBytes() -> UInt64 {
        var info = mach_task_basic_info()
        var count = mach_msg_type_number_t(MemoryLayout<mach_task_basic_info>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(MACH_TASK_BASIC_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? info.resident_size : 0
    }

    func machine() -> String {
        var size = 0
        sysctlbyname("hw.machine", nil, &size, nil, 0)
        var value = [CChar](repeating: 0, count: size)
        sysctlbyname("hw.machine", &value, &size, nil, 0)
        return String(cString: value)
    }

    func run() throws -> URL {
        guard let root = Bundle.main.url(forResource: "M60", withExtension: nil) else {
            throw M60Error.invalid("M60 resources missing; run the preparation script first")
        }
        let manifestURL = root.appendingPathComponent("manifest.json")
        let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: manifestURL))
        guard manifest.complete, manifest.compute_units == "ALL", manifest.samples.count == 16,
              manifest.samples.map({ $0.sample_id }) == manifest.sample_ids,
              manifest.warmups == 5, manifest.timed_predictions == 100, manifest.sustain_seconds == 60 else {
            throw M60Error.invalid("Unexpected M60 protocol")
        }
        let directory = try FileManager.default.url(for: .documentDirectory, in: .userDomainMask, appropriateFor: nil, create: true)
            .appendingPathComponent("M60-" + UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        #if targetEnvironment(simulator)
        let physicalDevice = false
        #else
        let physicalDevice = true
        #endif
        var report: [String: Any] = ["schema_version": 1, "revision": manifest.revision, "complete": false,
            "manifest_sha256": try hash(manifestURL), "compute_units": "ALL", "physical_device": physicalDevice,
            "hardware": machine(), "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
            "started_at": ISO8601DateFormatter().string(from: Date()), "samples": [], "prediction_ms": [],
            "sustain_prediction_ms": [], "warmups_completed": 0, "deployment_authorized": false,
            "camera_integration": false, "latency_scope": "synchronous MLModel.prediction only; no input loading, decode or file writes",
            "fixture_memory_note": "16 input tensors cached; RSS includes fixtures, app and model, not model-only allocation"]
        func save() throws {
            let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
            try data.write(to: directory.appendingPathComponent("report.json"), options: .atomic)
        }
        var peak: UInt64 = 0
        var observations: [[String: Any]] = []
        func observe(_ phase: String) throws {
            let memory = residentBytes()
            peak = max(peak, memory)
            let state = thermal()
            observations.append(["phase": phase, "uptime_seconds": ProcessInfo.processInfo.systemUptime,
                                 "thermal": state, "resident_bytes": memory])
            report["observations"] = observations
            report["peak_sampled_resident_bytes"] = peak
            try save()
            if state == "serious" || state == "critical" { throw M60Error.invalid("Thermal stop: \(state). Cool the phone before another run.") }
        }
        try save()
        do {
            progress("Verifying exact model files…")
            try observe("start")
            for (name, checksum) in manifest.model_files {
                guard try hash(root.appendingPathComponent(name)) == checksum else { throw M60Error.invalid("Model checksum: \(name)") }
            }
            progress("Compiling unchanged model for this device…")
            let compileStart = ProcessInfo.processInfo.systemUptime
            let compiled = try MLModel.compileModel(at: root.appendingPathComponent(manifest.model_directory))
            defer { try? FileManager.default.removeItem(at: compiled) }
            report["compile_ms"] = (ProcessInfo.processInfo.systemUptime - compileStart) * 1000
            try observe("compiled")
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .all
            let loadStart = ProcessInfo.processInfo.systemUptime
            let model = try MLModel(contentsOf: compiled, configuration: configuration)
            report["model_load_ms"] = (ProcessInfo.processInfo.systemUptime - loadStart) * 1000
            try observe("loaded")
            let inputs = try manifest.samples.map { try provider(root, $0) }
            report["fixture_input_bytes"] = manifest.samples.reduce(0) { sum, sample in
                sum + sample.inputs.values.reduce(0) { $0 + $1.shape.reduce(1, *) * 4 }
            }
            try observe("inputs_cached")
            var samples: [[String: Any]] = []
            var rawPassed = true
            for (index, fixture) in manifest.samples.enumerated() {
                progress("Device parity \(index + 1)/16: \(fixture.sample_id)")
                try autoreleasepool {
                    let start = ProcessInfo.processInfo.systemUptime
                    let prediction = try model.prediction(from: inputs[index])
                    let elapsed = (ProcessInfo.processInfo.systemUptime - start) * 1000
                    var outputs: [String: TensorFile] = [:]
                    var deltas: [String: Double] = [:]
                    for (name, reference) in fixture.mac_outputs {
                        guard let array = prediction.featureValue(for: name)?.multiArrayValue else { throw M60Error.invalid("Missing output \(name)") }
                        let values = try floats(array, shape: reference.shape)
                        let expected = try tensorData(root, reference)
                        var maxDelta: Float = 0
                        expected.withUnsafeBytes { bytes in
                            for i in values.indices {
                                let value = bytes.loadUnaligned(fromByteOffset: i * 4, as: Float.self)
                                maxDelta = max(maxDelta, abs(value - values[i]))
                            }
                        }
                        let family = name.hasPrefix("pred_region_prob") ? "pred_region_prob" : name
                        guard let limit = manifest.policy.raw_limits[family] else {
                            throw M60Error.invalid("Missing frozen limit for \(name)")
                        }
                        rawPassed = rawPassed && maxDelta <= limit
                        deltas[name] = Double(maxDelta)
                        let relative = "\(fixture.sample_id)_\(name).bin"
                        let url = directory.appendingPathComponent(relative)
                        try values.withUnsafeBytes { try Data($0).write(to: url, options: .atomic) }
                        outputs[name] = TensorFile(file: relative, shape: reference.shape, sha256: try hash(url))
                    }
                    let encoded = try JSONSerialization.jsonObject(with: JSONEncoder().encode(outputs))
                    samples.append(["sample_id": fixture.sample_id, "outputs": encoded,
                                    "raw_max_abs_deltas_vs_mac": deltas, "prediction_ms": elapsed])
                }
                report["samples"] = samples
                try observe("parity_\(index + 1)")
            }
            report["raw_parity_passed"] = rawPassed
            guard rawPassed else { throw M60Error.invalid("Raw parity failed. Outputs saved; stop for host review.") }
            for index in 0..<manifest.warmups {
                progress("Warmup \(index + 1)/5")
                try autoreleasepool { _ = try model.prediction(from: inputs[index % inputs.count]) }
                report["warmups_completed"] = index + 1
            }
            func predict(_ index: Int) throws -> Double {
                try autoreleasepool {
                    let start = ProcessInfo.processInfo.systemUptime
                    let output = try model.prediction(from: inputs[index % inputs.count])
                    let elapsed = (ProcessInfo.processInfo.systemUptime - start) * 1000
                    guard output.featureValue(for: "pred_depth") != nil else { throw M60Error.invalid("Timed prediction missing depth") }
                    return elapsed
                }
            }
            var timings: [Double] = []
            for index in 0..<manifest.timed_predictions {
                timings.append(try predict(index))
                if (index + 1) % 10 == 0 {
                    progress("Timing \(index + 1)/100")
                    report["prediction_ms"] = timings
                    try observe("timing_\(index + 1)")
                }
            }
            let sustainedStart = ProcessInfo.processInfo.systemUptime
            var sustained: [Double] = []
            var lastObservation = sustainedStart
            while ProcessInfo.processInfo.systemUptime - sustainedStart < Double(manifest.sustain_seconds) {
                sustained.append(try predict(sustained.count))
                if ProcessInfo.processInfo.systemUptime - lastObservation >= 5 {
                    let elapsed = ProcessInfo.processInfo.systemUptime - sustainedStart
                    progress("Stability \(Int(elapsed))/60 seconds")
                    report["sustain_prediction_ms"] = sustained
                    report["sustain_elapsed_seconds"] = elapsed
                    try observe("sustain_\(Int(elapsed))")
                    lastObservation = ProcessInfo.processInfo.systemUptime
                }
            }
            report["sustain_prediction_ms"] = sustained
            report["sustain_elapsed_seconds"] = ProcessInfo.processInfo.systemUptime - sustainedStart
            try observe("finished")
            report["complete"] = true
            report["finished_at"] = ISO8601DateFormatter().string(from: Date())
            try save()
            progress("Run saved. Host geometry/parity review still required.")
            return directory
        } catch {
            report["error"] = error.localizedDescription
            try? save()
            progress("Stopped; report saved in \(directory.lastPathComponent): \(error.localizedDescription)")
            throw error
        }
    }
}
