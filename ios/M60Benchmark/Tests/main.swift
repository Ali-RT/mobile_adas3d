// Host-only checks of the SAME resource reader/stride serializer used on iOS.
// No device-runtime or latency claim is made by this test executable.
import Foundation
import CoreML

let root = URL(fileURLWithPath: CommandLine.arguments[1]).standardizedFileURL
let manifest = try JSONDecoder().decode(Manifest.self, from: Data(contentsOf: root.appendingPathComponent("manifest.json")))
precondition(manifest.complete && manifest.samples.count == 16)
let runner = M60Runner { _ in }
for fixture in manifest.samples {
    let provider = try runner.provider(root, fixture)
    for (name, tensor) in fixture.inputs {
        let array = provider.featureValue(for: name)!.multiArrayValue!
        let values = try runner.floats(array, shape: tensor.shape)
        let data = try runner.tensorData(root, tensor)
        let serialized = values.withUnsafeBytes { Data($0) }
        precondition(serialized == data, "FP32 input bytes changed")
    }
}
let buffer = UnsafeMutablePointer<Float>.allocate(capacity: 8)
buffer.initialize(repeating: -123, count: 8)
buffer[0] = 1; buffer[1] = 2; buffer[2] = 3
buffer[4] = 4; buffer[5] = 5; buffer[6] = 6
let padded = try MLMultiArray(dataPointer: buffer, shape: [2, 3], dataType: .float32, strides: [4, 1], deallocator: { $0.deallocate() })
let values = try runner.floats(padded, shape: [2, 3])
precondition(values == [1, 2, 3, 4, 5, 6], "Output padding leaked")
print("PASS: all16 device inputs bit-exact; padded output serialization correct")
