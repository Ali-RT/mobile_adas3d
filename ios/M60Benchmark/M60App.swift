import SwiftUI

@main
struct M60App: App {
    var body: some Scene { WindowGroup { M60View() } }
}

struct M60View: View {
    @State private var running = false
    @State private var status = "Ready. Choose which existing model benchmark to run."
    @State private var result = ""

    private func run(_ model: String) {
        running = true
        result = ""
        UIApplication.shared.isIdleTimerDisabled = true
        Task {
            do {
                let url = try await Task.detached(priority: .userInitiated) {
                    let progress: @Sendable (String) -> Void = { message in
                        Task { @MainActor in status = message }
                    }
                    if model == "A2 ONNX" {
                        return try A2ONNXRunner(progress: progress).run()
                    }
                    return try M60Runner(progress: progress).run()
                }.value
                result = url.lastPathComponent
                status = "Run saved. Return the complete run folder for review."
            } catch {
                status = error.localizedDescription
            }
            UIApplication.shared.isIdleTimerDisabled = false
            running = false
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("MobileADAS3D Model Benchmarks").font(.title)
            Text("One existing benchmark app; separate model paths and reports.").font(.caption)
            Button(running ? "Running…" : "Run A2 MonoDETR — ONNX Runtime CPU") { run("A2 ONNX") }
                .buttonStyle(.borderedProminent).disabled(running)
            Button(running ? "Running…" : "Run MonoDGP M60 — Core ML") { run("M60") }
                .buttonStyle(.bordered).disabled(running)
            Text("A2 uses ONNX Runtime CPU only. This fixed-input pilot excludes camera capture, preprocessing, 3D decode and rendering.")
                .font(.callout)
            Text("5 warmups · 100 timed predictions · 60-second sustained run")
            if running { ProgressView() }
            Text(status).font(.callout)
            if !result.isEmpty { Text("Saved in this app’s Files folder: \(result)").font(.caption) }
            Spacer()
            Text("Model latency is not camera-to-label latency. A passing speed result alone does not qualify deployment.").font(.footnote)
        }.padding()
    }
}
