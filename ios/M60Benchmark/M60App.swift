import SwiftUI

@main
struct M60App: App {
    var body: some Scene { WindowGroup { M60View() } }
}

struct M60View: View {
    @State private var running = false
    @State private var status = "Ready. Keep the phone unlocked and this app in the foreground."
    @State private var result = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text("MonoDGP M60").font(.title)
            Text("Revision M60-2026-09-21-r1").font(.caption)
            Text("Fixed-input feasibility only. No camera, training or safety qualification.")
            Text("16 parity inputs · 5 warmups · 100 predictions · 60-second stability run")
            Button(running ? "Running…" : "Start device benchmark") {
                running = true
                result = ""
                UIApplication.shared.isIdleTimerDisabled = true
                Task {
                    do {
                        let url = try await Task.detached(priority: .userInitiated) {
                            try M60Runner { message in Task { @MainActor in status = message } }.run()
                        }.value
                        result = url.lastPathComponent
                    } catch { status = error.localizedDescription }
                    UIApplication.shared.isIdleTimerDisabled = false
                    running = false
                }
            }.buttonStyle(.borderedProminent).disabled(running)
            if running { ProgressView() }
            Text(status).font(.callout)
            if !result.isEmpty { Text("Saved in Files → On My iPhone → MonoDGP M60 → \(result)").font(.caption) }
            Spacer()
            Text("Model-only timings exclude preprocessing, decoding and I/O. Return the complete run folder for independent host review.").font(.footnote)
        }.padding()
    }
}
