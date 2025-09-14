import Combine
import SwiftUI

#if os(macOS)
struct ServerApp: App {
    @StateObject private var serverController = ServerController()

    var body: some Scene {
        MenuBarExtra("PI Server", systemImage: "bolt.horizontal.circle") {
            VStack(alignment: .leading, spacing: 8) {
                Text("Perspective Intelligence Server")
                    .font(.headline)
                ServerStatusView()
                    .environmentObject(serverController)
                Divider()
                // Standard macOS apps already have a Quit menu command; omit explicit Quit button to avoid AppKit.
            }
            .padding(12)
            .frame(width: 300)
        }
        .commands { // Ensure standard app commands (including Quit) are available
            CommandGroup(replacing: .appInfo) { }
        }
    }
}
#endif

@MainActor
final class ServerController: ObservableObject {
    @Published var isRunning: Bool = false
    @Published var port: UInt16 = 11434

    func start() {
        self.isRunning = true
        Task.detached(priority: .userInitiated) { [port] in
            await LocalHTTPServer.shared.setPort(port)
            await LocalHTTPServer.shared.start()
            // Do not immediately overwrite isRunning; the listener may report not-ready briefly.
        }
    }

    func stop() {
        // Immediately reflect stopped state in UI
        self.isRunning = false
        Task.detached(priority: .userInitiated) { [weak self] in
            await LocalHTTPServer.shared.stop()
            await self?.setRunning(false)
        }
    }

    func restart() {
        Task.detached(priority: .userInitiated) { [port, weak self] in
            await LocalHTTPServer.shared.stop()
            await LocalHTTPServer.shared.setPort(port)
            await LocalHTTPServer.shared.start()
            await self?.setRunning(true)
        }
    }

    private func setRunning(_ value: Bool) {
        // @MainActor context
        self.isRunning = value
    }
}

struct ServerStatusView: View {
    @EnvironmentObject private var server: ServerController
    @State private var localPort: UInt16 = 11434

    private static let portFormatter: NumberFormatter = {
        let nf = NumberFormatter()
        nf.numberStyle = .none
        nf.minimum = 1
        nf.maximum = 65535
        return nf
    }()

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Circle()
                    .fill(server.isRunning ? .green : .red)
                    .frame(width: 10, height: 10)
                Text(server.isRunning ? "Running on port \(server.port)" : "Stopped")
                    .font(.subheadline)
                    .foregroundStyle(.secondary)
                Spacer()
            }
            HStack(spacing: 8) {
                Button(server.isRunning ? "Restart" : "Start") {
                    server.port = localPort
                    if server.isRunning { server.restart() } else { server.start() }
                }
                Button("Stop") {
                    server.stop()
                }
                .disabled(!server.isRunning)
            }
            HStack(spacing: 6) {
                Text("Port:")
                TextField("Port", value: $localPort, formatter: Self.portFormatter)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 90)
            }
            Text("OpenAI-compatible endpoints:\nPOST /v1/chat/completions")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .id(server.isRunning) // Force menu to refresh button labels when state changes
        .animation(.default, value: server.isRunning)
        .onAppear {
            localPort = server.port
        }
        .onChange(of: server.port) { _, newValue in
            // Keep the text field in sync with external port changes
            localPort = newValue
        }
    }
}
