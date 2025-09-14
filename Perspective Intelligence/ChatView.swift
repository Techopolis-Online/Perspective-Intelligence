//
//  ChatView.swift
//  Perspective Intelligence
//
//  Created by GitHub Copilot on 9/14/25.
//

import SwiftUI
import Combine
#if os(macOS)
import AppKit
#endif

struct ChatMessageItem: Identifiable, Equatable {
    enum Role: String { case system, user, assistant }
    let id = UUID()
    let role: Role
    let content: String
}

@MainActor
final class ChatViewModel: ObservableObject {
    @Published var messages: [ChatMessageItem] = []
    @Published var input: String = ""
    @Published var isSending: Bool = false
    @Published var errorText: String?

    // Networking config
    @Published var port: UInt16 = 11434
    @Published var temperature: Double = 0.7

    func loadDefaultPortIfAvailable() {
        Task { [weak self] in
            guard let self else { return }
            // Ask the local server for its configured port if actor is running
            let running = await LocalHTTPServer.shared.getIsRunning()
            if running {
                let current = await LocalHTTPServer.shared.getPort()
                await MainActor.run { self.port = current }
            }
        }
    }

    func send() {
        let trimmed = input.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let userMsg = ChatMessageItem(role: .user, content: trimmed)
        input = ""
        messages.append(userMsg)
        isSending = true
        errorText = nil

        Task { [messages, port, temperature] in
            do {
                let assistant = try await self.callAPI(messages: messages, port: port, temperature: temperature)
                await MainActor.run {
                    self.messages.append(.init(role: .assistant, content: assistant))
                    self.isSending = false
                }
            } catch {
                await MainActor.run {
                    self.errorText = error.localizedDescription
                    self.isSending = false
                }
            }
        }
    }

    private func callAPI(messages: [ChatMessageItem], port: UInt16, temperature: Double) async throws -> String {
        // Convert to OpenAI-compatible request payload
        var mapped: [ChatCompletionRequest.Message] = []
        // Prepend system prompt from persisted settings if enabled (read via UserDefaults in ViewModel)
        let includeSystem = UserDefaults.standard.bool(forKey: "includeSystemPrompt")
        if includeSystem {
            if let system = UserDefaults.standard.string(forKey: "systemPrompt"),
               !system.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                mapped.append(.init(role: "system", content: system))
            }
        }
        // Include either full history or just the latest user message
        let includeHistory = UserDefaults.standard.bool(forKey: "includeHistory")
        if includeHistory {
            mapped += messages.map { .init(role: $0.role.rawValue, content: $0.content) }
        } else if let lastUser = messages.last(where: { $0.role == .user }) {
            mapped.append(.init(role: lastUser.role.rawValue, content: lastUser.content))
        }
        // Fixed model (non-configurable)
        let reqBody = ChatCompletionRequest(model: "apple.local", messages: mapped, temperature: temperature, max_tokens: nil, stream: false)
        let url = URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!
        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(reqBody)

        let (data, response) = try await URLSession.shared.data(for: request)
        if UserDefaults.standard.bool(forKey: "debugLogging") {
            print("[PI Chat] includeSystemPrompt=\(UserDefaults.standard.bool(forKey: "includeSystemPrompt")) includeHistory=\(UserDefaults.standard.bool(forKey: "includeHistory"))")
            print("[PI Chat] Request messages:\n\(mapped.map{"\($0.role): \($0.content)"}.joined(separator: "\n"))")
            if let http = response as? HTTPURLResponse {
                print("[PI Chat] Response status: \(http.statusCode)")
            }
            print("[PI Chat] Raw body: \(String(data: data, encoding: .utf8) ?? "<non-utf8>")")
        }
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            let body = String(data: data, encoding: .utf8) ?? "<no body>"
            throw NSError(domain: "ChatViewModel", code: http.statusCode, userInfo: [NSLocalizedDescriptionKey: "Server returned \(http.statusCode): \(body)"])
        }
        let decoded = try JSONDecoder().decode(ChatCompletionResponse.self, from: data)
        guard let first = decoded.choices.first?.message.content else {
            throw NSError(domain: "ChatViewModel", code: -1, userInfo: [NSLocalizedDescriptionKey: "Empty response from model"])
        }
        return first
    }
}

struct ChatView: View {
    @StateObject private var vm = ChatViewModel()
    @Environment(\.colorScheme) private var scheme
    #if os(macOS)
    @EnvironmentObject private var serverController: ServerController
    #endif
    @AppStorage("systemPrompt") private var systemPrompt: String = "You are a helpful assistant. Keep responses concise and relevant."
    @AppStorage("includeSystemPrompt") private var includeSystemPrompt: Bool = false
    @AppStorage("includeHistory") private var includeHistoryToggle: Bool = true
    @State private var serverReady: Bool = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            messagesList
            Divider()
            composer
        }
        .task {
            #if os(macOS)
            vm.port = serverController.port
            serverReady = await LocalHTTPServer.shared.getIsRunning()
            #endif
            vm.loadDefaultPortIfAvailable()
        }
        #if os(macOS)
        .onChange(of: serverController.isRunning) { _, newValue in
            serverReady = newValue
        }
        #endif
    }

    // No auto-start here; ChatView only reflects server status.

    private var header: some View {
        HStack(spacing: 8) {
            Circle()
                .fill(serverStatusColor)
                .frame(width: 10, height: 10)
                .accessibilityHidden(true)
            Text(serverReady ? "Local API: 127.0.0.1:\(vm.port)" : "Server offline")
                .font(.subheadline)
                .foregroundStyle(.secondary)
                .accessibilityLabel(serverReady ? "Local API on port \(vm.port)" : "Server offline")
            Spacer()
            if includeSystemPrompt && !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text("System Prompt On")
                    .font(.caption2)
                    .padding(.vertical, 2)
                    .padding(.horizontal, 6)
                    .background(RoundedRectangle(cornerRadius: 6).fill(Color.green.opacity(0.15)))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.green.opacity(0.3)))
                    .accessibilityLabel("System prompt enabled")
            }
            Menu {
                Toggle("Include System Prompt", isOn: $includeSystemPrompt)
                Toggle("Include Conversation History", isOn: $includeHistoryToggle)
                Divider()
                Button("New Chat") { vm.messages.removeAll() }
            } label: {
                Image(systemName: "gearshape")
            }
            .accessibilityLabel("Options menu")
            .menuIndicator(.visible)
            // New Chat remains accessible via the Options menu; keep header minimal.
        }
        .padding(8)
    }

    private var messagesList: some View {
        ScrollViewReader { proxy in
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 8) {
                    ForEach(vm.messages) { msg in
                        messageBubble(for: msg)
                            .frame(maxWidth: .infinity, alignment: msg.role == .user ? .trailing : .leading)
                            .id(msg.id)
                    }
                }
                .padding(12)
            }
            .onChange(of: vm.messages.count) { _, _ in
                if let last = vm.messages.last?.id {
                    withAnimation { proxy.scrollTo(last, anchor: .bottom) }
                }
            }
        }
    }

    private var composer: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let error = vm.errorText {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .padding(.horizontal, 8)
                    .accessibilityLabel("Error: \(error)")
            }
            HStack(spacing: 8) {
                TextField("Type a message…", text: $vm.input)
                    .textFieldStyle(.plain)
                    .padding(.vertical, 10)
                    .padding(.horizontal, 12)
                    .background(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .fill(Color.secondary.opacity(0.12))
                    )
                    .overlay(
                        RoundedRectangle(cornerRadius: 18, style: .continuous)
                            .stroke(.quaternary)
                    )
                    .accessibilityLabel("Message input")
                    .accessibilityHint("Type your message and press Return to send")
                    .submitLabel(.send)
                    .onSubmit { if serverReady { vm.send() } }
                Button(action: vm.send) {
                    if vm.isSending {
                        ProgressView().controlSize(.small)
                    } else {
                        Label("Send", systemImage: "paperplane.fill")
                            .labelStyle(.titleAndIcon)
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(vm.isSending || vm.input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || (!serverReady))
                .accessibilityLabel("Send")
                .accessibilityHint("Sends your message")
            }
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
        }
        .background(.ultraThinMaterial)
    }

    private var serverStatusColor: Color {
        #if os(macOS)
        serverController.isRunning ? .green : .red
        #else
        .gray
        #endif
    }

    // MARK: - Message bubble rendering (iMessage-like)
    @ViewBuilder
    private func messageBubble(for msg: ChatMessageItem) -> some View {
        let isUser = msg.role == .user
        VStack(alignment: isUser ? .trailing : .leading, spacing: 4) {
            Text(isUser ? "You" : (msg.role == .assistant ? "Assistant" : msg.role.rawValue.capitalized))
                .font(.caption2)
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            HStack(alignment: .bottom, spacing: 8) {
                if isUser { Spacer(minLength: 50) }
                Text(msg.content)
                    .padding(.vertical, 8)
                    .padding(.horizontal, 12)
                    .background(isUser ? Color.accentColor : Color.secondary.opacity(scheme == .dark ? 0.35 : 0.15))
                    .foregroundStyle(isUser ? Color.white : Color.primary)
                    .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
                    .textSelection(.enabled)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel(isUser ? "You: \(msg.content)" : "Assistant: \(msg.content)")
                if !isUser { Spacer(minLength: 50) }
            }
        }
        .contextMenu {
            #if os(macOS)
            Button("Copy") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(msg.content, forType: .string)
            }
            #endif
        }
    }
}

#Preview {
    ChatView()
}
