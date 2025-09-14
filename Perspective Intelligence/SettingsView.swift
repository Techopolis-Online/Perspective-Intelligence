//
//  SettingsView.swift
//  Perspective Intelligence
//
//  Created by GitHub Copilot on 9/14/25.
//

import SwiftUI

struct SettingsView: View {
    @AppStorage("systemPrompt") private var systemPrompt: String = "You are a helpful assistant. Keep responses concise and relevant."
    @AppStorage("includeSystemPrompt") private var includeSystemPrompt: Bool = false
    @AppStorage("debugLogging") private var debugLogging: Bool = false
    @AppStorage("includeHistory") private var includeHistory: Bool = true

    var body: some View {
        Form {
            Toggle("Include System Prompt", isOn: $includeSystemPrompt)
                .accessibilityLabel("Include system prompt")
                .accessibilityHint("Turn off to send chats without the system instruction")
            Toggle("Enable Debug Logging", isOn: $debugLogging)
                .accessibilityLabel("Enable debug logging")
                .accessibilityHint("Print requests and responses to the console for troubleshooting")
            Toggle("Include Conversation History", isOn: $includeHistory)
                .accessibilityLabel("Include conversation history")
                .accessibilityHint("Turn off to send only the latest user message")

            Section(header: Text("System Prompt")) {
                TextEditor(text: $systemPrompt)
                    .frame(minHeight: 120)
                    .overlay(RoundedRectangle(cornerRadius: 8).stroke(.quaternary))
                    .accessibilityLabel("System prompt")
                    .accessibilityHint("Text used as the assistant's system instruction")
                HStack {
                    Spacer()
                    Button("Reset to Default") {
                        systemPrompt = "You are a helpful assistant. Keep responses concise and relevant."
                        includeSystemPrompt = true
                    }
                }
            }
            Text("The system prompt (if enabled) is sent with each chat to guide the assistant's behavior.")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding()
        .frame(minWidth: 420, minHeight: 260)
    }
}

#Preview {
    SettingsView()
}
