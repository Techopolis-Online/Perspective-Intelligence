//
//  ChatCommands.swift
//  Perspective Intelligence
//
//  Created by GitHub Copilot on 9/14/25.
//

import SwiftUI

struct ChatCommands: Commands {
    @Environment(\.openWindow) private var openWindow

    var body: some Commands {
        CommandMenu("Chat") {
            Button("New Chat Window") {
                openWindow(id: "chat")
            }
            .keyboardShortcut("n", modifiers: [.command])
        }
    }
}
