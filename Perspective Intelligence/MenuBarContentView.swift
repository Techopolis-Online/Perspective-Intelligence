//
//  MenuBarContentView.swift
//  Perspective Intelligence
//
//  Created by GitHub Copilot on 9/14/25.
//

import SwiftUI

struct MenuBarContentView: View {
    @Environment(\.openWindow) private var openWindow
    @EnvironmentObject private var serverController: ServerController

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            ServerStatusView()
                .environmentObject(serverController)
            Divider()
            Button("Open Chat Window") {
                openWindow(id: "chat")
            }
        }
        .padding(12)
        .frame(width: 300)
        .id(serverController.isRunning)
    }
}

#Preview {
    MenuBarContentView()
        .environmentObject(ServerController())
}
