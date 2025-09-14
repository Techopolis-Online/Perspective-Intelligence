//
//  Perspective_IntelligenceApp.swift
//  Perspective Intelligence
//
//  Created by Michael Doise on 9/14/25.
//

import SwiftUI

@main
struct Perspective_IntelligenceApp: App {
    #if os(macOS)
    @StateObject private var serverController = ServerController()
    #endif
    var body: some Scene {
        #if os(macOS)
        MenuBarExtra("PI Server", systemImage: "bolt.horizontal.circle") {
            MenuBarContentView()
                .environmentObject(serverController)
        }
        #endif
        WindowGroup(id: "chat") {
            ChatView()
                #if os(macOS)
                .environmentObject(serverController)
                #endif
        }
        .commands {
            #if os(macOS)
            ChatCommands()
            #endif
        }
        #if os(macOS)
        Settings {
            SettingsView()
        }
        #endif
    }
}
