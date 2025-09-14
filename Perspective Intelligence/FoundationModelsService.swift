//
//  FoundationModelsService.swift
//  Perspective Intelligence
//
//  Created by Michael Doise on 9/14/25.
//

import Foundation
import OSLog
#if canImport(FoundationModels)
import FoundationModels
#endif
// We use system model APIs for on-device language model access

// MARK: - OpenAI-Compatible Types

struct ChatCompletionRequest: Codable {
    struct Message: Codable {
        let role: String
        let content: String
    }
    let model: String
    let messages: [Message]
    let temperature: Double?
    let max_tokens: Int?
    let stream: Bool?
}

struct ChatCompletionResponse: Codable {
    struct Choice: Codable {
        struct Message: Codable {
            let role: String
            let content: String
        }
        let index: Int
        let message: Message
        let finish_reason: String?
    }
    let id: String
    let object: String
    let created: Int
    let model: String
    let choices: [Choice]
}

// MARK: - Foundation Models Service

/// A service that bridges OpenAI-compatible requests to Apple's on-device Foundation Models.
final class FoundationModelsService: @unchecked Sendable {
    static let shared = FoundationModelsService()
    private let logger = Logger(subsystem: "com.example.PerspectiveIntelligence", category: "FoundationModelsService")
    
    private init() {}

    // MARK: Public API

    /// Handles an OpenAI-compatible chat completion request and returns a response.
    func handleChatCompletion(_ request: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        // Extract the prompt from messages according to OpenAI chat format.
        let prompt = buildPrompt(from: request.messages)

        // Call into Foundation Models.
        let output = try await generateText(model: request.model, prompt: prompt, temperature: request.temperature, maxTokens: request.max_tokens)

        let response = ChatCompletionResponse(
            id: "chatcmpl_" + UUID().uuidString.replacingOccurrences(of: "-", with: ""),
            object: "chat.completion",
            created: Int(Date().timeIntervalSince1970),
            model: request.model,
            choices: [
                .init(
                    index: 0,
                    message: .init(role: "assistant", content: output),
                    finish_reason: "stop"
                )
            ]
        )
        return response
    }

    // MARK: - Private helpers

    private func buildPrompt(from messages: [ChatCompletionRequest.Message]) -> String {
        // Simple concatenation of messages in role: content format.
        var parts: [String] = []
        for msg in messages {
            parts.append("\(msg.role): \(msg.content)")
        }
        parts.append("assistant:")
        return parts.joined(separator: "\n")
    }

    // Replace this with actual Foundation Models generation when available in your target.
    private func generateText(model: String, prompt: String, temperature: Double?, maxTokens: Int?) async throws -> String {
        // Prefer Apple Intelligence on supported platforms; otherwise return a graceful fallback
        logger.log("Generating text (FoundationModels if available, else fallback)")

        #if canImport(FoundationModels)
        if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
            return try await generateWithFoundationModels(model: model, prompt: prompt, temperature: temperature)
        }
        #endif

        // Fallback path when FoundationModels is not available on this platform/SDK.
        let trimmed = prompt.split(separator: "\n").last.map(String.init) ?? prompt
        let fallback = "(Local fallback) I can't access Apple Intelligence on this device/OS yet. Based on your prompt, here's an echo: \(trimmed.replacingOccurrences(of: "assistant:", with: "").trimmingCharacters(in: .whitespaces)))"
        return fallback
    }

    #if canImport(FoundationModels)
    @available(iOS 18.0, macOS 15.0, visionOS 2.0, *)
    private func generateWithFoundationModels(model: String, prompt: String, temperature: Double?) async throws -> String {
        // Use the system-managed on-device language model
        let systemModel = SystemLanguageModel.default

        // Check availability and provide descriptive errors for callers
        switch systemModel.availability {
        case .available:
            break
        case .unavailable(.deviceNotEligible):
            throw NSError(domain: "FoundationModelsService", code: 1, userInfo: [NSLocalizedDescriptionKey: "Device not eligible for Apple Intelligence."])
        case .unavailable(.appleIntelligenceNotEnabled):
            throw NSError(domain: "FoundationModelsService", code: 2, userInfo: [NSLocalizedDescriptionKey: "Apple Intelligence is not enabled. Please enable it in Settings."])
        case .unavailable(.modelNotReady):
            throw NSError(domain: "FoundationModelsService", code: 3, userInfo: [NSLocalizedDescriptionKey: "Model not ready (e.g., downloading). Try again later."])
        case .unavailable(let other):
            throw NSError(domain: "FoundationModelsService", code: 4, userInfo: [NSLocalizedDescriptionKey: "Model unavailable: \(String(describing: other))"])
        }

        // Build instructions from the requested model and temperature to guide style
        var instructionsParts: [String] = []
        instructionsParts.append("You are a helpful assistant. Keep responses clear and relevant.")
        instructionsParts.append("Requested model identifier: \(model)")
        if let temperature { instructionsParts.append("Use creativity level (temperature): \(temperature)") }
        let instructions = instructionsParts.joined(separator: "\n")

        // Create a short-lived session for this request
        let session = LanguageModelSession(instructions: instructions)

        // The current API does not expose maxTokens directly on respond; keep it in instructions.
        // You can also truncate on your side after response if needed.
        let response = try await session.respond(to: prompt)
        return response.content
    }
    #endif
}

