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

        // Support both classic string content and OpenAI-style structured content arrays.
        // We'll flatten any array of content parts into a single text string by concatenating text segments.
        init(from decoder: Decoder) throws {
            let c = try decoder.container(keyedBy: CodingKeys.self)
            self.role = (try? c.decode(String.self, forKey: .role)) ?? "user"
            // Try as plain string first
            if let s = try? c.decode(String.self, forKey: .content) {
                self.content = s
                return
            }
            // Try as array of strings
            if let arr = try? c.decode([String].self, forKey: .content) {
                self.content = arr.joined(separator: "\n")
                return
            }
            // Try as array of structured parts
            if let parts = try? c.decode([OAContentPart].self, forKey: .content) {
                let text = parts.compactMap { $0.text }.joined(separator: "")
                self.content = text
                return
            }
            // Try as a single structured part object
            if let part = try? c.decode(OAContentPart.self, forKey: .content) {
                self.content = part.text ?? ""
                return
            }
            // Fallback empty
            self.content = ""
        }

        init(role: String, content: String) {
            self.role = role
            self.content = content
        }

        enum CodingKeys: String, CodingKey { case role, content }
    }
    let model: String
    let messages: [Message]
    let temperature: Double?
    let max_tokens: Int?
    let stream: Bool?
    let multi_segment: Bool?
}

// Content part per OpenAI structured content. We only use text; non-text parts are ignored.
private struct OAContentPart: Codable {
    let type: String?
    let text: String?
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

// MARK: - OpenAI-Compatible Text Completions

struct TextCompletionRequest: Codable {
    let model: String
    let prompt: String
    let temperature: Double?
    let max_tokens: Int?
    let stream: Bool?

    // Support legacy clients that send prompt as either a string or an array of strings
    init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.model = try c.decode(String.self, forKey: .model)
        self.temperature = try? c.decode(Double.self, forKey: .temperature)
        self.max_tokens = try? c.decode(Int.self, forKey: .max_tokens)
        self.stream = try? c.decode(Bool.self, forKey: .stream)
        if let s = try? c.decode(String.self, forKey: .prompt) {
            self.prompt = s
        } else if let arr = try? c.decode([String].self, forKey: .prompt) {
            self.prompt = arr.joined(separator: "\n\n")
        } else {
            self.prompt = ""
        }
    }
}

struct TextCompletionResponse: Codable {
    struct Choice: Codable {
        let text: String
        let index: Int
        let logprobs: String? // null in our case
        let finish_reason: String?
    }
    let id: String
    let object: String // "text_completion"
    let created: Int
    let model: String
    let choices: [Choice]
}

// MARK: - OpenAI-Compatible Models

struct OpenAIModel: Codable {
    let id: String
    let object: String // "model"
    let created: Int
    let owned_by: String
}

struct OpenAIModelList: Codable {
    let object: String // "list"
    let data: [OpenAIModel]
}

// MARK: - Foundation Models Service

/// A service that bridges OpenAI-compatible requests to Apple's on-device Foundation Models.
final class FoundationModelsService: @unchecked Sendable {
    static let shared = FoundationModelsService()
    private let logger = Logger(subsystem: "com.example.PerspectiveIntelligence", category: "FoundationModelsService")
    private let createdEpoch: Int = Int(Date().timeIntervalSince1970)
    
    private init() {}

    // MARK: Public API

    /// Handles an OpenAI-compatible chat completion request and returns a response.
    func handleChatCompletion(_ request: ChatCompletionRequest) async throws -> ChatCompletionResponse {
        // Build a context-aware prompt that fits within the model's context by summarizing older content when needed.
        let prompt = await prepareChatPrompt(messages: request.messages, model: request.model, temperature: request.temperature, maxTokens: request.max_tokens)
        logger.log("[chat] model=\(request.model, privacy: .public) messages=\(request.messages.count) promptLen=\(prompt.count)")

        // Call into Foundation Models.
        let output = try await generateText(model: request.model, prompt: prompt, temperature: request.temperature, maxTokens: request.max_tokens)
        logger.log("[chat] outputLen=\(output.count)")

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

    // MARK: - Context management for Chat

    /// Prepares a prompt that fits within an approximate context budget by summarizing older
    /// messages into a compact system summary while preserving the most recent turns intact.
    /// This avoids naive truncation of the user's latest content.
    private func prepareChatPrompt(messages: [ChatCompletionRequest.Message], model: String, temperature: Double?, maxTokens: Int?) async -> String {
        // Build the full prompt first
        let full = buildPrompt(from: messages)
        let maxContextTokens = 4000
        let reserveForOutput = 512 // reserve headroom for the model's response
        let budget = max(1000, maxContextTokens - reserveForOutput)
        let fullTokens = approxTokenCount(full)
        if fullTokens <= budget {
            logger.log("[chat.ctx] fit=full tokens=\(fullTokens) budget=\(budget) messages=\(messages.count)")
            return full
        }

        // Strategy:
        // - Keep the last few messages intact (recent context is most relevant)
        // - Summarize the older messages into a short summary via FoundationModels when available
        // - Compose: system summary + recent messages
        let keepRecentCount = min(6, messages.count) // keep up to last 6 messages
        let recent = Array(messages.suffix(keepRecentCount))
        let older = Array(messages.dropLast(keepRecentCount))

        let olderText = older.isEmpty ? "" : buildPrompt(from: older)
        var summary: String = ""
        if !olderText.isEmpty {
            // Summarize older content into ~1500 chars; clamp input size to avoid overflows
            let clampInput = clampForSummarization(olderText, maxChars: 6000)
            summary = await summarizeText(clampInput, targetChars: 1500, model: model, temperature: temperature)
        }

        var parts: [String] = []
        if !summary.isEmpty {
            parts.append("system: Conversation summary (compressed): \n\(summary)")
        }
        parts.append(buildPrompt(from: recent))
        let compact = parts.joined(separator: "\n")
        let compactTokens = approxTokenCount(compact)
        logger.log("[chat.ctx] fit=summarized tokens=\(compactTokens) budget=\(budget) keptRecent=\(recent.count) olderSummarized=\(older.count)")

        // If still over budget, apply a second compression pass on the summary only.
        if compactTokens > budget, !summary.isEmpty {
            let tighter = await summarizeText(summary, targetChars: 800, model: model, temperature: temperature)
            let rebuilt = ["system: Conversation summary (compressed): \n\(tighter)", buildPrompt(from: recent)].joined(separator: "\n")
            let tokens = approxTokenCount(rebuilt)
            logger.log("[chat.ctx] fit=summary-tight tokens=\(tokens) budget=\(budget) keptRecent=\(recent.count)")
            return rebuilt
        }
        return compact
    }

    /// Rough token estimate (heuristic): ~4 chars per token.
    private func approxTokenCount(_ text: String) -> Int {
        return max(1, (text.count + 3) / 4)
    }

    /// Clamp very large input before summarization to avoid exceeding FM limits during the summarization step.
    private func clampForSummarization(_ text: String, maxChars: Int) -> String {
        guard text.count > maxChars else { return text }
        // Keep head and tail slices to retain both early and late context in the summary input
        let half = maxChars / 2
        let head = text.prefix(half)
        let tail = text.suffix(maxChars - half)
        return String(head) + "\n…\n" + String(tail)
    }

    /// Summarize text using FoundationModels when available; fallback to a naïve extract if not.
    private func summarizeText(_ text: String, targetChars: Int, model: String, temperature: Double?) async -> String {
        let instruction = "Summarize the following content in under \(targetChars) characters, preserving key technical details, APIs, and decisions relevant to the user’s most recent request. Use concise bullet points if helpful."
        let prompt = "Instructions:\n\(instruction)\n\nContent to summarize:\n\n\(text)"
        do {
            let out = try await generateText(model: model, prompt: prompt, temperature: temperature, maxTokens: nil)
            if out.count > targetChars {
                // Light clamp on the generated summary to respect target size
                return String(out.prefix(targetChars))
            }
            return out
        } catch {
            // Fall back to a naïve extract when FM is not available
            let sentences = text.split(separator: ".")
            let head = sentences.prefix(8).joined(separator: ". ")
            let tail = sentences.suffix(4).joined(separator: ". ")
            let combined = "\(head). … \(tail)."
            if combined.count > targetChars {
                return String(combined.prefix(targetChars))
            }
            return combined
        }
    }

    /// Handles an OpenAI-compatible text completion request and returns a response.
    func handleCompletion(_ request: TextCompletionRequest) async throws -> TextCompletionResponse {
        logger.log("[text] model=\(request.model, privacy: .public) promptLen=\(request.prompt.count)")
        let output = try await generateText(model: request.model, prompt: request.prompt, temperature: request.temperature, maxTokens: request.max_tokens)
        logger.log("[text] outputLen=\(output.count)")

        let response = TextCompletionResponse(
            id: "cmpl_" + UUID().uuidString.replacingOccurrences(of: "-", with: ""),
            object: "text_completion",
            created: Int(Date().timeIntervalSince1970),
            model: request.model,
            choices: [
                .init(text: output, index: 0, logprobs: nil, finish_reason: "stop")
            ]
        )
        return response
    }

    // MARK: - Ollama-compatible chat

    struct OllamaMessage: Codable {
        let role: String
        let content: String
    }

    struct OllamaChatRequest: Codable {
        let model: String
        let messages: [OllamaMessage]
        let stream: Bool?
        let options: OllamaChatOptions?
    }

    struct OllamaChatOptions: Codable {
        let temperature: Double?
        let num_predict: Int?
    }

    struct OllamaChatResponse: Codable {
        let model: String
        let created_at: String
        let message: OllamaMessage
        let done: Bool
        let total_duration: Int64?
    }

    func handleOllamaChat(_ request: OllamaChatRequest) async throws -> OllamaChatResponse {
        let temperature = request.options?.temperature
        let maxTokens = request.options?.num_predict
        // Reuse our chat completion pipeline by mapping roles/content
        let mapped = request.messages.map { ChatCompletionRequest.Message(role: $0.role, content: $0.content) }
    let chatReq = ChatCompletionRequest(model: request.model, messages: mapped, temperature: temperature, max_tokens: maxTokens, stream: false, multi_segment: nil)
        let resp = try await handleChatCompletion(chatReq)
        let iso = ISO8601DateFormatter()
        let createdAt = iso.string(from: Date(timeIntervalSince1970: TimeInterval(resp.created)))
        let outMessage = OllamaMessage(role: resp.choices.first?.message.role ?? "assistant", content: resp.choices.first?.message.content ?? "")
        return OllamaChatResponse(model: resp.model, created_at: createdAt, message: outMessage, done: true, total_duration: nil)
    }

    /// Returns the list of available models in OpenAI format. For now we expose a single on-device model id.
    func listModels() -> OpenAIModelList {
        let models = availableModels()
        return OpenAIModelList(object: "list", data: models)
    }

    /// Returns a single model by id in OpenAI format, if available.
    func getModel(id: String) -> OpenAIModel? {
        return availableModels().first { $0.id == id }
    }

    // MARK: Ollama-compatible models list (/api/tags)

    struct OllamaTagDetails: Codable {
        let format: String?
        let family: String?
        let families: [String]?
        let parameter_size: String?
        let quantization_level: String?
    }

    struct OllamaTagModel: Codable {
        let name: String
        let modified_at: String
        let size: Int64?
        let digest: String?
        let details: OllamaTagDetails?
    }

    struct OllamaTagsResponse: Codable {
        let models: [OllamaTagModel]
    }

    func listOllamaTags() -> OllamaTagsResponse {
        let iso = ISO8601DateFormatter()
        let modified = iso.string(from: Date(timeIntervalSince1970: TimeInterval(createdEpoch)))
        let model = OllamaTagModel(
            name: "apple.local:latest",
            modified_at: modified,
            size: nil,
            digest: nil,
            details: OllamaTagDetails(
                format: "system",
                family: "apple-intelligence",
                families: ["apple-intelligence"],
                parameter_size: nil,
                quantization_level: nil
            )
        )
        return OllamaTagsResponse(models: [model])
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
            do {
                return try await generateWithFoundationModels(model: model, prompt: prompt, temperature: temperature)
            } catch {
                logger.error("FoundationModels failed: \(String(describing: error))")
                // Fall through to fallback message below without truncating the prompt
            }
        }
        #endif

        // Fallback path when FoundationModels is not available on this platform/SDK.
        let trimmed = prompt.split(separator: "\n").last.map(String.init) ?? prompt
        let fallback = "(Local fallback) Apple Intelligence unavailable: returning a synthetic response. Based on your prompt, here's an echo: \(trimmed.replacingOccurrences(of: "assistant:", with: "").trimmingCharacters(in: .whitespaces)))"
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
        logger.log("[fm] requesting response len=\(prompt.count)")
        let response = try await session.respond(to: prompt)
        logger.log("[fm] got response len=\(response.content.count)")
        return response.content
    }
    #endif

    // MARK: - Models inventory

    private func availableModels() -> [OpenAIModel] {
        // Single logical model ID exposed to clients using OpenAI format. Keep stable for compatibility.
        // We report ownership as "system" since it's provided by on-device Apple Intelligence.
        let model = OpenAIModel(
            id: "apple.local",
            object: "model",
            created: createdEpoch,
            owned_by: "system"
        )
        return [model]
    }
}

// MARK: - Multi-segment chat generation (optional)

extension FoundationModelsService {
    /// Generate a long-form response in multiple segments by chaining short sessions.
    /// Each segment is streamed back via the `emit` callback as soon as it's generated.
    func generateChatSegments(messages: [ChatCompletionRequest.Message], model: String, temperature: Double?, segmentChars: Int = 900, maxSegments: Int = 4, emit: @escaping (String) async -> Void) async throws {
        // Prepare initial prompt within context budget
        let basePrompt = await prepareChatPrompt(messages: messages, model: model, temperature: temperature, maxTokens: nil)
        let tokens = approxTokenCount(basePrompt)
        logger.log("[chat.multi] basePromptLen=\(basePrompt.count) tokens=\(tokens) segChars=\(segmentChars) maxSeg=\(maxSegments)")
        var soFar = ""

        // Build a compact summary of the base prompt to use on continuation rounds to keep context small
        // If summarization is unavailable, fall back to clamping head/tail.
        let baseSummary: String = await summarizeText(basePrompt, targetChars: 800, model: model, temperature: temperature)
        let maxContextTokens = 4000
        // Leave extra headroom during multi-round streaming since we can't enforce output tokens on-device
        let reserveForRoundOutput = 800
        let roundBudget = max(1200, maxContextTokens - reserveForRoundOutput) // ~3200 tokens budget for input

        // Helper to build instructions for each segment
        func instructions(forRound round: Int) -> String {
            var parts: [String] = []
            parts.append("You are a helpful assistant. Continue the answer succinctly and cohesively.")
            parts.append("Aim for about \(segmentChars) characters in this segment; do not repeat prior content.")
            if round > 1 {
                parts.append("So far, you've written the following (do not repeat, only continue):\n\(soFar.suffix(1500))")
            }
            return parts.joined(separator: "\n")
        }

        // First segment uses the full prepared prompt
        for round in 1...maxSegments {
            // Construct prompt and instructions with adaptive trimming to fit within roundBudget
            var prompt: String
            var includeSoFarChars = 1500
            if round == 1 {
                prompt = basePrompt
            } else {
                // Use compact summary of the task/context for continuation rounds
                prompt = "Task/context summary (compressed):\n\(baseSummary)\n\nassistant:"
            }

            func buildInstructions(_ includeChars: Int) -> String {
                var parts: [String] = []
                parts.append("You are a helpful assistant. Continue the answer succinctly and cohesively.")
                parts.append("Aim for about \(segmentChars) characters in this segment; do not repeat prior content.")
                if round > 1 {
                    let tail = String(soFar.suffix(includeChars))
                    parts.append("So far, you've written the following (do not repeat, only continue):\n\(tail)")
                }
                return parts.joined(separator: "\n")
            }

            var instructionsStr = buildInstructions(includeSoFarChars)
            // If estimated tokens exceed budget, shrink the included previous-output tail
            while approxTokenCount(instructionsStr + "\n\n" + prompt) > roundBudget && includeSoFarChars > 200 {
                includeSoFarChars = max(200, includeSoFarChars / 2)
                instructionsStr = buildInstructions(includeSoFarChars)
            }

            do {
                #if canImport(FoundationModels)
                if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
                    // Create a fresh short-lived session per segment with tailored instructions
                    let session = LanguageModelSession(instructions: instructionsStr)
                    let response = try await session.respond(to: prompt)
                    let segment = response.content
                    logger.log("[chat.multi] round=\(round) segLen=\(segment.count)")
                    if !segment.isEmpty {
                        soFar += segment
                        await emit(segment)
                    }
                } else {
                    let segment = try await self.generateText(model: model, prompt: instructionsStr + "\n\n" + prompt, temperature: temperature, maxTokens: nil)
                    logger.log("[chat.multi] round=\(round) segLen=\(segment.count)")
                    if !segment.isEmpty {
                        soFar += segment
                        await emit(segment)
                    }
                }
                #else
                let segment = try await self.generateText(model: model, prompt: instructionsStr + "\n\n" + prompt, temperature: temperature, maxTokens: nil)
                logger.log("[chat.multi] round=\(round) segLen=\(segment.count)")
                if !segment.isEmpty {
                    soFar += segment
                    await emit(segment)
                }
                #endif
            } catch {
                // Propagate error so caller can send a friendly fallback and finalize the stream
                throw error
            }

            // Heuristic stop: if the last segment is short, assume completion
            if soFar.count >= segmentChars * (round - 1) + Int(Double(segmentChars) * 0.6) {
                // continue
            } else {
                break
            }
        }
    }
}

// (no prompt truncation utilities by design)

