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
    // OpenAI-style tools support (optional)
    let tools: [OAITool]?
    let tool_choice: ToolChoice?
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

// MARK: - OpenAI Tools Types

struct OAITool: Codable {
    let type: String // expecting "function"
    let function: OAIFunction?
}

struct OAIFunction: Codable {
    let name: String
    let description: String?
    let parameters: JSONValue? // arbitrary JSON schema, not used by executor
}

enum ToolChoice: Codable {
    case none
    case auto
    case required
    case function(name: String)

    init(from decoder: Decoder) throws {
        if let s = try? decoder.singleValueContainer().decode(String.self) {
            switch s {
            case "none": self = .none
            case "auto": self = .auto
            case "required": self = .required
            default: self = .auto
            }
            return
        }
        struct FuncWrap: Codable { let type: String?; let function: Func? }
        struct Func: Codable { let name: String }
        if let f = try? decoder.singleValueContainer().decode(FuncWrap.self), let name = f.function?.name {
            self = .function(name: name)
            return
        }
        self = .auto
    }

    func encode(to encoder: Encoder) throws {
        switch self {
        case .none: var c = encoder.singleValueContainer(); try c.encode("none")
        case .auto: var c = encoder.singleValueContainer(); try c.encode("auto")
        case .required: var c = encoder.singleValueContainer(); try c.encode("required")
        case .function(let name):
            struct Wrapper: Codable { let type: String; let function: Inner }
            struct Inner: Codable { let name: String }
            var c = encoder.singleValueContainer()
            try c.encode(Wrapper(type: "function", function: Inner(name: name)))
        }
    }
}

// A minimal JSON value tree for decoding arbitrary tool parameter shapes
enum JSONValue: Codable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null; return }
        if let b = try? c.decode(Bool.self) { self = .bool(b); return }
        if let d = try? c.decode(Double.self) { self = .number(d); return }
        if let s = try? c.decode(String.self) { self = .string(s); return }
        if let o = try? decoder.container(keyedBy: DynamicCodingKeys.self) {
            var dict: [String: JSONValue] = [:]
            for key in o.allKeys {
                let v = try o.decode(JSONValue.self, forKey: key)
                dict[key.stringValue] = v
            }
            self = .object(dict)
            return
        }
        if var a = try? decoder.unkeyedContainer() {
            var arr: [JSONValue] = []
            while !a.isAtEnd { arr.append(try a.decode(JSONValue.self)) }
            self = .array(arr)
            return
        }
        self = .null
    }

    func encode(to encoder: Encoder) throws {
        switch self {
        case .string(let s): var c = encoder.singleValueContainer(); try c.encode(s)
        case .number(let d): var c = encoder.singleValueContainer(); try c.encode(d)
        case .bool(let b): var c = encoder.singleValueContainer(); try c.encode(b)
        case .null: var c = encoder.singleValueContainer(); try c.encodeNil()
        case .object(let dict):
            var o = encoder.container(keyedBy: DynamicCodingKeys.self)
            for (k,v) in dict { try o.encode(v, forKey: DynamicCodingKeys(stringValue: k)!) }
        case .array(let arr):
            var a = encoder.unkeyedContainer()
            for v in arr { try a.encode(v) }
        }
    }
}

struct DynamicCodingKeys: CodingKey {
    var stringValue: String
    init?(stringValue: String) { self.stringValue = stringValue }
    var intValue: Int? { nil }
    init?(intValue: Int) { return nil }
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
        // If tools are provided, run the tool-calling orchestration flow.
        if let tools = request.tools, !tools.isEmpty {
            return try await handleChatCompletionWithTools(request, tools: tools)
        }

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

    /// Tool-calling orchestration: prompt for a tool call, execute it, then get the final answer.
    private func handleChatCompletionWithTools(_ request: ChatCompletionRequest, tools: [OAITool]) async throws -> ChatCompletionResponse {
        // Build an augmented system instruction describing available tools and the exact JSON contract.
        let toolIntro = toolsDescription(tools)
        var msgs: [ChatCompletionRequest.Message] = []
        msgs.append(.init(role: "system", content: toolIntro))
        msgs.append(contentsOf: request.messages)

        // First round: ask the model if it wants to call a tool; if so, it must reply ONLY with the JSON envelope.
        let prompt1 = await prepareChatPrompt(messages: msgs, model: request.model, temperature: request.temperature, maxTokens: request.max_tokens)
        let out1 = try await generateText(model: request.model, prompt: prompt1, temperature: request.temperature, maxTokens: request.max_tokens)
        logger.log("[tools] first-round len=\(out1.count)")

        if let call = parseToolCall(from: out1) {
            // Execute tool
            let result = try await ToolsRegistry.shared.execute(name: call.name, arguments: call.arguments)
            let resultText = jsonString(result) ?? String(describing: result)
            // Append tool call and tool result, then ask for the final answer.
            msgs.append(.init(role: "assistant", content: out1))
            msgs.append(.init(role: "tool", content: resultText))
            let prompt2 = await prepareChatPrompt(messages: msgs, model: request.model, temperature: request.temperature, maxTokens: request.max_tokens)
            let out2 = try await generateText(model: request.model, prompt: prompt2, temperature: request.temperature, maxTokens: request.max_tokens)
            return ChatCompletionResponse(
                id: "chatcmpl_" + UUID().uuidString.replacingOccurrences(of: "-", with: ""),
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: request.model,
                choices: [
                    .init(index: 0, message: .init(role: "assistant", content: out2), finish_reason: "stop")
                ]
            )
        } else {
            // No tool call requested; treat out1 as the final answer.
            return ChatCompletionResponse(
                id: "chatcmpl_" + UUID().uuidString.replacingOccurrences(of: "-", with: ""),
                object: "chat.completion",
                created: Int(Date().timeIntervalSince1970),
                model: request.model,
                choices: [
                    .init(index: 0, message: .init(role: "assistant", content: out1), finish_reason: "stop")
                ]
            )
        }
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
    let chatReq = ChatCompletionRequest(model: request.model, messages: mapped, temperature: temperature, max_tokens: maxTokens, stream: false, multi_segment: nil, tools: nil, tool_choice: nil)
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

    // Build a tool intro system message describing available tools and the JSON envelope to request them
    private func toolsDescription(_ tools: [OAITool]) -> String {
        var lines: [String] = []
        lines.append("You can call tools when helpful. If a tool is needed, reply ONLY with a single JSON line in this exact schema and no extra text:")
        lines.append("{\"tool_call\": {\"name\": \"<tool-name>\", \"arguments\": { /* args */ } }}")
        lines.append("")
        lines.append("Available tools:")
        for t in tools {
            if t.type == "function", let f = t.function {
                let desc = f.description ?? ""
                lines.append("- \(f.name): \(desc)")
            }
        }
        return lines.joined(separator: "\n")
    }

    // Parse a tool call from model output, expecting a JSON envelope as instructed
    private func parseToolCall(from text: String) -> (name: String, arguments: JSONValue)? {
        struct Envelope: Codable { let tool_call: Inner }
        struct Inner: Codable { let name: String; let arguments: JSONValue }
        // Try direct decode first
        if let data = text.data(using: .utf8), let env = try? JSONDecoder().decode(Envelope.self, from: data) {
            return (env.tool_call.name, env.tool_call.arguments)
        }
        // Try to find a JSON object substring
        if let start = text.firstIndex(of: "{"), let end = text.lastIndex(of: "}") {
            let sub = String(text[start...end])
            if let data = sub.data(using: .utf8), let env = try? JSONDecoder().decode(Envelope.self, from: data) {
                return (env.tool_call.name, env.tool_call.arguments)
            }
        }
        return nil
    }

    // Serialize JSONValue to a compact string
    private func jsonString(_ v: JSONValue) -> String? {
        func encode(_ v: JSONValue) -> Any {
            switch v {
            case .string(let s): return s
            case .number(let d): return d
            case .bool(let b): return b
            case .null: return NSNull()
            case .object(let o): return o.mapValues { encode($0) }
            case .array(let a): return a.map { encode($0) }
            }
        }
        let any = encode(v)
        guard JSONSerialization.isValidJSONObject(any) else { return nil }
        if let data = try? JSONSerialization.data(withJSONObject: any, options: []) {
            return String(data: data, encoding: .utf8)
        }
        return nil
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
            let prompt: String
            if round == 1 {
                prompt = basePrompt
            } else {
                prompt = "\(basePrompt)\n\nassistant:"
            }

            do {
                #if canImport(FoundationModels)
                if #available(iOS 18.0, macOS 15.0, visionOS 2.0, *) {
                    // Create a fresh short-lived session per segment with tailored instructions
                    let session = LanguageModelSession(instructions: instructions(forRound: round))
                    let response = try await session.respond(to: prompt)
                    let segment = response.content
                    logger.log("[chat.multi] round=\(round) segLen=\(segment.count)")
                    if !segment.isEmpty {
                        soFar += segment
                        await emit(segment)
                    }
                } else {
                    let segment = try await self.generateText(model: model, prompt: instructions(forRound: round) + "\n\n" + prompt, temperature: temperature, maxTokens: nil)
                    logger.log("[chat.multi] round=\(round) segLen=\(segment.count)")
                    if !segment.isEmpty {
                        soFar += segment
                        await emit(segment)
                    }
                }
                #else
                let segment = try await self.generateText(model: model, prompt: instructions(forRound: round) + "\n\n" + prompt, temperature: temperature, maxTokens: nil)
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


// MARK: - Tools Registry

private final class ToolsRegistry: @unchecked Sendable {
    static let shared = ToolsRegistry()
    private let logger = Logger(subsystem: "com.example.PerspectiveIntelligence", category: "ToolsRegistry")

    // Constrain all file operations to a single root directory.
    // Set PI_WORKSPACE_ROOT in the environment to point at your workspace; defaults to Documents.
    private let rootURL: URL
    private let fm = FileManager.default

    private init() {
        if let root = ProcessInfo.processInfo.environment["PI_WORKSPACE_ROOT"], !root.isEmpty {
            self.rootURL = URL(fileURLWithPath: root)
        } else {
            self.rootURL = fm.urls(for: .documentDirectory, in: .userDomainMask).first ?? URL(fileURLWithPath: NSTemporaryDirectory())
        }
        logger.log("[tools] root=\(self.rootURL.path, privacy: .public)")
    }

    enum ToolError: Error, LocalizedError {
        case invalidPath
        case notFound
        case ioFailed(String)
        var errorDescription: String? {
            switch self {
            case .invalidPath: return "Invalid or out-of-root path"
            case .notFound: return "Path not found"
            case .ioFailed(let m): return m
            }
        }
    }

    // Execute a tool by name with JSONValue arguments
    func execute(name: String, arguments: JSONValue) async throws -> JSONValue {
        switch name {
        case "read_file":
            guard let path = argString(arguments, key: "path") else { return .object(["error": .string("Missing 'path'")]) }
            let maxBytes = argInt(arguments, key: "max_bytes") ?? 256 * 1024
            let url = try resolvePath(path)
            guard fm.fileExists(atPath: url.path) else { throw ToolError.notFound }
            let data = try Data(contentsOf: url)
            let slice = data.prefix(maxBytes)
            let text = String(decoding: slice, as: UTF8.self)
            return .object(["path": .string(path), "content": .string(text), "truncated": .bool(slice.count < data.count)])
        case "write_file":
            guard let path = argString(arguments, key: "path") else { return .object(["error": .string("Missing 'path'")]) }
            let content = argString(arguments, key: "content") ?? ""
            let url = try resolvePath(path)
            do {
                try fm.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
                try content.data(using: .utf8)?.write(to: url, options: .atomic)
            } catch {
                throw ToolError.ioFailed(String(describing: error))
            }
            return .object(["path": .string(path), "written_bytes": .number(Double(content.utf8.count))])
        case "list_dir":
            guard let path = argString(arguments, key: "path") else { return .object(["error": .string("Missing 'path'")]) }
            let url = try resolvePath(path)
            do {
                let items = try fm.contentsOfDirectory(atPath: url.path)
                return .object(["path": .string(path), "items": .array(items.map { .string($0) })])
            } catch {
                throw ToolError.ioFailed(String(describing: error))
            }
        default:
            return .object(["error": .string("Unknown tool: \(name)")])
        }
    }

    // Helpers
    private func resolvePath(_ relative: String) throws -> URL {
        let candidate = rootURL.appendingPathComponent(relative)
        let standardized = candidate.standardized
        // Prevent escaping from root
        guard standardized.path.hasPrefix(rootURL.standardized.path) else { throw ToolError.invalidPath }
        return standardized
    }

    private func argString(_ args: JSONValue, key: String) -> String? {
        if case .object(let dict) = args, case .string(let s)? = dict[key] { return s }
        return nil
    }
    private func argInt(_ args: JSONValue, key: String) -> Int? {
        if case .object(let dict) = args, let v = dict[key] {
            switch v {
            case .number(let d): return Int(d)
            case .string(let s): return Int(s)
            default: return nil
            }
        }
        return nil
    }
}

