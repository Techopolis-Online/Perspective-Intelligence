import Foundation
import Network
import OSLog

actor LocalHTTPServer {
    static let shared = LocalHTTPServer()

    private let logger = Logger(subsystem: "com.example.PerspectiveIntelligence", category: "LocalHTTPServer")
    private var listener: NWListener?
    private var connections: Set<ConnectionWrapper> = []

    var port: UInt16 = 11434
    private(set) var isRunning: Bool = false

    private init() {}

    // MARK: Lifecycle

    func start() async {
        print("Server started.")
        guard !isRunning else { return }
        let currentPort = self.port
        do {
            let params = NWParameters.tcp
            listener = try NWListener(using: params, on: NWEndpoint.Port(rawValue: currentPort)!)
            listener?.stateUpdateHandler = { [weak self] state in
                guard let self else { return }
                Task { await self.handleListenerState(state, currentPort: currentPort) }
            }
            listener?.newConnectionHandler = { [weak self] newConn in
                guard let self else { return }
                Task { await self.accept(newConn) }
            }
            listener?.start(queue: DispatchQueue.global())
        } catch {
            logger.error("Failed to start listener: \(String(describing: error))")
        }
    }

    func stop() async {
        listener?.cancel()
        listener = nil
        connections.forEach { $0.cancel() }
        connections.removeAll()
        isRunning = false
    }

    // MARK: Request handling

    fileprivate func handleRequest(_ request: HTTPRequest) async -> ServerResponse {
        // Correlate logs for this request
        let rid = String(UUID().uuidString.prefix(8))
        // CORS preflight support
        if request.method == "OPTIONS" {
            return .normal(HTTPResponse(status: 204, headers: [
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS, HEAD",
                "Access-Control-Allow-Headers": "Content-Type, Authorization, Accept",
                "Access-Control-Max-Age": "600"
            ], body: Data()))
        }

        // Normalize path: strip query string and trailing slash
        let basePath: String = {
            if let q = request.path.firstIndex(of: "?") { return String(request.path[..<q]) }
            return request.path
        }()
        let path: String = {
            if basePath.count > 1 && basePath.hasSuffix("/") { return String(basePath.dropLast()) }
            return basePath
        }()

        // Basic request logging for troubleshooting
    let contentType = request.headers["content-type"] ?? request.headers["Content-Type"] ?? ""
    let contentLength = request.headers["content-length"] ?? request.headers["Content-Length"] ?? ""
        logger.log("[req:\(rid, privacy: .public)] HTTP \(request.method, privacy: .public) \(path, privacy: .public) ct=\(contentType, privacy: .public) cl=\(contentLength, privacy: .public)")
        if request.method == "POST" {
            logger.log("[req:\(rid, privacy: .public)] body: \(self.truncateBodyForLog(request.bodyData), privacy: .public)")
        }

        // Route GET /v1/models (list)
        if (request.method == "GET" || request.method == "HEAD") && path == "/v1/models" {
            do {
                let models = FoundationModelsService.shared.listModels()
                let data = try JSONEncoder().encode(models)
                let resp = HTTPResponse(status: 200, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data)
                if request.method == "HEAD" { return .normal(HTTPResponse(status: resp.status, headers: resp.headers, body: Data())) }
                return .normal(resp)
            } catch {
                logger.error("[req:\(rid, privacy: .public)] /v1/models error: \(String(describing: error), privacy: .public)")
                let msg = ["error": ["message": error.localizedDescription]]
                let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                return .normal(HTTPResponse(status: 400, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data ?? Data()))
            }
        }

        // Debug: GET /debug/health -> simple health check
        if (request.method == "GET" || request.method == "HEAD") && path == "/debug/health" {
            let obj: [String: Any] = [
                "status": "ok",
                "running": self.isRunning,
                "port": self.port,
            ]
            let data = (try? JSONSerialization.data(withJSONObject: obj, options: [])) ?? Data()
            let resp = HTTPResponse(status: 200, headers: [
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            ], body: data)
            if request.method == "HEAD" { return .normal(HTTPResponse(status: resp.status, headers: resp.headers, body: Data())) }
            return .normal(resp)
        }

        // Debug: POST /debug/echo -> echoes method, path, headers, and body
        if request.method == "POST" && path == "/debug/echo" {
            var payload: [String: Any] = [:]
            payload["method"] = request.method
            payload["path"] = request.path
            payload["headers"] = request.headers
            if let bodyStr = String(data: request.bodyData, encoding: .utf8) {
                payload["bodyUtf8"] = bodyStr
            } else {
                payload["bodyBytes"] = request.bodyData.count
            }
            let data = (try? JSONSerialization.data(withJSONObject: payload, options: [.prettyPrinted])) ?? Data()
            return .normal(HTTPResponse(status: 200, headers: [
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            ], body: data))
        }

        // Basic index for root, to satisfy HEAD/GET / pings from clients.
        if (request.method == "GET" || request.method == "HEAD") && path == "/" {
            let endpoints: [String] = [
                "/v1/models",
                "/v1/chat/completions",
                "/v1/completions",
                "/api/generate",
                "/api/tags",
                "/api/version",
                "/api/ps",
                "/api/chat",
                "/debug/health",
                "/debug/echo"
            ]
            let obj: [String: Any] = [
                "name": "Perspective Intelligence Local API",
                "endpoints": endpoints
            ]
            let data = (try? JSONSerialization.data(withJSONObject: obj, options: [.prettyPrinted])) ?? Data()
            let resp = HTTPResponse(status: 200, headers: [
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            ], body: data)
            if request.method == "HEAD" { return .normal(HTTPResponse(status: resp.status, headers: resp.headers, body: Data())) }
            return .normal(resp)
        }

        // Mirror GET /api/models (list)
        if (request.method == "GET" || request.method == "HEAD") && path == "/api/models" {
            do {
                let models = FoundationModelsService.shared.listModels()
                let data = try JSONEncoder().encode(models)
                let resp = HTTPResponse(status: 200, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data)
                if request.method == "HEAD" { return .normal(HTTPResponse(status: resp.status, headers: resp.headers, body: Data())) }
                return .normal(resp)
            } catch {
                logger.error("[req:\(rid, privacy: .public)] /api/models error: \(String(describing: error), privacy: .public)")
                let msg = ["error": ["message": error.localizedDescription]]
                let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                return .normal(HTTPResponse(status: 400, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data ?? Data()))
            }
        }

        // Ollama-compatible: GET /api/tags (list models)
        if (request.method == "GET" || request.method == "HEAD") && path == "/api/tags" {
            do {
                let tags = FoundationModelsService.shared.listOllamaTags()
                let data = try JSONEncoder().encode(tags)
                let resp = HTTPResponse(status: 200, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data)
                if request.method == "HEAD" { return .normal(HTTPResponse(status: resp.status, headers: resp.headers, body: Data())) }
                return .normal(resp)
            } catch {
                logger.error("[req:\(rid, privacy: .public)] /api/tags error: \(String(describing: error), privacy: .public)")
                let msg = ["error": ["message": error.localizedDescription]]
                let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                return .normal(HTTPResponse(status: 400, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data ?? Data()))
            }
        }

        // Ollama-compatible: GET /api/version
        if (request.method == "GET" || request.method == "HEAD") && path == "/api/version" {
            let obj = ["version": "0.1.0"] // minimal version string; adjust as needed
            let data = (try? JSONSerialization.data(withJSONObject: obj, options: [])) ?? Data()
            let resp = HTTPResponse(status: 200, headers: [
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            ], body: data)
            if request.method == "HEAD" { return .normal(HTTPResponse(status: resp.status, headers: resp.headers, body: Data())) }
            return .normal(resp)
        }

        // Ollama-compatible: GET /api/ps (list running models) – we don't manage sessions, so return empty
        if (request.method == "GET" || request.method == "HEAD") && path == "/api/ps" {
            let data = Data("{\"models\": []}".utf8)
            let resp = HTTPResponse(status: 200, headers: [
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"
            ], body: data)
            if request.method == "HEAD" { return .normal(HTTPResponse(status: resp.status, headers: resp.headers, body: Data())) }
            return .normal(resp)
        }

        // Ollama-compatible: POST /api/chat (non-streaming)
        if request.method == "POST" && path == "/api/chat" {
            do {
                let decoder = JSONDecoder()
                let req = try decoder.decode(FoundationModelsService.OllamaChatRequest.self, from: request.bodyData)
                let respObj = try await FoundationModelsService.shared.handleOllamaChat(req)
                let data = try JSONEncoder().encode(respObj)
                logger.log("[req:\(rid, privacy: .public)] /api/chat ok model=\(respObj.model, privacy: .public) msgLen=\(respObj.message.content.count)")
                return .normal(HTTPResponse(status: 200, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data))
            } catch {
                logger.error("[req:\(rid, privacy: .public)] /api/chat error: \(String(describing: error), privacy: .public) body=\(self.truncateBodyForLog(request.bodyData), privacy: .public)")
                let msg = ["error": ["message": error.localizedDescription]]
                let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                return .normal(HTTPResponse(status: 400, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data ?? Data()))
            }
        }

        // Route GET /v1/models/{id}
        if (request.method == "GET" || request.method == "HEAD") && path.hasPrefix("/v1/models/") {
            let id = String(path.dropFirst("/v1/models/".count))
            if let model = FoundationModelsService.shared.getModel(id: id) {
                do {
                    let data = try JSONEncoder().encode(model)
                    let resp = HTTPResponse(status: 200, headers: [
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    ], body: data)
                    if request.method == "HEAD" { return .normal(HTTPResponse(status: resp.status, headers: resp.headers, body: Data())) }
                    return .normal(resp)
                } catch {
                    logger.error("[req:\(rid, privacy: .public)] /v1/models/{id} encode error: \(String(describing: error), privacy: .public)")
                    let msg = ["error": ["message": error.localizedDescription]]
                    let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                    return .normal(HTTPResponse(status: 400, headers: [
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    ], body: data ?? Data()))
                }
            } else {
                logger.error("[req:\(rid, privacy: .public)] /v1/models/{id} not found")
                let msg = [
                    "error": [
                        "message": "Model not found",
                        "type": "invalid_request_error"
                    ]
                ]
                let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                return .normal(HTTPResponse(status: 404, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data ?? Data()))
            }
        }

        // Mirror GET /api/models/{id}
        if (request.method == "GET" || request.method == "HEAD") && path.hasPrefix("/api/models/") {
            let id = String(path.dropFirst("/api/models/".count))
            if let model = FoundationModelsService.shared.getModel(id: id) {
                do {
                    let data = try JSONEncoder().encode(model)
                    let resp = HTTPResponse(status: 200, headers: [
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    ], body: data)
                    if request.method == "HEAD" { return .normal(HTTPResponse(status: resp.status, headers: resp.headers, body: Data())) }
                    return .normal(resp)
                } catch {
                    logger.error("[req:\(rid, privacy: .public)] /api/models/{id} encode error: \(String(describing: error), privacy: .public)")
                    let msg = ["error": ["message": error.localizedDescription]]
                    let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                    return .normal(HTTPResponse(status: 400, headers: [
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    ], body: data ?? Data()))
                }
            } else {
                logger.error("[req:\(rid, privacy: .public)] /api/models/{id} not found")
                let msg = [
                    "error": [
                        "message": "Model not found",
                        "type": "invalid_request_error"
                    ]
                ]
                let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                return .normal(HTTPResponse(status: 404, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data ?? Data()))
            }
        }

        // Route POST /v1/completions (text completions)
        if request.method == "POST" && path == "/v1/completions" {
            do {
                let decoder = JSONDecoder()
                let req = try decoder.decode(TextCompletionRequest.self, from: request.bodyData)
                if req.stream == true {
                    // Simulate streaming via SSE with small text chunks
                    return .stream(HTTPStreamResponse.sse(handler: { emitter in
                        let resp = try await FoundationModelsService.shared.handleCompletion(req)
                        let full = resp.choices.first?.text ?? ""
                        self.logger.log("[req:\(rid, privacy: .public)] /v1/completions streaming text len=\(full.count)")
                        for chunk in StreamChunker.chunk(text: full) {
                            let event: [String: Any] = [
                                "id": resp.id,
                                "object": "text_completion.chunk",
                                "created": resp.created,
                                "model": resp.model,
                                "choices": [["text": chunk, "index": 0, "finish_reason": NSNull()]]
                            ]
                            try await emitter.emitSSE(json: event)
                        }
                        // Final event
                        try await emitter.emitSSE(raw: "[DONE]")
                    }))
                } else {
                    let resp = try await FoundationModelsService.shared.handleCompletion(req)
                    let data = try JSONEncoder().encode(resp)
                    logger.log("[req:\(rid, privacy: .public)] /v1/completions ok textLen=\(resp.choices.first?.text.count ?? 0)")
                    return .normal(HTTPResponse(status: 200, headers: [
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    ], body: data))
                }
            } catch {
                logger.error("[req:\(rid, privacy: .public)] /v1/completions error: \(String(describing: error), privacy: .public) body=\(self.truncateBodyForLog(request.bodyData), privacy: .public)")
                let msg = ["error": ["message": error.localizedDescription]]
                let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                return .normal(HTTPResponse(status: 400, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data ?? Data()))
            }
        }

        // Mirror POST /api/generate to the same text completions behavior
        if request.method == "POST" && path == "/api/generate" {
            do {
                let decoder = JSONDecoder()
                let req = try decoder.decode(TextCompletionRequest.self, from: request.bodyData)
                if req.stream == true {
                    // Ollama style NDJSON streaming with "response" chunks
                    return .stream(HTTPStreamResponse.ndjson(handler: { emitter in
                        let resp = try await FoundationModelsService.shared.handleCompletion(req)
                        let full = resp.choices.first?.text ?? ""
                        self.logger.log("[req:\(rid, privacy: .public)] /api/generate streaming text len=\(full.count)")
                        for chunk in StreamChunker.chunk(text: full) {
                            let event: [String: Any] = [
                                "model": resp.model,
                                "created_at": ISO8601DateFormatter().string(from: Date()),
                                "response": chunk,
                                "done": false
                            ]
                            try await emitter.emitNDJSON(json: event)
                        }
                        let final: [String: Any] = [
                            "model": resp.model,
                            "created_at": ISO8601DateFormatter().string(from: Date()),
                            "done": true
                        ]
                        try await emitter.emitNDJSON(json: final)
                    }))
                } else {
                    let resp = try await FoundationModelsService.shared.handleCompletion(req)
                    let data = try JSONEncoder().encode(resp)
                    logger.log("[req:\(rid, privacy: .public)] /api/generate ok textLen=\(resp.choices.first?.text.count ?? 0)")
                    return .normal(HTTPResponse(status: 200, headers: [
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    ], body: data))
                }
            } catch {
                logger.error("[req:\(rid, privacy: .public)] /api/generate error: \(String(describing: error), privacy: .public) body=\(self.truncateBodyForLog(request.bodyData), privacy: .public)")
                let msg = ["error": ["message": error.localizedDescription]]
                let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                return .normal(HTTPResponse(status: 400, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data ?? Data()))
            }
        }

        // Route only POST /v1/chat/completions
        if request.method == "POST" && path == "/v1/chat/completions" {
            do {
                let decoder = JSONDecoder()
                let req = try decoder.decode(ChatCompletionRequest.self, from: request.bodyData)
                if req.stream == true {
                    return .stream(HTTPStreamResponse.sse(handler: { emitter in
                        let useMulti = (req.multi_segment ?? true) == true
                        if useMulti {
                            // Stream multi-segment output piecewise
                            let streamId = "chatcmpl_" + UUID().uuidString.replacingOccurrences(of: "-", with: "")
                            let created = Int(Date().timeIntervalSince1970)
                            self.logger.log("[req:\(rid, privacy: .public)] /v1/chat/completions streaming mode=multi segmentChars=1400 maxSegments=6")
                            do {
                                try await FoundationModelsService.shared.generateChatSegments(messages: req.messages, model: req.model, temperature: req.temperature, segmentChars: 1400, maxSegments: 6) { segment in
                                    let event: [String: Any] = [
                                        "id": streamId,
                                        "object": "chat.completion.chunk",
                                        "created": created,
                                        "model": req.model,
                                        "choices": [["index": 0, "delta": ["content": segment]]]
                                    ]
                                    try? await emitter.emitSSE(json: event)
                                }
                                // Send terminal chunk with finish_reason before closing
                                let finalEvent: [String: Any] = [
                                    "id": streamId,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": req.model,
                                    "choices": [["index": 0, "delta": [:], "finish_reason": "stop"]]
                                ]
                                try? await emitter.emitSSE(json: finalEvent)
                                try? await emitter.emitSSE(raw: "[DONE]")
                            } catch {
                                // Safety guardrails or other FM errors: emit a friendly fallback and end stream
                                self.logger.error("[req:\(rid, privacy: .public)] multi-segment generation failed: \(String(describing: error), privacy: .public)")
                                let fallback = "(Local fallback) Unable to continue the streamed response right now. This may be due to safety guardrails or an unavailable on-device model. Please try rephrasing or reducing sensitive content."
                                let event: [String: Any] = [
                                    "id": streamId,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": req.model,
                                    "choices": [["index": 0, "delta": ["content": fallback]]]
                                ]
                                try? await emitter.emitSSE(json: event)
                                let finalEvent: [String: Any] = [
                                    "id": streamId,
                                    "object": "chat.completion.chunk",
                                    "created": created,
                                    "model": req.model,
                                    "choices": [["index": 0, "delta": [:], "finish_reason": "stop"]]
                                ]
                                try? await emitter.emitSSE(json: finalEvent)
                                try? await emitter.emitSSE(raw: "[DONE]")
                            }
                        } else {
                            let resp = try await FoundationModelsService.shared.handleChatCompletion(req)
                            let full = resp.choices.first?.message.content ?? ""
                            self.logger.log("[req:\(rid, privacy: .public)] /v1/chat/completions streaming msg len=\(full.count)")
                            for chunk in StreamChunker.chunk(text: full) {
                                let event: [String: Any] = [
                                    "id": resp.id,
                                    "object": "chat.completion.chunk",
                                    "created": resp.created,
                                    "model": resp.model,
                                    "choices": [["index": 0, "delta": ["content": chunk]]]
                                ]
                                try await emitter.emitSSE(json: event)
                            }
                            // Final terminal chunk with finish_reason
                            let finalEvent: [String: Any] = [
                                "id": resp.id,
                                "object": "chat.completion.chunk",
                                "created": resp.created,
                                "model": resp.model,
                                "choices": [["index": 0, "delta": [:], "finish_reason": "stop"]]
                            ]
                            try await emitter.emitSSE(json: finalEvent)
                            try await emitter.emitSSE(raw: "[DONE]")
                        }
                    }))
                } else {
                    let resp = try await FoundationModelsService.shared.handleChatCompletion(req)
                    let data = try JSONEncoder().encode(resp)
                    logger.log("[req:\(rid, privacy: .public)] /v1/chat/completions ok msgLen=\(resp.choices.first?.message.content.count ?? 0)")
                    return .normal(HTTPResponse(status: 200, headers: [
                        "Content-Type": "application/json",
                        "Access-Control-Allow-Origin": "*"
                    ], body: data))
                }
            } catch {
                logger.error("[req:\(rid, privacy: .public)] /v1/chat/completions error: \(String(describing: error), privacy: .public) body=\(self.truncateBodyForLog(request.bodyData), privacy: .public)")
                let msg = ["error": ["message": error.localizedDescription]]
                let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                return .normal(HTTPResponse(status: 400, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data ?? Data()))
            }
        }
        // Not Found
        logger.error("[req:\(rid, privacy: .public)] 404 Not Found \(path, privacy: .public)")
        let body = Data("Not Found".utf8)
        return .normal(HTTPResponse(status: 404, headers: [
            "Content-Type": "text/plain",
            "Access-Control-Allow-Origin": "*"
        ], body: body))
    }

    // MARK: - Actor-isolated helpers

    private func handleListenerState(_ state: NWListener.State, currentPort: UInt16) async {
        switch state {
        case .ready:
            logger.log("HTTP server listening on 0.0.0.0:\(currentPort)")
            isRunning = true
        case .failed(let error):
            logger.error("Listener failed: \(String(describing: error))")
            isRunning = false
        case .cancelled:
            logger.log("Listener cancelled")
            isRunning = false
        default:
            break
        }
    }

    private func accept(_ newConn: NWConnection) async {
        let wrapper = ConnectionWrapper(connection: newConn, server: self)
        connections.insert(wrapper)
        wrapper.start()
    }

    // Public actor APIs for cross-actor access
    func setPort(_ newPort: UInt16) {
        self.port = newPort
    }

    func getIsRunning() -> Bool {
        isRunning
    }

    func getPort() -> UInt16 {
        port
    }
}

// MARK: - Logging helpers

extension LocalHTTPServer {
    /// Whether to log full request bodies without truncation.
    /// Enabled if either:
    /// - UserDefaults: `debugFullRequestLog` or `debugLogging` is true, or
    /// - Env var `PI_DEBUG_FULL_LOG=1` is present.
    private func debugFullRequestLogEnabled() -> Bool {
        let defaults = UserDefaults.standard
        if defaults.bool(forKey: "debugFullRequestLog") { return true }
        if defaults.bool(forKey: "debugLogging") { return true }
        if ProcessInfo.processInfo.environment["PI_DEBUG_FULL_LOG"] == "1" { return true }
        return false
    }

    fileprivate func truncateBodyForLog(_ data: Data, limit: Int = 8192) -> String {
        guard let s = String(data: data, encoding: .utf8) else { return "<non-utf8 body \(data.count) bytes>" }
        let redacted = redactAuthorization(in: s)
        if debugFullRequestLogEnabled() { return redacted.replacingOccurrences(of: "\n", with: "\\n") }
        if redacted.count <= limit { return redacted.replacingOccurrences(of: "\n", with: "\\n") }
        let idx = redacted.index(redacted.startIndex, offsetBy: limit)
        return redacted[redacted.startIndex..<idx].replacingOccurrences(of: "\n", with: "\\n") + "… (truncated)"
    }

    private func redactAuthorization(in s: String) -> String {
        if s.lowercased().contains("authorization") {
            // Very simple masking for tokens in body if present
            return s.replacingOccurrences(of: "Authorization", with: "Authorization(REDACTED)")
        }
        return s
    }
}

// MARK: - Minimal HTTP over TCP

private final class ConnectionWrapper: @unchecked Sendable, Hashable {
    static func == (lhs: ConnectionWrapper, rhs: ConnectionWrapper) -> Bool { lhs === rhs }
    func hash(into hasher: inout Hasher) { hasher.combine(ObjectIdentifier(self)) }

    private let logger = Logger(subsystem: "com.example.PerspectiveIntelligence", category: "Connection")
    private let connection: NWConnection
    private unowned let server: LocalHTTPServer
    private var buffer = Data()
    private var connectionEnded = false
    private var didCancel = false

    init(connection: NWConnection, server: LocalHTTPServer) {
        self.connection = connection
        self.server = server
    }

    func start() {
        connection.stateUpdateHandler = { [weak self] state in
            guard let strongSelf = self else { return }
            switch state {
            case .ready:
                strongSelf.receive()
            case .failed, .cancelled:
                strongSelf.cancel()
            default: break
            }
        }
        connection.start(queue: DispatchQueue.global())
    }

    func cancel() {
        guard !didCancel else { return }
        didCancel = true
        connection.cancel()
    }

    private func receive() {
        connection.receive(minimumIncompleteLength: 1, maximumLength: 64 * 1024) { [weak self] data, _, isComplete, error in
            guard let strongSelf = self else { return }
            if let data, !data.isEmpty { strongSelf.buffer.append(data) }
            strongSelf.connectionEnded = strongSelf.connectionEnded || isComplete || (error != nil)

            if let request = strongSelf.tryParseRequest() {
                Task {
                    let response = await strongSelf.server.handleRequest(request)
                    strongSelf.sendServerResponse(response)
                }
                return
            }
            if strongSelf.connectionEnded {
                // If connection ended but we couldn't parse a full request, return Bad Request
                strongSelf.logger.error("Failed to parse full HTTP request before connection ended")
                strongSelf.send(HTTPResponse(status: 400, headers: ["Content-Type": "text/plain"], body: Data("Bad Request".utf8)))
                return
            }
            strongSelf.receive()
        }
    }

    private func send(_ response: HTTPResponse) {
        let data = response.serialize()
        connection.send(content: data, completion: .contentProcessed { [weak self] _ in
            self?.cancel()
        })
    }

    private func sendServerResponse(_ response: ServerResponse) {
        switch response {
        case .normal(let resp):
            send(resp)
        case .stream(let s):
            sendStream(s)
        }
    }

    private func sendStream(_ s: HTTPStreamResponse) {
        // Prepare headers for chunked transfer
        var lines: [String] = []
        lines.append("HTTP/1.1 200 OK")
        var headers = s.headers
        headers["Transfer-Encoding"] = "chunked"
        headers["Connection"] = "close"
        for (k, v) in headers { lines.append("\(k): \(v)") }
        lines.append("")
        let head = (lines.joined(separator: "\r\n") + "\r\n").data(using: .utf8) ?? Data()
        connection.send(content: head, completion: .contentProcessed { [weak self] _ in
            guard let self = self else { return }
            Task {
                let sender = StreamSender(connection: self.connection)
                await s.run { chunk in
                    await sender.sendChunked(chunk)
                }
                await sender.finish()
            }
        })
    }

    // Actor that owns writes to the NWConnection during a streaming response
    private actor StreamSender {
        private let connection: NWConnection

        init(connection: NWConnection) {
            self.connection = connection
        }

        func sendChunked(_ data: Data) {
            let prefix = String(format: "%@\r\n", String(data.count, radix: 16)).data(using: .utf8) ?? Data()
            var out = Data()
            out.append(prefix)
            out.append(data)
            out.append(Data("\r\n".utf8))
            connection.send(content: out, completion: .contentProcessed({ _ in }))
        }

        func finish() {
            connection.send(content: Data("0\r\n\r\n".utf8), completion: .contentProcessed({ _ in
                self.connection.cancel()
            }))
        }
    }

    // MARK: - Buffered HTTP parsing

    private func tryParseRequest() -> HTTPRequest? {
        // Look for end of headers \r\n\r\n
        guard let headerRange = buffer.range(of: Data([13,10,13,10])) else { // \r\n\r\n
            return nil
        }
        let headData = buffer.subdata(in: 0..<headerRange.lowerBound)
        guard let headText = String(data: headData, encoding: .utf8) else { return nil }
        let headLines = headText.components(separatedBy: "\r\n")
        guard let requestLine = headLines.first else { return nil }
        let comps = requestLine.split(separator: " ")
        guard comps.count >= 2 else { return nil }
        let method = String(comps[0])
        let path = String(comps[1])
        var headers: [String: String] = [:]
        for line in headLines.dropFirst() {
            if let sep = line.firstIndex(of: ":") {
                let key = String(line[..<sep]).trimmingCharacters(in: .whitespaces).lowercased()
                let value = String(line[line.index(after: sep)...]).trimmingCharacters(in: .whitespaces)
                headers[key] = value
            }
        }
        let bodyStart = headerRange.upperBound
        let contentLength = (headers["content-length"] ?? headers["Content-Length"]).flatMap { Int($0) }
        let availableBodyBytes = buffer.count - bodyStart
        let expectedBodyBytes = contentLength ?? availableBodyBytes
        guard availableBodyBytes >= expectedBodyBytes else {
            // Need more data
            return nil
        }
        let bodyData = buffer.subdata(in: bodyStart..<(bodyStart + expectedBodyBytes))
        // Consume used bytes from buffer
        if bodyStart + expectedBodyBytes <= buffer.count {
            buffer.removeSubrange(0..<(bodyStart + expectedBodyBytes))
        }
        return HTTPRequest(method: method, path: path, headers: headers, bodyData: bodyData)
    }
}

// MARK: - HTTP Types

struct HTTPRequest {
    let method: String
    let path: String
    let headers: [String: String]
    let bodyData: Data
}

struct HTTPResponse {
    let status: Int
    let headers: [String: String]
    let body: Data

    func serialize() -> Data {
        var lines: [String] = []
        lines.append("HTTP/1.1 \(status) \(statusText(status))")
        lines.append("Content-Length: \(body.count)")
        for (k, v) in headers { lines.append("\(k): \(v)") }
        lines.append("")
        let head = lines.joined(separator: "\r\n") + "\r\n"
        var data = Data(head.utf8)
        data.append(body)
        return data
    }

    private func statusText(_ code: Int) -> String {
        switch code {
        case 200: return "OK"
        case 400: return "Bad Request"
        case 404: return "Not Found"
        case 204: return "No Content"
        default: return "OK"
        }
    }
}

// MARK: - Streaming Support

enum ServerResponse {
    case normal(HTTPResponse)
    case stream(HTTPStreamResponse)
}

final class HTTPStreamResponse: @unchecked Sendable {
    typealias Emitter = StreamingEmitter

    let headers: [String: String]
    private let runner: (Emitter) async -> Void

    init(headers: [String: String], runner: @escaping (Emitter) async -> Void) {
        self.headers = headers
        self.runner = runner
    }

    static func sse(handler: @escaping (Emitter) async throws -> Void) -> HTTPStreamResponse {
        return HTTPStreamResponse(headers: [
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        ]) { emitter in
            try? await handler(emitter)
        }
    }

    static func ndjson(handler: @escaping (Emitter) async throws -> Void) -> HTTPStreamResponse {
        return HTTPStreamResponse(headers: [
            "Content-Type": "application/x-ndjson",
            "Cache-Control": "no-cache",
            "Access-Control-Allow-Origin": "*"
        ]) { emitter in
            try? await handler(emitter)
        }
    }

    func run(emit: @escaping @Sendable (Data) async -> Void) async {
        let emitter = StreamingEmitter(emit: emit)
        await runner(emitter)
    }

    struct StreamingEmitter {
        let emit: @Sendable (Data) async -> Void

        func emitSSE(raw: String) async throws {
            let line = "data: \(raw)\n\n"
            guard let data = line.data(using: .utf8) else { return }
            await emit(data)
        }

        func emitSSE(json: [String: Any]) async throws {
            let data = try JSONSerialization.data(withJSONObject: json, options: [])
            if let str = String(data: data, encoding: .utf8) {
                try await emitSSE(raw: str)
            }
        }

        func emitNDJSON(json: [String: Any]) async throws {
            let data = try JSONSerialization.data(withJSONObject: json, options: [])
            var nd = data
            nd.append(0x0A) // newline
            await emit(nd)
        }
    }
}

enum StreamChunker {
    static func chunk(text: String, size: Int = 64) -> [String] {
        guard !text.isEmpty else { return [] }
        var chunks: [String] = []
        var idx = text.startIndex
        while idx < text.endIndex {
            let end = text.index(idx, offsetBy: size, limitedBy: text.endIndex) ?? text.endIndex
            chunks.append(String(text[idx..<end]))
            idx = end
        }
        return chunks
    }
}

enum HTTPRequestParser {
    static func parse(data: Data) -> HTTPRequest? {
        guard let text = String(data: data, encoding: .utf8) else { return nil }
        let parts = text.components(separatedBy: "\r\n\r\n")
        guard parts.count >= 1 else { return nil }
        let head = parts[0]
        let bodyString = parts.dropFirst().joined(separator: "\r\n\r\n")
        let headLines = head.components(separatedBy: "\r\n")
        guard let requestLine = headLines.first else { return nil }
        let comps = requestLine.split(separator: " ")
        guard comps.count >= 2 else { return nil }
        let method = String(comps[0])
        let path = String(comps[1])
        var headers: [String: String] = [:]
        for line in headLines.dropFirst() {
            if let sep = line.firstIndex(of: ":") {
                let rawKey = String(line[..<sep]).trimmingCharacters(in: .whitespaces)
                let key = rawKey.lowercased()
                let value = String(line[line.index(after: sep)...]).trimmingCharacters(in: .whitespaces)
                headers[key] = value
            }
        }
        let bodyData = Data(bodyString.utf8)
        return HTTPRequest(method: method, path: path, headers: headers, bodyData: bodyData)
    }
}

