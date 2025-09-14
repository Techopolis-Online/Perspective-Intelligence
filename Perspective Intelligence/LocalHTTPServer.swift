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

    fileprivate func handleRequest(_ request: HTTPRequest) async -> HTTPResponse {
        // CORS preflight support
        if request.method == "OPTIONS" {
            return HTTPResponse(status: 204, headers: [
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type, Authorization",
                "Access-Control-Max-Age": "600"
            ], body: Data())
        }

        // Route only POST /v1/chat/completions
        if request.method == "POST" && request.path == "/v1/chat/completions" {
            do {
                let decoder = JSONDecoder()
                let req = try decoder.decode(ChatCompletionRequest.self, from: request.bodyData)
                let resp = try await FoundationModelsService.shared.handleChatCompletion(req)
                let data = try JSONEncoder().encode(resp)
                return HTTPResponse(status: 200, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data)
            } catch {
                let msg = ["error": ["message": error.localizedDescription]]
                let data = try? JSONSerialization.data(withJSONObject: msg, options: [])
                return HTTPResponse(status: 400, headers: [
                    "Content-Type": "application/json",
                    "Access-Control-Allow-Origin": "*"
                ], body: data ?? Data())
            }
        }
        // Not Found
        let body = Data("Not Found".utf8)
        return HTTPResponse(status: 404, headers: [
            "Content-Type": "text/plain",
            "Access-Control-Allow-Origin": "*"
        ], body: body)
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

// MARK: - Minimal HTTP over TCP

private final class ConnectionWrapper: @unchecked Sendable, Hashable {
    static func == (lhs: ConnectionWrapper, rhs: ConnectionWrapper) -> Bool { lhs === rhs }
    func hash(into hasher: inout Hasher) { hasher.combine(ObjectIdentifier(self)) }

    private let logger = Logger(subsystem: "com.example.PerspectiveIntelligence", category: "Connection")
    private let connection: NWConnection
    private unowned let server: LocalHTTPServer
    private var buffer = Data()
    private var connectionEnded = false

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
                    strongSelf.send(response)
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
            self?.connection.cancel()
        })
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
                let key = String(line[..<sep]).trimmingCharacters(in: .whitespaces)
                let value = String(line[line.index(after: sep)...]).trimmingCharacters(in: .whitespaces)
                headers[key] = value
            }
        }
        let bodyStart = headerRange.upperBound
        let contentLength = headers["Content-Length"].flatMap { Int($0) }
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
                let key = String(line[..<sep]).trimmingCharacters(in: .whitespaces)
                let value = String(line[line.index(after: sep)...]).trimmingCharacters(in: .whitespaces)
                headers[key] = value
            }
        }
        let bodyData = Data(bodyString.utf8)
        return HTTPRequest(method: method, path: path, headers: headers, bodyData: bodyData)
    }
}

