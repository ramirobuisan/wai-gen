import Foundation
import Network

/// One WebSocket (Network.framework) carrying `RemoteMessage` JSON in text frames and images in binary frames.
final class WSChannel: @unchecked Sendable {
    let connection: NWConnection
    private let queue = DispatchQueue(label: "waigen.remote.channel")

    enum Frame {
        case message(RemoteMessage)
        case binary(Data)
    }

    static func parameters() -> NWParameters {
        let params = NWParameters.tcp
        let ws = NWProtocolWebSocket.Options()
        ws.autoReplyPing = true
        ws.maximumMessageSize = RemoteProtocol.maxMessageBytes
        params.defaultProtocolStack.applicationProtocols.insert(ws, at: 0)
        return params
    }

    init(_ connection: NWConnection) { self.connection = connection }

    static let localNetworkHint = "macOS is blocking WAI Gen from the local network: turn WAI Gen on in System Settings → "
        + "Privacy & Security → Local Network (or use the other Mac's Tailscale address)."

    /// Starts the connection and waits until it's ready (or fails, or `timeout` passes). A connection that is
    /// "waiting" (e.g. while macOS asks for Local Network access the first time) is given until the timeout.
    func open(timeout: TimeInterval = 15) async throws {
        let once = Once()
        let lastWait = LockedBox<NWError?>(nil)
        try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
            let connection = connection
            func finish(_ r: Result<Void, Error>) {
                guard once.claim() else { return }
                if case .failure = r { connection.cancel() }   // or it keeps retrying in the background
                cont.resume(with: r)
            }
            connection.stateUpdateHandler = { state in
                if ProcessInfo.processInfo.environment["WAIGEN_REMOTE_DEBUG"] != nil {
                    FileHandle.standardError.write(Data("[channel \(connection.endpoint)] \(state)\n".utf8))
                }
                switch state {
                case .ready: finish(.success(()))
                case .failed(let e): finish(.failure(Self.describe(e, connection)))
                case .waiting(let e):
                    // keep waiting: macOS holds the first connection while it asks for Local Network access
                    // (and reports it as "denied" until the user answers), or the network may come back
                    lastWait.value = e
                case .cancelled: finish(.failure(RemoteError.connection("closed")))
                default: break
                }
            }
            connection.start(queue: queue)
            queue.asyncAfter(deadline: .now() + timeout) {
                finish(.failure(lastWait.value.map { Self.describe($0, connection) } ?? RemoteError.connection(
                    "no answer. Check that WAI Gen is open on the other Mac with “Let other Macs generate on this Mac” on, "
                    + "and that both Macs are on the same network or Tailscale.")))
            }
        }
    }

    /// An error with a hint when it's the Local Network permission (macOS reports it as "Network is down" /
    /// "No route to host" for addresses on the local network).
    static func describe(_ e: NWError, _ c: NWConnection) -> RemoteError {
        if #available(macOS 14.0, *), c.currentPath?.unsatisfiedReason == .localNetworkDenied {
            return .connection(localNetworkHint)
        }
        if case .posix(let code) = e, code == .ENETDOWN || code == .EHOSTUNREACH || code == .ENETUNREACH {
            return .connection("\(e.localizedDescription). " + localNetworkHint)
        }
        return .connection(e.localizedDescription)
    }

    func close() { connection.cancel() }

    func send(_ message: RemoteMessage) async throws {
        try await send(JSONEncoder().encode(message), opcode: .text)
    }

    func send(_ data: Data, opcode: NWProtocolWebSocket.Opcode) async throws {
        let meta = NWProtocolWebSocket.Metadata(opcode: opcode)
        let context = NWConnection.ContentContext(identifier: "waigen", metadata: [meta])
        try await withCheckedThrowingContinuation { (c: CheckedContinuation<Void, Error>) in
            connection.send(content: data, contentContext: context, isComplete: true, completion: .contentProcessed { error in
                if let error { c.resume(throwing: RemoteError.connection(error.localizedDescription)) } else { c.resume() }
            })
        }
    }

    /// The next text or binary frame (pings and pongs are skipped).
    func receive() async throws -> Frame {
        while true {
            let (data, opcode) = try await receiveRaw()
            switch opcode {
            case .text:
                do { return .message(try JSONDecoder().decode(RemoteMessage.self, from: data)) } catch {
                    throw RemoteError.invalid("unreadable message")
                }
            case .binary: return .binary(data)
            case .close: throw RemoteError.connection("closed by the other side")
            default: continue
            }
        }
    }

    private func receiveRaw() async throws -> (Data, NWProtocolWebSocket.Opcode) {
        try await withCheckedThrowingContinuation { c in
            connection.receiveMessage { data, context, _, error in
                if let error {
                    c.resume(throwing: RemoteError.connection(error.localizedDescription))
                    return
                }
                guard let meta = context?.protocolMetadata(definition: NWProtocolWebSocket.definition) as? NWProtocolWebSocket.Metadata
                else {
                    c.resume(throwing: RemoteError.connection("closed"))
                    return
                }
                c.resume(returning: (data ?? Data(), meta.opcode))
            }
        }
    }
}

final class LockedBox<T>: @unchecked Sendable {
    private let lock = NSLock()
    private var v: T
    init(_ v: T) { self.v = v }
    var value: T {
        get { lock.lock(); defer { lock.unlock() }; return v }
        set { lock.lock(); v = newValue; lock.unlock() }
    }
}

/// A thread-safe "first one wins" flag.
final class Once: @unchecked Sendable {
    private let lock = NSLock()
    private var done = false
    func claim() -> Bool {
        lock.lock()
        defer { lock.unlock() }
        if done { return false }
        done = true
        return true
    }
}

// MARK: - server

/// What the server needs from the app.
public protocol RemoteJobHandler: AnyObject, Sendable {
    func remoteInfo() async -> RemoteInfo
    /// Runs `job` (through the same queue as the app's own jobs) and returns the image as PNG.
    func runRemote(_ job: RemoteJob, id: UUID, client: String,
                   progress: @escaping @Sendable (RemoteProgress) -> Void,
                   preview: @escaping @Sendable (Data) -> Void) async throws -> (png: Data, seconds: Double, faceRepairSeconds: Double?)
    func cancelRemote(id: UUID)
}

/// Lets other Macs generate on this one: a WebSocket listener advertised with Bonjour, open only to
/// local-network and Tailscale addresses, and only after the pairing code.
public final class RemoteServer: @unchecked Sendable {
    public let port: UInt16
    public let code: String
    public let name: String
    private weak var handler: RemoteJobHandler?
    private var listener: NWListener?
    private let queue = DispatchQueue(label: "waigen.remote.server")
    /// "listening", "failed: ..." - for the settings window
    public var onStatus: (@Sendable (String) -> Void)?
    /// a client connected / finished a job (name, running)
    public var onActivity: (@Sendable (String, Bool) -> Void)?

    public init(handler: RemoteJobHandler, code: String, name: String, port: UInt16 = RemoteProtocol.defaultPort) {
        self.handler = handler
        self.code = code
        self.name = name
        self.port = port
    }

    public func start() throws {
        guard let p = NWEndpoint.Port(rawValue: port) else { throw RemoteError.invalid("port") }
        let l = try NWListener(using: WSChannel.parameters(), on: p)
        l.service = NWListener.Service(name: name, type: RemoteProtocol.serviceType)
        l.newConnectionHandler = { [weak self] c in self?.accept(c) }
        l.stateUpdateHandler = { [weak self] state in
            switch state {
            case .ready: self?.onStatus?("listening")
            case .failed(let e): self?.onStatus?("failed: \(e.localizedDescription)")
            default: break
            }
        }
        l.start(queue: queue)
        listener = l
    }

    public func stop() {
        listener?.cancel()
        listener = nil
    }

    private func accept(_ c: NWConnection) {
        guard Self.allowed(c.endpoint) else {
            log("refused \(c.endpoint)")
            c.cancel()
            return
        }
        let channel = WSChannel(c)
        Task { await self.serve(channel) }
    }

    /// WAIGEN_REMOTE_DEBUG=1: print what happens on each connection
    private let debug = ProcessInfo.processInfo.environment["WAIGEN_REMOTE_DEBUG"] != nil
    private func log(_ s: String) { if debug { FileHandle.standardError.write(Data("[server] \(s)\n".utf8)) } }

    private func serve(_ ch: WSChannel) async {
        defer { ch.close() }
        log("connection from \(ch.connection.endpoint)")
        do { try await ch.open() } catch {
            log("open failed: \(error)")
            return
        }
        log("open")
        // the pairing code must come first, within 10 s
        let authenticated = Once()
        queue.asyncAfter(deadline: .now() + 10) { if authenticated.claim() { ch.close() } }
        guard case .message(let hello)? = try? await ch.receive(), hello.type == .hello else { return }
        guard Self.same(hello.code ?? "", code), authenticated.claim() else {
            try? await Task.sleep(nanoseconds: 1_000_000_000)   // slows down guessing
            var e = RemoteMessage(.error)
            e.message = "Wrong pairing code"
            try? await ch.send(e)
            return
        }
        let client = hello.client ?? "another Mac"
        guard let handler else { return }
        var info = RemoteMessage(.info)
        info.info = await handler.remoteInfo()
        do { try await ch.send(info) } catch { return }

        var running: (id: UUID, task: Task<Void, Never>)?
        while true {
            guard let frame = try? await ch.receive() else { break }   // closed: cancel what it asked for
            guard case .message(let m) = frame else { continue }
            switch m.type {
            case .generate:
                guard running == nil, let job = m.job else { continue }
                let id = UUID()
                running = (id, Task {
                    onActivity?(client, true)
                    defer { onActivity?(client, false) }
                    do {
                        try job.validate()
                        let r = try await handler.runRemote(job, id: id, client: client, progress: { p in
                            var msg = RemoteMessage(.progress)
                            msg.progress = p
                            Task { try? await ch.send(msg) }
                        }, preview: { jpeg in
                            var msg = RemoteMessage(.preview)
                            msg.preview = jpeg
                            Task { try? await ch.send(msg) }
                        })
                        try await ch.send(r.png, opcode: .binary)
                        var done = RemoteMessage(.done)
                        done.seconds = r.seconds
                        done.faceRepairSeconds = r.faceRepairSeconds
                        try await ch.send(done)
                    } catch {
                        var e = RemoteMessage(.error)
                        e.message = error is CancellationError ? "cancelled" : error.localizedDescription
                        try? await ch.send(e)
                    }
                })
            case .cancel:
                if let r = running { handler.cancelRemote(id: r.id) }
            default:
                continue
            }
        }
        if let r = running {
            handler.cancelRemote(id: r.id)
            await r.task.value
        }
    }

    /// Constant-time comparison (no early exit on the first different character).
    static func same(_ a: String, _ b: String) -> Bool {
        let x = Array(a.uppercased().utf8), y = Array(b.uppercased().utf8)
        guard x.count == y.count else { return false }
        var diff: UInt8 = 0
        for i in 0 ..< x.count { diff |= x[i] ^ y[i] }
        return diff == 0
    }

    /// Loopback, private IPv4 (10/8, 172.16/12, 192.168/16), link-local, Tailscale (100.64/10, fd7a:115c:a1e0::/48)
    /// and other unique-local IPv6 addresses.
    static func allowed(_ endpoint: NWEndpoint) -> Bool {
        guard case .hostPort(let host, _) = endpoint else { return false }
        switch host {
        case .ipv4(let a): return allowedV4(Array(a.rawValue))
        case .ipv6(let a):
            let b = Array(a.rawValue)
            guard b.count == 16 else { return false }
            if b[0 ..< 15].allSatisfy({ $0 == 0 }) && b[15] == 1 { return true }               // ::1
            if b[0 ..< 10].allSatisfy({ $0 == 0 }) && b[10] == 0xFF && b[11] == 0xFF {          // ::ffff:a.b.c.d
                return allowedV4(Array(b[12 ..< 16]))
            }
            if b[0] == 0xFE && (b[1] & 0xC0) == 0x80 { return true }                             // fe80::/10
            if (b[0] & 0xFE) == 0xFC { return true }                                              // fc00::/7 (incl. Tailscale)
            return false
        default: return false
        }
    }

    static func allowedV4(_ b: [UInt8]) -> Bool {
        guard b.count == 4 else { return false }
        switch (b[0], b[1]) {
        case (127, _), (10, _): return true
        case (172, 16 ... 31), (192, 168), (169, 254): return true
        case (100, 64 ... 127): return true   // Tailscale (CGNAT range)
        default: return false
        }
    }

    /// A new random pairing code like "K7P-4QX" (no 0/O, 1/I).
    public static func newCode() -> String {
        let alphabet = Array("ABCDEFGHJKLMNPQRSTUVWXYZ23456789")
        var g = SystemRandomNumberGenerator()
        let chars = (0 ..< 6).map { _ in alphabet[Int.random(in: 0 ..< alphabet.count, using: &g)] }
        return String(chars[0 ..< 3]) + "-" + String(chars[3 ..< 6])
    }
}

// MARK: - client

public struct RemoteResult: Sendable {
    public let png: Data
    public let seconds: Double
    public let faceRepairSeconds: Double?
}

/// Talks to a generator (another Mac running WAI Gen with sharing on).
public struct RemoteClient: Sendable {
    public let endpoint: RemoteEndpoint
    public let code: String

    public init(endpoint: RemoteEndpoint, code: String) {
        self.endpoint = endpoint
        self.code = code
    }

    /// A Bonjour service name -> its current host and port (a plain TCP connection, closed right away).
    static func resolve(service: String) async throws -> (String, Int) {
        let c = NWConnection(to: .service(name: service, type: RemoteProtocol.serviceType, domain: "local.", interface: nil),
                             using: .tcp)
        let ch = WSChannel(c)
        try await ch.open()
        defer { c.cancel() }
        guard case .hostPort(let host, let port)? = c.currentPath?.remoteEndpoint else {
            throw RemoteError.connection("couldn't find \(service) on the network")
        }
        var name: String
        switch host {
        // no interface zone ("%en0") in a URL
        case .ipv4(let a): name = String("\(a)".split(separator: "%").first ?? "")
        case .ipv6(let a): name = "[\("\(a)".split(separator: "%").first ?? "")]"
        case .name(let n, _): name = n
        @unknown default: name = "\(host)"
        }
        if name.hasPrefix("[") == false, name.contains(":") { name = "[\(name)]" }
        return (name, Int(port.rawValue))
    }

    private func connect() async throws -> (WSChannel, RemoteInfo) {
        // Network.framework's WebSocket client needs a ws:// URL endpoint (a host/port endpoint aborts)
        let host: String, port: Int
        if let service = endpoint.service {
            (host, port) = try await Self.resolve(service: service)
        } else if let h = endpoint.host {
            host = h.contains(":") && !h.hasPrefix("[") ? "[\(h)]" : h
            port = endpoint.port ?? Int(RemoteProtocol.defaultPort)
        } else {
            throw RemoteError.invalid("no address")
        }
        guard let url = URL(string: "ws://\(host):\(port)/") else { throw RemoteError.invalid("bad address \(host)") }
        let ch = WSChannel(NWConnection(to: .url(url), using: WSChannel.parameters()))
        try await ch.open()
        var hello = RemoteMessage(.hello)
        hello.code = code
        hello.client = Host.current().localizedName ?? "another Mac"
        try await ch.send(hello)
        guard case .message(let m) = try await ch.receive() else { throw RemoteError.invalid("unexpected reply") }
        if m.type == .error {
            ch.close()
            throw RemoteError.refused(m.message ?? "refused")
        }
        guard m.type == .info, let info = m.info else { throw RemoteError.invalid("unexpected reply") }
        guard info.version == RemoteProtocol.version else {
            ch.close()
            throw RemoteError.refused("\(info.name) runs another version of WAI Gen; update both Macs")
        }
        return (ch, info)
    }

    /// The generator's name, models, LoRAs and upscalers (also checks the pairing code).
    public func info() async throws -> RemoteInfo {
        let (ch, info) = try await connect()
        ch.close()
        return info
    }

    /// Runs `job` on the generator. Cancelling the calling task cancels the job there.
    public func run(_ job: RemoteJob, progress: @escaping @Sendable (RemoteProgress) -> Void,
                    preview: @escaping @Sendable (Data) -> Void) async throws -> RemoteResult {
        let (ch, _) = try await connect()
        defer { ch.close() }
        var gen = RemoteMessage(.generate)
        gen.job = job
        try await ch.send(gen)
        return try await withTaskCancellationHandler {
            var png: Data?
            while true {
                switch try await ch.receive() {
                case .binary(let d):
                    png = d
                case .message(let m):
                    switch m.type {
                    case .progress: if let p = m.progress { progress(p) }
                    case .preview: if let d = m.preview { preview(d) }
                    case .done:
                        guard let png else { throw RemoteError.invalid("no image") }
                        return RemoteResult(png: png, seconds: m.seconds ?? 0, faceRepairSeconds: m.faceRepairSeconds)
                    case .error:
                        if m.message == "cancelled" { throw CancellationError() }
                        throw RemoteError.refused(m.message ?? "the generator failed")
                    default: continue
                    }
                }
            }
        } onCancel: {
            Task { try? await ch.send(RemoteMessage(.cancel)) }
        }
    }
}

// MARK: - discovery

/// Generators advertised on the local network (Bonjour). Tailscale doesn't carry Bonjour: type the address.
public final class RemoteBrowser: @unchecked Sendable {
    private var browser: NWBrowser?
    private let queue = DispatchQueue(label: "waigen.remote.browser")
    public var onChange: (@Sendable ([RemoteEndpoint]) -> Void)?

    public init() {}

    public func start() {
        let b = NWBrowser(for: .bonjour(type: RemoteProtocol.serviceType, domain: nil), using: .tcp)
        b.browseResultsChangedHandler = { [weak self] results, _ in
            let found = results.compactMap { r -> RemoteEndpoint? in
                if case .service(let name, _, _, _) = r.endpoint { return RemoteEndpoint(name: name, service: name) }
                return nil
            }
            self?.onChange?(Array(Set(found)).sorted { $0.name < $1.name })
        }
        b.start(queue: queue)
        browser = b
    }

    public func stop() {
        browser?.cancel()
        browser = nil
    }
}

/// This Mac's addresses other machines can use (Wi-Fi / Ethernet and Tailscale), IPv4.
public enum LocalAddresses {
    public static func current() -> [(kind: String, address: String)] {
        var out: [(String, String)] = []
        var ifaddr: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&ifaddr) == 0, let first = ifaddr else { return [] }
        defer { freeifaddrs(ifaddr) }
        var p: UnsafeMutablePointer<ifaddrs>? = first
        while let cur = p {
            defer { p = cur.pointee.ifa_next }
            guard let sa = cur.pointee.ifa_addr, sa.pointee.sa_family == UInt8(AF_INET) else { continue }
            var host = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            guard getnameinfo(sa, socklen_t(sa.pointee.sa_len), &host, socklen_t(host.count), nil, 0, NI_NUMERICHOST) == 0 else { continue }
            let addr = String(cString: host)
            let parts = addr.split(separator: ".").compactMap { UInt8($0) }
            guard parts.count == 4, parts[0] != 127, parts[0] != 169 else { continue }
            out.append(((100 ... 100).contains(Int(parts[0])) && (64 ... 127).contains(Int(parts[1])) ? "Tailscale" : "Local network", addr))
        }
        return out.sorted { $0.0 > $1.0 }
    }
}
