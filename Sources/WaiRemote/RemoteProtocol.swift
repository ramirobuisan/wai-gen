import Foundation

// WAI Gen's generator protocol: one WebSocket connection per request between the app on a laptop (client)
// and the app on the Mac with the models (server). Text frames carry `RemoteMessage` JSON; the finished image
// is a binary frame (PNG) sent right before the "done" message.
//
// client: hello(code) -> server: info | error
// client: generate(job) -> server: progress*, preview*, <PNG>, done | error
// client: cancel (while a job runs)

public enum RemoteProtocol {
    public static let version = 1
    public static let serviceType = "_waigen._tcp"
    public static let defaultPort: UInt16 = 47860
    /// largest message accepted (a gallery Upscale sends the source image inside the job)
    public static let maxMessageBytes = 96 << 20
}

public struct RemoteMessage: Codable, Sendable {
    public var type: Kind
    public var code: String?
    public var client: String?
    public var info: RemoteInfo?
    public var job: RemoteJob?
    public var progress: RemoteProgress?
    /// JPEG of the live preview
    public var preview: Data?
    public var seconds: Double?
    public var faceRepairSeconds: Double?
    public var message: String?

    public enum Kind: String, Codable, Sendable {
        case hello, info, generate, cancel, progress, preview, done, error
    }

    public init(_ type: Kind) { self.type = type }
}

/// What the server can generate with.
public struct RemoteInfo: Codable, Sendable, Equatable {
    public var name: String
    public var version: Int
    /// preset ids whose model files are on the server
    public var presets: [String]
    public var loras: [RemoteLoRAFile]
    public var upscalers: [RemoteUpscaler]
    public var faceDetector: Bool
    public var loadedModels: [String]
    public var busy: Bool

    public init(name: String, presets: [String], loras: [RemoteLoRAFile], upscalers: [RemoteUpscaler],
                faceDetector: Bool, loadedModels: [String], busy: Bool) {
        self.name = name
        version = RemoteProtocol.version
        self.presets = presets
        self.loras = loras
        self.upscalers = upscalers
        self.faceDetector = faceDetector
        self.loadedModels = loadedModels
        self.busy = busy
    }
}

public struct RemoteLoRAFile: Codable, Sendable, Equatable {
    public var relativePath: String
    public var baseModel: String?
    public var triggerTags: [String]
    public init(relativePath: String, baseModel: String?, triggerTags: [String]) {
        self.relativePath = relativePath
        self.baseModel = baseModel
        self.triggerTags = triggerTags
    }
}

public struct RemoteUpscaler: Codable, Sendable, Equatable {
    public var name: String
    public var scale: Int
    public init(name: String, scale: Int) {
        self.name = name
        self.scale = scale
    }
}

/// A fully resolved job (prompt assembled, seed fixed); file references are relative to the server's folders.
public struct RemoteJob: Codable, Sendable {
    public var presetID: String
    public var positive: String
    public var negative: String
    public var characters: [String]
    public var userPrompt: String
    public var width: Int
    public var height: Int
    public var steps: Int
    public var cfg: Double
    public var sampler: String
    public var scheduler: String
    public var seed: Int64
    public var hires: Hires?
    public var faceRepair: FaceRepair?
    public var loras: [LoRA]
    /// file name in the server's upscale_models folder
    public var hiresUpscaler: String?
    public var livePreview: Bool
    public var stepCache: Bool
    public var upscale: Upscale?

    public struct Hires: Codable, Sendable {
        public var scale: Double, steps: Int, denoise: Double
        public init(scale: Double, steps: Int, denoise: Double) { (self.scale, self.steps, self.denoise) = (scale, steps, denoise) }
    }
    public struct FaceRepair: Codable, Sendable {
        public var strength: Double, confidence: Double, prompt: String
        public init(strength: Double, confidence: Double, prompt: String) {
            (self.strength, self.confidence, self.prompt) = (strength, confidence, prompt)
        }
    }
    public struct LoRA: Codable, Sendable {
        public var relativePath: String, strength: Double
        public init(relativePath: String, strength: Double) { (self.relativePath, self.strength) = (relativePath, strength) }
    }
    /// gallery Upscale: the source image (PNG or JPEG bytes) and the target size
    public struct Upscale: Codable, Sendable {
        public var width: Int, height: Int, detailPass: Bool, source: Data
        public init(width: Int, height: Int, detailPass: Bool, source: Data) {
            (self.width, self.height, self.detailPass, self.source) = (width, height, detailPass, source)
        }
    }

    public init(presetID: String, positive: String, negative: String, characters: [String], userPrompt: String,
                width: Int, height: Int, steps: Int, cfg: Double, sampler: String, scheduler: String, seed: Int64,
                hires: Hires?, faceRepair: FaceRepair?, loras: [LoRA], hiresUpscaler: String?, livePreview: Bool,
                stepCache: Bool, upscale: Upscale?) {
        self.presetID = presetID
        self.positive = positive
        self.negative = negative
        self.characters = characters
        self.userPrompt = userPrompt
        self.width = width
        self.height = height
        self.steps = steps
        self.cfg = cfg
        self.sampler = sampler
        self.scheduler = scheduler
        self.seed = seed
        self.hires = hires
        self.faceRepair = faceRepair
        self.loras = loras
        self.hiresUpscaler = hiresUpscaler
        self.livePreview = livePreview
        self.stepCache = stepCache
        self.upscale = upscale
    }

    /// Rejects values that would only waste the server's GPU or memory.
    public func validate() throws {
        func bad(_ m: String) -> RemoteError { RemoteError.invalid(m) }
        guard (256 ... 2048).contains(width), (256 ... 2048).contains(height) else { throw bad("size out of range") }
        guard (1 ... 150).contains(steps), cfg >= 0, cfg <= 30 else { throw bad("steps / cfg out of range") }
        if let h = hires, !(h.scale >= 1 && h.scale <= 2 && (1 ... 100).contains(h.steps) && h.denoise > 0 && h.denoise <= 1) {
            throw bad("hi-res settings out of range")
        }
        if let u = upscale, !((256 ... 4096).contains(u.width) && (256 ... 4096).contains(u.height)) { throw bad("upscale size out of range") }
        guard loras.count <= 16 else { throw bad("too many LoRAs") }
        for l in loras where l.relativePath.contains("..") || l.relativePath.hasPrefix("/") { throw bad("bad LoRA path") }
        if let u = hiresUpscaler, u.contains("/") || u.contains("..") { throw bad("bad upscaler name") }
    }
}

public struct RemoteProgress: Codable, Sendable {
    public var kind: Kind
    public var name: String?
    public var step: Int = 0
    public var total: Int = 0
    public var hires = false

    public enum Kind: String, Codable, Sendable {
        case loadingModel, encoding, sampling, decoding, upscaling, preparingHires, loadingFaceRepair, faceRepair
    }

    public init(_ kind: Kind, name: String? = nil, step: Int = 0, total: Int = 0, hires: Bool = false) {
        self.kind = kind
        self.name = name
        self.step = step
        self.total = total
        self.hires = hires
    }
}

public enum RemoteError: LocalizedError, Sendable {
    case invalid(String)
    case refused(String)
    case connection(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .invalid(let m): "Invalid request: \(m)"
        case .refused(let m): m
        case .connection(let m): "Couldn't reach the generator: \(m)"
        case .cancelled: "Cancelled"
        }
    }
}

/// Where a generator is: a Bonjour service name on the local network, or a host name / address (e.g. a
/// Tailscale MagicDNS name or 100.x address) with a port.
public struct RemoteEndpoint: Codable, Sendable, Hashable {
    public var name: String
    public var service: String?
    public var host: String?
    public var port: Int?

    public init(name: String, service: String? = nil, host: String? = nil, port: Int? = nil) {
        self.name = name
        self.service = service
        self.host = host
        self.port = port
    }

    /// "host", "host:port" or "[ipv6]:port"
    public static func address(_ text: String) -> RemoteEndpoint? {
        let t = text.trimmingCharacters(in: .whitespaces)
        guard !t.isEmpty else { return nil }
        var host = t, port = Int(RemoteProtocol.defaultPort)
        if t.hasPrefix("["), let close = t.firstIndex(of: "]") {
            host = String(t[t.index(after: t.startIndex) ..< close])
            let rest = t[t.index(after: close)...]
            if rest.hasPrefix(":"), let p = Int(rest.dropFirst()) { port = p }
        } else if t.filter({ $0 == ":" }).count == 1, let colon = t.firstIndex(of: ":"), let p = Int(t[t.index(after: colon)...]) {
            host = String(t[..<colon])
            port = p
        }
        return RemoteEndpoint(name: host, host: host, port: port)
    }
}
