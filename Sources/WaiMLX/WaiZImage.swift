import Foundation
import MLX
import MLXRandom

/// How the sigmas are spread over the steps (ComfyUI schedulers on the flow model's 1000-entry sigma table).
public enum ZImageScheduler: String, CaseIterable, Sendable {
    /// ComfyUI `simple`: evenly spaced entries of the table
    case simple
    /// ComfyUI `beta` (alpha = beta = 0.6): more steps near the start and the end
    case beta
}

public struct ZImageParameters: Sendable {
    public var prompt: String
    /// Only used with cfg > 1; empty = ComfyUI's ConditioningZeroOut (zeros), as in the Turbo workflow.
    public var negativePrompt = ""
    public var width = 1024
    public var height = 1024
    /// The Turbo model is distilled for ~8-9 steps at CFG 1 (one model call per step).
    public var steps = 9
    public var cfg: Float = 1
    public var sampler: AnimaSampler = .eulerAncestral
    public var scheduler: ZImageScheduler = .beta
    public var seed: Int64 = 42

    public init(prompt: String) { self.prompt = prompt }
}

/// Z-Image (Turbo) text-to-image on MLX, from an all-in-one checkpoint (DiT + Qwen3-4B + Flux VAE), following
/// ComfyUI: Qwen3-4B layer −2 conditioning, NextDiT, discrete flow with shift 3 (multiplier 1), CONST denoising
/// (x0 = x − σ·out), Flux latent format (scale 0.3611, shift 0.1159).
public final class WaiZImage {
    public let dit: ZImageDiT
    public let textEncoder: ZImageTextEncoder
    let vae: Autoencoder
    public static let shift: Float = 3
    static let latentScale: Float = 0.3611
    static let latentShift: Float = 0.1159
    private var condCache: [(String, MLXArray)] = []

    public init(aio url: URL) throws {
        let encoder = try Qwen3TextEncoder(url: url, keyPrefix: "text_encoders.qwen3_4b.transformer.model.")
        textEncoder = ZImageTextEncoder(encoder: encoder, tokenizer: try AnimaTokenizer().qwen)
        dit = try ZImageDiT(url: url)
        let config = try JSONDecoder().decode(AutoencoderConfiguration.self, from: Data("""
            {"in_channels": 3, "out_channels": 3, "latent_channels": 16, "block_out_channels": [128, 256, 512, 512],
             "layers_per_block": 2, "norm_num_groups": 32, "scaling_factor": 1.0}
            """.utf8))
        vae = try loadLDMVAE(try loadArrays(url: url), prefix: "vae.", configuration: config)
        guard !vae.quantProjections else { throw WaiZImageError("\(url.lastPathComponent): unexpected VAE (quant_conv)") }
        Memory.cacheLimit = (Int(ProcessInfo.processInfo.environment["WAIGEN_CACHE_LIMIT_GB"] ?? "") ?? 2) << 30
    }

    // MARK: schedule

    /// ModelSamplingDiscreteFlow's table: entry k is time_snr_shift(shift, (k + 1) / 1000).
    static func tableSigma(_ k: Int) -> Float {
        let x = Float(k + 1) / 1000
        return shift * x / (1 + (shift - 1) * x)
    }

    public static func sigmas(steps n: Int, scheduler: ZImageScheduler) -> [Float] {
        guard n > 0 else { return [] }
        switch scheduler {
        case .simple:
            return WaiAnima.sigmas(steps: n, scheduler: .simple)   // same table (shift 3, 1000 entries)
        case .beta:
            var s: [Float] = []
            var last = -1
            for i in 0 ..< n {
                let q = 1 - Double(i) / Double(n)
                let t = Int((betaPPF(q, 0.6, 0.6) * 999).rounded(.toNearestOrEven))
                if t != last { s.append(tableSigma(t)) }
                last = t
            }
            s.append(0)
            return s
        }
    }

    /// Inverse of the regularized incomplete beta function (scipy.stats.beta.ppf), by bisection.
    static func betaPPF(_ q: Double, _ a: Double, _ b: Double) -> Double {
        if q <= 0 { return 0 }
        if q >= 1 { return 1 }
        var lo = 0.0, hi = 1.0
        for _ in 0 ..< 80 {
            let mid = (lo + hi) / 2
            if regularizedBeta(mid, a, b) < q { lo = mid } else { hi = mid }
        }
        return (lo + hi) / 2
    }

    /// I_x(a, b) via the continued fraction (Numerical Recipes `betai` / `betacf`).
    static func regularizedBeta(_ x: Double, _ a: Double, _ b: Double) -> Double {
        if x <= 0 { return 0 }
        if x >= 1 { return 1 }
        let front = exp(lgamma(a + b) - lgamma(a) - lgamma(b) + a * log(x) + b * log(1 - x))
        func cf(_ x: Double, _ a: Double, _ b: Double) -> Double {
            let tiny = 1e-300
            var c = 1.0, d = 1 - (a + b) * x / (a + 1)
            if abs(d) < tiny { d = tiny }
            d = 1 / d
            var h = d
            for m in 1 ... 300 {
                let m = Double(m)
                var num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
                d = 1 + num * d; if abs(d) < tiny { d = tiny }
                c = 1 + num / c; if abs(c) < tiny { c = tiny }
                d = 1 / d; h *= d * c
                num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
                d = 1 + num * d; if abs(d) < tiny { d = tiny }
                c = 1 + num / c; if abs(c) < tiny { c = tiny }
                d = 1 / d
                let delta = d * c
                h *= delta
                if abs(delta - 1) < 1e-15 { break }
            }
            return h
        }
        return x < (a + 1) / (a + b + 2) ? front * cf(x, a, b) / a : 1 - front * cf(1 - x, b, a) / b
    }

    // MARK: text, latents

    func condition(_ text: String) -> MLXArray {
        if let hit = condCache.first(where: { $0.0 == text }) { return hit.1 }
        let c = textEncoder.encode(text)
        condCache.append((text, c))
        if condCache.count > 4 { condCache.removeFirst() }
        return c
    }

    /// Sampler latent [1, 16, h, w] -> RGB8 [8h, 8w, 3] (process_latent_out, then the VAE, clamp((x + 1) / 2)).
    public func decode(_ latent: MLXArray) -> MLXArray {
        let z = (latent / Self.latentScale + Self.latentShift).transposed(0, 2, 3, 1)
        let img = vae.decode(z.asType(.bfloat16)).asType(.float32)
        let out = (clip(img / 2 + 0.5, min: 0, max: 1) * 255).round().asType(.uint8)[0]
        eval(out)
        return out
    }

    /// The VAE alone on a raw z [1, 16, h, w] (no latent-format scaling) -> [1, 8h, 8w, 3] in 0...1, like ComfyUI's
    /// `vae.decode` (for the reference check).
    public func decodeRaw(_ z: MLXArray) -> MLXArray {
        let img = vae.decode(z.transposed(0, 2, 3, 1).asType(.bfloat16)).asType(.float32)
        return clip(img / 2 + 0.5, min: 0, max: 1)
    }

    // MARK: generate

    /// Text to image; returns RGB8 [height, width, 3]. `noise` [1, 16, H/8, W/8] replaces the seeded start noise
    /// (to compare with ComfyUI on the same noise). `progress(step, total)`.
    public func generate(_ p: ZImageParameters, noise: MLXArray? = nil, isCancelled: () -> Bool = { false },
                         progress: (Int, Int) -> Void = { _, _ in }) throws -> MLXArray {
        defer { Memory.clearCache() }
        guard p.width % 16 == 0, p.height % 16 == 0 else { throw WaiZImageError("width and height must be multiples of 16") }
        let (h, w) = (p.height / 8, p.width / 8)
        let sigmas = Self.sigmas(steps: p.steps, scheduler: p.scheduler)
        let cond = condition(p.prompt)
        let uncond: MLXArray? = p.cfg > 1
            ? (p.negativePrompt.isEmpty ? MLXArray.zeros(like: cond) : condition(p.negativePrompt)) : nil

        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(bitPattern: p.seed)), into: sigmas.count + 1)
        let start = noise ?? MLXRandom.normal([1, 16, h, w], key: keys[0])
        var x = start * sigmas[0]   // CONST noise scaling on an empty latent: σ·noise
        eval(x)

        func denoise(_ x: MLXArray, _ sigma: Float) -> MLXArray {
            let c = dit(x, sigma: sigma, context: cond)
            guard let uncond else { return x - sigma * c }
            eval(c)
            let u = dit(x, sigma: sigma, context: uncond)
            return x - sigma * (u + p.cfg * (c - u))
        }

        var old: MLXArray? = nil
        let total = sigmas.count - 1
        for i in 0 ..< total {
            if isCancelled() { throw WaiCancelled() }
            let s = sigmas[i], next = sigmas[i + 1]
            let d = denoise(x, s)
            switch p.sampler {
            case .euler:
                x = x + (x - d) / s * (next - s)
            case .eulerAncestral:   // ComfyUI sample_euler_ancestral_RF, eta 1
                if next == 0 {
                    x = d
                } else {
                    let (down, up, alpha) = WaiAnima.ancestralStepFlow(s, next)
                    let ratio = down / s
                    x = ratio * x + (1 - ratio) * d
                    if up > 0 { x = x * alpha + MLXRandom.normal([1, 16, h, w], key: keys[i + 1]) * up }
                }
            case .dpmpp2m:
                let a = next / s
                if next == 0 {
                    x = d
                } else {
                    let hh = logf(s) - logf(next), b = expf(-hh) - 1
                    if i == 0 {
                        x = a * x - b * d
                    } else {
                        let r = (logf(sigmas[i - 1]) - logf(s)) / hh
                        x = a * x - b * ((1 + 1 / (2 * r)) * d - (1 / (2 * r)) * (old ?? d))
                    }
                }
                old = d
            }
            eval(x)
            progress(i + 1, total)
        }
        return decode(x)
    }
}

public struct WaiZImageError: LocalizedError {
    public let errorDescription: String?
    init(_ m: String) { errorDescription = m }
}
