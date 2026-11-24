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
    /// img2img: RGB8 image of width x height; `strength` = denoise (hi-res fix, Upscale, face repair)
    public var initImage: Data? = nil
    public var strength: Float = 0.75
    /// with `initImage`: one byte per pixel, 255 = repaint (masked img2img, as WAI-ANIMA's face repair)
    public var mask: Data? = nil

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

    /// RGB8 [H, W, 3] -> sampler latent [1, 16, H/8, W/8] (the VAE's mean, then process_latent_in).
    func encode(_ rgb: MLXArray) -> MLXArray {
        let x = (rgb.asType(.float32) / 255 * 2 - 1)[.newAxis].asType(.bfloat16)
        let (mean, _) = vae.encode(x)
        return (mean.asType(.float32).transposed(0, 3, 1, 2) - Self.latentShift) * Self.latentScale
    }

    /// Flux's latent -> RGB factors (ComfyUI `latent_formats.Flux`) for the live preview: RGB8 [h, w, 3].
    static let previewProj: [Float] = [
        -0.0346, 0.0244, 0.0681, 0.0034, 0.0210, 0.0687, 0.0275, -0.0668, -0.0433, -0.0174, 0.0160, 0.0617,
        0.0859, 0.0721, 0.0329, 0.0004, 0.0383, 0.0115, 0.0405, 0.0861, 0.0915, -0.0236, -0.0185, -0.0259,
        -0.0245, 0.0250, 0.1180, 0.1008, 0.0755, -0.0421, -0.0515, 0.0201, 0.0011, 0.0428, -0.0012, -0.0036,
        0.0817, 0.0765, 0.0749, -0.1264, -0.0522, -0.1103, -0.0280, -0.0881, -0.0499, -0.1262, -0.0982, -0.0778]
    static let previewBias: [Float] = [-0.0329, -0.0718, -0.0851]

    func preview(_ latent: MLXArray) -> MLXArray {
        let z = latent[0].transposed(1, 2, 0)   // [h, w, 16]: ComfyUI applies the factors to the sampler latent
        let rgb = matmul(z, MLXArray(Self.previewProj, [16, 3])) + MLXArray(Self.previewBias)
        return (clip(rgb * 0.5 + 0.5, min: 0, max: 1) * 255).asType(.uint8)
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
                         preview: ((MLXArray) -> Void)? = nil,
                         progress: (Int, Int) -> Void = { _, _ in }) throws -> MLXArray {
        defer { Memory.clearCache() }
        guard p.width % 16 == 0, p.height % 16 == 0 else { throw WaiZImageError("width and height must be multiples of 16") }
        let (h, w) = (p.height / 8, p.width / 8)
        var sigmas = Self.sigmas(steps: p.steps, scheduler: p.scheduler)
        var initLatent = MLXArray.zeros([1, 16, h, w])
        var mask: MLXArray? = nil
        if let initImage = p.initImage {
            guard initImage.count == p.width * p.height * 3 else { throw WaiZImageError("init image size mismatch") }
            initLatent = encode(MLXArray(initImage, [p.height, p.width, 3], type: UInt8.self))
            if p.strength < 1 {   // the last steps × strength of the schedule, as WAI-ANIMA (stable-diffusion.cpp's t_enc)
                let total = sigmas.count - 1
                var tEnc = Int(Float(total) * p.strength)
                if tEnc >= total { tEnc = total - 1 }
                sigmas = Array(sigmas[(total - tEnc - 1)...])
            }
            if let m = p.mask {
                guard m.count == p.width * p.height else { throw WaiZImageError("mask size mismatch") }
                // round(mask / 255), then the max over each 8×8 block (latent size)
                let full = (MLXArray(m, [p.height, p.width], type: UInt8.self).asType(.float32) / 255).round()
                let lm = full.reshaped(h, 8, w, 8).max(axes: [1, 3]).reshaped(1, 1, h, w)
                if lm.min().item(Float.self) < 1 { mask = lm }
            }
        }
        let cond = condition(p.prompt)
        let uncond: MLXArray? = p.cfg > 1
            ? (p.negativePrompt.isEmpty ? MLXArray.zeros(like: cond) : condition(p.negativePrompt)) : nil

        let keys = MLXRandom.split(key: MLXRandom.key(UInt64(bitPattern: p.seed)), into: sigmas.count + 1)
        let start = noise ?? MLXRandom.normal([1, 16, h, w], key: keys[0])
        var x = initLatent * (1 - sigmas[0]) + start * sigmas[0]   // CONST noise scaling
        eval(x)

        func denoise(_ x: MLXArray, _ sigma: Float) -> MLXArray {
            let c = dit(x, sigma: sigma, context: cond)
            var d: MLXArray
            if let uncond {
                eval(c)
                let u = dit(x, sigma: sigma, context: uncond)
                d = x - sigma * (u + p.cfg * (c - u))
            } else {
                d = x - sigma * c
            }
            if let mask { d = d * mask + initLatent * (1 - mask) }
            return d
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
            if let preview {
                let pv = self.preview(d)
                eval(x, pv)
                preview(pv)
            } else {
                eval(x)
            }
            progress(i + 1, total)
        }
        return decode(x)
    }

    /// Repaints each face like ADetailer: masked img2img of a square crop around it (WAI-ANIMA's approach).
    /// `p`: prompt, steps, cfg, sampler, scheduler, seed; `s.strength` = denoise. Returns RGB8 of width × height.
    public func repairFaces(
        rgb: Data, width: Int, height: Int, faces: [FaceBox], _ p: ZImageParameters, settings s: FaceRepairSettings,
        isCancelled: () -> Bool = { false }, progress: (Int, Int, Int) -> Void = { _, _, _ in }
    ) throws -> Data {
        let size = s.inpaintSize
        return try FaceRepairGeometry.apply(rgb: rgb, width: width, height: height, faces: faces, settings: s) { crop, mask, i in
            var q = p
            q.width = size
            q.height = size
            q.initImage = Data(crop)
            q.mask = Data(mask.map { $0 > 0 ? 255 : 0 })
            q.strength = s.strength
            q.seed = p.seed &+ Int64(i)
            let out = try generate(q, isCancelled: isCancelled) { progress(i, $0, $1) }
            return out.asArray(UInt8.self)
        }
    }
}

public struct WaiZImageError: LocalizedError {
    public let errorDescription: String?
    init(_ m: String) { errorDescription = m }
}
