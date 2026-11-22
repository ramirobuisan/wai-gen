import Foundation
import MLX
import MLXRandom

// WAI-ANIMA end to end in Swift + MLX, replacing stable-diffusion.cpp's pipeline line by line so the same seed
// gives the same image: Philox noise (rng_philox.hpp), the discrete-flow denoiser with shift 3 and its
// discrete / Karras / simple schedules (denoiser.hpp), Euler, flow Euler ancestral and DPM++ 2M, CFG, img2img
// (t_enc), masked img2img (denoise mask), EasyCache (easycache.hpp) and the Wan 2.1 latent normalisation
// and preview projection. The models are the MLX ports (AnimaDiT, Qwen3TextEncoder, QwenImageVAE).

/// Counter-based Philox 4x32-10 normal generator (stable-diffusion.cpp's default "cuda" RNG): every `randn`
/// call uses the next offset; element i of a call is a pure function of (seed, offset, i).
public struct PhiloxRNG {
    public var seed: UInt64
    public var offset: UInt32 = 0
    public init(seed: UInt64) { self.seed = seed }

    public mutating func randn(_ n: Int) -> [Float] {
        let key0 = UInt32(truncatingIfNeeded: seed), key1 = UInt32(truncatingIfNeeded: seed >> 32)
        let off = offset
        offset &+= 1
        let inv: Float = 2.3283064e-10, inv2pi: Float = 2.3283064e-10 * 6.2831855
        var out = [Float](repeating: 0, count: n)
        out.withUnsafeMutableBufferPointer { buf in
            DispatchQueue.concurrentPerform(iterations: max(1, min(16, n / 65536))) { chunk in
                let chunks = max(1, min(16, n / 65536))
                let lo = n * chunk / chunks, hi = n * (chunk + 1) / chunks
                for i in lo ..< hi {
                    var c0 = off, c1: UInt32 = 0, c2 = UInt32(truncatingIfNeeded: i), c3: UInt32 = 0
                    var k0 = key0, k1 = key1
                    for round in 0 ..< 10 {
                        let v1 = UInt64(c0) &* 0xD251_1F53
                        let v2 = UInt64(c2) &* 0xCD9E_8D57
                        let n0 = UInt32(truncatingIfNeeded: v2 >> 32) ^ c1 ^ k0
                        let n2 = UInt32(truncatingIfNeeded: v1 >> 32) ^ c3 ^ k1
                        c1 = UInt32(truncatingIfNeeded: v2)
                        c3 = UInt32(truncatingIfNeeded: v1)
                        c0 = n0
                        c2 = n2
                        if round < 9 {
                            k0 &+= 0x9E37_79B9
                            k1 &+= 0xBB67_AE85
                        }
                    }
                    // box_muller
                    let u = Float(c0) * inv + inv / 2
                    let v = Float(c1) * inv2pi + inv2pi / 2
                    // float math like stable-diffusion.cpp; ~1 % of values differ by <1e-6 from its vectorised build
                    buf[i] = (-2 * logf(u)).squareRoot() * sinf(v)
                }
            }
        }
        return out
    }
}

public enum AnimaSampler: String, Sendable { case euler, eulerAncestral = "euler_a", dpmpp2m = "dpm++2m" }
public enum AnimaScheduler: String, Sendable { case discrete, karras, simple }

public struct AnimaParameters: Sendable {
    public var prompt: String
    public var negativePrompt = ""
    public var width = 1024
    public var height = 1344
    public var steps = 25
    public var cfg: Float = 4.5
    public var sampler: AnimaSampler = .eulerAncestral
    public var scheduler: AnimaScheduler = .discrete
    public var seed: Int64 = 0
    /// img2img: RGB8 image of width x height; `strength` = denoise
    public var initImage: Data? = nil
    public var strength: Float = 0.75
    /// with `initImage`: one byte per pixel, 255 = repaint (masked img2img, as stable-diffusion.cpp's ADetailer)
    public var mask: Data? = nil
    /// EasyCache reuse threshold (stable-diffusion.cpp's `easycache`); nil = off
    public var stepCache: Float? = nil
    public init(prompt: String) { self.prompt = prompt }
}

public final class WaiAnima {
    public let dit: AnimaDiT
    public let textEncoder: Qwen3TextEncoder
    public let vae: QwenImageVAE
    let tokenizer: AnimaTokenizer
    /// last few prompt encodings (like stable-diffusion.cpp's conditioning cache of 4)
    private var condCache: [(String, Cond)] = []
    public private(set) var lastSkippedSteps = 0

    struct Cond {
        let context: MLXArray   // [1, n, 1024]
        let ids: MLXArray       // [m] int32
        let weights: MLXArray   // [m]
    }

    public static let shift: Float = 3
    static let latentMean: [Float] = [-0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
                                      0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921]
    static let latentStd: [Float] = [2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
                                     3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160]
    static let previewProj: [Float] = [
        0.015123, -0.148418, 0.479828, 0.003652, -0.010680, -0.037142, 0.212264, 0.063033, 0.016779,
        0.232999, 0.406476, 0.220125, -0.051864, -0.082384, -0.069396, 0.085005, -0.161492, 0.010689,
        -0.245369, -0.506846, -0.117010, -0.151145, 0.017721, 0.007207, -0.293239, -0.207936, -0.421135,
        -0.187721, 0.050783, 0.177649, -0.013067, 0.265964, 0.166578, 0.028327, 0.109329, 0.108642,
        -0.205343, 0.043991, 0.148914, 0.014307, -0.048647, -0.007219, 0.217150, 0.053074, 0.319923,
        0.155357, 0.083156, 0.064780]
    static let previewBias: [Float] = [-0.270270, -0.234976, -0.456853]

    public init(dit: AnimaDiT, textEncoder: Qwen3TextEncoder, vae: QwenImageVAE) throws {
        self.dit = dit
        self.textEncoder = textEncoder
        self.vae = vae
        tokenizer = try AnimaTokenizer()
        // keep MLX's buffer cache bounded (as WaiSDXL does): the hi-res pass and its VAE decode otherwise leave
        // tens of GB of freed buffers cached
        Memory.cacheLimit = (Int(ProcessInfo.processInfo.environment["WAIGEN_CACHE_LIMIT_GB"] ?? "") ?? 2) << 30
    }

    public convenience init(diffusionModel: URL, llm: URL, vae: URL) throws {
        try self.init(dit: AnimaDiT(url: diffusionModel), textEncoder: Qwen3TextEncoder(url: llm), vae: QwenImageVAE(url: vae))
    }

    // MARK: schedule

    static func tToSigma(_ t: Float) -> Float {
        let x = (t + 1) / 1000
        return shift * x / (1 + (shift - 1) * x)   // time_snr_shift
    }

    public static func sigmas(steps n: Int, scheduler: AnimaScheduler) -> [Float] {
        guard n > 0 else { return [] }
        let tMax: Float = 999
        var s: [Float] = []
        switch scheduler {
        case .discrete:
            if n == 1 { return [tToSigma(tMax), 0] }
            let step = tMax / Float(n - 1)
            for i in 0 ..< n { s.append(tToSigma(tMax - step * Float(i))) }
        case .karras:
            let rho: Float = 7
            let sMin = max(tToSigma(0), 1e-6), sMax = tToSigma(tMax)
            let minInv = powf(sMin, 1 / rho), maxInv = powf(sMax, 1 / rho)
            for i in 0 ..< n { s.append(powf(maxInv + Float(i) / (Float(n) - 1) * (minInv - maxInv), rho)) }
        case .simple:
            let factor = 1000 / Float(n)
            for i in 0 ..< n { s.append(tToSigma(Float(max(0, 999 - Int(Float(i) * factor))))) }
        }
        s.append(0)
        return s
    }

    // MARK: text

    func condition(_ text: String) -> Cond {
        if let hit = condCache.first(where: { $0.0 == text }) { return hit.1 }
        let t = tokenizer.tokenize(text)
        let context = textEncoder(t.qwen)[.newAxis]   // all Qwen weights are 1: no re-weighting
        eval(context)
        let c = Cond(context: context, ids: MLXArray(t.t5), weights: MLXArray(t.t5Weights))
        condCache.append((text, c))
        if condCache.count > 4 { condCache.removeFirst() }
        return c
    }

    // MARK: latents

    private var meanArray: MLXArray { MLXArray(Self.latentMean).reshaped(1, 16, 1, 1) }
    private var stdArray: MLXArray { MLXArray(Self.latentStd).reshaped(1, 16, 1, 1) }

    /// RGB8 [H, W, 3] -> normalised latent [1, 16, H/8, W/8]
    func encode(_ rgb: MLXArray) -> MLXArray {
        let x = (rgb.asType(.float32) / 255 * 2 - 1)[.newAxis]
        let z = vae.encode(x).asType(.float32).transposed(0, 3, 1, 2)
        return (z - meanArray) / stdArray
    }

    /// normalised latent [1, 16, h, w] -> RGB8 [8h, 8w, 3]
    func decode(_ latent: MLXArray) -> MLXArray {
        let z = (latent * stdArray + meanArray).transposed(0, 2, 3, 1)
        let img = (vae.decode(z).asType(.float32) + 1) * 0.5
        return (clip(img, min: 0, max: 1) * 255 + 0.5)[0].asType(.uint8)   // rounded, like stable-diffusion.cpp
    }

    func preview(_ latent: MLXArray) -> MLXArray {
        let x = latent[0].transposed(1, 2, 0)   // [h, w, 16]
        let rgb = matmul(x, MLXArray(Self.previewProj, [16, 3])) + MLXArray(Self.previewBias)
        return (clip(rgb * 0.5 + 0.5, min: 0, max: 1) * 255).asType(.uint8)
    }

    // MARK: generate

    /// Generates (or repaints) an image; returns RGB8 [height, width, 3]. `preview` gets each step's
    /// denoised latent projected to RGB8 (latent size); `progress(step, total)`.
    public func generate(
        _ p: AnimaParameters, isCancelled: () -> Bool = { false }, preview: ((MLXArray) -> Void)? = nil,
        progress: (Int, Int) -> Void = { _, _ in }
    ) throws -> MLXArray {
        defer { Memory.clearCache() }
        let (w, h) = (p.width / 8, p.height / 8)
        let n = 16 * h * w
        var rng = PhiloxRNG(seed: UInt64(bitPattern: p.seed))

        var sigmas = Self.sigmas(steps: p.steps, scheduler: p.scheduler)
        var initLatent = MLXArray.zeros([1, 16, h, w])
        var mask: MLXArray? = nil
        if let initImage = p.initImage {
            guard initImage.count == p.width * p.height * 3 else { throw WaiAnimaError("init image size mismatch") }
            initLatent = encode(MLXArray(initImage, [p.height, p.width, 3], type: UInt8.self))
            if p.strength < 1 {
                var tEnc = Int(Float(p.steps) * p.strength)
                if tEnc == p.steps { tEnc -= 1 }
                sigmas = Array(sigmas[(p.steps - tEnc - 1)...])
            }
            if let m = p.mask {
                guard m.count == p.width * p.height else { throw WaiAnimaError("mask size mismatch") }
                // round(mask / 255), then nearest-max down to latent size (max over each 8x8 block)
                let full = (MLXArray(m, [p.height, p.width], type: UInt8.self).asType(.float32) / 255).round()
                let lm = full.reshaped(h, 8, w, 8).max(axes: [1, 3]).reshaped(1, 1, h, w)
                if lm.min().item(Float.self) < 1 { mask = lm }
            }
        }
        let cond = condition(p.prompt)
        let uncond = condition(p.negativePrompt)

        // same seed for the init noise and the sampler's noise: one generator (offsets 0, 1, 2, ...)
        let noise = MLXArray(rng.randn(n), [1, 16, h, w])
        var x = initLatent * (1 - sigmas[0]) + noise * sigmas[0]
        eval(x)

        let cache = p.stepCache.map { EasyCache(threshold: $0) }
        let startSigma = Self.tToSigma((1 - 0.15) * 999), endSigma = Self.tToSigma((1 - 0.95) * 999)
        let total = sigmas.count - 1

        func model(_ x: MLXArray, _ c: Cond, _ t: MLXArray) -> MLXArray {
            dit(x, timesteps: t, context: c.context, tokenIDs: c.ids, tokenWeights: c.weights)
        }
        func denoise(_ x: MLXArray, _ sigma: Float, _ step: Int) -> MLXArray {
            let t = MLXArray([sigma * 1000 / 1000])
            let active = sigma <= startSigma && sigma > endSigma
            let condOut: MLXArray, uncondOut: MLXArray
            if let cache, active {
                condOut = cache.run(anchor: true, x, step: step) { model(x, cond, t) }
                uncondOut = cache.run(anchor: false, x, step: step) { model(x, uncond, t) }
            } else {
                // one pass at a time (as two separate graphs): measurably faster than one graph with both
                condOut = model(x, cond, t)
                eval(condOut)
                uncondOut = model(x, uncond, t)
            }
            let guided = uncondOut + p.cfg * (condOut - uncondOut)
            var d = x - sigma * guided   // c_skip 1, c_out -sigma
            if let mask { d = d * mask + initLatent * (1 - mask) }
            return d
        }

        var old: MLXArray? = nil
        for i in 0 ..< total {
            if isCancelled() { throw WaiCancelled() }
            let s = sigmas[i], next = sigmas[i + 1]
            if p.sampler == .euler, s == next { progress(i + 1, total); continue }
            let d = denoise(x, s, i)
            switch p.sampler {
            case .euler:
                x = x + (x - d) / s * (next - s)
            case .eulerAncestral:
                if next == 0 {
                    x = d
                } else {
                    let (down, up, alpha) = Self.ancestralStepFlow(s, next)
                    let ratio = down / s
                    x = ratio * x + (1 - ratio) * d
                    if up > 0 {
                        x = x * alpha + MLXArray(rng.randn(n), [1, 16, h, w]) * up
                    }
                }
            case .dpmpp2m:
                let tFn = { (s: Float) in -logf(s) }
                let a = next / s, hh = tFn(next) - tFn(s), b = expf(-hh) - 1
                if i == 0 || next == 0 {
                    x = a * x - b * d
                } else {
                    let r = (tFn(s) - tFn(sigmas[i - 1])) / hh
                    x = a * x - b * ((1 + 1 / (2 * r)) * d - (1 / (2 * r)) * (old ?? d))
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
        lastSkippedSteps = cache?.skipped ?? 0
        let out = decode(x)
        eval(out)
        return out
    }

    /// get_ancestral_step_flow with eta 1: (sigma_down, sigma_up, alpha_scale)
    static func ancestralStepFlow(_ from: Float, _ to: Float) -> (Float, Float, Float) {
        guard from > 0, to > 0 else { return (to, 0, 1) }
        let ratio = to / from
        var down = to * (1 + (ratio - 1) * 1)
        down = max(0, min(to, down))
        let denom = 1 - down
        guard denom > 0 else { return (to, 0, 1) }
        let alpha = (1 - to) / denom
        let term = max(-1, min(1, (down / to) * alpha))
        let up = to * max(1 - term * term, 0).squareRoot()
        return (down, up, alpha)
    }
}

public struct WaiAnimaError: LocalizedError {
    public let errorDescription: String?
    init(_ m: String) { errorDescription = m }
}

/// stable-diffusion.cpp's EasyCache for models called once per condition: the anchor (positive prompt)
/// decides, every condition reuses its own cached output - input offset on a skipped step.
final class EasyCache {
    let threshold: Float
    private var offsets: [Bool: MLXArray] = [:]
    private var prevInput: MLXArray?
    private var prevOutput: MLXArray?
    private var outputNorm: Float = 0
    private var rate: Float?
    private var cumulative: Float = 0
    private var lastInputChange: Float?
    private var currentStep = -1
    private var skipStep = false
    private(set) var skipped = 0

    init(threshold: Float) { self.threshold = threshold }

    func run(anchor: Bool, _ input: MLXArray, step: Int, compute: () -> MLXArray) -> MLXArray {
        if step != currentStep {
            currentStep = step
            skipStep = false
            lastInputChange = nil
        }
        if skipStep, let off = offsets[anchor] { return input + off }
        if anchor, let prevInput, prevOutput != nil, offsets[true] != nil {
            let change = abs(input - prevInput).mean().item(Float.self)
            lastInputChange = change
            if let rate, outputNorm > 0, change > 0 {
                cumulative += rate * change / outputNorm
                if cumulative < threshold {
                    skipStep = true
                    skipped += 1
                    return input + offsets[true]!
                }
                cumulative = 0
            }
        }
        let out = compute()
        offsets[anchor] = out - input
        if anchor {
            let outChange = prevOutput.map { abs(out - $0).mean() }
            let norm = abs(out).mean()
            eval(out, norm)
            if let oc = outChange?.item(Float.self), let ic = lastInputChange, ic > 0, oc > 0, oc.isFinite {
                rate = oc / ic
            }
            outputNorm = norm.item(Float.self)
            prevInput = input
            prevOutput = out
            cumulative = 0
            lastInputChange = nil
        }
        return out
    }
}

extension WaiAnima {
    /// Face repair like stable-diffusion.cpp's ADetailer (`FaceRepairGeometry`), each crop repainted with this
    /// pipeline's masked img2img: `p` gives prompt, negative, steps, cfg, sampler, scheduler, seed (face i
    /// uses seed + i) and step cache; `settings.strength` the denoise. `progress(face, step, steps)`.
    public func repairFaces(
        rgb: Data, width: Int, height: Int, faces: [FaceBox], _ p: AnimaParameters, settings s: FaceRepairSettings,
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
