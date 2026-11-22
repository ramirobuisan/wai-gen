// WAI-illustrious (SDXL, eps-prediction) text-to-image on MLX.

import Foundation
import MLX
import MLXNN
import MLXRandom

public struct SDXLParameters: Sendable {
    public var prompt: String
    public var negativePrompt: String = ""
    public var width = 832
    public var height = 1216
    public var steps = 28
    public var cfg: Float = 6
    public var sampler: SamplerKind = .eulerAncestral
    public var scheduler: SchedulerKind = .normal
    public var seed: UInt64 = 0

    /// Hi-res fix: second pass at `hiresScale` x the base size. nil = off.
    public var hiresScale: Float? = nil
    public var hiresSteps = 15
    public var hiresDenoise: Float = 0.4
    /// Step cache threshold (see `StepCache`): skips model calls on steps that would barely change the
    /// prediction. nil = off; 0.5 skips ~8 of 28 steps with a nearly identical image (docs/PERFORMANCE.md).
    public var stepCache: Float? = nil
    /// Guidance interval (Kynkäänniemi et al. 2024): classifier-free guidance only while
    /// `guidanceSigmaMin < sigma <= guidanceSigmaMax`; the other steps run the UNet once on the prompt alone
    /// (half the work). The defaults guide every step.
    public var guidanceSigmaMin: Float = 0
    public var guidanceSigmaMax: Float = .infinity
    /// false = the hi-res pass runs without guidance (prompt only).
    public var hiresGuidance = true
    /// > 1: the hi-res pass computes guidance only on every n-th model call and reuses the last guidance
    /// direction (cond − uncond) on the others, which run the UNet once (CFG caching, as in FasterCache).
    public var hiresGuidanceEvery = 1

    public init(prompt: String) { self.prompt = prompt }

    /// nil when every step is guided (the original graph), else which sigmas get guidance.
    var guidanceRule: ((Float) -> Bool)? {
        guard guidanceSigmaMin > 0 || guidanceSigmaMax.isFinite else { return nil }
        let (lo, hi) = (guidanceSigmaMin, guidanceSigmaMax)
        return { $0 > lo && $0 <= hi }
    }

    var hiresGuidanceRule: ((Float) -> Bool)? {
        if !hiresGuidance { return { _ in false } }
        return guidanceRule ?? (hiresGuidanceEvery > 1 ? { _ in true } : nil)
    }
}

/// Timings of one generation, in seconds.
public struct GenerationTimings: Sendable {
    public var textEncode: Double = 0
    public var sampling: Double = 0
    public var samplingSteps = 0
    public var hiresSampling: Double = 0
    public var hiresSteps = 0
    /// model calls the step cache skipped (base + hi-res)
    public var skippedSteps = 0
    public var decode: Double = 0
    public var total: Double = 0
}

public enum GenerationPhase: String, Sendable { case encode, sample, hires, decode }

/// Thrown by `WaiSDXL.generate` when the caller cancels.
public struct WaiCancelled: Error {}

public final class WaiSDXL {
    public let checkpoint: URL
    let unet: UNetModel
    let clipL: CLIPTextModel
    let clipG: CLIPTextModel
    let vae: Autoencoder
    let tokenizer: CLIPTokenizer
    let schedule = DiscreteSchedule()
    let dtype: DType = .float16

    public init(checkpoint: URL) throws {
        self.checkpoint = checkpoint
        let parts = try loadSDXLCheckpoint(url: checkpoint, dType: dtype)
        unet = parts.unet
        clipL = parts.clipL
        clipG = parts.clipG
        vae = parts.vae
        tokenizer = try loadCLIPTokenizer()
        eval(unet, clipL, clipG, vae)
        // keep MLX's buffer cache bounded so repeated generations don't push the Mac into swap
        Memory.cacheLimit = 2 << 30
    }

    let vaeDType: DType = .bfloat16
    let lora = LoRAApplier()

    /// Merge these LoRAs into the weights (undoing any previous set). No-op when unchanged.
    @discardableResult
    public func setLoRAs(_ specs: [LoRASpec]) throws -> LoRAReport {
        try lora.apply(specs, unet: unet, clipL: clipL, clipG: clipG)
    }

    /// Image upscaler for the hi-res fix: gets the decoded base image (float32 [1, H, W, 3] in 0...1)
    /// and the target width/height, returns float32 [1, targetH, targetW, 3] in 0...1.
    public typealias HiresUpscaler = (MLXArray, Int, Int) throws -> MLXArray

    /// Generate one image. Returns uint8 [H, W, 3]. Throws `WaiCancelled` when `isCancelled()`
    /// turns true (checked before every sampling step). `hiresUpscale` replaces the built-in bicubic
    /// upscale of the hi-res fix (e.g. an ESRGAN model).
    /// SDXL latent -> RGB projection (ComfyUI latent_formats.SDXL), used for live previews.
    static let previewFactors = MLXArray(converting: [
        0.3651, 0.4232, 0.4341,
        -0.2533, -0.0042, 0.1068,
        0.1076, 0.1111, -0.0362,
        -0.3165, -0.2492, -0.2188,
    ]).reshaped(4, 3)
    static let previewBias = MLXArray(converting: [0.1084, -0.0175, -0.0011])

    /// `preview` gets a small uint8 RGB image [h/8, w/8, 3] of the predicted result after every
    /// sampling step (base and hi-res pass). It is computed in the same evaluation as the step.
    public func generate(
        _ p: SDXLParameters,
        hiresUpscale: HiresUpscaler? = nil,
        isCancelled: () -> Bool = { false },
        preview: ((MLXArray) -> Void)? = nil,
        progress: (GenerationPhase, Int, Int) -> Void = { _, _, _ in }
    ) throws -> (image: MLXArray, timings: GenerationTimings) {
        let stepPreview = preview.map { deliver in
            StepPreview(
                make: { d in
                    let rgb = matmul(d[0].asType(.float32), Self.previewFactors) + Self.previewBias
                    return (clip((rgb + 1) / 2, min: 0, max: 1) * 255).round().asType(.uint8)
                },
                deliver: deliver)
        }
        defer { Memory.clearCache() }
        var tm = GenerationTimings()
        let t0 = Date()
        MLXRandom.seed(p.seed)

        // --- text conditioning
        progress(.encode, 0, 1)
        let (context, pooled) = encodeText(p)
        tm.textEncode = Date().timeIntervalSince(t0)

        // --- base pass
        let w8 = p.width / 8 * 8
        let h8 = p.height / 8 * 8
        let sigmas = schedule.sigmas(steps: p.steps, scheduler: p.scheduler)
        let noise = MLXRandom.normal([1, h8 / 8, w8 / 8, 4])
        let x = noise * (1 + sigmas[0] * sigmas[0]).squareRoot()   // EPS max-denoise noise scaling

        var ts = Date()
        let cache = p.stepCache.map { StepCache(threshold: $0, sigmas: sigmas) }
        guard var latent = sample(
            p.sampler,
            denoise: denoiser(context: context, pooled: pooled, cfg: p.cfg, width: w8, height: h8, cache: cache,
                              guided: p.guidanceRule),
            x: x, sigmas: sigmas, isCancelled: isCancelled, preview: stepPreview, progress: { progress(.sample, $0, $1) })
        else { throw WaiCancelled() }
        tm.skippedSteps += cache?.skipped ?? 0
        tm.sampling = Date().timeIntervalSince(ts)
        tm.samplingSteps = p.steps

        // --- hi-res fix
        if let scale = p.hiresScale, scale > 1, p.hiresSteps > 0, p.hiresDenoise > 0 {
            ts = Date()
            let hw = Int((Float(w8) * scale / 8).rounded()) * 8
            let hh = Int((Float(h8) * scale / 8).rounded()) * 8
            let image = vae.decode(latent.asType(vaeDType)).asType(.float32)   // [-1, 1], NHWC
            var skipped = 0
            latent = try refinePass(
                image, context: context, pooled: pooled, p, width: hw, height: hh, upscale: hiresUpscale,
                isCancelled: isCancelled, preview: stepPreview, progress: progress, skipped: &skipped)
            tm.skippedSteps += skipped
            tm.hiresSampling = Date().timeIntervalSince(ts)
            tm.hiresSteps = p.hiresSteps
        }

        // --- decode
        ts = Date()
        progress(.decode, 0, 1)
        var img = vae.decode(latent.asType(vaeDType)).asType(.float32)
        img = (clip(img / 2 + 0.5, min: 0, max: 1) * 255).round().asType(.uint8)
        let out = img[0]
        eval(out)
        tm.decode = Date().timeIntervalSince(ts)
        tm.total = Date().timeIntervalSince(t0)
        return (out, tm)
    }

    /// Upscale + detail pass on an existing image (gallery "Upscale"): uses `p.prompt`,
    /// `p.negativePrompt`, `p.seed`, `p.cfg`, sampler/scheduler and `p.hiresSteps` / `p.hiresDenoise`.
    /// `image` is uint8 [H, W, 3]; returns uint8 [height, width, 3] (sizes rounded to multiples of 8).
    public func refine(
        _ image: MLXArray, _ p: SDXLParameters, width: Int, height: Int,
        upscale: HiresUpscaler? = nil,
        isCancelled: () -> Bool = { false },
        preview: ((MLXArray) -> Void)? = nil,
        progress: (GenerationPhase, Int, Int) -> Void = { _, _, _ in }
    ) throws -> MLXArray {
        defer { Memory.clearCache() }
        MLXRandom.seed(p.seed)
        progress(.encode, 0, 1)
        let (context, pooled) = encodeText(p)
        let stepPreview = preview.map { deliver in
            StepPreview(
                make: { d in
                    let rgb = matmul(d[0].asType(.float32), Self.previewFactors) + Self.previewBias
                    return (clip((rgb + 1) / 2, min: 0, max: 1) * 255).round().asType(.uint8)
                },
                deliver: deliver)
        }
        let src = image.asType(.float32)[.newAxis] / 127.5 - 1      // [-1, 1], NHWC
        var skipped = 0
        let latent = try refinePass(
            src, context: context, pooled: pooled, p, width: width / 8 * 8, height: height / 8 * 8, upscale: upscale,
            isCancelled: isCancelled, preview: stepPreview, progress: progress, skipped: &skipped)
        progress(.decode, 0, 1)
        let img = vae.decode(latent.asType(vaeDType)).asType(.float32)
        let out = (clip(img / 2 + 0.5, min: 0, max: 1) * 255).round().asType(.uint8)[0]
        eval(out)
        return out
    }

    func encodeText(_ p: SDXLParameters) -> (MLXArray, MLXArray) {
        let cond = encodeSDXL(p.prompt, tokenizer: tokenizer, clipL: clipL, clipG: clipG)
        let uncond = encodeSDXL(p.negativePrompt, tokenizer: tokenizer, clipL: clipL, clipG: clipG)
        let (ctxC, ctxU) = matchContextLength(cond.context, uncond.context)
        let context = concatenated([ctxC, ctxU], axis: 0).asType(dtype)
        let pooled = concatenated([cond.pooled, uncond.pooled], axis: 0).asType(dtype)
        eval(context, pooled)
        return (context, pooled)
    }

    /// Hi-res style pass: upscale `image` ([-1, 1] NHWC) to width x height (model upscaler or
    /// bicubic), VAE-encode it and denoise the tail of the schedule (like ComfyUI's KSampler denoise).
    private func refinePass(
        _ image: MLXArray, context: MLXArray, pooled: MLXArray, _ p: SDXLParameters, width hw: Int, height hh: Int,
        upscale: HiresUpscaler?, isCancelled: () -> Bool, preview: StepPreview?,
        progress: (GenerationPhase, Int, Int) -> Void, skipped: inout Int
    ) throws -> MLXArray {
        var image = image
        if let upscale {
            image = try upscale(clip(image / 2 + 0.5, min: 0, max: 1), hw, hh) * 2 - 1
        }
        if image.dim(1) != hh || image.dim(2) != hw { image = resize(image, height: hh, width: hw) }
        var (x0, _) = vae.encode(clip(image, min: -1, max: 1).asType(vaeDType))
        x0 = x0.asType(.float32)
        let total = Int(Float(p.hiresSteps) / min(p.hiresDenoise, 1))
        let full = schedule.sigmas(steps: total, scheduler: p.scheduler)
        let hs = Array(full.suffix(p.hiresSteps + 1))
        let xh = x0 + MLXRandom.normal(x0.shape) * hs[0]
        let cache = p.stepCache.map { StepCache(threshold: $0, sigmas: hs) }
        guard let latent = sample(
            p.sampler,
            denoise: denoiser(context: context, pooled: pooled, cfg: p.cfg, width: hw, height: hh, cache: cache,
                              guided: p.hiresGuidanceRule, guidanceEvery: p.hiresGuidanceEvery),
            x: xh, sigmas: hs, isCancelled: isCancelled, preview: preview, progress: { progress(.hires, $0, $1) })
        else { throw WaiCancelled() }
        skipped = cache?.skipped ?? 0
        return latent
    }

    /// Run the UNet through `MLX.compile` (fuses its many small element-wise kernels). The model is
    /// passed as state, so LoRA weight changes are picked up; MLX recompiles when the image size changes.
    /// Off: WAIGEN_COMPILE=0.
    public var compileUNet = ProcessInfo.processInfo.environment["WAIGEN_COMPILE"] != "0"

    private lazy var compiledUNet: ([MLXArray]) -> [MLXArray] = {
        let unet = self.unet
        return compile(inputs: [unet], outputs: []) { a in
            [unet(a[0], timestep: a[1], encoderX: a[2], textTime: (a[3], a[4]))]
        }
    }()

    /// Seconds per sampling step per image (UNet with CFG, i.e. 2 passes per image) for `batch` images at
    /// once, measured over `steps` steps after one warm-up step. For deciding on batching / compile.
    public func benchmarkUNet(batch: Int, width: Int, height: Int, steps: Int = 6, compiled: Bool) -> Double {
        let (h8, w8) = (height / 8, width / 8)
        let x = MLXRandom.normal([2 * batch, h8, w8, 4]).asType(dtype)
        let context = MLXRandom.normal([2 * batch, 77, 2048]).asType(dtype)
        let pooled = MLXRandom.normal([2 * batch, 1280]).asType(dtype)
        let ids = tiled(MLXArray([Float(height), Float(width), 0, 0, Float(height), Float(width)]).reshaped(1, 6),
                        repetitions: [2 * batch, 1]).asType(dtype)
        let t = MLXArray(Array(repeating: Float(500), count: 2 * batch))
        let step: () -> MLXArray = compiled
            ? { self.compiledUNet([x, t, context, pooled, ids])[0] }
            : { self.unet(x, timestep: t, encoderX: context, textTime: (pooled, ids)) }
        eval(step())
        let t0 = Date()
        for _ in 0 ..< steps { eval(step()) }
        return Date().timeIntervalSince(t0) / Double(steps * batch)
    }

    /// CFG denoiser around the UNet (eps prediction -> denoised latent).
    /// `cache`: EasyCache step skipping on the UNet output (see `StepCache`).
    /// `guided`: which sigmas get classifier-free guidance (nil = all, the original graph); the others run the
    /// UNet once on the prompt. With a rule the step cache works on the guided prediction, not the CFG pair.
    /// `guidanceEvery` > 1: guidance is computed on every n-th guided call; the calls between run the UNet on
    /// the prompt alone and add the last guidance direction: eps_c + (cfg − 1)(eps_c − eps_u)_last.
    func denoiser(context: MLXArray, pooled: MLXArray, cfg: Float, width: Int, height: Int,
                  cache: StepCache? = nil, guided: ((Float) -> Bool)? = nil, guidanceEvery: Int = 1) -> Denoiser {
        let ids = MLXArray([Float(height), Float(width), 0, 0, Float(height), Float(width)])
            .reshaped(1, 6)
        let textTime = (pooled, tiled(ids, repetitions: [2, 1]).asType(dtype))
        let unet = self.unet
        let schedule = self.schedule
        let dtype = self.dtype
        let compiled = compileUNet ? compiledUNet : nil
        if let guided {
            let condContext = context[0 ..< 1]
            let condText = (textTime.0[0 ..< 1], textTime.1[0 ..< 1])
            final class Reuse { var direction: MLXArray?; var calls = 0 }
            let reuse = Reuse()
            return { x, sigma in
                let cIn = 1 / (sigma * sigma + 1).squareRoot()
                let t = schedule.timestep(sigma)
                let x1 = (x * cIn).asType(dtype)
                let predict: () -> MLXArray = {
                    let wantsGuidance = guided(sigma)
                    defer { if wantsGuidance { reuse.calls += 1 } }
                    if wantsGuidance, reuse.direction == nil || reuse.calls % max(guidanceEvery, 1) == 0 {
                        let xIn = concatenated([x1, x1], axis: 0)
                        let timesteps = MLXArray([t, t])
                        let eps = compiled.map { $0([xIn, timesteps, context, textTime.0, textTime.1])[0] }
                            ?? unet(xIn, timestep: timesteps, encoderX: context, textTime: textTime)
                        let (epsC, epsU) = eps.split()
                        if guidanceEvery > 1 { reuse.direction = epsC - epsU }
                        return epsU + cfg * (epsC - epsU)
                    }
                    let timesteps = MLXArray([t])
                    let epsC = compiled.map { $0([x1, timesteps, condContext, condText.0, condText.1])[0] }
                        ?? unet(x1, timestep: timesteps, encoderX: condContext, textTime: condText)
                    if wantsGuidance, let direction = reuse.direction { return epsC + (cfg - 1) * direction }
                    return epsC
                }
                let e = (cache.map { $0(x1.asType(.float32), sigma: sigma, compute: predict) } ?? predict()).asType(x.dtype)
                return x - sigma * e
            }
        }
        return { x, sigma in
            let cIn = 1 / (sigma * sigma + 1).squareRoot()
            let t = schedule.timestep(sigma)
            let xIn = (concatenated([x, x], axis: 0) * cIn).asType(dtype)
            let timesteps = MLXArray([t, t])
            let run = {
                compiled.map { $0([xIn, timesteps, context, textTime.0, textTime.1])[0] }
                    ?? unet(xIn, timestep: timesteps, encoderX: context, textTime: textTime)
            }
            // the cache sees the scaled input in float32 (as stable-diffusion.cpp's does); without it the
            // graph is exactly the uncached one
            let eps = cache.map { $0(xIn.asType(.float32), sigma: sigma, compute: run) } ?? run()
            let (epsC, epsU) = eps.split()
            let e = (epsU + cfg * (epsC - epsU)).asType(x.dtype)
            return x - sigma * e
        }
    }

    /// Bicubic resize of an NHWC image to an exact size.
    func resize(_ x: MLXArray, height: Int, width: Int) -> MLXArray {
        let sy = Float(height) / Float(x.dim(1))
        let sx = Float(width) / Float(x.dim(2))
        let up = Upsample(scaleFactor: [sy, sx], mode: .cubic(alignCorners: false))
        let y = up(x)
        return y[0..., ..<height, ..<width, 0...]
    }
}
