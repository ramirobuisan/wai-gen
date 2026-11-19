// WAI-ANIMA generation and ADetailer face repair through the stable-diffusion.cpp C API.
// The library is vendored at vendor/stable-diffusion.cpp and built with the Metal backend.

import CStableDiffusion
import CoreGraphics
import Foundation

public enum SDCppError: LocalizedError {
    case setup(String)
    case generation(String)
    case cancelled

    public var errorDescription: String? {
        switch self {
        case .setup(let m): "stable-diffusion.cpp setup failed: \(m)"
        case .generation(let m): "stable-diffusion.cpp generation failed: \(m)"
        case .cancelled: "Generation cancelled"
        }
    }
}

public struct SDCppLoRA: Hashable, Sendable {
    public let path: String
    public let multiplier: Float
    public init(path: String, multiplier: Float) {
        self.path = path
        self.multiplier = multiplier
    }
}

private func setLoRAs(_ opt: OpaquePointer, _ loras: [SDCppLoRA]) throws {
    let active = loras.filter { $0.multiplier != 0 }
    guard !active.isEmpty else { return }
    let paths = active.map { strdup($0.path)! }
    defer { paths.forEach { free($0) } }
    var list = zip(active, paths).map { sd_lora_t(is_high_noise: false, multiplier: $0.0.multiplier, path: UnsafePointer($0.1)) }
    try check(sd_opt_set_loras(opt, "loras", &list, list.count), "loras")
}

/// stable-diffusion.cpp's step cache ("cache.mode" plus its "cache.*" knobs). Step-level modes
/// (easycache, spectrum) skip whole model calls, so they also work with an external diffusion model.
public struct SDCppStepCache: Sendable, Equatable {
    /// easycache, spectrum, taylorseer, dbcache, cache_dit, ucache
    public var mode: String
    /// keys without the "cache." prefix, e.g. ["reuse_threshold": 0.2] or ["spectrum_warmup_steps": 4]
    public var options: [String: Double]
    public init(mode: String, options: [String: Double] = [:]) {
        self.mode = mode
        self.options = options
    }

    /// "mode" or "mode:key=value,key=value"
    public init?(argument: String) {
        let parts = argument.split(separator: ":", maxSplits: 1)
        guard let mode = parts.first.map(String.init), !mode.isEmpty else { return nil }
        var options = [String: Double]()
        if parts.count > 1 {
            for kv in parts[1].split(separator: ",") {
                let pair = kv.split(separator: "=")
                guard pair.count == 2, let v = Double(pair[1]) else { return nil }
                options[String(pair[0])] = v
            }
        }
        self.init(mode: mode, options: options)
    }
}

private func setStepCache(_ opt: OpaquePointer, _ cache: SDCppStepCache?) throws {
    guard let cache, cache.mode != "disabled" else { return }
    try check(sd_opt_set_string(opt, "cache.mode", cache.mode), "cache.mode")
    for (k, v) in cache.options.sorted(by: { $0.key < $1.key }) {
        let key = "cache." + k
        // the options are typed: try float, then int, then bool
        if sd_opt_set_float(opt, key, v) == SD_STATUS_OK { continue }
        if v == v.rounded(), sd_opt_set_int(opt, key, Int64(v)) == SD_STATUS_OK { continue }
        try check(sd_opt_set_bool(opt, key, v != 0), key)
    }
}

public struct SDCppParameters: Sendable {
    public var loras: [SDCppLoRA] = []
    /// nil = no step cache
    public var stepCache: SDCppStepCache? = nil
    /// img2img: start from this image (must already be width x height); `strength` = denoise.
    public var initImage: RGBImage? = nil
    public var strength: Float = 0.4
    public var prompt: String
    public var negativePrompt = ""
    public var width = 1024
    public var height = 1344
    public var steps = 25
    public var cfg: Float = 4.5
    /// stable-diffusion.cpp names: euler, euler_a, dpm++2m, ...
    public var sampler = "euler_a"
    /// nil = library default for the model (Anima: discrete, flow mode)
    public var scheduler: String? = nil
    public var seed: Int64 = 0
    public var hiresScale: Float? = nil
    public var hiresSteps = 20
    public var hiresDenoise: Float = 0.4
    /// C API names (lowercase): "lanczos", "nearest", "latent", "latent_bicubic", ...
    /// Ignored when `hiresModelPath` (ESRGAN) is set, which uses "model".
    public var hiresUpscaler = "lanczos"
    public var hiresModelPath: String? = nil

    public init(prompt: String) { self.prompt = prompt }
}

/// ADetailer pass settings (detect faces, repaint each crop).
public struct SDCppDetailParameters: Sendable {
    public var loras: [SDCppLoRA] = []
    public var stepCache: SDCppStepCache? = nil
    public var prompt: String
    public var negativePrompt = ""
    public var steps = 20
    public var cfg: Float = 6
    public var sampler = "euler_a"
    public var scheduler: String? = nil
    public var seed: Int64 = 0
    /// denoising strength of the repaint
    public var strength: Float = 0.4
    public var confidence: Float = 0.3
    /// crop is resized to this square for the repaint (1024 suits SDXL / Anima)
    public var inpaintSize = 1024
    public var padding = 32
    public var maskBlur = 4

    public init(prompt: String) { self.prompt = prompt }
}

/// One RGB8 image (interleaved rows).
public struct RGBImage: Sendable {
    public let width: Int
    public let height: Int
    public let pixels: Data

    public init(width: Int, height: Int, pixels: Data) {
        self.width = width
        self.height = height
        self.pixels = pixels
    }
}

private func lastError() -> String {
    if let e = sd_get_last_error() { return String(cString: e) }
    return "unknown error"
}

private func check(_ status: sd_status_t, _ what: String) throws {
    if status != SD_STATUS_OK { throw SDCppError.setup("\(what): \(lastError())") }
}

/// WAIGEN_SDCPP_LOG=info shows stable-diffusion.cpp's info lines too (its per-stage timings).
private let showInfoLog = ProcessInfo.processInfo.environment["WAIGEN_SDCPP_LOG"] == "info"

private let installLogger: Void = {
    sd_set_log_callback({ level, text, _ in
        guard let text else { return }
        if showInfoLog, level.rawValue >= SD_LOG_INFO.rawValue {
            FileHandle.standardError.write(Data("[sd.cpp] \(String(cString: text))".utf8))
            return
        }
        let line = String(cString: text)
        // warnings, plus the step cache's one-line summaries ("EasyCache skipped 6/25 steps ...")
        let cacheInfo = ["EasyCache", "UCache", "CacheDIT", "Spectrum"].contains { line.contains($0) }
        guard level.rawValue >= SD_LOG_WARN.rawValue || cacheInfo else { return }
        FileHandle.standardError.write(Data("[sd.cpp] \(line)".utf8))
    }, nil)
}()

/// Copy the first image out of a result (U8 tensor [C, W, H] = interleaved rows) as RGB.
private func rgbImage(status: sd_status_t, result: OpaquePointer?) throws -> RGBImage {
    if status == SD_STATUS_CANCELLED { throw SDCppError.cancelled }
    guard status == SD_STATUS_OK, let result, sd_result_get_image_count(result) > 0,
        let tensor = sd_result_get_image(result, 0), let image = rgbFromTensor(tensor)
    else { throw SDCppError.generation(lastError()) }
    return image
}

/// U8 image tensor [C, W, H] (interleaved rows) -> RGB.
private func rgbFromTensor(_ tensor: OpaquePointer) -> RGBImage? {
    guard sd_tensor_rank(tensor) >= 3, let data = sd_tensor_data(tensor) else { return nil }
    let channels = Int(sd_tensor_dim(tensor, 0))
    let width = Int(sd_tensor_dim(tensor, 1))
    let height = Int(sd_tensor_dim(tensor, 2))
    let raw = Data(bytes: data, count: sd_tensor_byte_size(tensor))
    if channels == 3 { return RGBImage(width: width, height: height, pixels: raw) }
    var rgb = Data(count: width * height * 3)
    raw.withUnsafeBytes { src in
        rgb.withUnsafeMutableBytes { dst in
            let s = src.bindMemory(to: UInt8.self)
            let d = dst.bindMemory(to: UInt8.self)
            for i in 0 ..< width * height {
                for c in 0 ..< 3 { d[i * 3 + c] = s[i * channels + min(c, channels - 1)] }
            }
        }
    }
    return RGBImage(width: width, height: height, pixels: rgb)
}

/// ESRGAN-family upscaler (e.g. RealESRGAN_x4plus_anime_6B.pth), run on the GPU by stable-diffusion.cpp.
public final class SDCppUpscaler: @unchecked Sendable {
    private var handle: OpaquePointer?
    public let scale: Int

    /// Scale factor if stable-diffusion.cpp can load this upscaler file, else nil.
    public static func probeScale(_ model: URL) -> Int? {
        let s = Int(sd_upscaler_probe_scale(model.path))
        return s > 0 ? s : nil
    }

    public init(model: URL, tileSize: Int = 256) throws {
        _ = installLogger
        guard let opt = sd_upscaler_opt_create() else { throw SDCppError.setup("out of memory") }
        defer { sd_opt_free(opt) }
        try check(sd_opt_set_string(opt, "model_path", model.path), "upscaler model_path")
        try check(sd_opt_set_int(opt, "tile_size", Int64(tileSize)), "tile_size")
        var h: OpaquePointer?
        try check(sd_upscaler_create(opt, &h), "loading upscaler")
        handle = h
        scale = Int(sd_upscaler_get_scale(h))
    }

    deinit { sd_upscaler_free(handle) }

    public func upscale(_ image: RGBImage) throws -> RGBImage {
        var shape: [Int64] = [3, Int64(image.width), Int64(image.height)]
        let tensor = image.pixels.withUnsafeBytes { buf in
            sd_tensor_create(sd_tensor_type_t(SD_TENSOR_U8), 3, &shape, buf.baseAddress, buf.count)
        }
        guard let tensor else { throw SDCppError.generation("could not wrap the image: \(lastError())") }
        defer { sd_tensor_free(tensor) }
        guard let opt = sd_upscale_opt_create() else { throw SDCppError.setup("out of memory") }
        defer { sd_opt_free(opt) }
        try check(sd_opt_set_tensor(opt, "image", tensor), "image")
        var result: OpaquePointer?
        let status = sd_upscale(handle, opt, &result)
        defer { sd_result_free(result) }
        return try rgbImage(status: status, result: result)
    }
}

extension RGBImage {
    /// High-quality resample (Core Graphics) to an exact size.
    public func resized(width w: Int, height h: Int) -> RGBImage {
        if w == width, h == height { return self }
        guard let provider = CGDataProvider(data: pixels as CFData),
            let src = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                              bytesPerRow: width * 3, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider, decode: nil,
                              shouldInterpolate: true, intent: .defaultIntent)
        else { return self }
        var rgba = Data(count: w * h * 4)
        rgba.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
            else { return }
            ctx.interpolationQuality = .high
            ctx.draw(src, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        var rgb = Data(count: w * h * 3)
        rgba.withUnsafeBytes { s in
            rgb.withUnsafeMutableBytes { d in
                let sp = s.bindMemory(to: UInt8.self)
                let dp = d.bindMemory(to: UInt8.self)
                for i in 0 ..< w * h { dp[i * 3] = sp[i * 4]; dp[i * 3 + 1] = sp[i * 4 + 1]; dp[i * 3 + 2] = sp[i * 4 + 2] }
            }
        }
        return RGBImage(width: w, height: h, pixels: rgb)
    }
}

/// YOLOv8 face detector for ADetailer (see scripts/convert_yolov8_safe.py).
public final class SDCppFaceDetector: @unchecked Sendable {
    fileprivate var handle: OpaquePointer?

    public init(model: URL) throws {
        _ = installLogger
        guard let opt = sd_adetailer_opt_create() else { throw SDCppError.setup("out of memory") }
        defer { sd_opt_free(opt) }
        try check(sd_opt_set_string(opt, "model_path", model.path), "detector model_path")
        var h: OpaquePointer?
        try check(sd_adetailer_create(opt, &h), "loading face detector")
        handle = h
    }

    /// Run only the YOLO detector (WAI Gen sd.cpp patch): face boxes as (x1, y1, x2, y2, confidence)
    /// in pixels, in ADetailer's order.
    public func detect(_ image: RGBImage, confidence: Float = 0.3) throws -> [(x1: Float, y1: Float, x2: Float, y2: Float, confidence: Float)] {
        var boxes = [Float](repeating: 0, count: 64 * 5)
        let n = image.pixels.withUnsafeBytes { raw in
            sd_adetailer_detect(handle, raw.bindMemory(to: UInt8.self).baseAddress, Int32(image.width), Int32(image.height),
                                confidence, &boxes, 64)
        }
        guard n >= 0 else { throw SDCppError.generation("face detection failed") }
        return (0 ..< min(Int(n), 64)).map { i in
            (boxes[i * 5], boxes[i * 5 + 1], boxes[i * 5 + 2], boxes[i * 5 + 3], boxes[i * 5 + 4])
        }
    }

    deinit { sd_adetailer_free(handle) }
}

/// A VAE that stable-diffusion.cpp should use instead of its own for single images (needs the WAI Gen
/// patch, patches/sdcpp-vae-hook.patch). Buffers are NCHW float32; shapes are [N, C, H, W].
/// Return false to let stable-diffusion.cpp use its built-in VAE for that call.
public protocol SDCppExternalVAE: AnyObject {
    /// image in 0...1 -> VAE latent (before the model's latent normalisation)
    func encode(_ input: UnsafePointer<Float>, shape: [Int], into output: UnsafeMutablePointer<Float>, outputShape: [Int]) -> Bool
    /// VAE latent -> image in 0...1
    func decode(_ input: UnsafePointer<Float>, shape: [Int], into output: UnsafeMutablePointer<Float>, outputShape: [Int]) -> Bool
}

/// A diffusion model that stable-diffusion.cpp should run instead of its own (WAI Gen patch; Anima).
/// Buffers are float32 in C order: x [N, C, H, W], context [N, L, D]; write the prediction (same shape
/// as x) to `output`. Return false to let stable-diffusion.cpp run its built-in model for that call.
public protocol SDCppExternalDiffusion: AnyObject {
    func forward(x: UnsafePointer<Float>, shape: [Int], timesteps: UnsafePointer<Float>, nTimesteps: Int,
                 context: UnsafePointer<Float>, contextShape: [Int], tokenIDs: UnsafePointer<Int32>?,
                 tokenWeights: UnsafePointer<Float>?, nTokens: Int, into output: UnsafeMutablePointer<Float>) -> Bool
}

/// Owns one stable-diffusion.cpp context (models stay loaded between generations).
public final class SDCppEngine: @unchecked Sendable {
    private var ctx: OpaquePointer?
    private var progressHandler: ((String, Int64, Int64) -> Void)?

    /// Runs the diffusion model's forward passes instead of stable-diffusion.cpp (Anima models only).
    /// Set it while the engine is idle; reading `externalDiffusionActive` tells whether the model took it.
    public private(set) var externalDiffusionActive = false
    public var externalDiffusion: SDCppExternalDiffusion? {
        didSet {
            guard let ctx else { return }
            guard externalDiffusion != nil else {
                _ = sd_set_diffusion_hook(ctx, nil, nil)
                externalDiffusionActive = false
                return
            }
            externalDiffusionActive = sd_set_diffusion_hook(ctx, { x, xs, ts, nts, c, cs, ids, wts, ntok, out, data in
                guard let x, let xs, let ts, let c, let cs, let out, let data else { return false }
                let engine = Unmanaged<SDCppEngine>.fromOpaque(data).takeUnretainedValue()
                guard let model = engine.externalDiffusion else { return false }
                // ggml order -> C order
                let shape = [Int(xs[3]), Int(xs[2]), Int(xs[1]), Int(xs[0])]
                let ctxShape = [Int(cs[2]), Int(cs[1]), Int(cs[0])]
                return model.forward(x: x, shape: shape, timesteps: ts, nTimesteps: Int(nts), context: c,
                                     contextShape: ctxShape, tokenIDs: ids, tokenWeights: wts, nTokens: Int(ntok), into: out)
            }, Unmanaged.passUnretained(self).toOpaque())
        }
    }

    /// Used for every VAE encode/decode of single images in this context (txt2img, img2img, hi-res,
    /// face repair). Set it while the engine is idle.
    public var externalVAE: SDCppExternalVAE? {
        didSet {
            guard let ctx else { return }
            if externalVAE == nil {
                sd_set_vae_hook(ctx, nil, nil)
                return
            }
            sd_set_vae_hook(ctx, { direction, input, inShape, output, outShape, data in
                guard let input, let inShape, let output, let outShape, let data else { return false }
                let engine = Unmanaged<SDCppEngine>.fromOpaque(data).takeUnretainedValue()
                guard let vae = engine.externalVAE else { return false }
                // ggml order [W, H, C, N] -> [N, C, H, W]
                let i = [Int(inShape[3]), Int(inShape[2]), Int(inShape[1]), Int(inShape[0])]
                let o = [Int(outShape[3]), Int(outShape[2]), Int(outShape[1]), Int(outShape[0])]
                return direction == 0
                    ? vae.encode(input, shape: i, into: output, outputShape: o)
                    : vae.decode(input, shape: i, into: output, outputShape: o)
            }, Unmanaged.passUnretained(self).toOpaque())
        }
    }

    /// Anima-style model: standalone diffusion model + LLM text encoder + VAE.
    /// `weightType`: convert the weights at load time (e.g. "q8_0"); nil keeps the file's types.
    public convenience init(diffusionModel: URL, llm: URL, vae: URL, flashAttention: Bool = true,
                            weightType: String? = nil) throws {
        try self.init(paths: ["diffusion_model_path": diffusionModel.path, "llm_path": llm.path, "vae_path": vae.path],
                      flashAttention: flashAttention, weightType: weightType)
    }

    /// Single-file checkpoint (e.g. WAI-illustrious SDXL), used for ADetailer repaints.
    public convenience init(checkpoint: URL, flashAttention: Bool = true) throws {
        try self.init(paths: ["model_path": checkpoint.path], flashAttention: flashAttention)
    }

    private init(paths: [String: String], flashAttention: Bool, weightType: String? = nil) throws {
        _ = installLogger
        guard let opt = sd_ctx_opt_create() else { throw SDCppError.setup("out of memory") }
        defer { sd_opt_free(opt) }
        for (key, value) in paths { try check(sd_opt_set_string(opt, key, value), key) }
        try check(sd_opt_set_bool(opt, "flash_attn", flashAttention), "flash_attn")
        // Merging LoRAs into the weights ("immediately", the default) aborts the process on Metal in
        // this stable-diffusion.cpp version ("pre-allocated tensor ... cannot run the operation (ADD)").
        // Applying them during compute works.
        try check(sd_opt_set_string(opt, "lora_apply_mode", "at_runtime"), "lora_apply_mode")
        if let weightType { try check(sd_opt_set_string(opt, "wtype", weightType), "wtype") }
        var c: OpaquePointer?
        try check(sd_ctx_create(opt, &c), "loading models")
        ctx = c

        sd_set_generation_progress_callback(ctx, { event, data in
            guard let event, let data else { return }
            let engine = Unmanaged<SDCppEngine>.fromOpaque(data).takeUnretainedValue()
            let stage = event.pointee.stage.map { String(cString: $0) } ?? ""
            engine.progressHandler?(stage, event.pointee.completed, event.pointee.total)
        }, Unmanaged.passUnretained(self).toOpaque())
    }

    deinit {
        if let ctx {
            sd_set_generation_progress_callback(ctx, nil, nil)
            sd_set_vae_hook(ctx, nil, nil)
            _ = sd_set_diffusion_hook(ctx, nil, nil)
            sd_ctx_free(ctx)
        }
    }

    /// Ask a running `generate` / `detail` to stop as soon as possible (thread-safe).
    public func cancel() {
        sd_cancel_generation(ctx, SD_CANCEL_ALL)
    }

    private var previewHandler: ((RGBImage) -> Void)?

    /// Blocking generation. `progress(stage, completed, total)` is called from the worker thread.
    /// `preview` gets a small latent-projection image of the predicted result after every step.
    public func generate(
        _ p: SDCppParameters,
        preview: ((RGBImage) -> Void)? = nil,
        progress: @escaping (String, Int64, Int64) -> Void = { _, _, _ in }
    ) throws -> RGBImage {
        guard let ctx else { throw SDCppError.setup("no context") }
        sd_cancel_generation(ctx, SD_CANCEL_RESET)
        progressHandler = progress
        defer { progressHandler = nil }
        if let preview {
            // the preview callback is process-wide: register it only for this generation
            previewHandler = preview
            sd_set_preview_callback({ _, frameCount, frames, _, data in
                guard let data, frameCount > 0, let frames, let tensor = frames[0] else { return }
                let engine = Unmanaged<SDCppEngine>.fromOpaque(data).takeUnretainedValue()
                if let image = rgbFromTensor(tensor) { engine.previewHandler?(image) }
            }, "proj", 1, true, false, Unmanaged.passUnretained(self).toOpaque())
        }
        defer {
            if preview != nil {
                sd_set_preview_callback(nil, "none", 0, false, false, nil)
                previewHandler = nil
            }
        }

        guard let opt = sd_img_gen_opt_create() else { throw SDCppError.setup("out of memory") }
        defer { sd_opt_free(opt) }
        try check(sd_opt_set_string(opt, "prompt", p.prompt), "prompt")
        try check(sd_opt_set_string(opt, "negative_prompt", p.negativePrompt), "negative_prompt")
        try check(sd_opt_set_int(opt, "width", Int64(p.width)), "width")
        try check(sd_opt_set_int(opt, "height", Int64(p.height)), "height")
        try check(sd_opt_set_int(opt, "seed", p.seed), "seed")
        try check(sd_opt_set_int(opt, "sample.steps", Int64(p.steps)), "sample.steps")
        try check(sd_opt_set_float(opt, "sample.guidance.txt_cfg", Double(p.cfg)), "cfg")
        try check(sd_opt_set_string(opt, "sample.method", p.sampler), "sample.method")
        if let s = p.scheduler { try check(sd_opt_set_string(opt, "sample.scheduler", s), "sample.scheduler") }
        try setLoRAs(opt, p.loras)
        try setStepCache(opt, p.stepCache)
        if let initImage = p.initImage {
            var shape: [Int64] = [3, Int64(initImage.width), Int64(initImage.height)]
            let tensor = initImage.pixels.withUnsafeBytes { buf in
                sd_tensor_create(sd_tensor_type_t(SD_TENSOR_U8), 3, &shape, buf.baseAddress, buf.count)
            }
            guard let tensor else { throw SDCppError.generation("could not wrap the image: \(lastError())") }
            defer { sd_tensor_free(tensor) }   // the options keep their own reference
            try check(sd_opt_set_tensor(opt, "init_image", tensor), "init_image")
            try check(sd_opt_set_float(opt, "strength", Double(p.strength)), "strength")
        }
        if let scale = p.hiresScale, scale > 1 {
            try check(sd_opt_set_bool(opt, "hires.enabled", true), "hires.enabled")
            try check(sd_opt_set_float(opt, "hires.scale", Double(scale)), "hires.scale")
            try check(sd_opt_set_int(opt, "hires.steps", Int64(p.hiresSteps)), "hires.steps")
            try check(sd_opt_set_float(opt, "hires.denoising_strength", Double(p.hiresDenoise)), "hires.denoising_strength")
            if let model = p.hiresModelPath {
                try check(sd_opt_set_string(opt, "hires.upscaler", "model"), "hires.upscaler")
                try check(sd_opt_set_string(opt, "hires.model_path", model), "hires.model_path")
            } else {
                try check(sd_opt_set_string(opt, "hires.upscaler", p.hiresUpscaler), "hires.upscaler")
            }
        }

        var result: OpaquePointer?
        let status = sd_generate_image(ctx, opt, &result)
        defer { sd_result_free(result) }
        return try firstImage(status: status, result: result)
    }

    /// ADetailer: detect faces in `image` with `detector` and repaint each one with this context's
    /// model. Returns the repaired image (unchanged when nothing is detected).
    public func detail(
        _ image: RGBImage, detector: SDCppFaceDetector, _ p: SDCppDetailParameters,
        progress: @escaping (String, Int64, Int64) -> Void = { _, _, _ in }
    ) throws -> RGBImage {
        guard let ctx, let det = detector.handle else { throw SDCppError.setup("no context") }
        sd_cancel_generation(ctx, SD_CANCEL_RESET)
        progressHandler = progress
        defer { progressHandler = nil }

        var shape: [Int64] = [3, Int64(image.width), Int64(image.height)]
        let tensor = image.pixels.withUnsafeBytes { buf in
            sd_tensor_create(sd_tensor_type_t(SD_TENSOR_U8), 3, &shape, buf.baseAddress, buf.count)
        }
        guard let tensor else { throw SDCppError.generation("could not wrap the image: \(lastError())") }
        defer { sd_tensor_free(tensor) }

        guard let opt = sd_adetail_opt_create() else { throw SDCppError.setup("out of memory") }
        defer { sd_opt_free(opt) }
        try check(sd_opt_set_tensor(opt, "image", tensor), "image")
        try check(sd_opt_set_string(opt, "prompt", p.prompt), "prompt")
        try check(sd_opt_set_string(opt, "negative_prompt", p.negativePrompt), "negative_prompt")
        try check(sd_opt_set_int(opt, "seed", p.seed), "seed")
        try check(sd_opt_set_int(opt, "sample.steps", Int64(p.steps)), "sample.steps")
        try check(sd_opt_set_float(opt, "sample.guidance.txt_cfg", Double(p.cfg)), "cfg")
        try check(sd_opt_set_string(opt, "sample.method", p.sampler), "sample.method")
        if let s = p.scheduler { try check(sd_opt_set_string(opt, "sample.scheduler", s), "sample.scheduler") }
        try check(sd_opt_set_float(opt, "strength", Double(p.strength)), "strength")
        try setLoRAs(opt, p.loras)
        try setStepCache(opt, p.stepCache)
        let args = "confidence=\(p.confidence),inpaint_width=\(p.inpaintSize),inpaint_height=\(p.inpaintSize),"
            + "inpaint_padding=\(p.padding),mask_blur=\(p.maskBlur)"
        try check(sd_opt_set_string(opt, "adetail.args", args), "adetail.args")

        var result: OpaquePointer?
        let status = sd_adetail(det, ctx, opt, &result)
        defer { sd_result_free(result) }
        return try firstImage(status: status, result: result)
    }

    private func firstImage(status: sd_status_t, result: OpaquePointer?) throws -> RGBImage {
        if status == SD_STATUS_CANCELLED { throw SDCppError.cancelled }
        guard status == SD_STATUS_OK, let result, sd_result_get_image_count(result) > 0,
            let tensor = sd_result_get_image(result, 0), let data = sd_tensor_data(tensor)
        else { throw SDCppError.generation(lastError()) }

        // U8 tensor with ggml dims [channels, width, height] = interleaved rows
        let channels = Int(sd_tensor_dim(tensor, 0))
        let width = Int(sd_tensor_dim(tensor, 1))
        let height = Int(sd_tensor_dim(tensor, 2))
        let raw = Data(bytes: data, count: sd_tensor_byte_size(tensor))
        if channels == 3 { return RGBImage(width: width, height: height, pixels: raw) }
        var rgb = Data(count: width * height * 3)
        raw.withUnsafeBytes { src in
            rgb.withUnsafeMutableBytes { dst in
                let s = src.bindMemory(to: UInt8.self)
                let d = dst.bindMemory(to: UInt8.self)
                for i in 0 ..< width * height {
                    for c in 0 ..< 3 { d[i * 3 + c] = s[i * channels + min(c, channels - 1)] }
                }
            }
        }
        return RGBImage(width: width, height: height, pixels: rgb)
    }
}
