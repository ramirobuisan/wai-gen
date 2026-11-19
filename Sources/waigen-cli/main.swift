// waigen-cli — command-line front end for the WAI engines (testing and benchmarks).
//
// WAI Illustrious (MLX):
//   waigen-cli --model <checkpoint.safetensors> --prompt "..." [options]
// WAI-ANIMA (stable-diffusion.cpp):
//   waigen-cli --diffusion-model <anima.safetensors> --llm <qwen_3_06b_base.safetensors>
//              --vae <qwen_image_vae.safetensors> --prompt "..." [options]
// Options: [--negative "..."] [-W 832] [-H 1216] [--steps 28] [--cfg 6] [--sampler euler_ancestral]
//          [--scheduler normal] [--seed 42] [--hires 1.5 --hires-steps 15 --hires-denoise 0.4]
//          [--repeat N] [-o out.png]

import CoreGraphics
import Foundation
import ImageIO
import MLX
import WaiMLX
import WaiSDCpp

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
    finish(1)
}

var args = Array(CommandLine.arguments.dropFirst())
func take(_ names: String...) -> String? {
    guard let i = args.firstIndex(where: { names.contains($0) }), i + 1 < args.count else { return nil }
    let v = args[i + 1]
    args.removeSubrange(i ... i + 1)
    return v
}

/// Upscaler model: MLX for ESRGAN (RRDBNet) files, stable-diffusion.cpp otherwise (same as the app).
func upscaleImage(_ img: RGBImage, model path: String) throws -> RGBImage {
    if let esrgan = try? RealESRGAN(url: URL(fileURLWithPath: path)) {
        let out = esrgan.upscale(MLXArray(img.pixels, [img.height, img.width, 3], type: UInt8.self))
        return RGBImage(width: out.dim(1), height: out.dim(0), pixels: out.asData(access: .copy).data)
    }
    return try SDCppUpscaler(model: URL(fileURLWithPath: path)).upscale(img)
}

/// MLX Anima model for stable-diffusion.cpp's diffusion forward passes
final class MLXDiTBridge: SDCppExternalDiffusion {
    let model: AnimaDiT
    init(_ model: AnimaDiT) { self.model = model }
    func forward(x: UnsafePointer<Float>, shape: [Int], timesteps: UnsafePointer<Float>, nTimesteps: Int,
                 context: UnsafePointer<Float>, contextShape: [Int], tokenIDs: UnsafePointer<Int32>?,
                 tokenWeights: UnsafePointer<Float>?, nTokens: Int, into output: UnsafeMutablePointer<Float>) -> Bool {
        model.run(x: x, shape: shape, timesteps: timesteps, nTimesteps: nTimesteps, context: context, contextShape: contextShape,
                  tokenIDs: tokenIDs, tokenWeights: tokenWeights, nTokens: nTokens, into: output)
    }
}

/// MLX VAE for stable-diffusion.cpp's single-image encode/decode
final class MLXVAEBridge: SDCppExternalVAE {
    let vae: QwenImageVAE
    init(_ vae: QwenImageVAE) { self.vae = vae }
    func encode(_ input: UnsafePointer<Float>, shape: [Int], into output: UnsafeMutablePointer<Float>, outputShape: [Int]) -> Bool {
        guard outputShape[1] == vae.latentChannels else { return false }
        return vae.encode(nchw: input, n: shape[0], channels: shape[1], height: shape[2], width: shape[3], into: output)
    }
    func decode(_ input: UnsafePointer<Float>, shape: [Int], into output: UnsafeMutablePointer<Float>, outputShape: [Int]) -> Bool {
        guard outputShape[1] == 3 else { return false }
        return vae.decode(nchw: input, n: shape[0], channels: shape[1], height: shape[2], width: shape[3], into: output)
    }
}

let modelPath = take("--model", "-m")
let diffusionPath = take("--diffusion-model")
let llmPath = take("--llm")
let vaePath = take("--vae")
guard let prompt = take("--prompt", "-p") else { fail("--prompt is required") }
let negative = take("--negative", "-n") ?? ""
let width = take("-W", "--width").flatMap(Int.init)
let height = take("-H", "--height").flatMap(Int.init)
let steps = take("--steps").flatMap(Int.init)
let cfg = take("--cfg").flatMap(Float.init)
let samplerName = take("--sampler")
let schedulerName = take("--scheduler")
let seed = take("--seed").flatMap(UInt64.init) ?? UInt64.random(in: 0 ... 0xFFFF_FFFF)
let hires = take("--hires").flatMap(Float.init)
let hiresSteps = take("--hires-steps").flatMap(Int.init)
let hiresDenoise = take("--hires-denoise").flatMap(Float.init)
var loraArgs = [String]()
while let l = take("--lora") { loraArgs.append(l) }   // path[:strength], repeatable
let loraSelfTest = take("--lora-selftest")
let benchUNet = take("--bench-unet")   // WxH, e.g. 1024x1344: UNet s/step per image, batch 1 vs 2, compile off/on
let loras: [LoRASpec] = loraArgs.map { a in
    let parts = a.split(separator: ":")
    let strength = parts.count > 1 ? Float(parts.last!) ?? 1 : 1
    let path = parts.count > 1 && Float(parts.last!) != nil ? parts.dropLast().joined(separator: ":") : a
    return LoRASpec(url: URL(fileURLWithPath: path), strength: strength)
}
let hiresUpscalerPath = take("--hires-upscaler")   // ESRGAN model, e.g. RealESRGAN_x4plus_anime_6B.pth
let previewDir = take("--preview-dir")   // save every step's live preview here (to measure its cost)
let refinePath = take("--refine")        // upscale + detail pass on an existing image (gallery Upscale)
let refineScale = take("--refine-scale").flatMap(Double.init) ?? 1.5
let vaeEngine = take("--vae-engine") ?? "mlx"      // WAI-ANIMA VAE: mlx (default) or sdcpp
let weightType = take("--wtype")                  // WAI-ANIMA: convert weights at load (e.g. q8_0)
let ditEngine = take("--dit-engine") ?? "mlx"      // WAI-ANIMA diffusion model: mlx (default) or sdcpp
let vaeRoundtrip = take("--vae-roundtrip")         // encode + decode an image with the MLX VAE and exit
let esrganTest = take("--esrgan-test")             // upscale an image with --hires-upscaler in MLX and sd.cpp, compare

/// Read an image file as RGB8 (exact size).
func loadRGB(_ path: String) -> RGBImage? {
    guard let src = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
        let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else { return nil }
    let w = cg.width, h = cg.height
    var rgba = Data(count: w * h * 4)
    rgba.withUnsafeMutableBytes { buf in
        let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                            space: CGColorSpace(name: CGColorSpace.sRGB)!,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
        ctx?.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
    }
    var rgb = Data(count: w * h * 3)
    for i in 0 ..< w * h { rgb[i * 3] = rgba[i * 4]; rgb[i * 3 + 1] = rgba[i * 4 + 1]; rgb[i * 3 + 2] = rgba[i * 4 + 2] }
    return RGBImage(width: w, height: h, pixels: rgb)
}
let faceDetector = take("--face-repair")   // path to face_yolov8m.safetensors
let faceEngine = take("--face-engine") ?? "mlx"   // face repainting for SDXL: mlx (default) or sdcpp
let faceStrength = take("--face-strength").flatMap(Float.init) ?? 0.4
let repeats = take("--repeat").flatMap(Int.init) ?? 1
let output = take("--output", "-o") ?? "waigen.png"
if !args.isEmpty { fail("unknown arguments: \(args)") }

func outputURL(_ run: Int) -> URL {
    if repeats == 1 { return URL(fileURLWithPath: output) }
    return URL(fileURLWithPath: "\((output as NSString).deletingPathExtension)_\(run).png")
}

func showProgress(_ label: String, _ i: Int, _ n: Int) {
    FileHandle.standardError.write(Data("\r\(label) \(i)/\(n)   ".utf8))
}

if let esrganTest {
    guard let upscalerPath = hiresUpscalerPath, let img = loadRGB(esrganTest) else { fail("--esrgan-test needs an image and --hires-upscaler") }
    do {
        let tl = Date()
        let mlx = try RealESRGAN(url: URL(fileURLWithPath: upscalerPath))
        print(String(format: "MLX ESRGAN (%d blocks) loaded in %.2fs", mlx.blocks, Date().timeIntervalSince(tl)))
        let x = MLXArray(img.pixels, [img.height, img.width, 3], type: UInt8.self)
        var a = MLXArray(0)
        for run in 1 ... 2 {
            let t = Date()
            a = mlx.upscale(x)
            print(String(format: "MLX run %d: %dx%d -> %dx%d in %.2fs", run, img.width, img.height, a.dim(1), a.dim(0),
                         Date().timeIntervalSince(t)))
        }
        try Image(a).save(url: outputURL(1))
        let ts = Date()
        let ref = try SDCppUpscaler(model: URL(fileURLWithPath: upscalerPath)).upscale(img)
        print(String(format: "sd.cpp: %.2fs", Date().timeIntervalSince(ts)))
        let b = MLXArray(ref.pixels, [ref.height, ref.width, 3], type: UInt8.self)
        let mse = (a.asType(.float32) - b.asType(.float32)).square().mean().item(Float.self)
        print(String(format: "PSNR MLX vs sd.cpp: %.2f dB", 10 * log10(255 * 255 / mse)))
    } catch { fail(error.localizedDescription) }
    finish(0)
}

if let diffusionPath {
    // ---------------- WAI-ANIMA through stable-diffusion.cpp
    guard let llmPath, let vaePath else { fail("--llm and --vae are required with --diffusion-model") }
    if let vaeRoundtrip {
        guard let img = loadRGB(vaeRoundtrip) else { fail("couldn't read \(vaeRoundtrip)") }
        do {
            let tl = Date()
            let vae = try QwenImageVAE(url: URL(fileURLWithPath: vaePath))
            print(String(format: "MLX VAE loaded in %.2fs", Date().timeIntervalSince(tl)))
            let (w, h) = (img.width / 8 * 8, img.height / 8 * 8)
            let src = img.resized(width: w, height: h)
            let x = MLXArray(src.pixels, [1, h, w, 3], type: UInt8.self).asType(.float32) / 127.5 - 1
            for run in 1 ... 2 {
                let te = Date()
                let z = vae.encode(x)
                eval(z)
                let td = Date()
                let y = vae.decode(z)
                eval(y)
                let mse = ((y.asType(.float32) - x) * 0.5).square().mean().item(Float.self)
                print(String(format: "run %d %dx%d: encode %.2fs, decode %.2fs, latent std %.3f, PSNR %.2f dB", run, w, h,
                             td.timeIntervalSince(te), Date().timeIntervalSince(td),
                             z.asType(.float32).variance().sqrt().item(Float.self), 10 * log10(1 / mse)))
                if run == 2 {
                    let out = clip(((y.asType(.float32) + 1) * 127.5).round(), min: 0, max: 255).asType(.uint8)[0]
                    try Image(out).save(url: outputURL(1))
                }
            }
        } catch { fail(error.localizedDescription) }
        finish(0)
    }
    let t0 = Date()
    let engine: SDCppEngine
    do {
        engine = try SDCppEngine(
            diffusionModel: URL(fileURLWithPath: diffusionPath), llm: URL(fileURLWithPath: llmPath),
            vae: URL(fileURLWithPath: vaePath), weightType: weightType)
        if vaeEngine == "mlx" { engine.externalVAE = MLXVAEBridge(try QwenImageVAE(url: URL(fileURLWithPath: vaePath))) }
        if ditEngine == "mlx" {
            let td = Date()
            engine.externalDiffusion = MLXDiTBridge(try AnimaDiT(url: URL(fileURLWithPath: diffusionPath)))
            print(String(format: "MLX Anima model loaded in %.1fs (hook %@)", Date().timeIntervalSince(td),
                         engine.externalDiffusionActive ? "active" : "NOT active"))
        }
    } catch { fail(error.localizedDescription) }
    print(String(format: "loaded %@ in %.1fs", (diffusionPath as NSString).lastPathComponent, Date().timeIntervalSince(t0)))

    if let refinePath {
        // like the app's gallery Upscale for sd.cpp models: upscale, then img2img at the new size
        guard var img = loadRGB(refinePath) else { fail("couldn't read \(refinePath)") }
        let ts = Date()
        do {
            let tw = Int(Double(img.width) * refineScale) / 16 * 16
            let th = Int(Double(img.height) * refineScale) / 16 * 16
            if let hiresUpscalerPath { img = try upscaleImage(img, model: hiresUpscalerPath) }
            img = img.resized(width: tw, height: th)
            var p = SDCppParameters(prompt: prompt)
            p.negativePrompt = negative
            p.width = tw
            p.height = th
            p.cfg = cfg ?? 4.5
            p.sampler = samplerName ?? "euler_a"
            p.seed = Int64(truncatingIfNeeded: seed)
            p.initImage = img
            p.strength = hiresDenoise ?? 0.4
            p.steps = Int((Double(hiresSteps ?? 20) / Double(p.strength)).rounded(.up))
            let out = try engine.generate(p) { stage, done, total in if total > 0 { showProgress(stage, Int(done), Int(total)) } }
            FileHandle.standardError.write(Data("\n".utf8))
            try Image(MLXArray(out.pixels, [out.height, out.width, 3], type: UInt8.self)).save(url: outputURL(1))
            print(String(format: "refine -> %dx%d in %.2fs -> %@", out.width, out.height, Date().timeIntervalSince(ts),
                         outputURL(1).lastPathComponent))
        } catch { fail(error.localizedDescription) }
        finish(0)
    }

    for run in 1 ... repeats {
        var p = SDCppParameters(prompt: prompt)
        p.negativePrompt = negative
        if let width { p.width = width }
        if let height { p.height = height }
        if let steps { p.steps = steps }
        if let cfg { p.cfg = cfg }
        if let samplerName { p.sampler = samplerName }
        p.scheduler = schedulerName
        p.seed = Int64(truncatingIfNeeded: seed &+ UInt64(run - 1))
        if let hires { p.hiresScale = hires }
        if let hiresSteps { p.hiresSteps = hiresSteps }
        if let hiresDenoise { p.hiresDenoise = hiresDenoise }
        p.hiresModelPath = hiresUpscalerPath
        let ts = Date()
        do {
            let img = try engine.generate(p) { stage, done, total in
                if total > 0 { showProgress(stage, Int(done), Int(total)) }
            }
            FileHandle.standardError.write(Data("\n".utf8))
            let array = MLXArray(img.pixels, [img.height, img.width, 3], type: UInt8.self)
            try Image(array).save(url: outputURL(run))
            print(String(format: "run %d seed %lld | %dx%d | total %.2fs -> %@", run, p.seed, img.width,
                         img.height, Date().timeIntervalSince(ts), outputURL(run).lastPathComponent))
        } catch { fail(error.localizedDescription) }
    }
} else {
    // ---------------- WAI Illustrious through MLX
    guard let modelPath else { fail("--model (or --diffusion-model for Anima) is required") }
    var params = SDXLParameters(prompt: prompt)
    params.negativePrompt = negative
    if let width { params.width = width }
    if let height { params.height = height }
    if let steps { params.steps = steps }
    if let cfg { params.cfg = cfg }
    if let samplerName {
        guard let s = SamplerKind(rawValue: samplerName) else { fail("sampler must be one of \(SamplerKind.allCases.map(\.rawValue))") }
        params.sampler = s
    }
    if let schedulerName {
        guard let s = SchedulerKind(rawValue: schedulerName) else { fail("scheduler must be one of \(SchedulerKind.allCases.map(\.rawValue))") }
        params.scheduler = s
    }
    if let hires { params.hiresScale = hires }
    if let hiresSteps { params.hiresSteps = hiresSteps }
    if let hiresDenoise { params.hiresDenoise = hiresDenoise }

    let tLoad = Date()
    let engine: WaiSDXL
    do { engine = try WaiSDXL(checkpoint: URL(fileURLWithPath: modelPath)) } catch {
        fail("loading \(modelPath): \(error.localizedDescription)")
    }
    print(String(format: "loaded %@ in %.1fs", (modelPath as NSString).lastPathComponent, Date().timeIntervalSince(tLoad)))

    if let benchUNet {
        let wh = benchUNet.split(separator: "x").compactMap { Int($0) }
        guard wh.count == 2 else { fail("--bench-unet WxH") }
        for (batch, compiled) in [(1, false), (1, true), (2, false), (2, true), (1, false), (1, true)] {
            let s = engine.benchmarkUNet(batch: batch, width: wh[0], height: wh[1], compiled: compiled)
            print(String(format: "%dx%d batch %d compile %@: %.3f s/step per image", wh[0], wh[1], batch,
                         compiled ? "on " : "off", s))
        }
        finish(0)
    }
    if let loraSelfTest {
        var p = params
        p.seed = seed
        do {
            let base = try engine.generate(p).image
            let r = try engine.setLoRAs([LoRASpec(url: URL(fileURLWithPath: loraSelfTest), strength: 1)])
            print("LoRA report: applied=\(r.applied) files=\(r.files) unsupported=\(r.unsupported) unmatched=\(r.unmatched)")
            let with = try engine.generate(p).image
            try engine.setLoRAs([])
            let after = try engine.generate(p).image
            func diff(_ a: MLXArray, _ b: MLXArray) -> Float { abs(a.asType(.float32) - b.asType(.float32)).mean().item(Float.self) }
            print(String(format: "mean |base - withLoRA| = %.3f (should be > 0)", diff(base, with)))
            print(String(format: "mean |base - afterRemove| = %.4f (should be 0)", diff(base, after)))
            try Image(with).save(url: outputURL(1))
        } catch { fail(error.localizedDescription) }
        finish(0)
    }
    if !loras.isEmpty {
        do {
            let r = try engine.setLoRAs(loras)
            print("LoRA: applied \(r.applied) layers from \(r.files) file(s), unmatched \(r.unmatched.count), unsupported \(r.unsupported)")
        } catch { fail("LoRA: \(error.localizedDescription)") }
    }

    // hi-res upscaler model (MLX for ESRGAN files), resized to the hi-res size
    var hiresUpscale: WaiSDXL.HiresUpscaler?
    if let hiresUpscalerPath {
        print("upscaler: \((hiresUpscalerPath as NSString).lastPathComponent)")
        hiresUpscale = { img, w, h in
            let u8 = (img[0] * 255).round().asType(.uint8)
            let rgb = RGBImage(width: u8.dim(1), height: u8.dim(0), pixels: u8.asData(access: .copy).data)
            let big = try upscaleImage(rgb, model: hiresUpscalerPath).resized(width: w, height: h)
            return MLXArray(big.pixels, [1, h, w, 3], type: UInt8.self).asType(.float32) / 255
        }
    }

    if let refinePath {
        guard let src = loadRGB(refinePath) else { fail("couldn't read \(refinePath)") }
        var p = params
        p.seed = seed
        let w = Int((Double(src.width) * refineScale / 8).rounded()) * 8
        let h = Int((Double(src.height) * refineScale / 8).rounded()) * 8
        let ts = Date()
        do {
            let out = try engine.refine(
                MLXArray(src.pixels, [src.height, src.width, 3], type: UInt8.self), p, width: w, height: h,
                upscale: hiresUpscale) { phase, i, n in if phase == .hires, i > 0 { showProgress("refine", i, n) } }
            FileHandle.standardError.write(Data("\n".utf8))
            try Image(out).save(url: outputURL(1))
            print(String(format: "refine %dx%d -> %dx%d in %.2fs -> %@", src.width, src.height, out.dim(1), out.dim(0),
                         Date().timeIntervalSince(ts), outputURL(1).lastPathComponent))
        } catch { fail(error.localizedDescription) }
        finish(0)
    }

    // face repair: stable-diffusion.cpp ADetailer with the same checkpoint (as the app does)
    var detailEngine: SDCppEngine?
    var detector: SDCppFaceDetector?
    if let faceDetector {
        do {
            detector = try SDCppFaceDetector(model: URL(fileURLWithPath: faceDetector))
            if faceEngine == "sdcpp" { detailEngine = try SDCppEngine(checkpoint: URL(fileURLWithPath: modelPath)) }
        } catch { fail("face repair setup: \(error.localizedDescription)") }
    }

    for run in 1 ... repeats {
        var p = params
        p.seed = seed &+ UInt64(run - 1)
        do {
            var previewCount = 0
            let previewSink: ((MLXArray) -> Void)? = previewDir.map { dir in
                { arr in
                    previewCount += 1
                    if dir == "-" {   // only copy the pixels out, like the app does
                        _ = arr.asData(access: .copy).data
                    } else {
                        try? Image(arr).save(url: URL(fileURLWithPath: dir).appendingPathComponent(String(format: "step_%02d.png", previewCount)))
                    }
                }
            }
            var (image, t) = try engine.generate(p, hiresUpscale: hiresUpscale, preview: previewSink) { phase, i, n in
                if phase == .sample || phase == .hires, i > 0 { showProgress(phase.rawValue, i, n) }
            }
            FileHandle.standardError.write(Data("\n".utf8))
            if detailEngine == nil, let detector {
                // MLX face repair: sd.cpp only finds the faces, this engine repaints them
                let tf = Date()
                try Image(image).save(url: URL(fileURLWithPath: (outputURL(run).path as NSString).deletingPathExtension + "_before.png"))
                let rgb = RGBImage(width: image.dim(1), height: image.dim(0), pixels: image.asData(access: .copy).data)
                let faces = try detector.detect(rgb).map { FaceBox(x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2, confidence: $0.confidence) }
                let td = Date().timeIntervalSince(tf)
                var settings = FaceRepairSettings()
                settings.strength = faceStrength
                let fixed = try engine.repairFaces(rgb: rgb.pixels, width: rgb.width, height: rgb.height, faces: faces, p,
                                                   settings: settings) { _, done, total in showProgress("face", done, total) }
                FileHandle.standardError.write(Data("\n".utf8))
                image = MLXArray(fixed, [rgb.height, rgb.width, 3], type: UInt8.self)
                print(String(format: "face repair (MLX) %.2fs: %d face(s), detection %.2fs", Date().timeIntervalSince(tf),
                             faces.count, td))
            }
            if let detailEngine, let detector {
                let tf = Date()
                try Image(image).save(url: URL(fileURLWithPath: (outputURL(run).path as NSString).deletingPathExtension + "_before.png"))
                let rgb = RGBImage(width: image.dim(1), height: image.dim(0), pixels: image.asData(access: .copy).data)
                var dp = SDCppDetailParameters(prompt: prompt)
                dp.negativePrompt = negative
                dp.steps = p.steps
                dp.cfg = p.cfg
                dp.seed = Int64(truncatingIfNeeded: p.seed)
                dp.strength = faceStrength
                let fixed = try detailEngine.detail(rgb, detector: detector, dp) { _, done, total in
                    if total > 0 { showProgress("face", Int(done), Int(total)) }
                }
                FileHandle.standardError.write(Data("\n".utf8))
                image = MLXArray(fixed.pixels, [fixed.height, fixed.width, 3], type: UInt8.self)
                print(String(format: "face repair %.2fs", Date().timeIntervalSince(tf)))
            }
            try Image(image).save(url: outputURL(run))
            var line = String(
                format: "run %d seed %llu | encode %.2fs | sample %d steps %.2fs (%.3f s/it)",
                run, p.seed, t.textEncode, t.samplingSteps, t.sampling, t.sampling / Double(max(t.samplingSteps, 1)))
            if t.hiresSteps > 0 {
                line += String(format: " | hires %d steps %.2fs (%.3f s/it)", t.hiresSteps, t.hiresSampling,
                               t.hiresSampling / Double(t.hiresSteps))
            }
            line += String(format: " | decode %.2fs | total %.2fs -> %@", t.decode, t.total, outputURL(run).lastPathComponent)
            print(line)
        } catch { fail(error.localizedDescription) }
    }
}

finish(0)

/// Leave without running C++ static destructors: ggml's Metal backend asserts in one of them that every
/// GPU buffer was freed, and the engines here live until exit, so a normal exit() aborts.
func finish(_ code: Int32) -> Never {
    fflush(stdout)
    fflush(stderr)
    _exit(code)
}
