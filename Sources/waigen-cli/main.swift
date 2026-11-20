// waigen-cli — command-line front end for the WAI engines (testing and benchmarks). Everything runs in MLX.
//
// WAI-illustrious SDXL:
//   waigen-cli --model <checkpoint.safetensors> --prompt "..." [options]
// WAI-ANIMA:
//   waigen-cli --diffusion-model <anima.safetensors> --llm <qwen_3_06b_base.safetensors>
//              --vae <qwen_image_vae.safetensors> --prompt "..." [options]
// Options: [--negative "..."] [-W 1024] [-H 1344] [--steps 28] [--cfg 6] [--sampler ...] [--scheduler ...]
//          [--seed 42] [--hires 1.5 --hires-steps 15 --hires-denoise 0.4 --hires-upscaler <esrgan>]
//          [--face-repair <face_yolov8m.safetensors> --face-strength 0.4] [--step-cache 0.5]
//          [--lora path[:strength]]... [--repeat N] [-o out.png]
// Samplers: SDXL euler_ancestral | euler | dpmpp_2m (schedulers normal | karras | simple);
//           WAI-ANIMA euler_a | euler | dpm++2m (schedulers discrete | karras | simple).

import CoreGraphics
import Foundation
import ImageIO
import MLX
import MLXRandom
import WaiMLX
import WaiRemote

func fail(_ msg: String) -> Never {
    FileHandle.standardError.write(Data("error: \(msg)\n".utf8))
    finish(1)
}

func finish(_ code: Int32) -> Never {
    fflush(stdout)
    fflush(stderr)
    exit(code)
}

var args = Array(CommandLine.arguments.dropFirst())
func take(_ names: String...) -> String? {
    guard let i = args.firstIndex(where: { names.contains($0) }), i + 1 < args.count else { return nil }
    let v = args[i + 1]
    args.removeSubrange(i ... i + 1)
    return v
}

/// A switch without a value.
func flag(_ name: String) -> Bool {
    guard let i = args.firstIndex(of: name) else { return false }
    args.remove(at: i)
    return true
}

/// ESRGAN (RRDBNet) upscaler model in MLX, x4.
func upscaleImage(_ img: RGBImage, model path: String) throws -> RGBImage {
    RGBImage(try RealESRGAN(url: URL(fileURLWithPath: path)).upscale(img.array))
}

let modelPath = take("--model", "-m")
let diffusionPath = take("--diffusion-model")
let llmPath = take("--llm")
let vaePath = take("--vae")
let prompt = take("--prompt", "-p") ?? ""
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
let hiresUpscalerPath = take("--hires-upscaler")   // ESRGAN model, e.g. RealESRGAN_x4plus_anime_6B.pth
var loraArgs = [String]()
while let l = take("--lora") { loraArgs.append(l) }   // path[:strength], repeatable
let loras: [LoRASpec] = loraArgs.map { a in
    let parts = a.split(separator: ":")
    let strength = parts.count > 1 ? Float(parts.last!) ?? 1 : 1
    let path = parts.count > 1 && Float(parts.last!) != nil ? parts.dropLast().joined(separator: ":") : a
    return LoRASpec(url: URL(fileURLWithPath: path), strength: strength)
}
let loraSelfTest = take("--lora-selftest")         // SDXL: generate with / without a LoRA and compare
let stepCacheThreshold = take("--step-cache").flatMap(Float.init)   // EasyCache threshold (0.5 in the app)
let previewDir = take("--preview-dir")            // save every step's live preview here
let refinePath = take("--refine")                 // upscale + detail pass on an existing image (gallery Upscale)
let refineScale = take("--refine-scale").flatMap(Double.init) ?? 1.5
let img2imgPath = take("--img2img")               // WAI-ANIMA: img2img from this image at --hires-denoise
let detailOnly = take("--detail-only")            // WAI-ANIMA: only the face repair, on this image
let faceDetector = take("--face-repair")          // path to face_yolov8m.safetensors
let faceStrength = take("--face-strength").flatMap(Float.init) ?? 0.4
let faceSize = take("--face-size").flatMap(Int.init)     // WAI-ANIMA face repaint size (default: the app's 768/1024 rule)
let faceSteps = take("--face-steps").flatMap(Int.init)   // WAI-ANIMA face repair schedule length (default --steps)
let detectPath = take("--detect")                 // print the faces found in these images (comma-separated) and the time
let benchUNet = take("--bench-unet")              // WxH: SDXL UNet s/step, batch 1 vs 2, compile off/on
let benchAnima = take("--bench-anima")            // WxH: Anima forward time per pass
let animaDType: DType = take("--anima-dtype") == "f16" ? .float16 : .bfloat16
let vaeRoundtrip = take("--vae-roundtrip")        // WAI-ANIMA: encode + decode an image with the VAE
let esrganTest = take("--esrgan-test")            // upscale an image with --hires-upscaler and time it
let benchConv = flag("--bench-conv")              // 3x3 convs: MLX's conv2d vs matmul rewrites (time + diff)
let serveTest = take("--serve-test")              // port: a test generator (dummy images) to check the protocol
let remoteAddress = take("--remote")             // host[:port] of a Mac sharing WAI Gen: generate there
let remoteCode = take("--code") ?? ""             // its pairing code
let remotePreset = take("--preset") ?? "wai-v16"  // with --remote: preset id (wai-v16, wai-anima)
let makeLycoris = take("--make-test-lycoris")     // dir: synthetic LoHa / LoKr / LoRA files for WAI-ANIMA
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

func previewSink() -> ((MLXArray) -> Void)? {
    var count = 0
    return previewDir.map { dir in
        { arr in
            count += 1
            if dir == "-" { _ = arr.asData(access: .copy).data } else {   // "-": only copy the pixels out, like the app
                try? Image(arr).save(url: URL(fileURLWithPath: dir).appendingPathComponent(String(format: "step_%02d.png", count)))
            }
        }
    }
}

if benchConv {
    // 3x3 stride-1 convolutions as the models run them, three ways: MLX's conv2d (implicit GEMM, no neural
    // accelerators), im2col + one matmul (9x the input in memory, matmul on the GPU's neural accelerators)
    // and nine shifted matmuls accumulated (no 9x copy, nine smaller matmuls)
    func im2col(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        let (n, h, wd, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3)), o = w.dim(0)
        let p = padded(x, widths: [0, 1, 1, 0])
        var cols: [MLXArray] = []
        for dy in 0 ..< 3 { for dx in 0 ..< 3 { cols.append(p[0..., dy ..< dy + h, dx ..< dx + wd, 0...]) } }
        let col = concatenated(cols, axis: -1).reshaped(n * h * wd, 9 * c)
        return matmul(col, w.reshaped(o, 9 * c).T).reshaped(n, h, wd, o)
    }
    func nineMatmuls(_ x: MLXArray, _ w: MLXArray) -> MLXArray {
        let (n, h, wd, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3)), o = w.dim(0)
        let p = padded(x, widths: [0, 1, 1, 0])
        var acc: MLXArray? = nil
        for dy in 0 ..< 3 {
            for dx in 0 ..< 3 {
                let tap = p[0..., dy ..< dy + h, dx ..< dx + wd, 0...].reshaped(n * h * wd, c)
                let y = matmul(tap, w[0..., dy, dx, 0...].T)
                acc = acc.map { $0 + y } ?? y
            }
        }
        return acc!.reshaped(n, h, wd, o)
    }
    func time(_ f: () -> MLXArray, runs: Int) -> Double {
        eval(f())   // warm-up (kernel compile, allocation)
        let t = Date()
        for _ in 0 ..< runs { eval(f()) }
        return Date().timeIntervalSince(t) / Double(runs)
    }
    // (label, N, H, W, Cin, Cout, dtype)
    let shapes: [(String, Int, Int, Int, Int, Int, DType)] = ProcessInfo.processInfo.environment["WAIGEN_CONV_SET"] == "low" ? [
        ("UNet 1024x1344 lowest res, 1280 -> 1280", 2, 42, 32, 1280, 1280, .float16),
        ("UNet 1024x1344 lowest res, 2560 -> 1280", 2, 42, 32, 2560, 1280, .float16),
        ("UNet 1024x1344 lowest res, 1920 -> 1280", 2, 42, 32, 1920, 1280, .float16),
        ("UNet 1024x1344, 640 res, 1920 -> 640", 2, 84, 64, 1920, 640, .float16),
        ("UNet 1024x1344, 640 res, 1280 -> 640", 2, 84, 64, 1280, 640, .float16),
        ("UNet 832x1216 lowest res, 1280 -> 1280", 2, 38, 26, 1280, 1280, .float16),
        ("UNet hi-res 1536x2016 lowest res, 1280 -> 1280", 2, 63, 48, 1280, 1280, .float16),
        ("UNet hi-res 1536x2016 lowest res, 2560 -> 1280", 2, 63, 48, 2560, 1280, .float16),
        ("UNet face 1024x1024 lowest res, 1280 -> 1280", 2, 32, 32, 1280, 1280, .float16),
        ("UNet 1024x1344 2nd level, 640 -> 640 (control)", 2, 84, 64, 640, 640, .float16),
    ] : [
        ("SDXL UNet 1024x1344, 320 ch", 2, 168, 128, 320, 320, .float16),
        ("SDXL UNet 1024x1344, 640 ch", 2, 84, 64, 640, 640, .float16),
        ("SDXL UNet 1024x1344, 1280 ch", 2, 42, 32, 1280, 1280, .float16),
        ("SDXL UNet hi-res 1536x2016, 320 ch", 2, 252, 192, 320, 320, .float16),
        ("SDXL VAE decode 1024x1344, 512 ch @ 1/4", 1, 336, 256, 512, 512, .bfloat16),
        ("SDXL VAE decode 1024x1344, 256 ch @ 1/2", 1, 672, 512, 256, 256, .bfloat16),
        ("SDXL VAE decode 1024x1344, 128 ch @ full", 1, 1344, 1024, 128, 128, .bfloat16),
        ("Qwen-Image VAE decode, 384 ch @ 1/4", 1, 336, 256, 384, 384, .bfloat16),
        ("Qwen-Image VAE decode, 96 ch @ full", 1, 1344, 1024, 96, 96, .bfloat16),
        ("ESRGAN tile 688x688, 64 -> 32", 1, 688, 688, 64, 32, .float16),
        ("ESRGAN tile 688x688, 160 -> 32", 1, 688, 688, 160, 32, .float16),
        ("ESRGAN tile 688x688, 192 -> 64", 1, 688, 688, 192, 64, .float16),
    ]
    MLXRandom.seed(1)
    print("shape | conv2d | im2col+matmul | 9 matmuls | mean |error| vs float32: conv2d / im2col / 9 matmuls")
    for (label, n, h, wd, c, o, dt) in shapes {
        let x = MLXRandom.normal([n, h, wd, c]).asType(dt)
        let w = (MLXRandom.normal([o, 3, 3, c]) * (1 / Float(9 * c).squareRoot())).asType(dt)
        // float32 reference through im2col + a float32 matmul (no Winograd)
        let exact = im2col(x.asType(.float32), w.asType(.float32))
        let d1 = abs(conv2d(x, w, padding: 1).asType(.float32) - exact).mean().item(Float.self)
        let d2 = abs(im2col(x, w).asType(.float32) - exact).mean().item(Float.self)
        let d3 = abs(nineMatmuls(x, w).asType(.float32) - exact).mean().item(Float.self)
        Memory.clearCache()
        let runs = h * wd > 300_000 ? 5 : 20
        let t0 = time({ conv2d(x, w, padding: 1) }, runs: runs)
        let t1 = time({ im2col(x, w) }, runs: runs)
        let t2 = time({ nineMatmuls(x, w) }, runs: runs)
        print(String(format: "%@ | %.2f ms | %.2f ms (%.2fx) | %.2f ms (%.2fx) | %.3g / %.3g / %.3g", label, t0 * 1000,
                     t1 * 1000, t0 / t1, t2 * 1000, t0 / t2, d1, d2, d3))
        Memory.clearCache()
    }
    finish(0)
}

if let serveTest {
    final class Dummy: RemoteJobHandler, @unchecked Sendable {
        func remoteInfo() async -> RemoteInfo {
            RemoteInfo(name: "test server", presets: ["wai-v16"], loras: [], upscalers: [], faceDetector: false, loadedModels: [], busy: false)
        }
        func runRemote(_ job: RemoteJob, id: UUID, client: String, progress: @escaping @Sendable (RemoteProgress) -> Void,
                       preview: @escaping @Sendable (Data) -> Void) async throws -> (png: Data, seconds: Double, faceRepairSeconds: Double?) {
            print("job from \(client): \(job.width)x\(job.height) \(job.steps) steps")
            for i in 1 ... 3 { progress(RemoteProgress(.sampling, step: i, total: 3)); try await Task.sleep(nanoseconds: 100_000_000) }
            return (Data(repeating: 7, count: 1000), 0.3, nil)
        }
        func cancelRemote(id: UUID) { print("cancel \(id)") }
    }
    let handler = Dummy()
    let server = RemoteServer(handler: handler, code: "TST-123", name: "waigen test", port: UInt16(serveTest) ?? 47861)
    server.onStatus = { print("server: \($0)") }
    server.onActivity = { c, r in print("activity: \(c) \(r)") }
    do { try server.start() } catch { fail(error.localizedDescription) }
    print("test server on port \(server.port), code TST-123")
    dispatchMain()
}

if let remoteAddress {
    // "bonjour:<name>" for a Mac found on the local network, else host[:port]
    let endpoint = remoteAddress.hasPrefix("bonjour:")
        ? RemoteEndpoint(name: String(remoteAddress.dropFirst(8)), service: String(remoteAddress.dropFirst(8)))
        : RemoteEndpoint.address(remoteAddress)
    guard let endpoint else { fail("--remote host[:port] or bonjour:<name>") }
    let client = RemoteClient(endpoint: endpoint, code: remoteCode)
    let semaphore = DispatchSemaphore(value: 0)
    Task {
        do {
            let t0 = Date()
            let info = try await client.info()
            print(String(format: "%@: models %@, %d LoRAs, upscalers %@, face repair %@, in memory %@ (%.2fs)", info.name,
                         info.presets.joined(separator: ", "), info.loras.count, info.upscalers.map(\.name).joined(separator: ", "),
                         info.faceDetector ? "yes" : "no", info.loadedModels.joined(separator: ", "), Date().timeIntervalSince(t0)))
            if prompt.isEmpty { finish(0) }
            let job = RemoteJob(
                presetID: remotePreset, positive: prompt, negative: negative, characters: [], userPrompt: prompt,
                width: width ?? 1024, height: height ?? 1344, steps: steps ?? 28, cfg: Double(cfg ?? 6),
                sampler: samplerName ?? "euler_ancestral", scheduler: schedulerName ?? "normal",
                seed: Int64(truncatingIfNeeded: seed),
                hires: hires.map { .init(scale: Double($0), steps: hiresSteps ?? 15, denoise: Double(hiresDenoise ?? 0.4)) },
                faceRepair: faceDetector != nil ? .init(strength: Double(faceStrength), confidence: 0.3, prompt: "") : nil,
                loras: loras.map { .init(relativePath: $0.url.lastPathComponent, strength: Double($0.strength)) },
                hiresUpscaler: hiresUpscalerPath, livePreview: previewDir != nil, stepCache: stepCacheThreshold != nil,
                upscale: nil)
            var previews = 0
            let ts = Date()
            let r = try await client.run(job, progress: { p in
                FileHandle.standardError.write(Data("\r\(p.kind.rawValue) \(p.step)/\(p.total)\(p.hires ? " hi-res" : "")    ".utf8))
            }, preview: { _ in previews += 1 })
            FileHandle.standardError.write(Data("\n".utf8))
            try r.png.write(to: outputURL(1))
            print(String(format: "done in %.2fs (server %.2fs%@), %d previews, %d KB PNG -> %@", Date().timeIntervalSince(ts),
                         r.seconds, r.faceRepairSeconds.map { String(format: ", face repair %.2fs", $0) } ?? "", previews,
                         r.png.count / 1024, outputURL(1).lastPathComponent))
        } catch { fail(error.localizedDescription) }
        semaphore.signal()
    }
    semaphore.wait()
    finish(0)
}

if let makeLycoris {
    // 12 blocks x 3 attention projections, small random factors (deterministic seed)
    MLXRandom.seed(7)
    var loha: [String: MLXArray] = [:], lokr: [String: MLXArray] = [:], lohaOnes: [String: MLXArray] = [:], loraEq: [String: MLXArray] = [:]
    var lokrFull: [String: MLXArray] = [:], loraKron: [String: MLXArray] = [:]
    for b in 0 ..< 12 {
        for proj in ["q_proj", "k_proj", "v_proj"] {
            let m = "diffusion_model.blocks.\(b).self_attn.\(proj)"
            let r = 8
            loha[m + ".hada_w1_a"] = MLXRandom.normal([2048, r]) * 0.05
            loha[m + ".hada_w1_b"] = MLXRandom.normal([r, 2048]) * 0.05
            loha[m + ".hada_w2_a"] = MLXRandom.normal([2048, r]) * 0.05
            loha[m + ".hada_w2_b"] = MLXRandom.normal([r, 2048]) * 0.05
            loha[m + ".alpha"] = MLXArray(Float(4))
            lokr[m + ".lokr_w1"] = MLXRandom.normal([8, 8]) * 0.05
            lokr[m + ".lokr_w2_a"] = MLXRandom.normal([256, r]) * 0.05
            lokr[m + ".lokr_w2_b"] = MLXRandom.normal([r, 256]) * 0.05
            lokr[m + ".alpha"] = MLXArray(Float(4))
            // LoHa whose second factor is all ones == LoRA(up: w1a, down: w1b); same alpha and rank -> same delta
            let a = MLXRandom.normal([2048, r]) * 0.2, bb = MLXRandom.normal([r, 2048]) * 0.2
            lohaOnes[m + ".hada_w1_a"] = a; lohaOnes[m + ".hada_w1_b"] = bb
            lohaOnes[m + ".hada_w2_a"] = MLXArray.ones([2048, 1]); lohaOnes[m + ".hada_w2_b"] = MLXArray.ones([1, 2048])
            lohaOnes[m + ".alpha"] = MLXArray(Float(4))
            loraEq[m + ".lora_up.weight"] = a; loraEq[m + ".lora_down.weight"] = bb; loraEq[m + ".alpha"] = MLXArray(Float(4))
            // LoKr with w1 = [[1]] (1x1) and full w2 == the full delta w2 (scale 1): same as LoRA(up, down) at alpha = rank
            let u = MLXRandom.normal([2048, r]) * 0.2, dn = MLXRandom.normal([r, 2048]) * 0.2
            lokrFull[m + ".lokr_w1"] = MLXArray.ones([1, 1]); lokrFull[m + ".lokr_w2"] = matmul(u, dn)
            loraKron[m + ".lora_up.weight"] = u; loraKron[m + ".lora_down.weight"] = dn
        }
    }
    do {
        try save(arrays: loha, url: URL(fileURLWithPath: makeLycoris).appendingPathComponent("test_loha.safetensors"))
        try save(arrays: lokr, url: URL(fileURLWithPath: makeLycoris).appendingPathComponent("test_lokr.safetensors"))
        for (name, a) in [("test_loha_ones", lohaOnes), ("test_lora_eq", loraEq), ("test_lokr_full", lokrFull), ("test_lora_kron", loraKron)] {
            try save(arrays: a, url: URL(fileURLWithPath: makeLycoris).appendingPathComponent(name + ".safetensors"))
        }
        print("wrote test_loha / test_lokr (36 modules each)")
    } catch { fail(error.localizedDescription) }
    finish(0)
}

if let detectPath {
    guard let faceDetector else { fail("--detect needs --face-repair <detector.safetensors>") }
    do {
        let det = try YOLOFaceDetector(url: URL(fileURLWithPath: faceDetector))
        for path in detectPath.split(separator: ",").map(String.init) {
            guard let img = RGBImage.load(URL(fileURLWithPath: path)) else { fail("couldn't read \(path)") }
            _ = det.detect(rgb: img.pixels, width: img.width, height: img.height)   // warm-up
            let t = Date()
            let faces = det.detect(rgb: img.pixels, width: img.width, height: img.height)
            print(String(format: "%@ %dx%d: %d face(s) in %.3fs", (path as NSString).lastPathComponent, img.width, img.height,
                         faces.count, Date().timeIntervalSince(t)))
            for f in faces { print(String(format: "  [%.1f %.1f %.1f %.1f] %.3f", f.x1, f.y1, f.x2, f.y2, f.confidence)) }
        }
    } catch { fail(error.localizedDescription) }
    finish(0)
}

if let esrganTest {
    guard let upscalerPath = hiresUpscalerPath, let img = RGBImage.load(URL(fileURLWithPath: esrganTest)) else {
        fail("--esrgan-test needs an image and --hires-upscaler")
    }
    do {
        let tl = Date()
        let model = try RealESRGAN(url: URL(fileURLWithPath: upscalerPath))
        print(String(format: "ESRGAN (%d blocks) loaded in %.2fs", model.blocks, Date().timeIntervalSince(tl)))
        var out = MLXArray(0)
        for run in 1 ... 2 {
            let t = Date()
            out = model.upscale(img.array)
            print(String(format: "run %d: %dx%d -> %dx%d in %.2fs", run, img.width, img.height, out.dim(1), out.dim(0),
                         Date().timeIntervalSince(t)))
        }
        try Image(out).save(url: outputURL(1))
    } catch { fail(error.localizedDescription) }
    finish(0)
}

if let diffusionPath {
    // ---------------- WAI-ANIMA
    guard let llmPath, let vaePath else { fail("WAI-ANIMA needs --diffusion-model, --llm and --vae") }
    if let benchAnima {
        let wh = benchAnima.split(separator: "x").compactMap { Int($0) }
        guard wh.count == 2 else { fail("--bench-anima WxH") }
        do {
            let dit = try AnimaDiT(url: URL(fileURLWithPath: diffusionPath), dtype: animaDType)
            for _ in 0 ..< 2 {
                let r = dit.benchmark(width: wh[0], height: wh[1], batch: 1, compiled: false)
                print(String(format: "%dx%d: %.3f s per forward (adapter %.4f s)", wh[0], wh[1], r.forward, r.adapter))
            }
        } catch { fail(error.localizedDescription) }
        finish(0)
    }
    if let vaeRoundtrip {
        guard let img = RGBImage.load(URL(fileURLWithPath: vaeRoundtrip)) else { fail("couldn't read \(vaeRoundtrip)") }
        do {
            let vae = try QwenImageVAE(url: URL(fileURLWithPath: vaePath))
            let (w, h) = (img.width / 8 * 8, img.height / 8 * 8)
            let x = img.resized(width: w, height: h).array[.newAxis].asType(.float32) / 127.5 - 1
            for run in 1 ... 2 {
                let te = Date()
                let z = vae.encode(x)
                eval(z)
                let td = Date()
                let y = vae.decode(z)
                eval(y)
                let mse = ((y.asType(.float32) - x) * 0.5).square().mean().item(Float.self)
                print(String(format: "run %d %dx%d: encode %.2fs, decode %.2fs, PSNR %.2f dB", run, w, h,
                             td.timeIntervalSince(te), Date().timeIntervalSince(td), 10 * log10(1 / mse)))
                if run == 2 {
                    try Image(clip(((y.asType(.float32) + 1) * 127.5).round(), min: 0, max: 255).asType(.uint8)[0]).save(url: outputURL(1))
                }
            }
        } catch { fail(error.localizedDescription) }
        finish(0)
    }
    do {
        let tl = Date()
        let anima = try WaiAnima(dit: AnimaDiT(url: URL(fileURLWithPath: diffusionPath), dtype: animaDType),
                                 textEncoder: Qwen3TextEncoder(url: URL(fileURLWithPath: llmPath)),
                                 vae: QwenImageVAE(url: URL(fileURLWithPath: vaePath)))
        if !loras.isEmpty {
            let r = try anima.dit.setLoRAs(loras)
            print("LoRA: \(r.applied) layers, \(r.unmatched.count) unmatched")
        }
        print(String(format: "WAI-ANIMA loaded in %.2fs", Date().timeIntervalSince(tl)))
        let detector = try faceDetector.map { try YOLOFaceDetector(url: URL(fileURLWithPath: $0)) }

        for run in 1 ... repeats {
            var p = AnimaParameters(prompt: prompt)
            p.negativePrompt = negative
            if let width { p.width = width }
            if let height { p.height = height }
            if let steps { p.steps = steps }
            if let cfg { p.cfg = cfg }
            if let samplerName {
                guard let s = AnimaSampler(rawValue: samplerName) else { fail("WAI-ANIMA sampler: euler_a, euler or dpm++2m") }
                p.sampler = s
            }
            if let schedulerName {
                guard let s = AnimaScheduler(rawValue: schedulerName) else { fail("WAI-ANIMA scheduler: discrete, karras or simple") }
                p.scheduler = s
            }
            p.seed = Int64(truncatingIfNeeded: seed &+ UInt64(run - 1))
            p.stepCache = stepCacheThreshold
            let ts = Date()
            var line = ""

            /// img2img at `w` x `h`: `n` steps of an n/denoise schedule (the app's hi-res fix and gallery Upscale)
            func refine(_ src: RGBImage, _ w: Int, _ h: Int, _ n: Int, _ denoise: Float) throws -> RGBImage {
                var q = p
                q.width = w / 16 * 16
                q.height = h / 16 * 16
                q.initImage = src.resized(width: q.width, height: q.height).pixels
                q.strength = denoise
                q.steps = Int((Float(n) / max(denoise, 0.05)).rounded(.up))
                let out = try anima.generate(q, preview: previewSink()) { i, k in showProgress("hires", i, k) }
                FileHandle.standardError.write(Data("\n".utf8))
                return RGBImage(out)
            }

            var img: RGBImage
            if let detailOnly {
                guard let src = RGBImage.load(URL(fileURLWithPath: detailOnly)) else { fail("couldn't read \(detailOnly)") }
                img = src
            } else if let refinePath {
                guard let src = RGBImage.load(URL(fileURLWithPath: refinePath)) else { fail("couldn't read \(refinePath)") }
                let up = try hiresUpscalerPath.map { try upscaleImage(src, model: $0) } ?? src
                img = try refine(up, Int(Double(src.width) * refineScale), Int(Double(src.height) * refineScale),
                                 hiresSteps ?? 15, hiresDenoise ?? 0.4)
            } else if let img2imgPath {
                guard let src = RGBImage.load(URL(fileURLWithPath: img2imgPath)) else { fail("couldn't read \(img2imgPath)") }
                p.width = src.width / 16 * 16
                p.height = src.height / 16 * 16
                p.initImage = src.resized(width: p.width, height: p.height).pixels
                p.strength = hiresDenoise ?? 0.4
                img = RGBImage(try anima.generate(p, preview: previewSink()) { i, n in showProgress("img2img", i, n) })
                FileHandle.standardError.write(Data("\n".utf8))
            } else {
                img = RGBImage(try anima.generate(p, preview: previewSink()) { i, n in showProgress("sampling", i, n) })
                FileHandle.standardError.write(Data("\n".utf8))
                line += String(format: " | base %.2fs", Date().timeIntervalSince(ts))
                if anima.lastSkippedSteps > 0 { line += " (cache skipped \(anima.lastSkippedSteps))" }
                if let hires, hires > 1 {
                    let th = Date()
                    let up = try hiresUpscalerPath.map { try upscaleImage(img, model: $0) } ?? img
                    img = try refine(up, Int(Float(p.width) * hires), Int(Float(p.height) * hires), hiresSteps ?? 15,
                                     hiresDenoise ?? 0.4)
                    line += String(format: " | hires %.2fs", Date().timeIntervalSince(th))
                }
            }
            if let detector {
                let tf = Date()
                let faces = detector.detect(rgb: img.pixels, width: img.width, height: img.height)
                var settings = FaceRepairSettings()
                settings.strength = faceStrength
                settings.inpaintSize = faceSize ?? WaiAnima.faceRepaintSize(faces, settings: settings)
                var fp = p
                if let faceSteps { fp.steps = faceSteps }
                let fixed = try anima.repairFaces(rgb: img.pixels, width: img.width, height: img.height, faces: faces, fp,
                                                  settings: settings) { _, i, n in showProgress("face", i, n) }
                FileHandle.standardError.write(Data("\n".utf8))
                img = RGBImage(width: img.width, height: img.height, pixels: fixed)
                line += String(format: " | face repair %.2fs (%d face(s), %d px)", Date().timeIntervalSince(tf), faces.count,
                               settings.inpaintSize)
            }
            try Image(img.array).save(url: outputURL(run))
            print(String(format: "run %d seed %lld | %dx%d%@ | total %.2fs -> %@", run, p.seed, img.width, img.height, line,
                         Date().timeIntervalSince(ts), outputURL(run).lastPathComponent))
        }
    } catch { fail(error.localizedDescription) }
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
    params.stepCache = stepCacheThreshold
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
            let big = try upscaleImage(RGBImage(u8), model: hiresUpscalerPath).resized(width: w, height: h)
            return MLXArray(big.pixels, [1, h, w, 3], type: UInt8.self).asType(.float32) / 255
        }
    }

    if let refinePath {
        guard let src = RGBImage.load(URL(fileURLWithPath: refinePath)) else { fail("couldn't read \(refinePath)") }
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

    let detector = faceDetector.flatMap { try? YOLOFaceDetector(url: URL(fileURLWithPath: $0)) }
    if faceDetector != nil, detector == nil { fail("couldn't load the face detector") }

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
            if let detector {
                let tf = Date()
                try Image(image).save(url: URL(fileURLWithPath: (outputURL(run).path as NSString).deletingPathExtension + "_before.png"))
                let rgb = RGBImage(image)
                let faces = detector.detect(rgb: rgb.pixels, width: rgb.width, height: rgb.height)
                let td = Date().timeIntervalSince(tf)
                var settings = FaceRepairSettings()
                settings.strength = faceStrength
                let fixed = try engine.repairFaces(rgb: rgb.pixels, width: rgb.width, height: rgb.height, faces: faces, p,
                                                   settings: settings) { _, done, total in showProgress("face", done, total) }
                FileHandle.standardError.write(Data("\n".utf8))
                image = MLXArray(fixed, [rgb.height, rgb.width, 3], type: UInt8.self)
                print(String(format: "face repair %.2fs: %d face(s), detection %.3fs", Date().timeIntervalSince(tf), faces.count, td))
            }
            try Image(image).save(url: outputURL(run))
            var line = String(
                format: "run %d seed %llu | encode %.2fs | sample %d steps %.2fs (%.3f s/it)",
                run, p.seed, t.textEncode, t.samplingSteps, t.sampling, t.sampling / Double(max(t.samplingSteps, 1)))
            if t.hiresSteps > 0 {
                line += String(format: " | hires %d steps %.2fs (%.3f s/it)", t.hiresSteps, t.hiresSampling,
                               t.hiresSampling / Double(t.hiresSteps))
            }
            if t.skippedSteps > 0 { line += " | cache skipped \(t.skippedSteps)" }
            line += String(format: " | decode %.2fs | total %.2fs -> %@", t.decode, t.total, outputURL(run).lastPathComponent)
            print(line)
        } catch { fail(error.localizedDescription) }
    }
}

finish(0)
