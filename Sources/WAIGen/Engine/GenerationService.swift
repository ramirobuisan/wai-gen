import CoreGraphics
import Foundation
import ImageIO
import MLX
import WaiMLX
import WaiSDCpp

struct JobLoRA: Sendable {
    let url: URL
    let relativePath: String
    let strength: Double
}

struct FaceRepairOptions: Sendable {
    let strength: Double
    let confidence: Double
    let prompt: String
}

/// Gallery "Upscale": start from an existing image instead of noise.
struct UpscaleRequest: Sendable {
    let source: URL
    let width: Int
    let height: Int
    /// false = upscaler model / resize only, no diffusion pass
    let detailPass: Bool
}

/// A fully resolved request (prompt already assembled, seed fixed).
struct GenerationJob: Identifiable, Sendable {
    let id = UUID()
    let preset: ModelPreset
    let modelsRoot: URL
    let positive: String
    let negative: String
    let characters: [String]
    let userPrompt: String
    let width: Int
    let height: Int
    let steps: Int
    let cfg: Double
    let sampler: Sampler
    let scheduler: Scheduler
    let seed: Int64
    let hires: (scale: Double, steps: Int, denoise: Double)?
    let faceRepair: FaceRepairOptions?
    let loras: [JobLoRA]
    /// ESRGAN-family model for the hi-res upscale; nil = built-in resize
    let hiresUpscaler: URL?
    let livePreview: Bool
    var upscale: UpscaleRequest? = nil
}

enum JobProgress: Sendable {
    case loadingModel(String)
    case encoding
    case sampling(step: Int, total: Int, hires: Bool)
    case decoding
    case upscaling
    case preparingHires
    case loadingFaceRepair
    case faceRepair(step: Int, total: Int)
    case preview(CGImage)
}

struct JobResult: Sendable {
    let image: CGImage
    let seconds: Double
    let faceRepairSeconds: Double?
}

extension ModelPreset: @unchecked Sendable {}

/// Runs jobs one at a time on a dedicated queue and keeps the models loaded between jobs: the selected
/// image model with its helpers (upscaler, face detector, face-repair context). With "keep every image
/// model loaded" (off by default) the other image models stay in memory too, so switching is instant.
/// `preload` loads them in the background, always after any waiting job.
final class GenerationService: @unchecked Sendable {
    static let keepLoadedKey = "keepModelsLoaded"
    static let preloadKey = "preloadModels"
    static var keepAllLoaded: Bool { UserDefaults.standard.object(forKey: keepLoadedKey) as? Bool ?? false }

    private let queue = DispatchQueue(label: "waigen.engine", qos: .userInitiated)
    // everything below is only touched on `queue`
    private var mlx: [String: WaiSDXL] = [:]
    private var sdcpp: [String: SDCppEngine] = [:]
    /// MLX diffusion models standing in for stable-diffusion.cpp's (WAI-ANIMA), by preset id
    private var mlxDiT: [String: MLXDiTBridge] = [:]
    private var detector: SDCppFaceDetector?
    private var upscalers: [URL: ModelUpscaler] = [:]
    private var preloadSteps: [(name: String, load: () throws -> Void)] = []
    private var names: [String: String] = [:]   // preset id -> name, for the "in memory" list

    /// Called (on the engine queue) when the set of loaded models or the background loading changes.
    var onModelsChanged: (@Sendable (_ loaded: [String], _ loading: String?) -> Void)?

    private let lock = NSLock()
    private var cancelled = false
    private var activeSDCpp: SDCppEngine?
    private var waitingJobs = 0

    /// Stops any running job and frees every model. Call before the app exits: ggml's Metal backend
    /// asserts at process exit (in a static destructor) that no GPU buffers are still alive, so quitting
    /// with stable-diffusion.cpp contexts loaded aborts ("quit unexpectedly"). False if the running job
    /// didn't stop within `timeout`.
    func shutdown(timeout: TimeInterval = 20) -> Bool {
        cancel()
        let done = DispatchSemaphore(value: 0)
        queue.async {
            self.setActive(nil)
            self.preloadSteps.removeAll()
            self.sdcpp.removeAll()
            self.mlxDiT.removeAll()
            self.detector = nil
            self.upscalers.removeAll()
            self.mlx.removeAll()
            done.signal()
        }
        return done.wait(timeout: .now() + timeout) == .success
    }

    func cancel() {
        lock.lock()
        cancelled = true
        let sd = activeSDCpp
        lock.unlock()
        sd?.cancel()
    }

    private var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return cancelled
    }

    private func setActive(_ e: SDCppEngine?) {
        lock.lock()
        activeSDCpp = e
        lock.unlock()
    }

    func run(_ job: GenerationJob, progress: @escaping @Sendable (JobProgress) -> Void) async throws -> JobResult {
        lock.lock()
        cancelled = false
        waitingJobs += 1
        lock.unlock()
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                self.lock.lock()
                self.waitingJobs -= 1
                self.lock.unlock()
                defer { self.schedulePreload() }   // carry on loading the other models afterwards
                do {
                    let result: JobResult
                    if let up = job.upscale {
                        result = try self.executeUpscale(job, up, progress: progress)
                    } else {
                        result = try self.execute(job, progress: progress)
                    }
                    cont.resume(returning: result)
                } catch is WaiCancelled {
                    cont.resume(throwing: CancellationError())
                } catch SDCppError.cancelled {
                    cont.resume(throwing: CancellationError())
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    // MARK: engines

    private func path(_ job: GenerationJob, _ rel: String?) -> URL { job.modelsRoot.appendingPathComponent(rel ?? "") }

    // MARK: model cache

    /// Unless every model may stay loaded, drop the ones `keep` doesn't need.
    private func makeRoom(keeping id: String) {
        guard !Self.keepAllLoaded else { return }
        mlx = mlx.filter { $0.key == id }
        sdcpp = sdcpp.filter { $0.key == id }
        mlxDiT = mlxDiT.filter { $0.key == id }
        Memory.clearCache()
    }

    private func loadMLX(_ p: ModelPreset, _ root: URL) throws -> WaiSDXL {
        if let e = mlx[p.id] { return e }
        makeRoom(keeping: p.id)
        let e = try WaiSDXL(checkpoint: root.appendingPathComponent(p.checkpoint ?? ""))
        mlx[p.id] = e
        names[p.id] = p.name
        Memory.clearCache()
        reportModels()
        return e
    }

    private func loadSDCpp(_ p: ModelPreset, _ root: URL) throws -> SDCppEngine {
        if let e = sdcpp[p.id] { return e }
        makeRoom(keeping: p.id)
        func file(_ rel: String?) -> URL { root.appendingPathComponent(rel ?? "") }
        let e = try SDCppEngine(diffusionModel: file(p.diffusionModel), llm: file(p.llm), vae: file(p.vae))
        // its VAE runs in MLX (sd.cpp's Metal VAE takes minutes at hi-res sizes); sd.cpp's own is the fallback
        if let vae = try? QwenImageVAE(url: file(p.vae)) { e.externalVAE = MLXVAEBridge(vae) }
        // and so does the diffusion model (sd.cpp then never loads its own copy of the weights)
        if let dit = try? AnimaDiT(url: file(p.diffusionModel)) {
            let bridge = MLXDiTBridge(dit)
            e.externalDiffusion = bridge
            if e.externalDiffusionActive { mlxDiT[p.id] = bridge } else { e.externalDiffusion = nil }
        }
        sdcpp[p.id] = e
        names[p.id] = p.name
        reportModels()
        return e
    }

    private func loadDetector(_ root: URL) throws -> SDCppFaceDetector {
        if let d = detector { return d }
        let d = try SDCppFaceDetector(model: root.appendingPathComponent("detectors/face_yolov8m.safetensors"))
        detector = d
        return d
    }

    private func reportModels(loading: String? = nil) {
        let loaded = Set(mlx.keys).union(sdcpp.keys).compactMap { names[$0] }.sorted()
        onModelsChanged?(loaded, loading)
    }

    /// Load these models in the background, one step at a time, each step queued behind any job that
    /// is waiting, so generating never waits for more than the step already running.
    func preload(_ presets: [ModelPreset], modelsRoot root: URL, faceRepair: Bool, upscaler: URL?) {
        queue.async {
            var steps: [(String, () throws -> Void)] = []
            var sharedAdded = false
            for p in presets where p.missingFiles(in: root).isEmpty {
                switch p.engine {
                case .mlx:
                    steps.append((p.name, { _ = try self.loadMLX(p, root) }))   // repaints faces itself
                case .sdcpp:
                    steps.append((p.name, { _ = try self.loadSDCpp(p, root) }))
                }
                if !sharedAdded {   // right after the first model: what its first image needs
                    sharedAdded = true
                    if faceRepair { steps.append(("face detector", { _ = try self.loadDetector(root) })) }
                    if let upscaler {
                        steps.append(("upscaler", {
                            if self.upscalers[upscaler] == nil { self.upscalers[upscaler] = try ModelUpscaler(url: upscaler) }
                        }))
                    }
                }
            }
            self.preloadSteps = steps.map { (name: $0.0, load: $0.1) }
            self.schedulePreload()
        }
    }

    /// Queue the next preload step (on the engine queue, behind anything already queued).
    private func schedulePreload() {
        queue.async {
            self.lock.lock()
            let busy = self.waitingJobs > 0
            self.lock.unlock()
            guard !busy, !self.preloadSteps.isEmpty else {
                if self.preloadSteps.isEmpty { self.reportModels() }
                return
            }
            let step = self.preloadSteps.removeFirst()
            self.reportModels(loading: step.name)
            do { try step.load() } catch {
                FileHandle.standardError.write(Data("preload \(step.name): \(error.localizedDescription)\n".utf8))
            }
            self.schedulePreload()
        }
    }

    /// Free every model now (they load again when needed).
    func unloadAll() {
        queue.async {
            self.preloadSteps.removeAll()
            self.mlx.removeAll()
            self.sdcpp.removeAll()
            self.mlxDiT.removeAll()
            self.detector = nil
            self.upscalers.removeAll()
            Memory.clearCache()
            self.reportModels()
        }
    }

    // MARK: engines for a job

    private func mlxEngine(_ job: GenerationJob, _ progress: (JobProgress) -> Void) throws -> WaiSDXL {
        let p = job.preset
        if mlx[p.id] == nil { progress(.loadingModel(p.name)) }
        let engine = try loadMLX(p, job.modelsRoot)
        let report = try engine.setLoRAs(job.loras.map { LoRASpec(url: $0.url, strength: Float($0.strength)) })
        if !report.unmatched.isEmpty || report.unsupported > 0 {
            FileHandle.standardError.write(Data(
                "LoRA: \(report.unmatched.count) unmatched layers, \(report.unsupported) unsupported tensors (LoHa/LoKr)\n".utf8))
        }
        return engine
    }

    private func sdcppEngine(_ job: GenerationJob, _ progress: (JobProgress) -> Void) throws -> SDCppEngine {
        let p = job.preset
        if sdcpp[p.id] == nil { progress(.loadingModel(p.name)) }
        let engine = try loadSDCpp(p, job.modelsRoot)
        // LoRAs are applied inside stable-diffusion.cpp's own model: use it for jobs with LoRAs
        if let bridge = mlxDiT[p.id] {
            let wanted: SDCppExternalDiffusion? = job.loras.isEmpty ? bridge : nil
            if (engine.externalDiffusion == nil) != (wanted == nil) { engine.externalDiffusion = wanted }
        }
        return engine
    }

    private func modelUpscaler(_ url: URL) throws -> ModelUpscaler {
        if let up = upscalers[url] { return up }
        let up = try ModelUpscaler(url: url)
        upscalers[url] = up
        return up
    }

    /// Hi-res upscale hook for the MLX engine: the upscaler model (x4), then a high-quality
    /// resample to the hi-res size.
    private func mlxUpscaler(_ url: URL) throws -> WaiSDXL.HiresUpscaler {
        let up = try modelUpscaler(url)
        return { img, w, h in
            let u8 = (img[0] * 255).round().asType(.uint8)
            let rgb = RGBImage(width: u8.dim(1), height: u8.dim(0), pixels: u8.asData(access: .copy).data)
            let big = try up.upscale(rgb).resized(width: w, height: h)
            return MLXArray(big.pixels, [1, h, w, 3], type: UInt8.self).asType(.float32) / 255
        }
    }

    private func mlxPreview(_ job: GenerationJob, _ progress: @escaping (JobProgress) -> Void) -> ((MLXArray) -> Void)? {
        guard job.livePreview else { return nil }
        return { arr in
            if let cg = ImageFiles.cgImage(rgb: arr.asData(access: .copy).data, width: arr.dim(1), height: arr.dim(0)) {
                progress(.preview(cg))
            }
        }
    }

    private func sdPreview(_ job: GenerationJob, _ progress: @escaping (JobProgress) -> Void) -> ((RGBImage) -> Void)? {
        guard job.livePreview else { return nil }
        return { img in
            if let cg = ImageFiles.cgImage(rgb: img.pixels, width: img.width, height: img.height) { progress(.preview(cg)) }
        }
    }

    private func mlxPhase(_ progress: @escaping (JobProgress) -> Void) -> (GenerationPhase, Int, Int) -> Void {
        { phase, i, n in
            switch phase {
            case .encode: progress(.encoding)
            case .sample: progress(.sampling(step: i, total: n, hires: false))
            case .hires: progress(.sampling(step: i, total: n, hires: true))
            case .decode: progress(.decoding)
            }
        }
    }

    /// `hasHiresPass`: a second sampling pass will follow the first one, so a step counter that
    /// starts over means the hi-res pass has begun. Without it every pass is the base pass.
    private func sdProgress(_ progress: @escaping (JobProgress) -> Void, firstPassIsHires: Bool = false,
                            hasHiresPass: Bool = false)
        -> (String, Int64, Int64) -> Void
    {
        var hiresPass = firstPassIsHires
        var lastDone: Int64 = -1
        // between the upscale and the first hi-res step sd.cpp re-reads the prompt and VAE-encodes the
        // upscaled image without step events (minutes for WAI-ANIMA at 1536x2016): keep one label for it
        var preparingHires = false
        return { stage, done, total in
            let s = stage.lowercased()
            if s.contains("decode") {
                progress(.decoding)
            } else if s.contains("upscal") {
                // hi-res upscale between the passes (the ESRGAN model can take a while on big images)
                if hasHiresPass { hiresPass = true; preparingHires = true }
                progress(.upscaling)
            } else if s.contains("sampl") || s.contains("denois"), total > 0 {
                if hasHiresPass, lastDone > 1, done < lastDone { hiresPass = true }
                lastDone = done
                preparingHires = false
                progress(.sampling(step: Int(done), total: Int(total), hires: hiresPass))
            } else if preparingHires {
                progress(.preparingHires)
            } else if s.contains("cond") || s.contains("encod") || s.contains("text") {
                progress(.encoding)
            }
        }
    }

    private func sdxlParameters(_ job: GenerationJob) -> SDXLParameters {
        var sp = SDXLParameters(prompt: job.positive)
        sp.negativePrompt = job.negative
        sp.width = job.width
        sp.height = job.height
        sp.steps = job.steps
        sp.cfg = Float(job.cfg)
        sp.sampler = SamplerKind(rawValue: job.sampler.rawValue) ?? .eulerAncestral
        sp.scheduler = SchedulerKind(rawValue: job.scheduler.rawValue) ?? .normal
        sp.seed = UInt64(bitPattern: job.seed)
        if let h = job.hires {
            sp.hiresScale = Float(h.scale)
            sp.hiresSteps = h.steps
            sp.hiresDenoise = Float(h.denoise)
        }
        return sp
    }

    private func sdcppParameters(_ job: GenerationJob) -> SDCppParameters {
        var sp = SDCppParameters(prompt: job.positive)
        sp.loras = job.loras.map { SDCppLoRA(path: $0.url.path, multiplier: Float($0.strength)) }
        sp.negativePrompt = job.negative
        sp.width = job.width
        sp.height = job.height
        sp.steps = job.steps
        sp.cfg = Float(job.cfg)
        sp.sampler = job.sampler.sdcppName
        sp.scheduler = job.scheduler.sdcppName
        sp.seed = job.seed
        return sp
    }

    // MARK: text to image

    private func execute(_ job: GenerationJob, progress: @escaping (JobProgress) -> Void) throws -> JobResult {
        let t0 = Date()
        var image: RGBImage

        switch job.preset.engine {
        case .mlx:
            let engine = try mlxEngine(job, progress)
            let hiresUpscale = try job.hires != nil ? job.hiresUpscaler.map { try mlxUpscaler($0) } : nil
            let (array, _) = try engine.generate(
                sdxlParameters(job), hiresUpscale: hiresUpscale, isCancelled: { self.isCancelled },
                preview: mlxPreview(job, progress), progress: mlxPhase(progress))
            image = RGBImage(width: array.dim(1), height: array.dim(0), pixels: array.asData(access: .copy).data)

        case .sdcpp:
            let engine = try sdcppEngine(job, progress)
            if isCancelled { throw CancellationError() }
            setActive(engine)
            defer { setActive(nil) }
            var sp = sdcppParameters(job)
            if let h = job.hires, let u = job.hiresUpscaler, let up = try? modelUpscaler(u), up.runsInMLX {
                // hi-res fix done here so the upscaler runs in MLX: base pass, upscale, img2img at the new size
                let base = try engine.generate(sp, preview: sdPreview(job, progress), progress: sdProgress(progress))
                if isCancelled { throw CancellationError() }
                progress(.upscaling)
                let tw = Int(Double(job.width) * h.scale / 16) * 16
                let th = Int(Double(job.height) * h.scale / 16) * 16
                sp.initImage = try up.upscale(base).resized(width: tw, height: th)
                sp.width = tw
                sp.height = th
                sp.strength = Float(h.denoise)
                sp.steps = Int((Double(h.steps) / max(h.denoise, 0.05)).rounded(.up))   // sd.cpp runs steps x strength
                if isCancelled { throw CancellationError() }
                image = try engine.generate(sp, preview: sdPreview(job, progress), progress: sdProgress(progress, firstPassIsHires: true))
            } else {
                if let h = job.hires {
                    sp.hiresScale = Float(h.scale)
                    sp.hiresSteps = h.steps
                    sp.hiresDenoise = Float(h.denoise)
                    sp.hiresModelPath = job.hiresUpscaler?.path
                }
                image = try engine.generate(sp, preview: sdPreview(job, progress),
                                            progress: sdProgress(progress, hasHiresPass: job.hires != nil))
            }
        }

        let repair = try repairFaces(image, job, progress)
        return try result(repair.image, start: t0, faceRepairSeconds: repair.seconds)
    }

    // MARK: gallery upscale

    private func executeUpscale(_ job: GenerationJob, _ up: UpscaleRequest, progress: @escaping (JobProgress) -> Void)
        throws -> JobResult
    {
        let t0 = Date()
        guard let source = loadRGB(up.source) else { throw SDCppError.generation("couldn't read \(up.source.lastPathComponent)") }

        if !up.detailPass {
            progress(.upscaling)
            var img = source
            if let u = job.hiresUpscaler { img = try modelUpscaler(u).upscale(img) }
            return try result(img.resized(width: up.width, height: up.height), start: t0, faceRepairSeconds: nil)
        }

        let steps = job.hires?.steps ?? 20
        let denoise = job.hires?.denoise ?? 0.4
        var image: RGBImage
        switch job.preset.engine {
        case .mlx:
            let engine = try mlxEngine(job, progress)
            var sp = sdxlParameters(job)
            sp.hiresSteps = steps
            sp.hiresDenoise = Float(denoise)
            let src = MLXArray(source.pixels, [source.height, source.width, 3], type: UInt8.self)
            progress(.upscaling)
            let out = try engine.refine(
                src, sp, width: up.width, height: up.height, upscale: try job.hiresUpscaler.map { try mlxUpscaler($0) },
                isCancelled: { self.isCancelled }, preview: mlxPreview(job, progress), progress: mlxPhase(progress))
            image = RGBImage(width: out.dim(1), height: out.dim(0), pixels: out.asData(access: .copy).data)

        case .sdcpp:
            let engine = try sdcppEngine(job, progress)
            progress(.upscaling)
            var img = source
            if let u = job.hiresUpscaler { img = try modelUpscaler(u).upscale(img) }
            img = img.resized(width: up.width / 16 * 16, height: up.height / 16 * 16)
            if isCancelled { throw CancellationError() }
            setActive(engine)
            defer { setActive(nil) }
            var sp = sdcppParameters(job)
            sp.width = img.width
            sp.height = img.height
            sp.initImage = img
            sp.strength = Float(denoise)
            sp.steps = Int((Double(steps) / max(denoise, 0.05)).rounded(.up))   // sd.cpp runs steps x strength
            image = try engine.generate(sp, preview: sdPreview(job, progress), progress: sdProgress(progress, firstPassIsHires: true))
        }

        let repair = try repairFaces(image, job, progress)
        return try result(repair.image, start: t0, faceRepairSeconds: repair.seconds)
    }

    // MARK: shared

    private func repairFaces(_ image: RGBImage, _ job: GenerationJob, _ progress: @escaping (JobProgress) -> Void)
        throws -> (image: RGBImage, seconds: Double?)
    {
        guard let fr = job.faceRepair else { return (image, nil) }
        if isCancelled { throw CancellationError() }
        let tf = Date()
        let detector = try loadDetector(job.modelsRoot)
        if job.preset.engine == .mlx {
            // WAI v16 / v12: sd.cpp's YOLO only finds the faces; the MLX engine (already loaded, LoRAs
            // applied) repaints them the way sd.cpp's ADetailer would, ~3x faster
            let faces = try detector.detect(image, confidence: Float(fr.confidence))
                .map { FaceBox(x1: $0.x1, y1: $0.y1, x2: $0.x2, y2: $0.y2, confidence: $0.confidence) }
            guard !faces.isEmpty else { return (image, Date().timeIntervalSince(tf)) }
            let engine = try loadMLX(job.preset, job.modelsRoot)
            var p = sdxlParameters(job)
            if !fr.prompt.isEmpty { p.prompt = fr.prompt }
            var settings = FaceRepairSettings()
            settings.strength = Float(fr.strength)
            let fixed = try engine.repairFaces(
                rgb: image.pixels, width: image.width, height: image.height, faces: faces, p, settings: settings,
                isCancelled: { self.isCancelled }) { _, step, total in progress(.faceRepair(step: step, total: total)) }
            return (RGBImage(width: image.width, height: image.height, pixels: fixed), Date().timeIntervalSince(tf))
        }
        let engine = try sdcppEngine(job, progress)
        setActive(engine)
        defer { setActive(nil) }
        var dp = SDCppDetailParameters(prompt: fr.prompt.isEmpty ? job.positive : fr.prompt)
        dp.loras = job.loras.map { SDCppLoRA(path: $0.url.path, multiplier: Float($0.strength)) }
        dp.negativePrompt = job.negative
        dp.steps = job.steps
        dp.cfg = Float(job.cfg)
        dp.sampler = job.sampler.sdcppName
        dp.scheduler = job.scheduler.sdcppName
        dp.seed = job.seed
        dp.strength = Float(fr.strength)
        dp.confidence = Float(fr.confidence)
        let fixed = try engine.detail(image, detector: detector, dp) { stage, done, total in
            if total > 0, !stage.lowercased().contains("decode") {
                progress(.faceRepair(step: Int(done), total: Int(total)))
            }
        }
        return (fixed, Date().timeIntervalSince(tf))
    }

    private func result(_ image: RGBImage, start: Date, faceRepairSeconds: Double?) throws -> JobResult {
        guard let cg = ImageFiles.cgImage(rgb: image.pixels, width: image.width, height: image.height) else {
            throw SDCppError.generation("could not build the image")
        }
        return JobResult(image: cg, seconds: Date().timeIntervalSince(start), faceRepairSeconds: faceRepairSeconds)
    }

    /// Read any image file as RGB8.
    private func loadRGB(_ url: URL) -> RGBImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { return nil }
        let w = cg.width, h = cg.height
        var rgba = Data(count: w * h * 4)
        let ok = rgba.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
            else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { return nil }
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
