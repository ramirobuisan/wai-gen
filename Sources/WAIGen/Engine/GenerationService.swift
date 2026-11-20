import CoreGraphics
import Foundation
import ImageIO
import MLX
import WaiMLX

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
    var id = UUID()
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
    /// EasyCache step skipping (GenerationService.sdxlStepCache / animaStepCache)
    var stepCache = false
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

struct GenerationError: LocalizedError {
    let errorDescription: String?
    init(_ message: String) { errorDescription = message }
}

/// Runs jobs one at a time on a dedicated queue and keeps the models loaded between jobs: the selected
/// image model with its helpers (upscaler, face detector). With "keep every image model loaded" (off by
/// default) the other image models stay in memory too, so switching is instant. `preload` loads them in
/// the background, always after any waiting job. Everything runs in MLX: WaiSDXL (WAI-illustrious) and
/// WaiAnima (WAI-ANIMA).
final class GenerationService: @unchecked Sendable {
    static let keepLoadedKey = "keepModelsLoaded"
    static let preloadKey = "preloadModels"
    static var keepAllLoaded: Bool { UserDefaults.standard.object(forKey: keepLoadedKey) as? Bool ?? false }

    private let queue = DispatchQueue(label: "waigen.engine", qos: .userInitiated)
    // everything below is only touched on `queue`
    private var sdxl: [String: WaiSDXL] = [:]
    private var anima: [String: WaiAnima] = [:]
    private var detector: YOLOFaceDetector?
    private var upscalers: [URL: ModelUpscaler] = [:]
    private var preloadSteps: [(name: String, load: () throws -> Void)] = []
    private var names: [String: String] = [:]   // preset id -> name, for the "in memory" list

    /// Called (on the engine queue) when the set of loaded models or the background loading changes.
    var onModelsChanged: (@Sendable (_ loaded: [String], _ loading: String?) -> Void)?

    private let lock = NSLock()
    /// jobs asked to stop (by id: this Mac's jobs and other Macs' jobs share the queue)
    private var cancelledJobs = Set<UUID>()
    private var currentJob: UUID?
    private var waitingJobs = 0

    /// Stops any running job and frees every model (call before the app exits). False if the running job
    /// didn't stop within `timeout`.
    func shutdown(timeout: TimeInterval = 20) -> Bool {
        cancel()
        let done = DispatchSemaphore(value: 0)
        queue.async {
            self.preloadSteps.removeAll()
            self.anima.removeAll()
            self.detector = nil
            self.upscalers.removeAll()
            self.sdxl.removeAll()
            done.signal()
        }
        return done.wait(timeout: .now() + timeout) == .success
    }

    /// Stops the job that is running now.
    func cancel() {
        lock.lock()
        if let c = currentJob { cancelledJobs.insert(c) }
        lock.unlock()
    }

    /// Stops this job (running or still waiting).
    func cancel(jobID: UUID) {
        lock.lock()
        cancelledJobs.insert(jobID)
        lock.unlock()
    }

    private var isCancelled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return currentJob.map { cancelledJobs.contains($0) } ?? false
    }

    func run(_ job: GenerationJob, progress: @escaping @Sendable (JobProgress) -> Void) async throws -> JobResult {
        lock.lock()
        waitingJobs += 1
        lock.unlock()
        return try await withCheckedThrowingContinuation { cont in
            queue.async {
                self.lock.lock()
                self.waitingJobs -= 1
                self.currentJob = job.id
                self.lock.unlock()
                defer {
                    self.lock.lock()
                    self.cancelledJobs.remove(job.id)
                    self.currentJob = nil
                    self.lock.unlock()
                    self.schedulePreload()   // carry on loading the other models afterwards
                }
                do {
                    if self.isCancelled { throw CancellationError() }
                    let result: JobResult
                    if let up = job.upscale {
                        result = try self.executeUpscale(job, up, progress: progress)
                    } else {
                        result = try self.execute(job, progress: progress)
                    }
                    cont.resume(returning: result)
                } catch is WaiCancelled {
                    cont.resume(throwing: CancellationError())
                } catch {
                    cont.resume(throwing: error)
                }
            }
        }
    }

    // MARK: model cache

    /// Unless every model may stay loaded, drop the ones `keep` doesn't need.
    private func makeRoom(keeping id: String) {
        guard !Self.keepAllLoaded else { return }
        sdxl = sdxl.filter { $0.key == id }
        anima = anima.filter { $0.key == id }
        Memory.clearCache()
    }

    private func loadSDXL(_ p: ModelPreset, _ root: URL) throws -> WaiSDXL {
        if let e = sdxl[p.id] { return e }
        makeRoom(keeping: p.id)
        let e = try WaiSDXL(checkpoint: root.appendingPathComponent(p.checkpoint ?? ""))
        sdxl[p.id] = e
        names[p.id] = p.name
        Memory.clearCache()
        reportModels()
        return e
    }

    private func loadAnima(_ p: ModelPreset, _ root: URL) throws -> WaiAnima {
        if let e = anima[p.id] { return e }
        makeRoom(keeping: p.id)
        func file(_ rel: String?) -> URL { root.appendingPathComponent(rel ?? "") }
        let e = try WaiAnima(diffusionModel: file(p.diffusionModel), llm: file(p.llm), vae: file(p.vae))
        anima[p.id] = e
        names[p.id] = p.name
        Memory.clearCache()
        reportModels()
        return e
    }

    /// YOLOv8 face detector in MLX (docs/PERFORMANCE.md round 5)
    private func loadDetector(_ root: URL) throws -> YOLOFaceDetector {
        if let d = detector { return d }
        let d = try YOLOFaceDetector(url: root.appendingPathComponent("detectors/face_yolov8m.safetensors"))
        detector = d
        return d
    }

    private func reportModels(loading: String? = nil) {
        let loaded = Set(sdxl.keys).union(anima.keys).compactMap { names[$0] }.sorted()
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
                case .sdxl: steps.append((p.name, { _ = try self.loadSDXL(p, root) }))
                case .anima: steps.append((p.name, { _ = try self.loadAnima(p, root) }))
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
            self.sdxl.removeAll()
            self.anima.removeAll()
            self.detector = nil
            self.upscalers.removeAll()
            Memory.clearCache()
            self.reportModels()
        }
    }

    // MARK: engines for a job

    private var loraSpecs: (GenerationJob) -> [LoRASpec] = { job in
        job.loras.map { LoRASpec(url: $0.url, strength: Float($0.strength)) }
    }

    private func sdxlEngine(_ job: GenerationJob, _ progress: (JobProgress) -> Void) throws -> WaiSDXL {
        let p = job.preset
        if sdxl[p.id] == nil { progress(.loadingModel(p.name)) }
        let engine = try loadSDXL(p, job.modelsRoot)
        let report = try engine.setLoRAs(loraSpecs(job))
        if !report.unmatched.isEmpty || report.unsupported > 0 {
            FileHandle.standardError.write(Data(
                "LoRA: \(report.unmatched.count) unmatched layers, \(report.unsupported) unsupported tensors\n".utf8))
        }
        return engine
    }

    private func animaEngine(_ job: GenerationJob, _ progress: (JobProgress) -> Void) throws -> WaiAnima {
        let p = job.preset
        if anima[p.id] == nil { progress(.loadingModel(p.name)) }
        let engine = try loadAnima(p, job.modelsRoot)
        // LoRA, LoHa and LoKr merge into the MLX model; anything else (e.g. text-encoder layers) is skipped
        let report = try engine.dit.setLoRAs(loraSpecs(job))
        if !report.unmatched.isEmpty {
            FileHandle.standardError.write(Data("WAI-ANIMA LoRA: \(report.unmatched.count) modules not applied\n".utf8))
        }
        return engine
    }

    private func modelUpscaler(_ url: URL) throws -> ModelUpscaler {
        if let up = upscalers[url] { return up }
        let up = try ModelUpscaler(url: url)
        upscalers[url] = up
        return up
    }

    /// Hi-res upscale hook for WaiSDXL: the upscaler model (x4), then a high-quality resample.
    private func sdxlUpscaler(_ url: URL) throws -> WaiSDXL.HiresUpscaler {
        let up = try modelUpscaler(url)
        return { img, w, h in
            let u8 = (img[0] * 255).round().asType(.uint8)
            let big = try up.upscale(RGBImage(u8)).resized(width: w, height: h)
            return MLXArray(big.pixels, [1, h, w, 3], type: UInt8.self).asType(.float32) / 255
        }
    }

    private func preview(_ job: GenerationJob, _ progress: @escaping (JobProgress) -> Void) -> ((MLXArray) -> Void)? {
        guard job.livePreview else { return nil }
        return { arr in
            if let cg = ImageFiles.cgImage(rgb: arr.asData(access: .copy).data, width: arr.dim(1), height: arr.dim(0)) {
                progress(.preview(cg))
            }
        }
    }

    private func sdxlPhase(_ progress: @escaping (JobProgress) -> Void) -> (GenerationPhase, Int, Int) -> Void {
        { phase, i, n in
            switch phase {
            case .encode: progress(.encoding)
            case .sample: progress(.sampling(step: i, total: n, hires: false))
            case .hires: progress(.sampling(step: i, total: n, hires: true))
            case .decode: progress(.decoding)
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
        if job.stepCache { sp.stepCache = Self.stepCacheThreshold }
        return sp
    }

    private func animaParameters(_ job: GenerationJob) -> AnimaParameters {
        var p = AnimaParameters(prompt: job.positive)
        p.negativePrompt = job.negative
        p.width = job.width
        p.height = job.height
        p.steps = job.steps
        p.cfg = Float(job.cfg)
        p.sampler = switch job.sampler {
        case .eulerA: .eulerAncestral
        case .euler: .euler
        case .dpmpp2m: .dpmpp2m
        }
        p.scheduler = switch job.scheduler {
        case .normal: .discrete   // WAI-ANIMA's default (stable-diffusion.cpp's "discrete")
        case .karras: .karras
        case .simple: .simple
        }
        p.seed = job.seed
        if job.stepCache { p.stepCache = Self.stepCacheThreshold }
        return p
    }

    /// Step cache (EasyCache) threshold for both models, measured in docs/PERFORMANCE.md (round 4): the
    /// most steps skipped while the image stays visibly the same.
    static let stepCacheThreshold: Float = 0.5

    /// WAI-ANIMA img2img at `size` from `image` (hi-res fix and gallery Upscale): runs `steps` steps of a
    /// steps/denoise schedule, like the hi-res pass of WAI v16.
    private func animaRefine(_ engine: WaiAnima, _ job: GenerationJob, _ image: RGBImage, width: Int, height: Int,
                             steps: Int, denoise: Double, progress: @escaping (JobProgress) -> Void) throws -> RGBImage {
        var p = animaParameters(job)
        p.width = width
        p.height = height
        p.initImage = image.resized(width: width, height: height).pixels
        p.strength = Float(denoise)
        p.steps = Int((Double(steps) / max(denoise, 0.05)).rounded(.up))   // runs steps x strength of them
        if isCancelled { throw CancellationError() }
        progress(.preparingHires)
        let out = try engine.generate(p, isCancelled: { self.isCancelled }, preview: preview(job, progress)) { i, n in
            progress(.sampling(step: i, total: n, hires: true))
        }
        return RGBImage(out)
    }

    // MARK: text to image

    private func execute(_ job: GenerationJob, progress: @escaping (JobProgress) -> Void) throws -> JobResult {
        let t0 = Date()
        var image: RGBImage

        switch job.preset.engine {
        case .sdxl:
            let engine = try sdxlEngine(job, progress)
            let hiresUpscale = try job.hires != nil ? job.hiresUpscaler.map { try sdxlUpscaler($0) } : nil
            let (array, _) = try engine.generate(
                sdxlParameters(job), hiresUpscale: hiresUpscale, isCancelled: { self.isCancelled },
                preview: preview(job, progress), progress: sdxlPhase(progress))
            image = RGBImage(array)

        case .anima:
            let engine = try animaEngine(job, progress)
            if isCancelled { throw CancellationError() }
            progress(.encoding)
            let base = try engine.generate(animaParameters(job), isCancelled: { self.isCancelled },
                                           preview: preview(job, progress)) { i, n in
                progress(.sampling(step: i, total: n, hires: false))
            }
            image = RGBImage(base)
            if let h = job.hires {
                // hi-res fix: upscaler model (or a resize), then img2img at the new size
                progress(.upscaling)
                if let u = job.hiresUpscaler { image = try modelUpscaler(u).upscale(image) }
                image = try animaRefine(engine, job, image, width: Int(Double(job.width) * h.scale / 16) * 16,
                                        height: Int(Double(job.height) * h.scale / 16) * 16, steps: h.steps,
                                        denoise: h.denoise, progress: progress)
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
        guard let source = RGBImage.load(up.source) else { throw GenerationError("couldn't read \(up.source.lastPathComponent)") }

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
        case .sdxl:
            let engine = try sdxlEngine(job, progress)
            var sp = sdxlParameters(job)
            sp.hiresSteps = steps
            sp.hiresDenoise = Float(denoise)
            progress(.upscaling)
            let out = try engine.refine(
                source.array, sp, width: up.width, height: up.height, upscale: try job.hiresUpscaler.map { try sdxlUpscaler($0) },
                isCancelled: { self.isCancelled }, preview: preview(job, progress), progress: sdxlPhase(progress))
            image = RGBImage(out)

        case .anima:
            let engine = try animaEngine(job, progress)
            progress(.upscaling)
            var img = source
            if let u = job.hiresUpscaler { img = try modelUpscaler(u).upscale(img) }
            image = try animaRefine(engine, job, img, width: up.width / 16 * 16, height: up.height / 16 * 16,
                                    steps: steps, denoise: denoise, progress: progress)
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
        let faces = detector.detect(rgb: image.pixels, width: image.width, height: image.height,
                                    confidence: Float(fr.confidence))
        guard !faces.isEmpty else { return (image, Date().timeIntervalSince(tf)) }
        var settings = FaceRepairSettings()
        settings.strength = Float(fr.strength)
        let faceProgress: (Int, Int, Int) -> Void = { _, step, total in progress(.faceRepair(step: step, total: total)) }
        let fixed: Data
        switch job.preset.engine {
        case .sdxl:
            // the engine that made the image (LoRAs applied) repaints each face like ADetailer
            let engine = try loadSDXL(job.preset, job.modelsRoot)
            var p = sdxlParameters(job)
            if !fr.prompt.isEmpty { p.prompt = fr.prompt }
            fixed = try engine.repairFaces(rgb: image.pixels, width: image.width, height: image.height, faces: faces, p,
                                           settings: settings, isCancelled: { self.isCancelled }, progress: faceProgress)
        case .anima:
            // 768 px repaints unless a face crop is bigger (close-ups): half the time of 1024 px, same look
            let engine = try loadAnima(job.preset, job.modelsRoot)
            var p = animaParameters(job)
            if !fr.prompt.isEmpty { p.prompt = fr.prompt }
            settings.inpaintSize = WaiAnima.faceRepaintSize(faces, settings: settings)
            fixed = try engine.repairFaces(rgb: image.pixels, width: image.width, height: image.height, faces: faces, p,
                                           settings: settings, isCancelled: { self.isCancelled }, progress: faceProgress)
        }
        return (RGBImage(width: image.width, height: image.height, pixels: fixed), Date().timeIntervalSince(tf))
    }

    private func result(_ image: RGBImage, start: Date, faceRepairSeconds: Double?) throws -> JobResult {
        guard let cg = ImageFiles.cgImage(rgb: image.pixels, width: image.width, height: image.height) else {
            throw GenerationError("could not build the image")
        }
        return JobResult(image: cg, seconds: Date().timeIntervalSince(start), faceRepairSeconds: faceRepairSeconds)
    }
}
