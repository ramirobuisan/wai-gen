import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import WaiMLX
import WaiRemote

/// Runs other Macs' jobs on this Mac ("Share this Mac" in Settings): turns a `RemoteJob` into a
/// `GenerationJob` for the same `GenerationService` (so they queue behind and in front of this Mac's own jobs
/// and reuse the loaded models), with every file reference resolved inside this Mac's models folder.
final class GeneratorHost: RemoteJobHandler, @unchecked Sendable {
    private let service: GenerationService
    /// models in memory now, for the info reply (set on the main actor)
    private let lock = NSLock()
    private var loadedModels: [String] = []
    private var busy = false

    init(service: GenerationService) { self.service = service }

    func setLoaded(_ names: [String]) {
        lock.lock()
        loadedModels = names
        lock.unlock()
    }

    func remoteInfo() async -> RemoteInfo {
        let root = AppPaths.models
        let presets = ModelCatalog.presets.filter { $0.missingFiles(in: root).isEmpty }.map(\.id)
        let loras = LoRACatalog.scan().map {
            RemoteLoRAFile(relativePath: $0.relativePath, baseModel: $0.baseModel, triggerTags: $0.triggerTags)
        }
        let folder = AppModel.upscalerFolder
        let names = ((try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? []).sorted()
        let upscalers = names.compactMap { n in RealESRGAN.probe(folder.appendingPathComponent(n)).map { RemoteUpscaler(name: n, scale: $0) } }
        lock.lock()
        defer { lock.unlock() }
        return RemoteInfo(name: Host.current().localizedName ?? "Mac", presets: presets, loras: loras, upscalers: upscalers,
                          faceDetector: FileManager.default.fileExists(atPath: AppPaths.faceDetector.path),
                          loadedModels: loadedModels, busy: busy)
    }

    func runRemote(_ r: RemoteJob, id: UUID, client: String, progress: @escaping @Sendable (RemoteProgress) -> Void,
                   preview: @escaping @Sendable (Data) -> Void) async throws -> (png: Data, seconds: Double, faceRepairSeconds: Double?) {
        let root = AppPaths.models
        guard let preset = ModelCatalog.presets.first(where: { $0.id == r.presetID }) else {
            throw RemoteError.invalid("unknown model \(r.presetID)")
        }
        let missing = preset.missingFiles(in: root)
        guard missing.isEmpty else { throw RemoteError.refused("\(preset.name) isn't installed on this Mac") }
        if r.faceRepair != nil, !FileManager.default.fileExists(atPath: AppPaths.faceDetector.path) {
            throw RemoteError.refused("Face repair isn't set up on this Mac")
        }
        guard let sampler = Sampler(rawValue: r.sampler), let scheduler = Scheduler(rawValue: r.scheduler) else {
            throw RemoteError.invalid("unknown sampler or schedule")
        }
        let loraRoot = LoRACatalog.folder.standardizedFileURL
        let loras = try r.loras.map { l -> JobLoRA in
            let url = loraRoot.appendingPathComponent(l.relativePath).standardizedFileURL
            guard url.path.hasPrefix(loraRoot.path + "/"), FileManager.default.fileExists(atPath: url.path) else {
                throw RemoteError.refused("LoRA \(l.relativePath) isn't on this Mac")
            }
            return JobLoRA(url: url, relativePath: l.relativePath, strength: l.strength)
        }
        var upscaler: URL? = nil
        if let name = r.hiresUpscaler {
            let url = AppModel.upscalerFolder.appendingPathComponent(name)
            guard RealESRGAN.probe(url) != nil else { throw RemoteError.refused("Upscaler \(name) isn't on this Mac") }
            upscaler = url
        }
        var upscale: UpscaleRequest? = nil
        var sourceFile: URL? = nil
        if let u = r.upscale {
            let tmp = FileManager.default.temporaryDirectory.appendingPathComponent("waigen-remote-\(id.uuidString).img")
            try u.source.write(to: tmp)
            sourceFile = tmp
            upscale = UpscaleRequest(source: tmp, width: u.width, height: u.height, detailPass: u.detailPass)
        }
        defer { if let sourceFile { try? FileManager.default.removeItem(at: sourceFile) } }

        var job = GenerationJob(
            preset: preset, modelsRoot: root, positive: r.positive, negative: r.negative, characters: r.characters,
            userPrompt: r.userPrompt, width: r.width / 8 * 8, height: r.height / 8 * 8, steps: r.steps, cfg: r.cfg,
            sampler: sampler, scheduler: scheduler, seed: r.seed,
            hires: r.hires.map { ($0.scale, $0.steps, $0.denoise) },
            faceRepair: r.faceRepair.map { FaceRepairOptions(strength: $0.strength, confidence: $0.confidence, prompt: $0.prompt) },
            loras: loras, hiresUpscaler: upscaler, livePreview: r.livePreview, stepCache: r.stepCache && sampler.supportsStepCache, upscale: upscale)
        job.id = id
        job.hiresGuidanceEvery = min(max(r.hiresGuidanceEvery ?? 1, 1), 4)

        setBusy(true)
        defer { setBusy(false) }
        let activity = ProcessInfo.processInfo.beginActivity(
            options: [.userInitiated, .idleSystemSleepDisabled], reason: "Generating for \(client)")
        defer { ProcessInfo.processInfo.endActivity(activity) }
        let result = try await service.run(job) { p in
            switch p {
            case .preview(let cg): if let jpeg = Self.encode(cg, as: .jpeg) { preview(jpeg) }
            default: progress(Self.remote(p))
            }
        }
        guard let png = Self.encode(result.image, as: .png) else { throw RemoteError.refused("couldn't encode the image") }
        return (png, result.seconds, result.faceRepairSeconds)
    }

    func cancelRemote(id: UUID) { service.cancel(jobID: id) }

    private func setBusy(_ b: Bool) {
        lock.lock()
        busy = b
        lock.unlock()
    }

    static func remote(_ p: JobProgress) -> RemoteProgress {
        switch p {
        case .loadingModel(let n): RemoteProgress(.loadingModel, name: n)
        case .encoding: RemoteProgress(.encoding)
        case .sampling(let s, let t, let h): RemoteProgress(.sampling, step: s, total: t, hires: h)
        case .decoding: RemoteProgress(.decoding)
        case .upscaling: RemoteProgress(.upscaling)
        case .preparingHires: RemoteProgress(.preparingHires)
        case .loadingFaceRepair: RemoteProgress(.loadingFaceRepair)
        case .faceRepair(let s, let t): RemoteProgress(.faceRepair, step: s, total: t)
        case .preview: RemoteProgress(.encoding)   // not sent as progress
        }
    }

    static func local(_ p: RemoteProgress) -> JobProgress {
        switch p.kind {
        case .loadingModel: .loadingModel(p.name ?? "the model")
        case .encoding: .encoding
        case .sampling: .sampling(step: p.step, total: p.total, hires: p.hires)
        case .decoding: .decoding
        case .upscaling: .upscaling
        case .preparingHires: .preparingHires
        case .loadingFaceRepair: .loadingFaceRepair
        case .faceRepair: .faceRepair(step: p.step, total: p.total)
        }
    }

    static func encode(_ image: CGImage, as type: UTType) -> Data? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, type.identifier as CFString, 1, nil) else { return nil }
        CGImageDestinationAddImage(dest, image, type == .jpeg ? [kCGImageDestinationLossyCompressionQuality: 0.8] as CFDictionary : nil)
        return CGImageDestinationFinalize(dest) ? data as Data : nil
    }
}

extension GenerationJob {
    /// This job for another Mac's generator (file references by name, a gallery Upscale's source image inlined).
    func remote() throws -> RemoteJob {
        var job = RemoteJob(
            presetID: preset.id, positive: positive, negative: negative, characters: characters, userPrompt: userPrompt,
            width: width, height: height, steps: steps, cfg: cfg, sampler: sampler.rawValue, scheduler: scheduler.rawValue,
            seed: seed, hires: hires.map { .init(scale: $0.scale, steps: $0.steps, denoise: $0.denoise) },
            faceRepair: faceRepair.map { .init(strength: $0.strength, confidence: $0.confidence, prompt: $0.prompt) },
            loras: loras.map { .init(relativePath: $0.relativePath, strength: $0.strength) },
            hiresUpscaler: hiresUpscaler?.lastPathComponent, livePreview: livePreview, stepCache: stepCache,
            upscale: try upscale.map { .init(width: $0.width, height: $0.height, detailPass: $0.detailPass,
                                             source: try Data(contentsOf: $0.source)) })
        if hiresGuidanceEvery > 1 { job.hiresGuidanceEvery = hiresGuidanceEvery }
        return job
    }
}
