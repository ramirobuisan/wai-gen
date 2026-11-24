import AppKit
import Foundation
import ImageIO
import Observation
import WaiMLX
import WaiRemote

struct ProgressInfo: Equatable {
    var title: String
    var fraction: Double?      // nil = indeterminate
    var detail: String = ""
}

/// SplitMix64 so wildcard picks are reproducible from the seed.
struct SeededRNG: RandomNumberGenerator {
    var state: UInt64
    mutating func next() -> UInt64 {
        state &+= 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }
}

@MainActor @Observable
final class AppModel {
    var settings = GenerationSettings.load() {
        didSet { if settings != oldValue { settings.save() } }
    }
    let characters = CharacterStore()
    let styles = StyleStore()
    @ObservationIgnored let tags = TagIndex()
    /// new versions from GitHub releases
    let updater = Updater()
    var tagsReady = false

    /// Character list shown in the sidebar (defaults to the selected model's list).
    var characterList: CharacterListID = .wai160 {
        didSet { characters.load(characterList) }
    }

    var pending: [GenerationJob] = []
    var running: GenerationJob?
    var progress: ProgressInfo?
    var gallery: [GalleryItem] = []
    var selection: GalleryItem?
    var displayedImage: NSImage?
    /// The image forming during the current generation (small latent projection).
    var livePreview: NSImage?
    var errorMessage: String?
    var showInspector = true
    /// the sidebar thumbnail under the pointer, shown bigger over the canvas
    var hoverPreview: HoverPreviewItem?
    /// the app's main window (set by ContentView)
    @ObservationIgnored weak var mainWindow: NSWindow?
    var loraFiles: [LoRAFile] = []
    /// Upscaler models WAI Gen can run: (file name, scale)
    var upscalerFiles: [(name: String, scale: Int)] = []

    nonisolated static var upscalerFolder: URL { AppPaths.models.appendingPathComponent("upscale_models", isDirectory: true) }

    func refreshUpscalers() {
        if let info = remoteInfo, isRemote {   // the generator's upscalers
            upscalerFiles = info.upscalers.map { ($0.name, $0.scale) }
            return
        }
        Task.detached(priority: .utility) {
            let folder = AppModel.upscalerFolder
            let names = ((try? FileManager.default.contentsOfDirectory(atPath: folder.path)) ?? [])
                .filter { ["pth", "safetensors", "gguf"].contains(($0 as NSString).pathExtension.lowercased()) }
                .sorted()
            // only models the MLX upscaler runs (ESRGAN / RRDBNet); others such as 4x-UltraSharpV2 (DAT) are hidden
            let usable = names.compactMap { n in RealESRGAN.probe(folder.appendingPathComponent(n)).map { (n, $0) } }
            await MainActor.run {
                self.upscalerFiles = usable
                // a saved choice that exists but can't run (e.g. 4x-UltraSharpV2) falls back to a usable one
                let chosen = self.settings.hiresUpscaler
                if !chosen.isEmpty, names.contains(chosen), !usable.contains(where: { $0.0 == chosen }) {
                    self.settings.hiresUpscaler = usable.first?.0 ?? ""
                }
            }
        }
    }

    func refreshLoRAs() {
        if let info = remoteInfo, isRemote {   // the generator's LoRAs
            loraFiles = info.loras.map { LoRAFile(relativePath: $0.relativePath, baseModel: $0.baseModel, triggerTags: $0.triggerTags) }
            return
        }
        Task.detached(priority: .utility) {
            let files = LoRACatalog.scan()
            await MainActor.run { self.loraFiles = files }
        }
    }

    func addLoRA(_ file: LoRAFile) {
        guard !settings.loras.contains(where: { $0.relativePath == file.relativePath }) else { return }
        settings.loras.append(LoRASelection(relativePath: file.relativePath))
    }

    @ObservationIgnored private let service = GenerationService()
    /// runs other Macs' jobs here when sharing is on
    @ObservationIgnored private lazy var generatorHost = GeneratorHost(service: service)
    @ObservationIgnored private var server: RemoteServer?
    @ObservationIgnored private var sharingActivity: NSObjectProtocol?

    // MARK: generating on another Mac

    static let generatorKey = "generator.v1"
    static let shareKey = "shareGenerator"
    static let shareCodeKey = "shareCode"

    /// The Mac that generates when it isn't this one (Settings → Generate on).
    struct GeneratorChoice: Codable, Equatable {
        var endpoint: RemoteEndpoint
        var code: String
    }
    var generator: GeneratorChoice? = {
        guard let d = UserDefaults.standard.data(forKey: AppModel.generatorKey) else { return nil }
        return try? JSONDecoder().decode(GeneratorChoice.self, from: d)
    }()
    /// what the chosen generator offers (nil until connected)
    var remoteInfo: RemoteInfo?
    var remoteStatus: String?
    var isRemote: Bool { generator != nil }
    @ObservationIgnored private var remoteTask: Task<Void, Never>?
    /// "Sharing as … on port …" or an error, for Settings
    var sharingStatus: String?
    /// a job for another Mac is running here
    var servingClient: String?

    /// Generate on `choice` (nil = this Mac). Connects to check the pairing code and read its models.
    func setGenerator(_ choice: GeneratorChoice?) {
        generator = choice
        if let choice, let d = try? JSONEncoder().encode(choice) {
            UserDefaults.standard.set(d, forKey: Self.generatorKey)
        } else {
            UserDefaults.standard.removeObject(forKey: Self.generatorKey)
        }
        remoteInfo = nil
        remoteStatus = nil
        if choice != nil {
            service.unloadAll()   // this Mac's own models aren't needed while another Mac generates
            connectGenerator()
        } else {
            refreshLoRAs()
            refreshUpscalers()
            if Self.preloadAtLaunch { preloadModels() }
        }
    }

    /// (Re)reads the generator's models, LoRAs and upscalers.
    func connectGenerator() {
        guard let g = generator else { return }
        remoteStatus = "Connecting to \(g.endpoint.name)…"
        Task {
            do {
                let info = try await RemoteClient(endpoint: g.endpoint, code: g.code).info()
                guard generator == g else { return }
                remoteInfo = info
                remoteStatus = nil
                refreshLoRAs()
                refreshUpscalers()
            } catch {
                guard generator == g else { return }
                remoteInfo = nil
                remoteStatus = error.localizedDescription
            }
        }
    }

    var generatorName: String { remoteInfo?.name ?? generator?.endpoint.name ?? "this Mac" }

    /// Files a preset needs that the generating Mac doesn't have.
    func missingFiles(_ p: ModelPreset) -> [String] {
        guard isRemote else { return p.missingFiles(in: AppPaths.models) }
        guard let info = remoteInfo else { return ["Not connected to \(generatorName)\(remoteStatus.map { ": \($0)" } ?? "")"] }
        return info.presets.contains(p.id) ? [] : ["\(p.name) isn't installed on \(info.name)"]
    }

    var faceDetectorAvailable: Bool {
        isRemote ? remoteInfo?.faceDetector == true : FileManager.default.fileExists(atPath: AppPaths.faceDetector.path)
    }

    func loraAvailable(_ relativePath: String) -> Bool {
        isRemote ? remoteInfo?.loras.contains { $0.relativePath == relativePath } == true
            : FileManager.default.fileExists(atPath: LoRACatalog.folder.appendingPathComponent(relativePath).path)
    }

    func upscalerAvailable(_ name: String) -> Bool {
        isRemote ? remoteInfo?.upscalers.contains { $0.name == name } == true
            : FileManager.default.fileExists(atPath: Self.upscalerFolder.appendingPathComponent(name).path)
    }

    // MARK: sharing this Mac

    static var shareCode: String {
        if let c = UserDefaults.standard.string(forKey: shareCodeKey), !c.isEmpty { return c }
        let c = RemoteServer.newCode()
        UserDefaults.standard.set(c, forKey: shareCodeKey)
        return c
    }

    /// Lets other Macs generate on this one (or stops it).
    func setSharing(_ on: Bool, newCode: Bool = false) {
        UserDefaults.standard.set(on, forKey: Self.shareKey)
        if newCode { UserDefaults.standard.set(RemoteServer.newCode(), forKey: Self.shareCodeKey) }
        server?.stop()
        server = nil
        if let a = sharingActivity { ProcessInfo.processInfo.endActivity(a) }
        sharingActivity = nil
        guard on else {
            sharingStatus = nil
            return
        }
        let s = RemoteServer(handler: generatorHost, code: Self.shareCode, name: Host.current().localizedName ?? "WAI Gen")
        s.onStatus = { status in Task { @MainActor in self.sharingStatus = status } }
        s.onActivity = { client, running in
            Task { @MainActor in
                self.servingClient = running ? client : nil
                if running { self.idleTimer?.invalidate() } else if !self.isBusy { self.scheduleIdleUnload() }
            }
        }
        do {
            try s.start()
            server = s
            // keep answering while the app is in the background (no App Nap)
            sharingActivity = ProcessInfo.processInfo.beginActivity(
                options: .userInitiatedAllowingIdleSystemSleep, reason: "Generating for other Macs")
        } catch {
            sharingStatus = "failed: \(error.localizedDescription)"
        }
    }
    /// zoom commands for the image stage (View menu)
    @ObservationIgnored let zoom = StageZoom()
    @ObservationIgnored private var stepStart: Date?
    @ObservationIgnored private var started = false

    // MARK: models in memory

    /// Names of the models in memory, and the one loading in the background (for the inspector).
    var loadedModels: [String] = []
    var backgroundLoading: String?
    @ObservationIgnored private var idleTimer: Timer?
    @ObservationIgnored private var unloadedForIdle = false

    static let preloadKey = GenerationService.preloadKey
    static let idleMinutesKey = "unloadAfterIdleMinutes"
    /// Free the models after this many minutes without generating (0 = never).
    static var idleMinutes: Int { UserDefaults.standard.object(forKey: idleMinutesKey) as? Int ?? 5 }
    static var preloadAtLaunch: Bool { UserDefaults.standard.object(forKey: preloadKey) as? Bool ?? true }

    /// Load in the background what the next image needs: the selected image model, the upscaler and face
    /// repair (and the other image models too if "keep every image model loaded" is on). Generating
    /// always goes before the next load step.
    func preloadModels() {
        guard !isRemote else { return }
        unloadedForIdle = false
        let others = ModelCatalog.presets.filter { $0.id != preset.id }
        let presets = GenerationService.keepAllLoaded ? [preset] + others : [preset]
        let upscaler = settings.hiresUpscaler.isEmpty ? nil : Self.upscalerFolder.appendingPathComponent(settings.hiresUpscaler)
        service.preload(presets, modelsRoot: AppPaths.models,
                        faceRepair: settings.faceRepair && FileManager.default.fileExists(atPath: AppPaths.faceDetector.path),
                        upscaler: upscaler.flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil })
        scheduleIdleUnload()
    }

    func unloadModels() {
        idleTimer?.invalidate()
        service.unloadAll()
    }

    /// (Re)start the countdown that frees the models while nothing is generating.
    private func scheduleIdleUnload() {
        idleTimer?.invalidate()
        let minutes = Self.idleMinutes
        guard minutes > 0 else { return }
        idleTimer = Timer.scheduledTimer(withTimeInterval: TimeInterval(minutes * 60), repeats: false) { [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, !self.isBusy else { return }
                self.service.unloadAll()
                self.unloadedForIdle = true
            }
        }
    }

    var preset: ModelPreset { ModelCatalog.preset(settings.modelID) }
    var isBusy: Bool { running != nil }
    var queueCount: Int { pending.count + (running == nil ? 0 : 1) }
    var missingModelFiles: [String] { missingFiles(preset) }

    func start() {
        guard !started else { return }
        started = true
        if let upgraded = ModelCatalog.legacyNegatives[settings.negative] { settings.negative = upgraded }
        // 2026-11: the presets went from 20 to 15 hi-res steps (same look, a quarter less hi-res time)
        if UserDefaults.standard.integer(forKey: "settingsRevision") < 2 {
            if settings.hiresSteps == 20 { settings.hiresSteps = preset.defaults.hiresSteps }
            UserDefaults.standard.set(2, forKey: "settingsRevision")
        }
        // 2026-11-22: Step cache and Faster hi-res guidance are on by default, the fastest combination that keeps the
        // look (docs/PERFORMANCE.md, round 11); switched on once for saved settings too
        if UserDefaults.standard.integer(forKey: "settingsRevision") < 3 {
            settings.stepCache = true
            settings.fastHiresGuidance = true
            UserDefaults.standard.set(3, forKey: "settingsRevision")
        }
        service.onModelsChanged = { [generatorHost] loaded, loading in
            generatorHost.setLoaded(loaded)
            Task { @MainActor in
                self.loadedModels = loaded
                self.backgroundLoading = loading
            }
        }
        if UserDefaults.standard.bool(forKey: Self.shareKey) { setSharing(true) }
        updater.startAutomaticChecks()
        if isRemote { connectGenerator() }
        // back to the app after the models were freed for being idle: load them again
        NotificationCenter.default.addObserver(forName: NSApplication.didBecomeActiveNotification, object: nil, queue: .main) {
            [weak self] _ in
            MainActor.assumeIsolated {
                guard let self, self.unloadedForIdle, Self.preloadAtLaunch else { return }
                self.preloadModels()
            }
        }
        if Self.preloadAtLaunch { preloadModels() } else { scheduleIdleUnload() }
        // ← / → move through the gallery, unless a text field or the prompt editor is being edited
        NSEvent.addLocalMonitorForEvents(matching: .keyDown) { [weak self] event in
            let arrows: [UInt16: Int] = [123: -1, 124: 1]
            guard let delta = arrows[event.keyCode],
                  event.modifierFlags.intersection([.command, .option, .control, .shift]).isEmpty,
                  let window = event.window, MainActor.assumeIsolated({ window === self?.mainWindow }),   // not in Settings
                  let focus = window.firstResponder,
                  // controls that use the arrows themselves keep them: text, sliders, pop-ups, lists
                  !(focus is NSText || focus is NSControl || focus is NSTableView || focus is NSCollectionView)
            else { return event }
            MainActor.assumeIsolated { self?.showAdjacent(delta) }
            return nil
        }
        // free the models before exit() runs static destructors (see GenerationService.shutdown).
        // queue nil = delivered synchronously on the main thread inside terminate:
        NotificationCenter.default.addObserver(forName: NSApplication.willTerminateNotification, object: nil, queue: nil) {
            [service] _ in
            if !service.shutdown() {
                // a job is still holding the GPU: leave without running the static destructors
                CFPreferencesAppSynchronize(kCFPreferencesCurrentApplication)
                _exit(0)
            }
        }
        characters.tagIndex = tags
        characterList = preset.characterList
        if UserDefaults.standard.string(forKey: "sidebarTab") == "styles" { styles.loadIfNeeded() }
        Wildcards.seedFromSAA()
        refreshLoRAs()
        refreshUpscalers()
        reloadGallery()
        Task.detached(priority: .utility) { [tags] in
            do {
                let url = try await AppPaths.ensureDataFile(AppPaths.tagListName, url: AppPaths.tagListURL)
                try tags.load(from: url)
                await MainActor.run {
                    self.tagsReady = true
                    self.characters.refreshPopularity()   // sort characters by popularity
                }
            } catch {
                // autocomplete stays off; generation still works
            }
        }
    }

    // MARK: model and presets

    func selectModel(_ p: ModelPreset) {
        guard p.id != settings.modelID else { return }
        settings.apply(p)
        characterList = p.characterList
        if Self.preloadAtLaunch { preloadModels() }   // start loading it (and its face repair) right away
    }

    func resetToPreset() { settings.apply(preset) }

    // MARK: characters

    func toggleCharacter(_ tag: String) {
        if let i = settings.characters.firstIndex(of: tag) {
            settings.characters.remove(at: i)
        } else {
            settings.characters.append(tag)
        }
    }

    // MARK: styles

    func toggleStyle(_ tag: String) {
        if let i = settings.artists.firstIndex(of: tag) {
            settings.artists.remove(at: i)
        } else {
            settings.artists.append(tag)
        }
    }

    /// An artist's tag as the selected model expects it: Anima (WAI-ANIMA) wants "@name", Illustrious the plain tag.
    func styleTag(_ tag: String) -> String {
        preset.engine == .anima ? "@" + escapeTag(tag) : escapeTag(tag)
    }

    /// The prompt sent to the engine: characters (+ tag assist), artist styles, quality tags, then the user's prompt.
    func assemblePositive(seed: Int64, userPrompt: String? = nil) -> String {
        var parts = [String]()
        for c in settings.characters {
            parts.append(escapeTag(c))
            if settings.tagAssist, let extra = characters.assistTags(for: c) { parts.append(extra) }
        }
        parts += settings.artists.map(styleTag)
        if settings.useQualityPrefix, !settings.qualityPrefix.isEmpty { parts.append(settings.qualityPrefix) }
        let user = (userPrompt ?? settings.prompt).trimmingCharacters(in: .whitespacesAndNewlines)
        if !user.isEmpty { parts.append(user) }
        var rng = SeededRNG(state: UInt64(bitPattern: seed))
        return Wildcards.expand(parts.joined(separator: ", "), using: &rng)
    }

    // MARK: generation

    func generate() {
        let missing = missingModelFiles
        guard missing.isEmpty else {
            errorMessage = isRemote ? missing.joined(separator: "\n")
                : "\(preset.name) needs files that aren't in \(AppPaths.models.path):\n" + missing.joined(separator: "\n")
            return
        }
        if settings.faceRepair, isRemote, !faceDetectorAvailable {
            errorMessage = "Face repair isn't set up on \(generatorName). Turn it off in the inspector."
            return
        }
        if settings.faceRepair, !isRemote, !faceDetectorAvailable {
            errorMessage = "Face repair needs the face detector at \(AppPaths.faceDetector.path). "
                + "Turn face repair off in the inspector, or convert face_yolov8m.pt with scripts/convert_yolov8_safe.py."
            return
        }
        let s = settings
        // LoRAs made for another model family (e.g. an SDXL LoRA on WAI-ANIMA) are skipped; the inspector flags them
        let activeLoRAs = s.loras.filter { $0.enabled && $0.strength != 0 && loraFits($0.relativePath, preset.engine) }
        let missingLoRAs = activeLoRAs.filter { !loraAvailable($0.relativePath) }
        guard missingLoRAs.isEmpty else {
            errorMessage = "These LoRAs aren't in \(isRemote ? "\(generatorName)'s LoRA folder" : LoRACatalog.folder.path):\n"
                + missingLoRAs.map(\.relativePath).joined(separator: "\n")
            return
        }
        let jobLoRAs = activeLoRAs.map {
            JobLoRA(url: LoRACatalog.folder.appendingPathComponent($0.relativePath), relativePath: $0.relativePath,
                    strength: $0.strength)
        }
        let upscalerURL: URL? = s.hiresUpscaler.isEmpty ? nil : Self.upscalerFolder.appendingPathComponent(s.hiresUpscaler)
        if s.hiresEnabled, upscalerURL != nil, !upscalerAvailable(s.hiresUpscaler) {
            errorMessage = "The hi-res upscaler \(s.hiresUpscaler) isn't on \(generatorName). Pick another one in the inspector."
            return
        }
        // Fast mode: the preset's quicker sampler / steps (Z-Image Turbo is already a few-step model)
        let fast = s.fastMode && preset.engine != .zimage ? preset.defaults.fast : nil
        for i in 0 ..< max(1, s.batchCount) {
            let seed = s.seed < 0 ? Int64(UInt32.random(in: 0 ... UInt32.max)) : s.seed + Int64(i)
            var rng = SeededRNG(state: UInt64(bitPattern: seed) ^ 0xA5A5)
            let job = GenerationJob(
                preset: preset, modelsRoot: AppPaths.models,
                positive: assemblePositive(seed: seed),
                negative: Wildcards.expand(s.negative, using: &rng),
                characters: s.characters, userPrompt: s.prompt,
                width: s.width, height: s.height, steps: fast?.steps ?? s.steps, cfg: s.cfg,
                sampler: fast?.sampler ?? s.sampler, scheduler: fast?.scheduler ?? s.scheduler, seed: seed,
                hires: s.hiresEnabled && s.hiresScale > 1
                    ? (min(s.hiresScale, preset.defaults.maxHiresScale), s.hiresSteps, s.hiresDenoise) : nil,
                faceRepair: s.faceRepair
                    ? FaceRepairOptions(strength: s.faceStrength, confidence: s.faceConfidence,
                                        prompt: s.facePrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                            ? "" : assemblePositive(seed: seed, userPrompt: s.facePrompt))
                    : nil,
                loras: jobLoRAs,
                hiresUpscaler: upscalerURL,
                livePreview: s.livePreview,
                stepCache: s.stepCache && preset.engine.supportsStepCache && (fast?.sampler ?? s.sampler).supportsStepCache,
                artists: s.artists,
                hiresGuidanceEvery: s.fastHiresGuidance ? preset.engine.fastHiresGuidanceEvery : 1)
            pending.append(job)
        }
        processNext()
        if unloadedForIdle, Self.preloadAtLaunch { preloadModels() }   // the rest load after this job
    }

    /// False only when the LoRA's metadata says it was trained for a different model family.
    func loraFits(_ relativePath: String, _ engine: EngineKind) -> Bool {
        loraFiles.first { $0.relativePath == relativePath }?.fits(engine) != false
    }

    /// Pixel size of an image file without decoding it.
    nonisolated static func imageSize(_ url: URL) -> (Int, Int)? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let p = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
            let w = p[kCGImagePropertyPixelWidth] as? Int, let h = p[kCGImagePropertyPixelHeight] as? Int
        else { return nil }
        return (w, h)
    }

    /// Target size for the gallery Upscale action.
    func upscaleTarget(_ item: GalleryItem, detail: Bool) -> (Int, Int)? {
        guard let (w, h) = Self.imageSize(item.url) else { return nil }
        let p = item.record.flatMap { r in ModelCatalog.presets.first { $0.id == r.modelID } } ?? preset
        let scale = detail ? min(settings.hiresScale, p.defaults.maxHiresScale) : 2
        return (Int((Double(w) * scale / 8).rounded()) * 8, Int((Double(h) * scale / 8).rounded()) * 8)
    }

    /// Queue an upscale of a saved image: model upscaler + (optionally) a diffusion detail pass with
    /// the image's own prompt, seed, model and LoRAs, then face repair. Saved as a new image.
    func upscale(_ item: GalleryItem, detail: Bool) {
        guard let (tw, th) = upscaleTarget(item, detail: detail) else {
            errorMessage = "Couldn't read \(item.url.lastPathComponent)."
            return
        }
        if detail, tw * th > 4_500_000 {
            errorMessage = "\(tw) × \(th) is too big for a detail pass (SDXL gets very slow and memory-hungry above about 4.5 MP). "
                + "Use “Upscale ×2 (upscaler only)”, or lower the hi-res scale in the inspector."
            return
        }
        let r = item.record
        let p = r.flatMap { rec in ModelCatalog.presets.first { $0.id == rec.modelID } } ?? preset
        let missing = missingFiles(p)
        if detail, !missing.isEmpty {
            errorMessage = isRemote ? missing.joined(separator: "\n")
                : "\(p.name) needs files that aren't in \(AppPaths.models.path):\n" + missing.joined(separator: "\n")
            return
        }
        let s = settings
        let upscalerURL: URL? = s.hiresUpscaler.isEmpty ? nil : Self.upscalerFolder.appendingPathComponent(s.hiresUpscaler)
        if !detail, upscalerURL == nil {
            errorMessage = "Pick an upscaler model in the inspector's Hi-res fix section for “upscaler only”."
            return
        }
        let loras = (r?.loras ?? []).compactMap { l -> JobLoRA? in
            let url = LoRACatalog.folder.appendingPathComponent(l.path)
            guard loraFits(l.path, p.engine) else { return nil }
            return loraAvailable(l.path) ? JobLoRA(url: url, relativePath: l.path, strength: l.strength) : nil
        }
        let seed = r?.seed ?? Int64(UInt32.random(in: 0 ... UInt32.max))
        let upscaleSampler = r.flatMap { rec in Sampler.allCases.first { $0.label == rec.sampler } } ?? s.sampler
        let job = GenerationJob(
            preset: p, modelsRoot: AppPaths.models,
            positive: r?.positive ?? assemblePositive(seed: seed),
            negative: r?.negative ?? s.negative,
            characters: r?.characters ?? s.characters, userPrompt: r?.userPrompt ?? s.prompt,
            width: tw, height: th, steps: r?.steps ?? s.steps, cfg: r?.cfg ?? s.cfg,
            sampler: upscaleSampler,
            scheduler: r.flatMap { rec in Scheduler.allCases.first { $0.label == rec.scheduler } } ?? s.scheduler,
            seed: seed,
            hires: (min(s.hiresScale, p.defaults.maxHiresScale), s.hiresSteps, s.hiresDenoise),
            faceRepair: detail && s.faceRepair && faceDetectorAvailable
                ? FaceRepairOptions(strength: s.faceStrength, confidence: s.faceConfidence, prompt: "") : nil,
            loras: loras, hiresUpscaler: upscalerURL, livePreview: s.livePreview,
            stepCache: s.stepCache && p.engine.supportsStepCache && upscaleSampler.supportsStepCache,
            upscale: UpscaleRequest(source: item.url, width: tw, height: th, detailPass: detail),
            artists: r.map { $0.artists ?? [] } ?? s.artists,
            hiresGuidanceEvery: detail && s.fastHiresGuidance ? p.engine.fastHiresGuidanceEvery : 1)
        pending.append(job)
        processNext()
    }

    func cancel() {
        guard isBusy else { return }
        progress = ProgressInfo(title: "Stopping…", fraction: nil)
        if let remoteTask {
            remoteTask.cancel()
        } else if let running {
            service.cancel(jobID: running.id)
        }
    }

    func clearQueue() { pending.removeAll() }

    private func processNext() {
        guard running == nil, !pending.isEmpty else {
            if running == nil { scheduleIdleUnload() }   // queue empty: start the idle countdown
            return
        }
        idleTimer?.invalidate()
        let job = pending.removeFirst()
        running = job
        progress = ProgressInfo(title: "Starting…", fraction: nil)
        stepStart = nil
        let generator = generator
        let task = Task {
            do {
                let result: JobResult
                if let generator {
                    progress = ProgressInfo(title: "Sending to \(generatorName)…", fraction: nil)
                    let r = try await RemoteClient(endpoint: generator.endpoint, code: generator.code).run(
                        try job.remote(),
                        progress: { p in Task { @MainActor in self.update(GeneratorHost.local(p)) } },
                        preview: { jpeg in
                            Task { @MainActor in
                                if let img = NSImage(data: jpeg), let cg = img.cgImage(forProposedRect: nil, context: nil, hints: nil) {
                                    self.update(.preview(cg))
                                }
                            }
                        })
                    guard let src = CGImageSourceCreateWithData(r.png as CFData, nil),
                          let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
                    else { throw RemoteError.invalid("unreadable image from \(generatorName)") }
                    result = JobResult(image: cg, seconds: r.seconds, faceRepairSeconds: r.faceRepairSeconds)
                } else {
                    result = try await service.run(job) { p in
                        Task { @MainActor in self.update(p) }
                    }
                }
                finish(job, result)
            } catch is CancellationError {
                pending.removeAll()
            } catch {
                errorMessage = error.localizedDescription
            }
            remoteTask = nil
            running = nil
            progress = nil
            livePreview = nil
            processNext()
        }
        if generator != nil { remoteTask = task }
    }

    private func update(_ p: JobProgress) {
        guard running != nil else { return }
        switch p {
        case .loadingModel(let name):
            progress = ProgressInfo(title: "Loading \(name)…", fraction: nil)
        case .encoding:
            progress = ProgressInfo(title: "Reading the prompt…", fraction: nil)
        case .sampling(let step, let total, let hires):
            if step <= 1 { stepStart = Date() }
            var detail = ""
            if let t0 = stepStart, step > 1 {
                let per = Date().timeIntervalSince(t0) / Double(step - 1)
                detail = "about \(Int((per * Double(total - step)).rounded())) s left"
            }
            progress = ProgressInfo(
                title: "\(hires ? "Hi-res pass" : "Sampling") \(step) / \(total)",
                fraction: total > 0 ? Double(step) / Double(total) : nil, detail: detail)
        case .decoding:
            progress = ProgressInfo(title: "Decoding…", fraction: nil)
        case .upscaling:
            progress = ProgressInfo(title: "Upscaling…", fraction: nil)
        case .preparingHires:
            progress = ProgressInfo(title: "Preparing the hi-res pass…", fraction: nil)
        case .loadingFaceRepair:
            progress = ProgressInfo(title: "Loading face repair…", fraction: nil)
        case .faceRepair(let step, let total):
            if step <= 1 { stepStart = Date() }
            progress = ProgressInfo(
                title: "Repairing faces \(step) / \(total)",
                fraction: total > 0 ? Double(step) / Double(total) : nil)
        case .preview(let cg):
            livePreview = NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
        }
    }

    private func finish(_ job: GenerationJob, _ result: JobResult) {
        var record = GenerationRecord(
            modelID: job.preset.id, modelName: job.preset.name, engine: job.preset.engine.rawValue,
            positive: job.positive, negative: job.negative, characters: job.characters,
            userPrompt: job.userPrompt, width: job.width, height: job.height, steps: job.steps, cfg: job.cfg,
            sampler: job.sampler.label, scheduler: job.scheduler.label, seed: job.seed,
            hiresScale: job.hires?.scale, hiresSteps: job.hires?.steps, hiresDenoise: job.hires?.denoise,
            seconds: result.seconds, date: Date(),
            faceRepairStrength: job.faceRepair?.strength, faceRepairConfidence: job.faceRepair?.confidence,
            faceRepairPrompt: job.faceRepair.flatMap { $0.prompt.isEmpty ? nil : $0.prompt },
            faceRepairSeconds: result.faceRepairSeconds,
            loras: job.loras.isEmpty ? nil : job.loras.map { LoRARecord(path: $0.relativePath, strength: $0.strength) },
            hiresUpscaler: job.hires == nil ? nil : (job.hiresUpscaler?.lastPathComponent ?? "built-in"))
        if job.stepCache { record.stepCache = true }
        if job.hiresGuidanceEvery > 1, job.hires != nil || job.upscale?.detailPass == true {
            record.hiresGuidanceEvery = job.hiresGuidanceEvery
        }
        if !job.artists.isEmpty { record.artists = job.artists }
        if let up = job.upscale {
            record.upscaledFrom = up.source.lastPathComponent
            record.upscaleDetailPass = up.detailPass
            record.width = result.image.width
            record.height = result.image.height
            if !up.detailPass { record.hiresSteps = nil; record.hiresDenoise = nil; record.hiresScale = 2 }
        }
        do {
            let url = try ImageFiles.save(result.image, record: record)
            let item = GalleryItem(url: url, record: record)
            gallery.insert(item, at: 0)
            selection = item
            displayedImage = NSImage(cgImage: result.image, size: NSSize(width: result.image.width, height: result.image.height))
        } catch {
            errorMessage = "Couldn't save the image: \(error.localizedDescription)"
        }
    }

    // MARK: gallery

    /// Read the gallery from the images folder again (at launch and after choosing another folder).
    func reloadGallery() {
        gallery = ImageFiles.recentImages()
        if let first = gallery.first {
            show(first)
        } else {
            selection = nil
            displayedImage = nil
        }
    }

    /// Copies the image file: Finder pastes the PNG itself (with its settings), other apps get the PNG image.
    func copyImage(_ item: GalleryItem) {
        let entry = NSPasteboardItem()
        if let png = try? Data(contentsOf: item.url) { entry.setData(png, forType: .png) }
        entry.setString(item.url.absoluteString, forType: .fileURL)
        NSPasteboard.general.clearContents()
        NSPasteboard.general.writeObjects([entry])
    }

    func show(_ item: GalleryItem) {
        selection = item
        displayedImage = NSImage(contentsOf: item.url)
    }

    /// Restore prompt, characters and settings from a saved image. An upscaled image's record holds the
    /// upscaled size, so its size and hi-res settings come from the image it was upscaled from (when
    /// that file is still around; otherwise they are left as they are).
    func reuse(_ item: GalleryItem, seed: Bool) {
        guard let record = item.record else { return }
        var base: GenerationRecord? = record
        var url = item.url
        for _ in 0 ..< 4 {
            guard let src = base?.upscaledFrom else { break }
            if let g = gallery.first(where: { $0.url.lastPathComponent == src }), g.record != nil {
                url = g.url
                base = g.record
            } else {
                url = url.deletingLastPathComponent().appendingPathComponent(src)
                base = ImageFiles.readRecord(url)
            }
        }
        reuse(record, sizeFrom: base?.upscaledFrom == nil ? base : nil, seed: seed)
    }

    private func reuse(_ record: GenerationRecord, sizeFrom sizeRecord: GenerationRecord?, seed: Bool) {
        if record.modelID != settings.modelID, let p = ModelCatalog.presets.first(where: { $0.id == record.modelID }) {
            selectModel(p)
        }
        settings.characters = record.characters
        settings.artists = record.artists ?? []
        settings.prompt = record.userPrompt
        settings.negative = record.negative
        settings.steps = record.steps
        settings.cfg = record.cfg
        if let s = Sampler.allCases.first(where: { $0.label == record.sampler }) { settings.sampler = s }
        if let s = Scheduler.allCases.first(where: { $0.label == record.scheduler }) { settings.scheduler = s }
        if let r = sizeRecord {
            settings.width = r.width
            settings.height = r.height
            if let h = r.hiresScale {
                settings.hiresEnabled = true
                settings.hiresScale = h
                settings.hiresSteps = r.hiresSteps ?? settings.hiresSteps
                settings.hiresDenoise = r.hiresDenoise ?? settings.hiresDenoise
            } else {
                settings.hiresEnabled = false
            }
        }
        if let fs = record.faceRepairStrength {
            settings.faceRepair = true
            settings.faceStrength = fs
            settings.faceConfidence = record.faceRepairConfidence ?? settings.faceConfidence
        }
        if let l = record.loras {
            settings.loras = l.map { LoRASelection(relativePath: $0.path, strength: $0.strength) }
        } else {
            settings.loras = []
        }
        settings.stepCache = record.stepCache == true
        settings.fastHiresGuidance = (record.hiresGuidanceEvery ?? 1) > 1
        if seed { settings.seed = record.seed }
    }

    /// Select the neighbouring gallery image (newest is on the left): -1 = left / newer, +1 = right / older.
    func showAdjacent(_ delta: Int) {
        guard !gallery.isEmpty else { return }
        let current = selection.flatMap { gallery.firstIndex(of: $0) } ?? (delta > 0 ? -1 : gallery.count)
        let next = min(max(current + delta, 0), gallery.count - 1)
        if gallery[next] != selection { show(gallery[next]) }
    }

    /// False while a queued or running upscale still needs the image's file.
    func canTrash(_ item: GalleryItem) -> Bool {
        !(pending + [running].compactMap { $0 }).contains { $0.upscale?.source == item.url }
    }

    /// Image → Move to Trash (⌘⌫). While text is being edited ⌘⌫ keeps its usual meaning there.
    func trashSelectionOrDeleteText() {
        // the window the keystroke went to (keyWindow is nil while the app isn't active)
        if let text = (NSApp.currentEvent?.window ?? NSApp.keyWindow)?.firstResponder as? NSText {
            text.doCommand(by: #selector(NSStandardKeyBindingResponding.deleteToBeginningOfLine(_:)))
        } else if let selection {
            moveToTrash(selection)
        }
    }

    /// the main window's undo stack (Edit → Undo / ⌘Z)
    private var undoManager: UndoManager? { (mainWindow ?? NSApp.keyWindow)?.undoManager }

    func moveToTrash(_ item: GalleryItem) {
        guard canTrash(item) else { return }
        do {
            var trashed: NSURL?
            try FileManager.default.trashItem(at: item.url, resultingItemURL: &trashed)
            let index = gallery.firstIndex(of: item) ?? 0
            gallery.removeAll { $0.id == item.id }
            if selection == item {
                // like Photos: select the neighbour, not the newest image
                if gallery.isEmpty { selection = nil; displayedImage = nil } else { show(gallery[min(index, gallery.count - 1)]) }
            }
            if let trashed = trashed as URL? {
                undoManager?.registerUndo(withTarget: self) { $0.putBack(item, from: trashed, at: index) }
                undoManager?.setActionName("Move to Trash")
            }
        } catch {
            errorMessage = "Couldn't move the image to the Trash: \(error.localizedDescription)"
        }
    }

    /// Undo of Move to Trash: the file comes back from the Trash to its place in the gallery.
    private func putBack(_ item: GalleryItem, from trashed: URL, at index: Int) {
        do {
            try FileManager.default.createDirectory(at: item.url.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.moveItem(at: trashed, to: item.url)
            gallery.insert(item, at: min(index, gallery.count))
            show(item)
            undoManager?.registerUndo(withTarget: self) { $0.moveToTrash(item) }   // redo
            undoManager?.setActionName("Move to Trash")
        } catch {
            errorMessage = "Couldn't put the image back: \(error.localizedDescription)"
        }
    }
}
