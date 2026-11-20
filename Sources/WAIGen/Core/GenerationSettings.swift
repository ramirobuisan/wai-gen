import Foundation

/// Everything the inspector and prompt editor control. Persisted between launches.
struct GenerationSettings: Codable, Equatable {
    var modelID = ModelCatalog.presets[0].id
    var characters: [String] = []          // unescaped tags, e.g. "rebecca (cyberpunk)"
    var artists: [String] = []             // artist styles, unescaped tags (Styles in the sidebar)
    var prompt = ""
    var negative = ModelCatalog.waiNegative
    var qualityPrefix = ModelCatalog.presets[0].defaults.qualityPrefix
    var useQualityPrefix = true
    var tagAssist = true

    var width = 1024
    var height = 1344
    var steps = 28
    var cfg = 6.0
    var sampler = Sampler.eulerA
    var scheduler = Scheduler.normal
    var seed: Int64 = -1                    // -1 = random for every image
    var batchCount = 1

    var hiresEnabled = true
    var hiresScale = 1.5
    var hiresSteps = 15
    var hiresDenoise = 0.4

    /// ADetailer-style face repair after generation (YOLOv8 face detector + cropped repaint).
    var faceRepair = true
    var faceStrength = 0.4
    var faceConfidence = 0.3
    var facePrompt = ""            // empty = same prompt as the image

    /// Show the image forming step by step (latent projection, computed with each step).
    var livePreview = true
    /// use the preset's fast sampler / step count instead of the ones above
    var fastMode = false
    /// skip model calls on steps that would barely change the image (EasyCache, see docs/PERFORMANCE.md)
    var stepCache = false

    /// Hi-res upscaler model in models/upscale_models ("" = built-in bicubic / Lanczos).
    var hiresUpscaler = "RealESRGAN_x4plus_anime_6B.pth"

    /// LoRA slots (paths relative to models/loras).
    var loras: [LoRASelection] = []

    mutating func apply(_ preset: ModelPreset) {
        let d = preset.defaults
        modelID = preset.id
        width = d.width
        height = d.height
        steps = d.steps
        cfg = d.cfg
        sampler = d.sampler
        scheduler = d.scheduler
        hiresEnabled = d.hiresEnabled
        hiresScale = min(d.hiresScale, d.maxHiresScale)
        hiresSteps = d.hiresSteps
        hiresDenoise = d.hiresDenoise
        qualityPrefix = d.qualityPrefix
        negative = d.negative
    }

    /// True when every preset-controlled value still matches the preset.
    func matches(_ preset: ModelPreset) -> Bool {
        var copy = self
        copy.apply(preset)
        return copy == self
    }

    static let storageKey = "generationSettings.v1"

    static func load() -> GenerationSettings {
        // Merge saved values over the defaults so settings added in newer versions don't
        // make the whole saved state fail to decode.
        if let data = UserDefaults.standard.data(forKey: storageKey),
            let saved = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let defaultsData = try? JSONEncoder().encode(GenerationSettings()),
            let defaults = try? JSONSerialization.jsonObject(with: defaultsData) as? [String: Any],
            let merged = try? JSONSerialization.data(withJSONObject: defaults.merging(saved) { _, new in new }),
            let s = try? JSONDecoder().decode(GenerationSettings.self, from: merged)
        {
            return s
        }
        var s = GenerationSettings()
        s.apply(ModelCatalog.presets[0])
        return s
    }

    func save() {
        if let data = try? JSONEncoder().encode(self) {
            UserDefaults.standard.set(data, forKey: Self.storageKey)
        }
    }
}

/// Escape a Danbooru tag for prompts: `rebecca (cyberpunk)` -> `rebecca \(cyberpunk\)`.
func escapeTag(_ tag: String) -> String {
    tag.replacingOccurrences(of: "\\(", with: "(").replacingOccurrences(of: "\\)", with: ")")
        .replacingOccurrences(of: "(", with: "\\(").replacingOccurrences(of: ")", with: "\\)")
}
