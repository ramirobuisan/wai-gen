import Foundation

enum EngineKind: String, Codable {
    case mlx = "MLX"
    case sdcpp = "stable-diffusion.cpp"
}

/// Sampler choices shown in the app, mapped to each engine's own names.
enum Sampler: String, CaseIterable, Codable, Identifiable {
    case eulerA = "euler_ancestral"
    case euler
    case dpmpp2m = "dpmpp_2m"

    var id: String { rawValue }
    var label: String {
        switch self {
        case .eulerA: "Euler a"
        case .euler: "Euler"
        case .dpmpp2m: "DPM++ 2M"
        }
    }
    var sdcppName: String {
        switch self {
        case .eulerA: "euler_a"
        case .euler: "euler"
        case .dpmpp2m: "dpm++2m"
        }
    }
}

enum Scheduler: String, CaseIterable, Codable, Identifiable {
    case normal, karras, simple
    var id: String { rawValue }
    var label: String { rawValue.capitalized }
    /// stable-diffusion.cpp: nil = the model's own default (Anima uses its flow schedule)
    var sdcppName: String? {
        switch self {
        case .normal: nil
        case .karras: "karras"
        case .simple: "simple"
        }
    }
}

/// Character list (thumbnails + tags) from Character Select SAA's dataset.
enum CharacterListID: String, CaseIterable, Codable, Identifiable {
    case wai160 = "waiIllustriousSDXL_v160"
    case wai120 = "waiNSFWIllustrious_v120"
    case anima = "waiANIMA_v10Base10"

    var id: String { rawValue }
    var label: String {
        switch self {
        case .wai160: "WAI v16"
        case .wai120: "WAI v12"
        case .anima: "WAI-ANIMA"
        }
    }

    private static let hf = "https://huggingface.co/datasets/flagrantia/character_select_stand_alone_app"

    /// (local file name, download URL)
    var files: [(name: String, url: String)] {
        switch self {
        case .wai160:
            [("\(rawValue)_characters.csv", "\(Self.hf)/raw/main/wai_characters_v160.csv"),
             ("\(rawValue)_tag_assist.json", "\(Self.hf)/raw/main/waiIllustriousSDXL_v160_tag_assist.json"),
             ("\(rawValue)_thumbs.json", "\(Self.hf)/resolve/main/wai_character_thumbs_v160.json?download=true")]
        case .wai120:
            [("\(rawValue)_characters.csv", "\(Self.hf)/raw/main/wai_characters_v120.csv"),
             ("\(rawValue)_tag_assist.json", "\(Self.hf)/raw/main/waiIllustriousSDXL_v160_tag_assist.json"),
             ("\(rawValue)_thumbs.json", "\(Self.hf)/resolve/main/wai_character_thumbs_v120.json?download=true")]
        case .anima:
            [("\(rawValue)_characters.csv", "\(Self.hf)/raw/main/waiANIMA_v10Base10_characters.csv"),
             ("\(rawValue)_tag_assist.json", "\(Self.hf)/raw/main/waiANIMA_v10Base10_tag_assist.json"),
             ("\(rawValue)_thumbs.json", "\(Self.hf)/resolve/main/waiANIMA_v10Base10_thumbs.json?download=true")]
        }
    }
}

/// The settings a preset applies when its model is selected.
struct PresetDefaults: Equatable {
    var width: Int
    var height: Int
    var steps: Int
    var cfg: Double
    var sampler: Sampler
    var scheduler: Scheduler
    var hiresEnabled: Bool
    var hiresScale: Double
    var hiresSteps: Int
    var hiresDenoise: Double
    var maxHiresScale: Double
    var qualityPrefix: String
    var negative: String
}

struct ModelPreset: Identifiable, Hashable {
    let id: String
    let name: String
    let engine: EngineKind
    /// Paths relative to the models folder.
    let checkpoint: String?
    let diffusionModel: String?
    let llm: String?
    let vae: String?
    let characterList: CharacterListID
    let defaults: PresetDefaults
    let source: String

    static func == (a: ModelPreset, b: ModelPreset) -> Bool { a.id == b.id }
    func hash(into h: inout Hasher) { h.combine(id) }

    var requiredFiles: [String] { [checkpoint, diffusionModel, llm, vae].compactMap { $0 } }

    func missingFiles(in root: URL) -> [String] {
        requiredFiles.filter { !FileManager.default.fileExists(atPath: root.appendingPathComponent($0).path) }
    }
}

enum ModelCatalog {
    // Recommended settings from the WAI0731 model pages on Civitai (2026-11).
    // "nsfw" is added to the negatives as the WAI author suggests; remove it in the app if wanted.
    // The watermark tags stop the Patreon / signature text these models learned from their training images.
    static let waiNegative =
        "bad quality, worst quality, worst detail, sketch, censor, watermark, signature, artist name, patreon username, patreon logo, web address, nsfw"
    static let animaNegative =
        "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, lowres, censor, watermark, signature, patreon username, patreon logo, web address, nsfw"
    /// Earlier default negatives, upgraded to the current ones when found unchanged in saved settings.
    static let legacyNegatives: [String: String] = [
        "bad quality, worst quality, worst detail, sketch, censor, nsfw": waiNegative,
        "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, lowres, censor, nsfw": animaNegative,
    ]

    static let presets: [ModelPreset] = [
        ModelPreset(
            id: "wai-v16", name: "WAI Illustrious v16", engine: .mlx,
            checkpoint: "checkpoints/waiIllustriousSDXL_v160.safetensors",
            diffusionModel: nil, llm: nil, vae: nil, characterList: .wai160,
            defaults: PresetDefaults(
                width: 1024, height: 1344, steps: 28, cfg: 6, sampler: .eulerA, scheduler: .normal,
                hiresEnabled: true, hiresScale: 1.5, hiresSteps: 15, hiresDenoise: 0.4, maxHiresScale: 2,
                qualityPrefix: "masterpiece, best quality, amazing quality", negative: waiNegative),
            source: "Civitai 827184 v16.0: Euler a, 15-30 steps, CFG 5-7, >1024², hires 1.5x/20 steps/denoise 0.35-0.5"),
        ModelPreset(
            id: "wai-v12", name: "WAI Illustrious v12", engine: .mlx,
            checkpoint: "checkpoints/waiNSFWIllustrious_v120.safetensors",
            diffusionModel: nil, llm: nil, vae: nil, characterList: .wai120,
            defaults: PresetDefaults(
                width: 1024, height: 1344, steps: 30, cfg: 6, sampler: .eulerA, scheduler: .normal,
                hiresEnabled: true, hiresScale: 1.5, hiresSteps: 15, hiresDenoise: 0.4, maxHiresScale: 2,
                qualityPrefix: "masterpiece, best quality, amazing quality", negative: waiNegative),
            source: "Civitai 827184 v12.0: Euler a, 25-40 steps, CFG 5-7, >1024², hires 1.5x/20 steps/denoise 0.35-0.5"),
        ModelPreset(
            id: "wai-anima", name: "WAI-ANIMA v1.0", engine: .sdcpp, checkpoint: nil,
            diffusionModel: "diffusion_models/waiANIMA_v10Base10.safetensors",
            llm: "text_encoders/qwen_3_06b_base.safetensors", vae: "vae/qwen_image_vae.safetensors",
            characterList: .anima,
            defaults: PresetDefaults(
                width: 1024, height: 1344, steps: 25, cfg: 4.5, sampler: .eulerA, scheduler: .normal,
                hiresEnabled: true, hiresScale: 1.5, hiresSteps: 15, hiresDenoise: 0.4, maxHiresScale: 1.5,
                qualityPrefix: "masterpiece, best quality, score_7", negative: animaNegative),
            source: "Civitai 2544636 v1.0: Euler a normal, 20-30 steps, CFG 4-5, hires ≤1.5x/20 steps/denoise 0.35-0.5"),
    ]

    static func preset(_ id: String) -> ModelPreset { presets.first { $0.id == id } ?? presets[0] }
}
