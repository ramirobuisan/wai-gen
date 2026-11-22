import Foundation

/// Which pipeline runs a preset (both are MLX).
enum EngineKind: String, Codable {
    case sdxl = "MLX"
    case anima = "MLX (Anima)"

    /// "Faster hi-res guidance": the hi-res pass computes the guidance on every n-th step. WAI v16: every 2nd
    /// (every 3rd saves little more, the step cache already skips many); WAI-ANIMA: every 3rd, which looks the same,
    /// where every 2nd deepens shadows a little (docs/PERFORMANCE.md, rounds 9 and 10).
    var fastHiresGuidanceEvery: Int {
        switch self {
        case .sdxl: 2
        case .anima: 3
        }
    }
}

/// Sampler choices shown in the app, mapped to each engine's own names.
enum Sampler: String, CaseIterable, Codable, Identifiable {
    case eulerA = "euler_ancestral"
    case euler
    case dpmpp2m = "dpmpp_2m"

    /// The step cache (EasyCache) was tuned on Euler a, which adds fresh noise every step. With Euler or
    /// DPM++ 2M the latent changes smoothly, the cache skips far more steps (23-25 of 43 instead of 13) and the
    /// image comes out washed out or grainy (docs/PERFORMANCE.md, round 10), so it only applies to Euler a.
    var supportsStepCache: Bool { self == .eulerA }

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
    /// "Fast mode": fewer steps with a sampler that converges sooner (compared in docs/PERFORMANCE.md)
    var fast: FastSampling = .init(sampler: .dpmpp2m, scheduler: .karras, steps: 16)
}

struct FastSampling: Equatable {
    var sampler: Sampler
    var scheduler: Scheduler
    var steps: Int
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
    /// NoobAI-XL's recommended negative (Hugging Face Laxhar/noobai-XL-1.1)
    static let noobaiNegative =
        "nsfw, worst quality, old, early, low quality, lowres, signature, username, logo, bad hands, mutated hands, mammal, anthro, furry, ambiguous form, feral, semi-anthro"
    /// Earlier default negatives, upgraded to the current ones when found unchanged in saved settings.
    static let legacyNegatives: [String: String] = [
        "bad quality, worst quality, worst detail, sketch, censor, nsfw": waiNegative,
        "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, jpeg artifacts, lowres, censor, nsfw": animaNegative,
    ]

    static let presets: [ModelPreset] = [
        ModelPreset(
            id: "wai-v16", name: "WAI Illustrious v16", engine: .sdxl,
            checkpoint: "checkpoints/waiIllustriousSDXL_v160.safetensors",
            diffusionModel: nil, llm: nil, vae: nil, characterList: .wai160,
            defaults: PresetDefaults(
                width: 1024, height: 1344, steps: 28, cfg: 6, sampler: .eulerA, scheduler: .normal,
                hiresEnabled: true, hiresScale: 1.5, hiresSteps: 15, hiresDenoise: 0.4, maxHiresScale: 2,
                qualityPrefix: "masterpiece, best quality, amazing quality", negative: waiNegative),
            source: "Civitai 827184 v16.0: Euler a, 15-30 steps, CFG 5-7, >1024², hires 1.5x/20 steps/denoise 0.35-0.5"),
        ModelPreset(
            id: "wai-anima", name: "WAI-ANIMA v1.0", engine: .anima, checkpoint: nil,
            diffusionModel: "diffusion_models/waiANIMA_v10Base10.safetensors",
            llm: "text_encoders/qwen_3_06b_base.safetensors", vae: "vae/qwen_image_vae.safetensors",
            characterList: .anima,
            defaults: PresetDefaults(
                width: 1024, height: 1344, steps: 25, cfg: 4.5, sampler: .eulerA, scheduler: .normal,
                hiresEnabled: true, hiresScale: 1.5, hiresSteps: 15, hiresDenoise: 0.4, maxHiresScale: 1.5,
                qualityPrefix: "masterpiece, best quality, score_7", negative: animaNegative,
                // flow model: DPM++ 2M leaves blotches at 16 steps, plain Euler stays clean
                fast: .init(sampler: .euler, scheduler: .normal, steps: 16)),
            source: "Civitai 2544636 v1.0: Euler a normal, 20-30 steps, CFG 4-5, hires ≤1.5x/20 steps/denoise 0.35-0.5"),
        ModelPreset(
            // epsilon-prediction SDXL (Illustrious-based), so the WAI v16 engine and its speed-ups apply as they are;
            // the V-Pred versions need a different prediction type and schedule
            id: "noobai-xl-11", name: "NoobAI-XL 1.1", engine: .sdxl,
            checkpoint: "checkpoints/NoobAI-XL-v1.1.safetensors",
            diffusionModel: nil, llm: nil, vae: nil, characterList: .wai160,
            defaults: PresetDefaults(
                width: 832, height: 1216, steps: 28, cfg: 5, sampler: .eulerA, scheduler: .normal,
                hiresEnabled: true, hiresScale: 1.5, hiresSteps: 15, hiresDenoise: 0.4, maxHiresScale: 2,
                qualityPrefix: "masterpiece, best quality, newest, absurdres, highres, safe", negative: noobaiNegative),
            source: "Hugging Face Laxhar/noobai-XL-1.1 (EPS): Euler a, 25-30 steps, CFG 5-6, ~1024² (832×1216 recommended)"),
    ]

    static func preset(_ id: String) -> ModelPreset { presets.first { $0.id == id } ?? presets[0] }
}
