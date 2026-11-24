import Foundation
import WaiMLX

/// Which pipeline runs a preset (both are MLX).
enum EngineKind: String, Codable {
    case sdxl = "MLX"
    case anima = "MLX (Anima)"
    case zimage = "MLX (Z-Image)"

    /// The step cache (EasyCache) is built into the WAI v16 and WAI-ANIMA engines; Z-Image Turbo already runs ~9 steps.
    var supportsStepCache: Bool { self != .zimage }

    /// "Faster hi-res guidance": the hi-res pass computes the guidance on every n-th step. WAI v16: every 2nd
    /// (every 3rd saves little more, the step cache already skips many); WAI-ANIMA: every 3rd, which looks the same,
    /// where every 2nd deepens shadows a little (docs/PERFORMANCE.md, rounds 9 and 10).
    var fastHiresGuidanceEvery: Int {
        switch self {
        case .sdxl: 2
        case .anima: 3
        case .zimage: 1   // CFG 1: no guidance to reuse
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
    /// ComfyUI's beta schedule (Z-Image only)
    case beta
    var id: String { rawValue }
    var label: String { rawValue.capitalized }
    /// Which engines offer it in the inspector.
    func available(for engine: EngineKind) -> Bool {
        switch self {
        case .beta: engine == .zimage
        case .karras: engine != .zimage
        case .normal, .simple: true
        }
    }
    /// stable-diffusion.cpp: nil = the model's own default (Anima uses its flow schedule)
    var sdcppName: String? {
        switch self {
        case .normal: nil
        case .karras: "karras"
        case .simple: "simple"
        case .beta: nil
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
    /// What the SDXL UNet predicts; the checkpoint decides it, the card records it (epsilon checkpoints leave .epsilon)
    var prediction: Prediction = .epsilon
    /// CFG Rescale the model card recommends (0 = off). ComfyUI convention, blend fixed at 0.7 in the engine
    var cfgRescale: Double = 0
    /// A distilled few-step model at CFG 1 (Z-Image Turbo, Anima Turbo): Fast mode, the step cache and faster
    /// hi-res guidance don't apply, so they're hidden and off
    var distilled = false
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

    /// The step cache needs an engine that has it and a model that isn't a distilled few-step one.
    var allowsStepCache: Bool { engine.supportsStepCache && !defaults.distilled }

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
    /// WAI-SHUFFLE-NOOB's negative (its Civitai page), plus "nsfw" as in the app's other presets
    static let shuffleNoobNegative =
        "low quality, worst quality, normal quality, text, jpeg artifacts, bad anatomy, old, early, copyright name, watermark, artist name, signature, nsfw"
    /// Nova Anime XL's Illustrious negative (its Civitai page, duplicates removed), plus "nsfw"
    static let novaAnimeNegative =
        "modern, recent, old, oldest, cartoon, graphic, text, painting, crayon, graphite, abstract, glitch, deformed, mutated, ugly, disfigured, long body, lowres, bad anatomy, bad hands, missing fingers, extra digits, fewer digits, cropped, very displeasing, (worst quality, bad quality:1.2), sketch, jpeg artifacts, signature, watermark, username, simple background, conjoined, bad ai-generated, nsfw"
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
        ModelPreset(
            // the V-Pred 1.0 release of the same base model: v-prediction + zero terminal SNR, which the engine
            // detects from the checkpoint itself; Euler only, no Karras, CFG Rescale recommended
            id: "noobai-xl-vpred-10", name: "NoobAI-XL V-Pred 1.0", engine: .sdxl,
            checkpoint: "checkpoints/noobaiXLNAIXL_vPred10Version.safetensors",
            diffusionModel: nil, llm: nil, vae: nil, characterList: .wai160,
            defaults: PresetDefaults(
                // Model card (Civitai 833294 v1190596): euler + normal ("Karras does not work"), CFG 4-5,
                // 28-35 steps at 832x1216, CFG Rescale 0.7 on. Card quality prefix + NoobAI negative.
                width: 832, height: 1216, steps: 28, cfg: 4.5, sampler: .euler, scheduler: .normal,
                hiresEnabled: true, hiresScale: 1.5, hiresSteps: 15, hiresDenoise: 0.4, maxHiresScale: 2,
                qualityPrefix: "masterpiece, best quality, newest, absurdres, highres, safe", negative: noobaiNegative,
                // fewer steps on Euler-a would drift on a v-pred schedule; keep plain Euler, just 20 steps
                fast: .init(sampler: .euler, scheduler: .normal, steps: 20),
                prediction: .v, cfgRescale: 0.7),
            source: "Civitai 833294 v1190596 (V-Pred 1.0): Euler normal only (no Karras), 28-35 steps, CFG 4-5, "
                + "832x1216, CFG Rescale ~0.7 recommended, zero terminal SNR"),
        ModelPreset(
            // Z-Image Turbo fine-tuned for anime, all in one file (NextDiT + Qwen3-4B + Flux VAE); distilled for ~9 steps
            // at CFG 1 (one model call per step, the negative prompt isn't used). No LoRAs yet.
            id: "z-image-anime-turbo", name: "Z-Image Anime (Turbo)", engine: .zimage,
            checkpoint: "checkpoints/z-image-anime-aio-bf16.safetensors",
            diffusionModel: nil, llm: nil, vae: nil, characterList: .wai160,
            defaults: PresetDefaults(
                // the creator's ComfyUI workflow: 832×1216, 9 steps, CFG 1, Euler a with the beta schedule, shift 3
                width: 832, height: 1216, steps: 9, cfg: 1, sampler: .eulerA, scheduler: .beta,
                // hi-res: 3 steps at denoise 0.30 looks like 5 at 0.35 for ~30 % less time (PERFORMANCE.md, round 12)
                hiresEnabled: false, hiresScale: 1.5, hiresSteps: 3, hiresDenoise: 0.30, maxHiresScale: 1.5,
                qualityPrefix: "anime style, masterpiece, best quality", negative: "",
                fast: .init(sampler: .eulerA, scheduler: .beta, steps: 9), distilled: true),
            source: "Civitai 2259646 v2543657 (Z-Image-Turbo-Anime AIO BF16): ~9 steps, CFG 1, Euler a, beta, 832×1216"),
        ModelPreset(
            // WAI0731's NoobAI v-prediction merge: the V-Pred engine (v-prediction + zero terminal SNR + CFG Rescale)
            id: "wai-shuffle-noob-vpred", name: "WAI-SHUFFLE-NOOB V-Pred", engine: .sdxl,
            checkpoint: "checkpoints/WAI-SHUFFLE-NOOB-v-pred-04.safetensors",
            diffusionModel: nil, llm: nil, vae: nil, characterList: .wai160,
            defaults: PresetDefaults(
                // the author's settings: Euler a, 30 steps, CFG 5-7, 1024×1360, hi-res 1.5×
                width: 1024, height: 1344, steps: 30, cfg: 5, sampler: .eulerA, scheduler: .normal,
                hiresEnabled: true, hiresScale: 1.5, hiresSteps: 15, hiresDenoise: 0.4, maxHiresScale: 2,
                qualityPrefix: "masterpiece, best quality, newest, absurdres, highres, very awa", negative: shuffleNoobNegative,
                fast: .init(sampler: .euler, scheduler: .normal, steps: 20),
                // CFG Rescale 0.7 washes this merge out (grey, low contrast); without it the colours are right
                prediction: .v, cfgRescale: 0),
            source: "Civitai 989367 v2444683 (v-pred-04, by WAI0731): Euler a, 30 steps, CFG 5-7, 1024x1360, hires 1.5x"),
        ModelPreset(
            // Illustrious anime merge (NoobAI EPS 1.1 + Illustrious 2.0 + ChenkinNoob): the WAI v16 engine as it is
            id: "nova-anime-xl-il19", name: "Nova Anime XL IL v19", engine: .sdxl,
            checkpoint: "checkpoints/NovaAnimeILV190.safetensors",
            diffusionModel: nil, llm: nil, vae: nil, characterList: .wai160,
            defaults: PresetDefaults(
                // its page (Illustrious): Euler a, 20-30 steps, CFG 4-6, its quality tags and negative
                width: 832, height: 1216, steps: 28, cfg: 5, sampler: .eulerA, scheduler: .normal,
                hiresEnabled: true, hiresScale: 1.5, hiresSteps: 15, hiresDenoise: 0.4, maxHiresScale: 2,
                // + "safe" (NoobAI's rating tag; this merge leans to fan service without it)
                qualityPrefix: "masterpiece, best quality, amazing quality, very aesthetic, high resolution, ultra-detailed, absurdres, newest, safe",
                negative: novaAnimeNegative),
            source: "Civitai 376130 v2940478 (IL v19.0): Euler a, 20-30 steps, CFG 4-6 (Illustrious settings)"),
        ModelPreset(
            // the distilled Anima from its makers: WAI-ANIMA's engine, text encoder and VAE, CFG 1 and ~10 steps
            id: "anima-turbo-11", name: "Anima Turbo v1.1", engine: .anima, checkpoint: nil,
            diffusionModel: "diffusion_models/anima-turbo-v1.1.safetensors",
            llm: "text_encoders/qwen_3_06b_base.safetensors", vae: "vae/qwen_image_vae.safetensors",
            characterList: .anima,
            defaults: PresetDefaults(
                // its page: CFG 1, 8-12 steps; prefix "masterpiece, best quality, score_7, safe"
                width: 1024, height: 1344, steps: 10, cfg: 1, sampler: .eulerA, scheduler: .normal,
                hiresEnabled: false, hiresScale: 1.5, hiresSteps: 4, hiresDenoise: 0.35, maxHiresScale: 1.5,
                qualityPrefix: "masterpiece, best quality, score_7, safe", negative: "",
                fast: .init(sampler: .eulerA, scheduler: .normal, steps: 10), distilled: true),
            source: "Civitai 2458426 v3263843 (turbo v1.1, CircleStone Labs): CFG 1, 8-12 steps"),
    ]

    /// Where each model file comes from (the setup window's "Get It" buttons), by path in the models folder.
    static let sources: [String: (site: String, url: URL)] = [
        "checkpoints/waiIllustriousSDXL_v160.safetensors": ("Civitai", URL(string: "https://civitai.com/models/827184")!),
        "checkpoints/NoobAI-XL-v1.1.safetensors": ("Hugging Face", URL(string: "https://huggingface.co/Laxhar/noobai-XL-1.1")!),
        "checkpoints/noobaiXLNAIXL_vPred10Version.safetensors":
            ("Civitai", URL(string: "https://civitai.com/models/833294?modelVersionId=1190596")!),
        "diffusion_models/waiANIMA_v10Base10.safetensors": ("Civitai", URL(string: "https://civitai.com/models/2544636")!),
        "text_encoders/qwen_3_06b_base.safetensors":
            ("Hugging Face", URL(string: "https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/text_encoders")!),
        "vae/qwen_image_vae.safetensors":
            ("Hugging Face", URL(string: "https://huggingface.co/circlestone-labs/Anima/tree/main/split_files/vae")!),
        "upscale_models/RealESRGAN_x4plus_anime_6B.pth":
            ("GitHub", URL(string: "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.2.4")!),
        "detectors/face_yolov8m.safetensors": ("the README", URL(string: "https://github.com/ramirobuisan/wai-gen#models")!),
        "checkpoints/WAI-SHUFFLE-NOOB-v-pred-04.safetensors":
            ("Civitai", URL(string: "https://civitai.com/models/989367?modelVersionId=2444683")!),
        "checkpoints/NovaAnimeILV190.safetensors":
            ("Civitai", URL(string: "https://civitai.com/models/376130?modelVersionId=2940478")!),
        "diffusion_models/anima-turbo-v1.1.safetensors":
            ("Civitai", URL(string: "https://civitai.com/models/2458426?modelVersionId=3263843")!),
        "checkpoints/z-image-anime-aio-bf16.safetensors":
            ("Civitai", URL(string: "https://civitai.com/models/2259646?modelVersionId=2543657")!),
    ]

    static func preset(_ id: String) -> ModelPreset { presets.first { $0.id == id } ?? presets[0] }
}
