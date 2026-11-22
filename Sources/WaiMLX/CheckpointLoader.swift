// Loads an original-layout ("ldm" / single-file, e.g. Civitai) SDXL checkpoint such as
// WAI-illustrious-SDXL into the MLX models.
//
// The key translation mirrors ComfyUI's ldm <-> diffusers converters (comfy/utils.py
// `unet_to_diffusers`, comfy/diffusers_convert.py) for the fixed SDXL layout. The final
// diffusers -> MLX renames (`unetRemap`, `clipRemap`, `vaeRemap`) are from Apple's
// mlx-swift-examples StableDiffusion/Load.swift (MIT).

import Foundation
import MLX
import MLXNN

public enum WaiError: LocalizedError {
    case missingResource(String)
    case badCheckpoint(String)

    public var errorDescription: String? {
        switch self {
        case .missingResource(let r): "Missing bundled resource: \(r)"
        case .badCheckpoint(let r): "Checkpoint is not a usable SDXL file: \(r)"
        }
    }
}

// MARK: - diffusers -> MLX module names (from Apple's Load.swift)

func keyReplace(_ replace: String, _ with: String) -> (String) -> String? {
    { key in key.contains(replace) ? key.replacingOccurrences(of: replace, with: with) : nil }
}

func dropPrefix(_ prefix: String) -> (String) -> String? {
    { key in key.hasPrefix(prefix) ? String(key.dropFirst(prefix.count)) : nil }
}

let unetRules: [(String) -> String?] = [
    keyReplace("downsamplers.0.conv", "downsample"),
    keyReplace("upsamplers.0.conv", "upsample"),
    keyReplace("mid_block.resnets.0", "mid_blocks.0"),
    keyReplace("mid_block.attentions.0", "mid_blocks.1"),
    keyReplace("mid_block.resnets.1", "mid_blocks.2"),
    keyReplace("to_k", "key_proj"),
    keyReplace("to_out.0", "out_proj"),
    keyReplace("to_q", "query_proj"),
    keyReplace("to_v", "value_proj"),
    keyReplace("ff.net.2", "linear3"),
]

func unetRemap(key: String, value: MLXArray) -> [(String, MLXArray)] {
    var key = key
    var value = value
    for rule in unetRules { key = rule(key) ?? key }

    // GEGLU projection is split into two linears
    if key.contains("ff.net.0") {
        let k1 = key.replacingOccurrences(of: "ff.net.0.proj", with: "linear1")
        let k2 = key.replacingOccurrences(of: "ff.net.0.proj", with: "linear2")
        let (v1, v2) = value.split()
        return [(k1, v1), (k2, v2)]
    }
    if key.contains("conv_shortcut.weight") { value = value.squeezed() }
    if value.ndim == 4 && (key.contains("proj_in") || key.contains("proj_out")) {
        value = value.squeezed()
    }
    if value.ndim == 4 {
        // PyTorch OIHW -> MLX OHWI
        value = value.transposed(0, 2, 3, 1)
        value = value.reshaped(-1).reshaped(value.shape)
    }
    return [(key, value)]
}

let clipRules: [(String) -> String?] = [
    dropPrefix("text_model."),
    dropPrefix("embeddings."),
    dropPrefix("encoder."),
    keyReplace("self_attn.", "attention."),
    keyReplace("q_proj.", "query_proj."),
    keyReplace("k_proj.", "key_proj."),
    keyReplace("v_proj.", "value_proj."),
    keyReplace("mlp.fc1", "linear1"),
    keyReplace("mlp.fc2", "linear2"),
]

func clipRemap(key: String, value: MLXArray) -> [(String, MLXArray)] {
    var key = key
    for rule in clipRules { key = rule(key) ?? key }
    if key == "position_ids" { return [] }
    return [(key, value)]
}

let vaeRules: [(String) -> String?] = [
    keyReplace("downsamplers.0.conv", "downsample"),
    keyReplace("upsamplers.0.conv", "upsample"),
    keyReplace("to_k", "key_proj"),
    keyReplace("to_out.0", "out_proj"),
    keyReplace("to_q", "query_proj"),
    keyReplace("to_v", "value_proj"),
    keyReplace("mid_block.resnets.0", "mid_blocks.0"),
    keyReplace("mid_block.attentions.0", "mid_blocks.1"),
    keyReplace("mid_block.resnets.1", "mid_blocks.2"),
]

func vaeRemap(key: String, value: MLXArray) -> [(String, MLXArray)] {
    var key = key
    var value = value
    for rule in vaeRules { key = rule(key) ?? key }
    if key.contains("quant_conv") {
        key = key.replacingOccurrences(of: "quant_conv", with: "quant_proj")
        value = value.squeezed()
    }
    if key.contains("conv_shortcut.weight") { value = value.squeezed() }
    if value.ndim == 4 {
        value = value.transposed(0, 2, 3, 1)
        value = value.reshaped(-1).reshaped(value.shape)
    }
    return [(key, value)]
}

// MARK: - ldm (single-file) -> diffusers names for SDXL

/// SDXL UNet: diffusers name -> ldm name (ComfyUI `unet_to_diffusers` with the SDXL config).
func sdxlUnetDiffusersToLdm() -> [String: String] {
    let resnet: [String: String] = [
        "in_layers.2.weight": "conv1.weight", "in_layers.2.bias": "conv1.bias",
        "emb_layers.1.weight": "time_emb_proj.weight", "emb_layers.1.bias": "time_emb_proj.bias",
        "out_layers.3.weight": "conv2.weight", "out_layers.3.bias": "conv2.bias",
        "skip_connection.weight": "conv_shortcut.weight", "skip_connection.bias": "conv_shortcut.bias",
        "in_layers.0.weight": "norm1.weight", "in_layers.0.bias": "norm1.bias",
        "out_layers.0.weight": "norm2.weight", "out_layers.0.bias": "norm2.bias",
    ]
    let attention = ["proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias",
                     "norm.weight", "norm.bias"]
    let block = ["norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "norm3.weight",
                 "norm3.bias", "attn1.to_q.weight", "attn1.to_k.weight", "attn1.to_v.weight",
                 "attn1.to_out.0.weight", "attn1.to_out.0.bias", "attn2.to_q.weight",
                 "attn2.to_k.weight", "attn2.to_v.weight", "attn2.to_out.0.weight",
                 "attn2.to_out.0.bias", "ff.net.0.proj.weight", "ff.net.0.proj.bias",
                 "ff.net.2.weight", "ff.net.2.bias"]

    // SDXL: 3 levels, 2 res blocks each, transformer depth 0/2/10, middle 10
    let numRes = [2, 2, 2]
    var depth = [0, 0, 2, 2, 10, 10]
    var depthOut = [0, 0, 0, 2, 2, 2, 10, 10, 10]
    let midDepth = 10
    var map = [String: String]()

    for x in 0 ..< 3 {
        var n = 1 + (numRes[x] + 1) * x
        for i in 0 ..< numRes[x] {
            for (ldm, dif) in resnet { map["down_blocks.\(x).resnets.\(i).\(dif)"] = "input_blocks.\(n).0.\(ldm)" }
            let t = depth.removeFirst()
            if t > 0 {
                for b in attention { map["down_blocks.\(x).attentions.\(i).\(b)"] = "input_blocks.\(n).1.\(b)" }
                for j in 0 ..< t {
                    for b in block {
                        map["down_blocks.\(x).attentions.\(i).transformer_blocks.\(j).\(b)"] =
                            "input_blocks.\(n).1.transformer_blocks.\(j).\(b)"
                    }
                }
            }
            n += 1
        }
        for k in ["weight", "bias"] { map["down_blocks.\(x).downsamplers.0.conv.\(k)"] = "input_blocks.\(n).0.op.\(k)" }
    }

    for b in attention { map["mid_block.attentions.0.\(b)"] = "middle_block.1.\(b)" }
    for j in 0 ..< midDepth {
        for b in block { map["mid_block.attentions.0.transformer_blocks.\(j).\(b)"] = "middle_block.1.transformer_blocks.\(j).\(b)" }
    }
    for (i, n) in [0, 2].enumerated() {
        for (ldm, dif) in resnet { map["mid_block.resnets.\(i).\(dif)"] = "middle_block.\(n).\(ldm)" }
    }

    let numResRev = Array(numRes.reversed())
    for x in 0 ..< 3 {
        var n = (numResRev[x] + 1) * x
        let l = numResRev[x] + 1
        for i in 0 ..< l {
            var c = 0
            for (ldm, dif) in resnet { map["up_blocks.\(x).resnets.\(i).\(dif)"] = "output_blocks.\(n).0.\(ldm)" }
            c += 1
            let t = depthOut.removeLast()
            if t > 0 {
                c += 1
                for b in attention { map["up_blocks.\(x).attentions.\(i).\(b)"] = "output_blocks.\(n).1.\(b)" }
                for j in 0 ..< t {
                    for b in block {
                        map["up_blocks.\(x).attentions.\(i).transformer_blocks.\(j).\(b)"] =
                            "output_blocks.\(n).1.transformer_blocks.\(j).\(b)"
                    }
                }
            }
            if i == l - 1 {
                for k in ["weight", "bias"] { map["up_blocks.\(x).upsamplers.0.conv.\(k)"] = "output_blocks.\(n).\(c).conv.\(k)" }
            }
            n += 1
        }
    }

    let basic: [(String, String)] = [
        ("label_emb.0.0.weight", "add_embedding.linear_1.weight"),
        ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
        ("label_emb.0.2.weight", "add_embedding.linear_2.weight"),
        ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
        ("input_blocks.0.0.weight", "conv_in.weight"), ("input_blocks.0.0.bias", "conv_in.bias"),
        ("out.0.weight", "conv_norm_out.weight"), ("out.0.bias", "conv_norm_out.bias"),
        ("out.2.weight", "conv_out.weight"), ("out.2.bias", "conv_out.bias"),
        ("time_embed.0.weight", "time_embedding.linear_1.weight"),
        ("time_embed.0.bias", "time_embedding.linear_1.bias"),
        ("time_embed.2.weight", "time_embedding.linear_2.weight"),
        ("time_embed.2.bias", "time_embedding.linear_2.bias"),
    ]
    for (ldm, dif) in basic { map[dif] = ldm }
    return map
}

/// VAE: ldm name -> diffusers name (ComfyUI `vae_conversion_map`, applied as substring replaces).
private let vaeLdmToDiffusers: [(String, String)] = {
    var m: [(String, String)] = [("nin_shortcut", "conv_shortcut"), ("norm_out", "conv_norm_out"),
                                 ("mid.attn_1.", "mid_block.attentions.0.")]
    for i in 0 ..< 4 {
        for j in 0 ..< 2 { m.append(("encoder.down.\(i).block.\(j).", "encoder.down_blocks.\(i).resnets.\(j).")) }
        if i < 3 {
            m.append(("down.\(i).downsample.", "down_blocks.\(i).downsamplers.0."))
            m.append(("up.\(3 - i).upsample.", "up_blocks.\(i).upsamplers.0."))
        }
        for j in 0 ..< 3 { m.append(("decoder.up.\(3 - i).block.\(j).", "decoder.up_blocks.\(i).resnets.\(j).")) }
    }
    for i in 0 ..< 2 { m.append(("mid.block_\(i + 1).", "mid_block.resnets.\(i).")) }
    return m
}()

// MARK: - Loading

/// The tensor names stored in a safetensors file, read from its JSON header alone (no tensor
/// data touched), so configs can ship marker tensors without the loader having to materialise them.
func safetensorsTensorNames(contentsOf url: URL) throws -> [String] {
    let handle = try FileHandle(forReadingFrom: url)
    defer { try? handle.close() }
    guard let sizeData = try handle.read(upToCount: 8), sizeData.count == 8 else {
        throw WaiError.badCheckpoint("\(url.lastPathComponent): truncated safetensors header")
    }
    let headerSize = Int(sizeData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) })
    guard headerSize > 2, headerSize < 1 << 28,
        let headerData = try handle.read(upToCount: headerSize), headerData.count == headerSize,
        let header = try JSONSerialization.jsonObject(with: headerData) as? [String: Any]
    else { throw WaiError.badCheckpoint("\(url.lastPathComponent): unreadable safetensors header") }
    return header.keys.filter { $0 != "__metadata__" }
}

/// The four SDXL components loaded from one single-file checkpoint.
struct SDXLComponents {
    let unet: UNetModel
    let clipL: CLIPTextModel
    let clipG: CLIPTextModel
    let vae: Autoencoder
    /// what the checkpoint's own config says it predicts (`v_pred`/`ztsnr` markers); presets and
    /// CLI flags override the maths, but it picks the noise schedule: the model was trained on it
    let prediction: Prediction
}

func bundledJSON<T: Decodable>(_ name: String, as: T.Type) throws -> T {
    guard let url = Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Resources")
    else { throw WaiError.missingResource("\(name).json") }
    return try JSONDecoder().decode(T.self, from: Data(contentsOf: url))
}

private func apply(_ weights: [(String, MLXArray)], to model: Module) throws {
    try model.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
}

/// Load a single-file SDXL checkpoint (WAI-illustrious) into MLX modules.
/// UNet and text encoders use `dType` (fp16); the VAE uses bfloat16 (SDXL's VAE overflows in
/// fp16; bf16 has fp32's range, like ComfyUI's default on Apple silicon).
func loadSDXLCheckpoint(url: URL, dType: DType = .float16) throws -> SDXLComponents {
    let all = try loadArrays(url: url)

    let unetPrefix = "model.diffusion_model."
    let clipLPrefix = "conditioner.embedders.0.transformer."
    let clipGPrefix = "conditioner.embedders.1.model."
    let vaePrefix = "first_stage_model."
    guard all.keys.contains(where: { $0.hasPrefix(unetPrefix) }),
        all.keys.contains(where: { $0.hasPrefix(clipGPrefix) })
    else { throw WaiError.badCheckpoint("no SDXL UNet / OpenCLIP-G weights in \(url.lastPathComponent)") }

    // --- UNet
    var ldmToDif = [String: String]()
    for (dif, ldm) in sdxlUnetDiffusersToLdm() { ldmToDif[ldm] = dif }
    var unetWeights = [(String, MLXArray)]()
    var unmapped = 0
    for (k, v) in all where k.hasPrefix(unetPrefix) {
        let ldm = String(k.dropFirst(unetPrefix.count))
        guard let dif = ldmToDif[ldm] else { unmapped += 1; continue }
        unetWeights += unetRemap(key: dif, value: v.asType(dType))
    }
    if unmapped > 0 { throw WaiError.badCheckpoint("\(unmapped) UNet tensors did not map to SDXL") }
    let unet = UNetModel(configuration: try bundledJSON("sdxl_unet", as: UNetConfiguration.self))
    try apply(unetWeights, to: unet)

    // --- CLIP-L (stored in Hugging Face transformers layout)
    var clipLWeights = [(String, MLXArray)]()
    for (k, v) in all where k.hasPrefix(clipLPrefix) {
        clipLWeights += clipRemap(key: String(k.dropFirst(clipLPrefix.count)), value: v.asType(dType))
    }
    let clipL = CLIPTextModel(configuration: try bundledJSON("sdxl_clip_l", as: CLIPTextModelConfiguration.self))
    try apply(clipLWeights, to: clipL)

    // --- OpenCLIP-G (open_clip layout: fused qkv, c_fc/c_proj, text_projection matrix)
    var clipGWeights = [(String, MLXArray)]()
    for (k, v) in all where k.hasPrefix(clipGPrefix) {
        let key = String(k.dropFirst(clipGPrefix.count))
        let v = v.asType(dType)
        switch key {
        case "token_embedding.weight": clipGWeights.append(("token_embedding.weight", v))
        case "positional_embedding": clipGWeights.append(("position_embedding.weight", v))
        case "ln_final.weight": clipGWeights.append(("final_layer_norm.weight", v))
        case "ln_final.bias": clipGWeights.append(("final_layer_norm.bias", v))
        case "text_projection": clipGWeights.append(("text_projection.weight", v.T))  // x @ P == Linear(P^T)
        case "logit_scale": break
        default:
            guard key.hasPrefix("transformer.resblocks.") else { continue }
            let parts = key.dropFirst("transformer.resblocks.".count).split(separator: ".", maxSplits: 1)
            let layer = "layers.\(parts[0])."
            let rest = String(parts[1])
            switch rest {
            case "attn.in_proj_weight", "attn.in_proj_bias":
                let suffix = rest.hasSuffix("weight") ? "weight" : "bias"
                let qkv = v.split(parts: 3, axis: 0)
                clipGWeights.append((layer + "attention.query_proj.\(suffix)", qkv[0]))
                clipGWeights.append((layer + "attention.key_proj.\(suffix)", qkv[1]))
                clipGWeights.append((layer + "attention.value_proj.\(suffix)", qkv[2]))
            default:
                let mapped = rest
                    .replacingOccurrences(of: "attn.out_proj.", with: "attention.out_proj.")
                    .replacingOccurrences(of: "ln_1.", with: "layer_norm1.")
                    .replacingOccurrences(of: "ln_2.", with: "layer_norm2.")
                    .replacingOccurrences(of: "mlp.c_fc.", with: "linear1.")
                    .replacingOccurrences(of: "mlp.c_proj.", with: "linear2.")
                clipGWeights.append((layer + mapped, v))
            }
        }
    }
    let clipG = CLIPTextModel(configuration: try bundledJSON("sdxl_clip_g", as: CLIPTextModelConfiguration.self))
    try apply(clipGWeights, to: clipG)

    // --- VAE (bfloat16)
    var vaeWeights = [(String, MLXArray)]()
    for (k, v) in all where k.hasPrefix(vaePrefix) {
        var key = String(k.dropFirst(vaePrefix.count))
        var value = v.asType(.bfloat16)
        for (ldm, dif) in vaeLdmToDiffusers where key.contains(ldm) {
            key = key.replacingOccurrences(of: ldm, with: dif)
        }
        if key.contains("mid_block.attentions.0.") {
            // single-head attention stored as 1x1 convs -> linear
            for (ldm, dif) in [("norm.", "group_norm."), ("q.", "to_q."), ("k.", "to_k."),
                               ("v.", "to_v."), ("proj_out.", "to_out.0.")] {
                let marker = "mid_block.attentions.0."
                if key.hasPrefix(marker + ldm) || key.contains("." + marker + ldm) {
                    key = key.replacingOccurrences(of: marker + ldm, with: marker + dif)
                    break
                }
            }
            if value.ndim == 4 { value = value.reshaped(value.dim(0), value.dim(1)) }
        }
        vaeWeights += vaeRemap(key: key, value: value)
    }
    let vae = Autoencoder(configuration: try bundledJSON("sdxl_vae", as: AutoencoderConfiguration.self))
    try apply(vaeWeights, to: vae)

    // --- checkpoint markers: NoobAI-XL V-Pred 1.0 stores empty `v_pred`/`ztsnr` tensors in its
    // header; they say the model predicts v and was trained with zero terminal SNR sigmas.
    // (Read from the safetensors header, not `all`: a streaming load may skip zero-size tensors.)
    let names = (try? safetensorsTensorNames(contentsOf: url)) ?? []
    let prediction: Prediction = names.contains("v_pred") || names.contains("ztsnr") ? .v : .epsilon

    return SDXLComponents(unet: unet, clipL: clipL, clipG: clipG, vae: vae, prediction: prediction)
}

/// CLIP BPE tokenizer shared by both SDXL text encoders (they use the same vocabulary).
func loadCLIPTokenizer() throws -> CLIPTokenizer {
    guard let vocabURL = Bundle.module.url(forResource: "vocab", withExtension: "json", subdirectory: "Resources"),
        let mergesURL = Bundle.module.url(forResource: "merges", withExtension: "txt", subdirectory: "Resources")
    else { throw WaiError.missingResource("vocab.json / merges.txt") }
    let vocabulary = try JSONDecoder().decode([String: Int].self, from: Data(contentsOf: vocabURL))
    let merges = try String(contentsOf: mergesURL, encoding: .utf8)
        .components(separatedBy: .newlines).dropFirst().filter { !$0.isEmpty }
    return CLIPTokenizer(merges: Array(merges), vocabulary: vocabulary)
}
