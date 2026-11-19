// LoRA for the MLX SDXL engine: kohya (lora_unet_* / lora_te1_* / lora_te2_*, original or diffusers
// layer names) and PEFT/diffusers (unet.* / text_encoder.* with lora_A/lora_B) files.
// Deltas `strength * alpha / rank * up @ down` are merged into the weights; the original weights of
// every touched layer are kept so a different LoRA set can be applied without reloading the model.
// LyCORIS variants (LoHa, LoKr) are reported as unsupported.

import Foundation
import MLX
import MLXNN

public struct LoRASpec: Hashable, Sendable {
    public let url: URL
    public let strength: Float
    public init(url: URL, strength: Float) {
        self.url = url
        self.strength = strength
    }
}

public struct LoRAReport: Sendable {
    public var applied = 0          // layers changed
    public var unmatched: [String] = []
    public var unsupported = 0      // LoHa/LoKr/... tensors
    public var files = 0
}

/// Where a LoRA layer lands in the MLX models.
struct LoRATarget {
    enum Model { case unet, clipL, clipG }
    enum Kind {
        case linear
        case conv        // delta (O,I,kh,kw) -> MLX (O,kh,kw,I)
        case squeezed    // 1x1 conv stored as linear in MLX
        case gegluHalf(Int)  // diffusers ff.net.0.proj split into linear1 / linear2
    }
    let model: Model
    let key: String
    let kind: Kind
}

/// Maps normalised LoRA layer names ("unet:input_blocks_4_1_...", "te1:text_model_encoder_layers_0_self_attn_q_proj")
/// to MLX parameter keys.
func buildLoRATargets() -> [String: [LoRATarget]] {
    var map = [String: [LoRATarget]]()
    func add(_ name: String, _ t: LoRATarget) { map[name, default: []].append(t) }
    func us(_ s: String) -> String { s.replacingOccurrences(of: ".", with: "_") }

    for (dif, ldm) in sdxlUnetDiffusersToLdm() where dif.hasSuffix(".weight") {
        let difBase = String(dif.dropLast(".weight".count))
        let ldmBase = String(ldm.dropLast(".weight".count))
        var mlxKey = dif
        for rule in unetRules { mlxKey = rule(mlxKey) ?? mlxKey }
        var targets: [LoRATarget]
        if dif.contains("ff.net.0.proj") {
            targets = [
                LoRATarget(model: .unet, key: mlxKey.replacingOccurrences(of: "ff.net.0.proj", with: "linear1"), kind: .gegluHalf(0)),
                LoRATarget(model: .unet, key: mlxKey.replacingOccurrences(of: "ff.net.0.proj", with: "linear2"), kind: .gegluHalf(1)),
            ]
        } else if dif.contains("conv_shortcut") {
            targets = [LoRATarget(model: .unet, key: mlxKey, kind: .squeezed)]
        } else if dif.contains("conv") || dif.contains("downsamplers") || dif.contains("upsamplers") {
            targets = [LoRATarget(model: .unet, key: mlxKey, kind: .conv)]
        } else if dif.contains("norm") {
            continue
        } else {
            targets = [LoRATarget(model: .unet, key: mlxKey, kind: .linear)]
        }
        for t in targets {
            add("unet:" + us(ldmBase), t)
            add("unet:" + us(difBase), t)
        }
    }

    for (prefix, model, layers) in [("te1", LoRATarget.Model.clipL, 12), ("te2", .clipG, 32)] {
        for i in 0 ..< layers {
            let hf = "text_model_encoder_layers_\(i)_"
            for (src, dst) in [("self_attn_q_proj", "attention.query_proj"), ("self_attn_k_proj", "attention.key_proj"),
                               ("self_attn_v_proj", "attention.value_proj"), ("self_attn_out_proj", "attention.out_proj"),
                               ("mlp_fc1", "linear1"), ("mlp_fc2", "linear2")] {
                add("\(prefix):\(hf)\(src)", LoRATarget(model: model, key: "layers.\(i).\(dst).weight", kind: .linear))
            }
        }
    }
    return map
}

/// "lora_unet_x_y" / "unet.x.y" / "lora_te1_..." / "text_encoder.x" -> "unet:x_y", "te1:..."
func normaliseLoRAName(_ base: String) -> String? {
    let pairs: [(String, String)] = [
        ("lora_unet_", "unet:"), ("lora_te1_", "te1:"), ("lora_te2_", "te2:"), ("lora_te_", "te1:"),
        ("unet.", "unet:"), ("text_encoder_2.", "te2:"), ("text_encoder.", "te1:"),
        ("base_model.model.", "unet:"),
    ]
    for (p, n) in pairs where base.hasPrefix(p) {
        return n + String(base.dropFirst(p.count)).replacingOccurrences(of: ".", with: "_")
    }
    return nil
}

/// Applies and removes LoRA deltas on the SDXL modules.
final class LoRAApplier {
    private let targets = buildLoRATargets()
    private var originals: [String: (LoRATarget.Model, MLXArray)] = [:]   // "model|key" -> weight
    private(set) var current: [LoRASpec] = []

    func apply(_ specs: [LoRASpec], unet: UNetModel, clipL: CLIPTextModel, clipG: CLIPTextModel) throws -> LoRAReport {
        var report = LoRAReport()
        if specs == current { return report }
        restore(unet: unet, clipL: clipL, clipG: clipG)

        func module(_ m: LoRATarget.Model) -> Module {
            switch m {
            case .unet: unet
            case .clipL: clipL
            case .clipG: clipG
            }
        }
        var params: [LoRATarget.Model: [String: MLXArray]] = [:]
        for m in [LoRATarget.Model.unet, .clipL, .clipG] {
            params[m] = Dictionary(module(m).parameters().flattened(), uniquingKeysWith: { a, _ in a })
        }
        var updates: [LoRATarget.Model: [String: MLXArray]] = [:]

        for spec in specs where spec.strength != 0 {
            report.files += 1
            let tensors = try loadArrays(url: spec.url)
            var bases = Set<String>()
            for k in tensors.keys {
                for suffix in [".lora_down.weight", ".lora_A.weight"] where k.hasSuffix(suffix) {
                    bases.insert(String(k.dropLast(suffix.count)))
                }
                if k.contains(".hada_") || k.contains(".lokr_") { report.unsupported += 1 }
            }
            for base in bases.sorted() {
                guard let name = normaliseLoRAName(base), let ts = targets[name] else {
                    report.unmatched.append(base)
                    continue
                }
                let down = (tensors[base + ".lora_down.weight"] ?? tensors[base + ".lora_A.weight"])!.asType(.float32)
                guard let up = (tensors[base + ".lora_up.weight"] ?? tensors[base + ".lora_B.weight"])?.asType(.float32)
                else { report.unmatched.append(base); continue }
                let rank = down.dim(0)
                let alpha = tensors[base + ".alpha"].map { $0.asType(.float32).item(Float.self) } ?? Float(rank)
                let scale = spec.strength * alpha / Float(rank)
                // full-shape delta in PyTorch layout
                var delta = matmul(up.reshaped(up.dim(0), rank), down.reshaped(rank, -1)) * scale
                if down.ndim == 4 { delta = delta.reshaped(up.dim(0), down.dim(1), down.dim(2), down.dim(3)) }

                for t in ts {
                    var d = delta
                    switch t.kind {
                    case .linear, .squeezed:
                        d = d.reshaped(d.dim(0), -1)
                    case .conv:
                        if d.ndim == 2 { d = d.reshaped(d.dim(0), d.dim(1), 1, 1) }
                        d = d.transposed(0, 2, 3, 1)
                    case .gegluHalf(let half):
                        let parts = d.reshaped(d.dim(0), -1).split(parts: 2, axis: 0)
                        d = parts[half]
                    }
                    let current = updates[t.model]?[t.key] ?? params[t.model]?[t.key]
                    guard let w = current, w.shape == d.shape else {
                        report.unmatched.append(base)
                        continue
                    }
                    let id = "\(t.model)|\(t.key)"
                    if originals[id] == nil, let orig = params[t.model]?[t.key] {
                        // Module.update replaces array contents in place, so keep a materialised copy
                        let saved = orig * 1
                        eval(saved)
                        originals[id] = (t.model, saved)
                    }
                    updates[t.model, default: [:]][t.key] = (w.asType(.float32) + d).asType(w.dtype)
                    report.applied += 1
                }
            }
        }

        for (m, u) in updates where !u.isEmpty {
            try module(m).update(parameters: ModuleParameters.unflattened(u.map { ($0.key, $0.value) }), verify: .none)
            eval(module(m))
        }
        current = specs
        return report
    }

    func restore(unet: UNetModel, clipL: CLIPTextModel, clipG: CLIPTextModel) {
        guard !originals.isEmpty else { current = []; return }
        var byModel: [LoRATarget.Model: [(String, MLXArray)]] = [:]
        for (id, (m, w)) in originals {
            byModel[m, default: []].append((String(id.split(separator: "|", maxSplits: 1)[1]), w))
        }
        for (m, list) in byModel {
            let mod: Module = m == .unet ? unet : (m == .clipL ? clipL : clipG)
            try? mod.update(parameters: ModuleParameters.unflattened(list), verify: .none)
            eval(mod)
        }
        originals.removeAll()
        current = []
    }
}
