// LoRA for the MLX SDXL engine: kohya (lora_unet_* / lora_te1_* / lora_te2_*, original or diffusers
// layer names) and PEFT/diffusers (unet.* / text_encoder.* with lora_A/lora_B) files.
// Deltas `strength * alpha / rank * up @ down` are merged into the weights; the original weights of
// every touched layer are kept so a different LoRA set can be applied without reloading the model.
// LyCORIS LoCon, LoHa and LoKr (stable-diffusion.cpp's / ComfyUI's scale rules) and its trained norm layers
// (train_norm: w_norm / b_norm added to the norm weights) are merged too; DoRA and Tucker-decomposed LyCORIS
// tensors are reported as unsupported.

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
    public var unsupported = 0      // DoRA / Tucker LyCORIS modules
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
        case norm            // LyCORIS train_norm: w_norm -> weight, b_norm -> bias
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
            for name in ["unet:" + us(ldmBase), "unet:" + us(difBase)] {
                add(name, LoRATarget(model: .unet, key: mlxKey, kind: .norm))
            }
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
    // CLIP-G's final projection (some kohya LoRAs train it)
    add("te2:text_projection", LoRATarget(model: .clipG, key: "text_projection.weight", kind: .linear))
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
            // group the file's tensors by module: "lora_unet_x.lokr_w1" -> ["lora_unet_x": ["lokr_w1": ...]]
            var groups: [String: [String: MLXArray]] = [:]
            for (k, v) in tensors {
                guard let (base, role) = splitLoRAKey(k) else { continue }
                groups[base, default: [:]][role] = v
            }
            for (base, p) in groups.sorted(by: { $0.key < $1.key }) {
                guard let name = normaliseLoRAName(base), let ts = targets[name] else {
                    report.unmatched.append(base)
                    continue
                }
                func add(_ t: LoRATarget, _ key: String, _ d: MLXArray) -> Bool {
                    guard let w = updates[t.model]?[key] ?? params[t.model]?[key], w.shape == d.shape else { return false }
                    let id = "\(t.model)|\(key)"
                    if originals[id] == nil, let orig = params[t.model]?[key] {
                        // Module.update replaces array contents in place, so keep a materialised copy
                        let saved = orig * 1
                        eval(saved)
                        originals[id] = (t.model, saved)
                    }
                    updates[t.model, default: [:]][key] = (w.asType(.float32) + d).asType(w.dtype)
                    return true
                }
                // LyCORIS trained norm layers
                if p["w_norm"] != nil || p["b_norm"] != nil {
                    guard let t = ts.first(where: { if case .norm = $0.kind { true } else { false } }) else {
                        report.unmatched.append(base)
                        continue
                    }
                    let bias = String(t.key.dropLast("weight".count)) + "bias"
                    var ok = true
                    if let wn = p["w_norm"] { ok = add(t, t.key, wn.asType(.float32) * spec.strength) && ok }
                    if let bn = p["b_norm"] { ok = add(t, bias, bn.asType(.float32) * spec.strength) && ok }
                    if ok { report.applied += 1 } else { report.unmatched.append(base) }
                    continue
                }
                if p["dora_scale"] != nil || p["hada_t1"] != nil || p["hada_t2"] != nil || p["lokr_t1"] != nil
                    || p["lokr_t2"] != nil {
                    report.unsupported += 1
                    continue
                }
                guard let (delta2D, scale) = loraDelta(p) else {
                    report.unmatched.append(base)
                    continue
                }
                // full delta in PyTorch layout, flattened to (out, in * kh * kw)
                let delta = delta2D * (scale * spec.strength)
                var applied = false
                for t in ts {
                    var d: MLXArray
                    switch t.kind {
                    case .norm:
                        continue
                    case .linear, .squeezed:
                        d = delta
                    case .conv:
                        // MLX conv weights are (O, kh, kw, I); PyTorch's (O, I, kh, kw)
                        guard let w = params[t.model]?[t.key], w.ndim == 4, delta.size == w.size else { continue }
                        d = delta.reshaped(w.dim(0), w.dim(3), w.dim(1), w.dim(2)).transposed(0, 2, 3, 1)
                    case .gegluHalf(let half):
                        d = delta.split(parts: 2, axis: 0)[half]
                    }
                    if add(t, t.key, d) {
                        report.applied += 1
                        applied = true
                    }
                }
                if !applied { report.unmatched.append(base) }
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

/// "lora_unet_x.lokr_w1" -> ("lora_unet_x", "lokr_w1"); "x.lora_down.weight" -> ("x", "down"). nil for other tensors.
func splitLoRAKey(_ key: String) -> (String, String)? {
    let roles: [(String, String)] = [
        (".lora_down.weight", "down"), (".lora_A.weight", "down"), (".lora_up.weight", "up"), (".lora_B.weight", "up"),
        (".hada_w1_a", "hada_w1_a"), (".hada_w1_b", "hada_w1_b"), (".hada_w2_a", "hada_w2_a"), (".hada_w2_b", "hada_w2_b"),
        (".hada_t1", "hada_t1"), (".hada_t2", "hada_t2"),
        (".lokr_w1_a", "lokr_w1_a"), (".lokr_w1_b", "lokr_w1_b"), (".lokr_w2_a", "lokr_w2_a"), (".lokr_w2_b", "lokr_w2_b"),
        (".lokr_w1", "lokr_w1"), (".lokr_w2", "lokr_w2"), (".lokr_t1", "lokr_t1"), (".lokr_t2", "lokr_t2"),
        (".w_norm", "w_norm"), (".b_norm", "b_norm"), (".dora_scale", "dora_scale"), (".alpha", "alpha"),
    ]
    for (suffix, role) in roles where key.hasSuffix(suffix) {
        return (String(key.dropLast(suffix.count)), role)
    }
    return nil
}

/// The module's weight delta in PyTorch layout flattened to (out, in * kh * kw), and its scale (before strength).
/// LoRA / LoCon: up · down, alpha / rank. LoHa: (w1a · w1b) ⊙ (w2a · w2b), alpha / rank. LoKr: w1 ⊗ w2 (each
/// optionally a low-rank pair), alpha / rank when a factor is decomposed, else 1 (stable-diffusion.cpp, ComfyUI).
func loraDelta(_ p: [String: MLXArray]) -> (MLXArray, Float)? {
    func f(_ k: String) -> MLXArray? { p[k].map { $0.asType(.float32) } }
    func flat(_ a: MLXArray) -> MLXArray { a.reshaped(a.dim(0), -1) }
    let alpha = p["alpha"].map { $0.asType(.float32).item(Float.self) }
    if let down = f("down"), let up = f("up") {
        let rank = down.dim(0)
        return (matmul(flat(up), flat(down)), (alpha ?? Float(rank)) / Float(rank))
    }
    if let a1 = f("hada_w1_a"), let b1 = f("hada_w1_b"), let a2 = f("hada_w2_a"), let b2 = f("hada_w2_b") {
        let rank = b1.dim(0)
        return (matmul(flat(a1), flat(b1)) * matmul(flat(a2), flat(b2)), alpha.map { $0 / Float(rank) } ?? 1)
    }
    var w1 = f("lokr_w1"), w2 = f("lokr_w2")
    var rank: Int? = nil
    if w1 == nil, let a = f("lokr_w1_a"), let b = f("lokr_w1_b") { w1 = matmul(flat(a), flat(b)); rank = b.dim(0) }
    if w2 == nil, let a = f("lokr_w2_a"), let b = f("lokr_w2_b") { w2 = matmul(flat(a), flat(b)); rank = b.dim(0) }
    guard let w1, let w2 else { return nil }
    // Kronecker product; with a conv w2 (o2, i2, kh, kw) flattened, rows/columns land as (o1 o2, i1 i2 kh kw)
    let a = flat(w1), b = flat(w2)
    let kron = (a[0..., .newAxis, 0..., .newAxis] * b[.newAxis, 0..., .newAxis, 0...])
        .reshaped(a.dim(0) * b.dim(0), a.dim(1) * b.dim(1))
    let scale: Float = if let rank, let alpha { alpha / Float(rank) } else { 1 }
    return (kron, scale)
}
