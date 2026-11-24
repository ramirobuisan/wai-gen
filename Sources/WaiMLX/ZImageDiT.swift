import Foundation
import MLX
import MLXNN

/// The Z-Image NextDiT (Lumina-Image-2.0 family with `z_image_modulation`), as in the z-image-anime Turbo AIO, in
/// MLX. A port of `NextDiT` in ComfyUI's `comfy/ldm/lumina/model.py`, checked against tensors dumped from it
/// (`scripts/zimage_reference.py`, `waigen-cli --zimage-check`):
/// - caption: RMSNorm + Linear (2560 → 3840), padded to a multiple of 32 with the learned `cap_pad_token`, RoPE
///   positions (i + 1, 0, 0), then 2 `context_refiner` blocks without modulation;
/// - image: 2×2 patches of the 16-channel latent (features ordered row, column, channel), Linear (64 → 3840),
///   padded to a multiple of 32 with `x_pad_token` (pad positions (0, 0, 0)), positions (capLen + 1, row, column),
///   then 2 `noise_refiner` blocks with modulation;
/// - 30 blocks over [caption, image]; blocks: x += tanh(gate) · norm2(attn(norm1(x) · (1 + scale))), same for the
///   SwiGLU feed-forward; the adaLN is Linear(256 → 4·3840) on the time embedding (no SiLU);
/// - attention: fused qkv, per-head RMSNorm on q and k (eps = float32 epsilon, as ComfyUI's RMSNorm(eps=None)),
///   Flux-style 3-axis RoPE (32/48/48 dims, theta 256) rotating adjacent pairs;
/// - final layer: affine-free LayerNorm · (1 + Linear(SiLU(t))), Linear (3840 → 64), unpatchify, **negated**:
///   the sampler's denoised latent is x − σ·out (ComfyUI's CONST flow). The time input is t = 1 − σ, ×1000.
public final class ZImageDiT {
    private var w: [String: MLXArray]
    public let layers: Int
    public let refinerLayers: Int
    public let dtype: DType

    private let dim: Int
    private let heads: Int
    private let headDim: Int
    private let inChannels = 16
    private let patch = 2
    // from ComfyUI's model_detection for this model (the dim == 3840 branch)
    private let axes = [32, 48, 48]
    private let theta: Float = 256
    private let timeScale: Float = 1000
    private let padMultiple = 32
    private let normEps: Float = 1e-5
    /// q_norm / k_norm are RMSNorm(eps=None): torch.finfo(float32).eps
    private let qkEps: Float = Float.ulpOfOne
    private var ropeCache: [String: (cos: MLXArray, sin: MLXArray)] = [:]
    /// Quantized block linears (see `quantize`): name -> (packed weight, scales, biases)
    private var qw: [String: (wq: MLXArray, scales: MLXArray, biases: MLXArray?)] = [:]
    public private(set) var quantBits: Int? = nil
    private var quantGroup = 64
    /// LoRAs, applied at run time as one low-rank term per layer: y += (x · downᵀ) · upᵀ, the scales folded into `up`
    /// (works on bf16 and quantized weights; nothing to undo when they change)
    private var loraTerms: [String: (down: MLXArray, up: MLXArray)] = [:]
    private var appliedLoRAs: [String] = []

    public struct Unsupported: LocalizedError {
        public let errorDescription: String?
    }

    /// Loads the DiT from an AIO checkpoint (keys under `model.diffusion_model.`) or a bare DiT file. `quantizeBits`:
    /// quantize each block linear as it's read, so the bf16 weights are never all in memory at once.
    public init(url: URL, dtype: DType = .bfloat16, quantizeBits: Int? = nil) throws {
        self.dtype = dtype
        let prefix = "model.diffusion_model."
        var weights: [String: MLXArray] = [:]
        var packed: [String: (wq: MLXArray, scales: MLXArray, biases: MLXArray?)] = [:]
        do {
            // MLX loads lazily: each tensor is read when evaluated. With quantization, a block linear is packed as
            // soon as it's read and its bf16 copy dropped from `all`, so the bf16 weights are never all in memory
            var all = try loadArrays(url: url)
            let hasPrefix = all.keys.contains { $0.hasPrefix(prefix) }
            for k in Array(all.keys) where !hasPrefix || k.hasPrefix(prefix) {
                guard let v = all.removeValue(forKey: k) else { continue }
                let key = hasPrefix ? String(k.dropFirst(prefix.count)) : k
                if let bits = quantizeBits, Self.quantizable(key), v.ndim == 2, v.dim(1) % 64 == 0 {
                    let q = quantized(v.asType(dtype), groupSize: 64, bits: bits)
                    eval([q.wq, q.scales] + (q.biases.map { [$0] } ?? []))
                    packed[key] = (q.wq, q.scales, q.biases)
                } else {
                    weights[key] = v.asType(dtype)
                }
            }
        }
        func rows(_ key: String) -> Int? { weights[key]?.dim(0) ?? packed[key]?.wq.dim(0) }
        guard let qkvRows = rows("layers.0.attention.qkv.weight"), let qn = weights["layers.0.attention.q_norm.weight"],
              let xe = weights["x_embedder.weight"], weights["cap_embedder.1.weight"] != nil,
              weights["cap_pad_token"] != nil, weights["x_pad_token"] != nil
        else { throw Unsupported(errorDescription: "\(url.lastPathComponent) isn't a Z-Image NextDiT") }
        dim = xe.dim(0)
        headDim = qn.dim(0)
        heads = dim / headDim
        guard qkvRows == 3 * dim, axes.reduce(0, +) == headDim, xe.dim(1) == inChannels * patch * patch else {
            throw Unsupported(errorDescription: "\(url.lastPathComponent): unexpected NextDiT shapes")
        }
        var n = 0
        while rows("layers.\(n).attention.qkv.weight") != nil { n += 1 }
        var r = 0
        while rows("noise_refiner.\(r).attention.qkv.weight") != nil { r += 1 }
        guard n > 0, r > 0, rows("context_refiner.\(r - 1).attention.qkv.weight") != nil else {
            throw Unsupported(errorDescription: "\(url.lastPathComponent): missing NextDiT blocks")
        }
        layers = n
        refinerLayers = r
        w = weights
        qw = packed
        if !packed.isEmpty { quantBits = quantizeBits }
        eval(Array(w.values))
        Memory.clearCache()
    }

    /// The block linears `quantize` packs (attention, feed-forward, adaLN: ~99 % of the weights).
    static func quantizable(_ key: String) -> Bool {
        [".attention.qkv.weight", ".attention.out.weight", ".feed_forward.w1.weight", ".feed_forward.w2.weight",
         ".feed_forward.w3.weight", ".adaLN_modulation.0.weight"].contains { key.hasSuffix($0) }
    }

    // MARK: building blocks

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let y: MLXArray
        if let q = qw[name + ".weight"] {
            y = quantizedMM(x, q.wq, scales: q.scales, biases: q.biases, transpose: true, groupSize: quantGroup, bits: quantBits)
        } else {
            y = matmul(x, w[name + ".weight"]!.T)
        }
        var out = y
        if let t = loraTerms[name] { out = out + matmul(matmul(x, t.down.T), t.up.T) }
        if let b = w[name + ".bias"] { return out + b }
        return out
    }

    /// Sets the LoRAs (replacing the previous ones). LoRA / LoCon in ai-toolkit (diffusers names, `lora_A`/`lora_B`),
    /// kohya (`lora_down`/`lora_up` + alpha, underscore names) and ComfyUI (fused `qkv`) layouts; diffusers' separate
    /// `to_q`/`to_k`/`to_v` go into rows of the fused qkv. LoHa / LoKr aren't supported here yet (counted as unsupported).
    public func setLoRAs(_ loras: [LoRASpec]) throws -> LoRAReport {
        let signature = loras.map { "\($0.url.path)@\($0.strength)" }
        var report = LoRAReport()
        if signature == appliedLoRAs { return report }
        loraTerms.removeAll()
        appliedLoRAs = []
        compiled.removeAll()
        guard !loras.isEmpty else { return report }

        // our linear modules, plus the diffusers names that map onto them (and kohya's underscore forms)
        var targets: [String: (module: String, rows: Range<Int>?)] = [:]
        var blocks: [String] = (0 ..< layers).map { "layers.\($0)" }
        for i in 0 ..< refinerLayers { blocks += ["noise_refiner.\(i)", "context_refiner.\(i)"] }
        for b in blocks {
            for m in ["attention.qkv", "attention.out", "feed_forward.w1", "feed_forward.w2", "feed_forward.w3",
                      "adaLN_modulation.0"] { targets["\(b).\(m)"] = ("\(b).\(m)", nil) }
            targets["\(b).attention.to_q"] = ("\(b).attention.qkv", 0 ..< dim)
            targets["\(b).attention.to_k"] = ("\(b).attention.qkv", dim ..< 2 * dim)
            targets["\(b).attention.to_v"] = ("\(b).attention.qkv", 2 * dim ..< 3 * dim)
            targets["\(b).attention.to_out.0"] = ("\(b).attention.out", nil)
        }
        var underscored: [String: String] = [:]
        for k in targets.keys { underscored[k.replacingOccurrences(of: ".", with: "_")] = k }

        let roles: [(String, String)] = [(".lora_down.weight", "down"), (".lora_A.default.weight", "down"),
                                         (".lora_A.weight", "down"), (".lora_up.weight", "up"),
                                         (".lora_B.default.weight", "up"), (".lora_B.weight", "up"), (".alpha", "alpha")]
        let other = [".hada_w1_a", ".lokr_w1", ".lokr_w2", ".lokr_w1_a", ".lokr_w2_a", ".dora_scale"]
        var downs: [String: [MLXArray]] = [:], ups: [String: [MLXArray]] = [:]
        for lora in loras {
            report.files += 1
            var parts: [String: [String: MLXArray]] = [:]
            var unsupported = Set<String>()
            for (key, value) in try loadArrays(url: lora.url) {
                var k = key
                for prefix in ["model.diffusion_model.", "diffusion_model.", "transformer.", "base_model.model.",
                               "lora_unet_"] where k.hasPrefix(prefix) {
                    k = String(k.dropFirst(prefix.count))
                    break
                }
                if let o = other.first(where: { k.contains($0) }) {
                    unsupported.insert(String(k[..<k.range(of: o)!.lowerBound]))
                    continue
                }
                guard let (suffix, role) = roles.first(where: { k.hasSuffix($0.0) }) else { continue }
                var module = String(k.dropLast(suffix.count))
                if targets[module] == nil, let m = underscored[module] { module = m }
                parts[module, default: [:]][role] = value
            }
            report.unsupported += unsupported.count
            for (module, p) in parts {
                guard let t = targets[module], let down = p["down"], let up = p["up"], down.ndim == 2, up.ndim == 2,
                      down.dim(0) == up.dim(1) else {
                    report.unmatched.append(module)
                    continue
                }
                let rank = Float(down.dim(0))
                let alpha = p["alpha"].map { $0.asType(.float32).item(Float.self) } ?? rank
                let outDim = t.module.hasSuffix("qkv") ? 3 * dim : up.dim(0)
                guard down.dim(1) == inputDim(t.module), up.dim(0) == (t.rows?.count ?? outDim) else {
                    report.unmatched.append(module)
                    continue
                }
                var u = up.asType(.float32) * (lora.strength * alpha / rank)
                if let rows = t.rows {   // q / k / v: place the rows in the fused qkv's output
                    u = concatenated([MLXArray.zeros([rows.lowerBound, u.dim(1)]), u,
                                      MLXArray.zeros([outDim - rows.upperBound, u.dim(1)])], axis: 0)
                }
                downs[t.module, default: []].append(down.asType(dtype))
                ups[t.module, default: []].append(u.asType(dtype))
                report.applied += 1
            }
        }
        // one term per layer: the ranks of every LoRA (and of q, k, v) side by side
        for (module, d) in downs {
            let term = (down: concatenated(d, axis: 0), up: concatenated(ups[module]!, axis: 1))
            eval(term.down, term.up)
            loraTerms[module] = term
        }
        appliedLoRAs = signature
        return report
    }

    /// Input width of one of our linear modules.
    private func inputDim(_ module: String) -> Int {
        if let q = qw[module + ".weight"] { return q.wq.dim(1) * 32 / (quantBits ?? 32) }
        return w[module + ".weight"]?.dim(1) ?? -1
    }

    /// Quantizes the block linears (attention, feed-forward, adaLN: ~99 % of the weights) to `bits` (MLX affine,
    /// groups of `groupSize`); embedders, norms and the final layer stay bf16. Once per model.
    public func quantize(bits: Int, groupSize: Int = 64) {
        guard quantBits == nil else { return }
        let names = w.keys.filter(Self.quantizable)
        for k in names {
            guard let v = w[k], v.ndim == 2, v.dim(1) % groupSize == 0 else { continue }
            let r = quantized(v, groupSize: groupSize, bits: bits)
            eval([r.wq, r.scales] + (r.biases.map { [$0] } ?? []))
            qw[k] = (r.wq, r.scales, r.biases)
            w[k] = nil
        }
        quantBits = bits
        quantGroup = groupSize
        compiled.removeAll()
        Memory.clearCache()
    }

    private func rms(_ x: MLXArray, _ name: String, eps: Float? = nil) -> MLXArray {
        MLXFast.rmsNorm(x, weight: w[name + ".weight"]!, eps: eps ?? normEps)
    }

    private func feedForward(_ x: MLXArray, _ p: String) -> MLXArray {
        linear(silu(linear(x, p + ".feed_forward.w1")) * linear(x, p + ".feed_forward.w3"), p + ".feed_forward.w2")
    }

    /// x [n, l, heads, headDim] rotated by the (cos, sin) tables [l, headDim/2]: adjacent pairs (2i, 2i+1).
    private func rope(_ x: MLXArray, _ table: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let (n, l, h, d) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let pairs = x.asType(.float32).reshaped(n, l, h, d / 2, 2)
        let x0 = pairs[.ellipsis, 0], x1 = pairs[.ellipsis, 1]
        let c = table.cos.reshaped(1, l, 1, d / 2), s = table.sin.reshaped(1, l, 1, d / 2)
        return stacked([x0 * c - x1 * s, x0 * s + x1 * c], axis: -1).reshaped(n, l, h, d).asType(x.dtype)
    }

    private func attention(_ x: MLXArray, _ p: String, rope table: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let (n, l) = (x.dim(0), x.dim(1))
        let qkv = linear(x, p + ".attention.qkv")
        var q = qkv[.ellipsis, 0 ..< dim].reshaped(n, l, heads, headDim)
        var k = qkv[.ellipsis, dim ..< 2 * dim].reshaped(n, l, heads, headDim)
        let v = qkv[.ellipsis, (2 * dim)...].reshaped(n, l, heads, headDim)
        q = rope(rms(q, p + ".attention.q_norm", eps: qkEps), table)
        k = rope(rms(k, p + ".attention.k_norm", eps: qkEps), table)
        let o = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: v.transposed(0, 2, 1, 3),
            scale: 1 / Float(headDim).squareRoot(), mask: nil)
        return linear(o.transposed(0, 2, 1, 3).reshaped(n, l, dim), p + ".attention.out")
    }

    /// JointTransformerBlock; `t` (the 256-dim time embedding) switches the z-image modulation on.
    private func block(_ x: MLXArray, _ p: String, rope table: (cos: MLXArray, sin: MLXArray), t: MLXArray?) -> MLXArray {
        guard let t else {
            let a = x + rms(attention(rms(x, p + ".attention_norm1"), p, rope: table), p + ".attention_norm2")
            return a + rms(feedForward(rms(a, p + ".ffn_norm1"), p), p + ".ffn_norm2")
        }
        // scale_msa, gate_msa, scale_mlp, gate_mlp, each [n, 1, dim]
        let m = split(linear(t, p + ".adaLN_modulation.0"), parts: 4, axis: 1).map { $0.expandedDimensions(axis: 1) }
        let attn = attention(rms(x, p + ".attention_norm1") * (1 + m[0]), p, rope: table)
        let a = x + tanh(m[1]) * rms(attn, p + ".attention_norm2")
        let ff = feedForward(rms(a, p + ".ffn_norm1") * (1 + m[2]), p)
        return a + tanh(m[3]) * rms(ff, p + ".ffn_norm2")
    }

    // MARK: positions

    /// Flux EmbedND: per axis of `d` dims, angles = position · theta^(−2i/d), i < d/2; the axes are concatenated.
    /// `ids` [l, 3] float32 -> (cos, sin) [l, headDim/2] float32.
    private func ropeTable(_ ids: MLXArray) -> (cos: MLXArray, sin: MLXArray) {
        var angles: [MLXArray] = []
        for (a, d) in axes.enumerated() {
            let omega = MLXArray((0 ..< d / 2).map { 1 / pow(theta, Float(2 * $0) / Float(d)) })
            angles.append(ids[0..., a ..< a + 1] * omega.reshaped(1, d / 2))
        }
        let all = concatenated(angles, axis: 1)
        return (cos(all), sin(all))
    }

    private func padded(_ n: Int) -> Int { (n + padMultiple - 1) / padMultiple * padMultiple }

    /// Caption ids (i + 1, 0, 0) for the padded caption, then the image ids (capLen + 1, row, col) and (0, 0, 0)
    /// for the image pad tokens.
    private func ropeTables(capLen: Int, h: Int, w wd: Int) -> (cap: (cos: MLXArray, sin: MLXArray), img: (cos: MLXArray, sin: MLXArray)) {
        let key = "\(capLen)/\(h)x\(wd)"
        let imgKey = "img " + key
        if let cap = ropeCache["cap \(capLen)"], let img = ropeCache[imgKey] { return (cap, img) }
        var capIDs = [Float](repeating: 0, count: capLen * 3)
        for i in 0 ..< capLen { capIDs[i * 3] = Float(i + 1) }
        let tokens = h * wd, padTo = padded(tokens)
        var imgIDs = [Float](repeating: 0, count: padTo * 3)
        for r in 0 ..< h {
            for c in 0 ..< wd {
                let i = (r * wd + c) * 3
                imgIDs[i] = Float(capLen + 1); imgIDs[i + 1] = Float(r); imgIDs[i + 2] = Float(c)
            }
        }
        let cap = ropeTable(MLXArray(capIDs, [capLen, 3]))
        let img = ropeTable(MLXArray(imgIDs, [padTo, 3]))
        eval(cap.cos, cap.sin, img.cos, img.sin)
        ropeCache["cap \(capLen)"] = cap
        ropeCache[imgKey] = img
        return (cap, img)
    }

    // MARK: forward

    /// Run the forward pass through `MLX.compile` (WAIGEN_COMPILE=1 or set it): traced once per shape.
    public var compileForward = ProcessInfo.processInfo.environment["WAIGEN_COMPILE"] == "1"
    private var compiled: [Bool: @Sendable ([MLXArray]) -> [MLXArray]] = [:]

    /// x [1, 16, H, W] (latent, sampler space), sigma (1 = pure noise), context [l, 2560] or [1, l, 2560] (the
    /// Qwen3-4B conditioning) -> [1, 16, H, W] float32, negated: denoised = x − σ·out.
    public func callAsFunction(_ xIn: MLXArray, sigma: Float, context: MLXArray) -> MLXArray {
        precondition(xIn.dim(0) == 1, "ZImageDiT runs one image at a time")
        precondition(xIn.dim(1) == inChannels, "expected a \(inChannels)-channel latent, got \(xIn.dim(1))")
        var ctx = context
        if ctx.ndim == 2 { ctx = ctx.expandedDimensions(axis: 0) }
        // RoPE tables are built (and cached) outside the traced function
        let tables = ropeTables(capLen: padded(ctx.dim(1)), h: (xIn.dim(2) + patch - 1) / patch,
                                w: (xIn.dim(3) + patch - 1) / patch)
        let inputs = [xIn.asType(dtype), MLXArray([sigma]), ctx.asType(dtype),
                      tables.cap.cos, tables.cap.sin, tables.img.cos, tables.img.sin]
        guard compileForward else { return forward(inputs) }
        let f = compiled[true] ?? {
            let f = compile { [unowned self] a in [self.forward(a)] }
            compiled[true] = f
            return f
        }()
        return f(inputs)[0]
    }

    /// [x, sigma [1], context [1, l, 2560], cap cos, cap sin, image cos, image sin] -> output (see callAsFunction)
    private func forward(_ a: [MLXArray]) -> MLXArray {
        let xIn = a[0]
        let (c, hh, ww) = (xIn.dim(1), xIn.dim(2), xIn.dim(3))

        // time: t = 1 − σ, ×1000, 256-d sinusoid (cos, sin), MLP 256 → 1024 → 256
        let t = (1 - a[1].asType(.float32)) * timeScale
        let half = 128
        let freqs = MLXArray((0 ..< half).map { expf(-logf(10000) * Float($0) / Float(half)) })
        let args = t.reshaped(1, 1) * freqs.reshaped(1, half)
        let tFreq = concatenated([cos(args), sin(args)], axis: 1).asType(dtype)
        let temb = linear(silu(linear(tFreq, "t_embedder.mlp.0")), "t_embedder.mlp.2")   // [1, 256]

        // caption: RMSNorm + Linear, pad with cap_pad_token
        var cap = linear(rms(a[2], "cap_embedder.0"), "cap_embedder.1")
        let capTokens = cap.dim(1), capLen = padded(capTokens)
        if capLen > capTokens {
            cap = concatenated([cap, repeated(w["cap_pad_token"]!.reshaped(1, 1, dim), count: capLen - capTokens, axis: 1)],
                               axis: 1)
        }

        // image: pad to the patch size (circular, at the end, as comfy's pad_to_patch_size), 2×2 patches ordered
        // (row, column, channel), Linear, pad with x_pad_token
        let x = padToPatch(xIn.asType(dtype))
        let (h, wd) = (x.dim(2) / patch, x.dim(3) / patch)
        var img = x.reshaped(1, c, h, patch, wd, patch)
            .transposed(0, 2, 4, 3, 5, 1)
            .reshaped(1, h * wd, patch * patch * c)
        img = linear(img, "x_embedder")
        let imgTokens = h * wd, imgLen = padded(imgTokens)
        if imgLen > imgTokens {
            img = concatenated([img, repeated(w["x_pad_token"]!.reshaped(1, 1, dim), count: imgLen - imgTokens, axis: 1)],
                               axis: 1)
        }

        let capTable = (cos: a[3], sin: a[4]), imgTable = (cos: a[5], sin: a[6])
        for i in 0 ..< refinerLayers { cap = block(cap, "context_refiner.\(i)", rope: capTable, t: nil) }
        for i in 0 ..< refinerLayers { img = block(img, "noise_refiner.\(i)", rope: imgTable, t: temb) }

        var all = concatenated([cap, img], axis: 1)
        let full = (cos: concatenated([capTable.cos, imgTable.cos], axis: 0),
                    sin: concatenated([capTable.sin, imgTable.sin], axis: 0))
        for i in 0 ..< layers { all = block(all, "layers.\(i)", rope: full, t: temb) }

        // final layer: LayerNorm (no affine, eps 1e-6) · (1 + Linear(SiLU(t))), Linear 3840 → 64
        let scale = linear(silu(temb), "final_layer.adaLN_modulation.1").expandedDimensions(axis: 1)
        all = linear(MLXFast.layerNorm(all, weight: nil, bias: nil, eps: 1e-6) * (1 + scale), "final_layer.linear")

        // unpatchify the image tokens: (row, column, ph, pw, channel) -> [1, C, H, W]; crop, negate
        let out = all[0..., capLen ..< capLen + imgTokens]
            .reshaped(1, h, wd, patch, patch, c)
            .transposed(0, 5, 1, 3, 2, 4)
            .reshaped(1, c, h * patch, wd * patch)
        return -(out[0..., 0..., 0 ..< hh, 0 ..< ww].asType(.float32))
    }

    /// torch's F.pad(..., mode="circular") on the bottom/right edge, up to a multiple of the patch size.
    private func padToPatch(_ x: MLXArray) -> MLXArray {
        let (hh, ww) = (x.dim(2), x.dim(3))
        let ph = (patch - hh % patch) % patch, pw = (patch - ww % patch) % patch
        var x = x
        if ph > 0 { x = concatenated([x, x[0..., 0..., 0 ..< ph, 0...]], axis: 2) }
        if pw > 0 { x = concatenated([x, x[0..., 0..., 0..., 0 ..< pw]], axis: 3) }
        return x
    }
}
