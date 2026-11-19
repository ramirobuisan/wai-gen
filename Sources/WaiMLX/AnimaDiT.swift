import Foundation
import MLX
import MLXNN
import MLXRandom

/// The Anima diffusion model (Cosmos-Predict2-style DiT with an LLM adapter) used by WAI-ANIMA, in MLX.
/// A direct port of stable-diffusion.cpp's `src/model/diffusion/anima.hpp`, so it can stand in for it
/// through `sd_set_diffusion_hook`: same inputs (latent, timestep, Qwen3 hidden states, adapter token ids
/// and weights), same output (the flow prediction).
public final class AnimaDiT {
    private var w: [String: MLXArray]
    /// weights as loaded, for the ones a LoRA changed (to undo it)
    private var original: [String: MLXArray] = [:]
    private var appliedLoRAs: [String] = []
    public let layers: Int
    public let dtype: DType
    private let hidden = 2048, heads = 16, headDim = 128, patch = 2, outChannels = 16
    private let axes = [44, 42, 42]
    private var ropeCache: [String: (MLXArray, MLXArray)] = [:]

    public init(url: URL, dtype: DType = .bfloat16) throws {
        self.dtype = dtype
        var w: [String: MLXArray] = [:]
        for (k, v) in try loadArrays(url: url) {
            var key = k
            for prefix in ["model.diffusion_model.net.", "model.diffusion_model.", "net."] where key.hasPrefix(prefix) {
                key = String(key.dropFirst(prefix.count))
                break
            }
            w[key] = v.asType(dtype)
        }
        var n = 0
        while w["blocks.\(n).self_attn.q_proj.weight"] != nil { n += 1 }
        guard n > 0, w["x_embedder.proj.1.weight"] != nil, w["llm_adapter.embed.weight"] != nil else {
            throw NSError(domain: "AnimaDiT", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "\(url.lastPathComponent) isn't an Anima diffusion model"])
        }
        layers = n
        // (8-bit, mxfp8 and 4-bit quantized linears were measured: none is faster than bf16 here,
        // see docs/PERFORMANCE.md)
        self.w = w
        eval(Array(w.values))
    }

    // MARK: LoRA

    public struct LoRAReport: Sendable {
        public var applied = 0
        public var unmatched: [String] = []
    }

    /// Merge these LoRAs into the weights (W += strength · alpha/rank · up·down), undoing earlier ones
    /// first; same key formats as stable-diffusion.cpp (kohya `lora_down/lora_up/alpha`, PEFT
    /// `lora_A/lora_B`, `lora_unet_` underscores; `diffusion_model.`/`transformer.`/`net.` prefixes).
    /// `unmatched` lists the modules it couldn't map (e.g. LoHa/LoKr or text-encoder parts).
    public func setLoRAs(_ loras: [LoRASpec]) throws -> LoRAReport {
        let signature = loras.map { "\($0.url.path)@\($0.strength)" }
        var report = LoRAReport()
        if signature == appliedLoRAs { return report }
        for (k, v) in original { w[k] = v }
        original.removeAll()
        appliedLoRAs = []
        compiled.removeAll()   // weights changed: re-trace if compile is on
        guard !loras.isEmpty else { return report }

        // module names in this model, also in kohya's underscore form
        var underscored: [String: String] = [:]
        for key in w.keys where key.hasSuffix(".weight") {
            let module = String(key.dropLast(".weight".count))
            underscored[module.replacingOccurrences(of: ".", with: "_")] = module
        }
        for lora in loras {
            var parts: [String: (down: MLXArray?, up: MLXArray?, alpha: Float?)] = [:]
            for (key, value) in try loadArrays(url: lora.url) {
                var k = key
                for prefix in ["model.diffusion_model.net.", "model.diffusion_model.", "diffusion_model.net.", "diffusion_model.",
                               "transformer.", "base_model.model.", "net.", "lora_unet_"] where k.hasPrefix(prefix) {
                    k = String(k.dropFirst(prefix.count))
                    break
                }
                let suffixes: [(String, Int)] = [(".lora_down.weight", 0), (".lora_A.default.weight", 0), (".lora_A.weight", 0),
                                                  (".lora_up.weight", 1), (".lora_B.default.weight", 1), (".lora_B.weight", 1),
                                                  (".alpha", 2)]
                guard let (suffix, kind) = suffixes.first(where: { k.hasSuffix($0.0) }) else { continue }
                var module = String(k.dropLast(suffix.count))
                if w[module + ".weight"] == nil, let m = underscored[module] { module = m }
                var entry = parts[module] ?? (nil, nil, nil)
                switch kind {
                case 0: entry.down = value
                case 1: entry.up = value
                default: entry.alpha = value.asType(.float32).item(Float.self)
                }
                parts[module] = entry
            }
            for (module, p) in parts {
                let key = module + ".weight"
                guard let down = p.down, let up = p.up, let base = w[key], down.ndim == 2, up.ndim == 2,
                      up.dim(0) == base.dim(0), down.dim(1) == base.dim(1)
                else { report.unmatched.append(module); continue }
                let rank = Float(down.dim(0))
                let scale = lora.strength * (p.alpha ?? rank) / rank
                if original[key] == nil { original[key] = base }
                w[key] = (base.asType(.float32) + matmul(up.asType(.float32), down.asType(.float32)) * scale).asType(dtype)
                report.applied += 1
            }
        }
        eval(Array(original.keys).compactMap { w[$0] })
        appliedLoRAs = signature
        return report
    }

    // MARK: layers

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        var y = matmul(x, w[name + ".weight"]!.T)
        if let b = w[name + ".bias"] { y = y + b }
        return y
    }

    private func rms(_ x: MLXArray, _ name: String) -> MLXArray {
        MLXFast.rmsNorm(x, weight: w[name + ".weight"]!, eps: 1e-6)
    }

    private func layerNorm(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x, weight: nil, bias: nil, eps: 1e-6)
    }

    private func gelu(_ x: MLXArray) -> MLXArray { geluApproximate(x) }   // ggml_gelu: tanh approximation

    /// Rotate-half RoPE (stable-diffusion.cpp `apply_rope(..., interleaved = false)`):
    /// x [N, H, L, D]; cos/sin [L, D/2].
    private func rope(_ x: MLXArray, _ cs: (MLXArray, MLXArray)) -> MLXArray {
        let half = x.dim(-1) / 2
        let x0 = x[.ellipsis, 0 ..< half], x1 = x[.ellipsis, half...]
        let (c, s) = cs
        return concatenated([x0 * c - x1 * s, x0 * s + x1 * c], axis: -1)
    }

    /// AnimaAttention: q/k/v projections, per-head RMS norm on q and k, optional RoPE, attention, output.
    private func attention(_ x: MLXArray, context: MLXArray?, _ p: String, heads: Int, headDim: Int,
                           ropeQ: (MLXArray, MLXArray)?, ropeK: (MLXArray, MLXArray)?, out: String) -> MLXArray {
        let src = context ?? x
        let (n, lq, lk) = (x.dim(0), x.dim(1), src.dim(1))
        var q = linear(x, p + ".q_proj").reshaped(n, lq, heads, headDim)
        var k = linear(src, p + ".k_proj").reshaped(n, lk, heads, headDim)
        let v = linear(src, p + ".v_proj").reshaped(n, lk, heads, headDim).transposed(0, 2, 1, 3)
        q = rms(q, p + ".q_norm").transposed(0, 2, 1, 3)
        k = rms(k, p + ".k_norm").transposed(0, 2, 1, 3)
        if let ropeQ { q = rope(q, ropeQ) }
        if let ropeK { k = rope(k, ropeK) }
        let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v,
                                                  scale: 1 / Float(headDim).squareRoot(), mask: nil)
        return linear(o.transposed(0, 2, 1, 3).reshaped(n, lq, heads * headDim), p + "." + out)
    }

    /// AdaLayerNorm(Zero): LayerNorm(x) * (1 + scale) + shift, from the timestep (+ temb); `chunks` 3 also returns the gate.
    private func modulation(_ x: MLXArray, _ p: String, et: MLXArray, temb: MLXArray, chunks: Int)
        -> (MLXArray, MLXArray?)
    {
        var emb = linear(linear(silu(et), p + ".1"), p + ".2")
        emb = emb + temb[.ellipsis, 0 ..< chunks * hidden]
        let parts = split(emb, parts: chunks, axis: -1).map { $0.expandedDimensions(axis: 1) }
        return (layerNorm(x) * (1 + parts[1]) + parts[0], chunks == 3 ? parts[2] : nil)
    }

    // MARK: positions

    private static func freqs(_ dim: Int, theta: Float) -> [Float] {
        (0 ..< dim / 2).map { 1 / pow(theta, Float(2 * $0) / Float(dim)) }
    }

    /// cos/sin [h*w, 64] for the image tokens: ids (0, row, col), axes 44/42/42, NTK-scaled theta for rows/cols.
    private func imageRope(h: Int, w: Int) -> (MLXArray, MLXArray) {
        let key = "\(h)x\(w)"
        if let c = ropeCache[key] { return c }
        func ntk(_ ratio: Float, _ d: Int) -> Float { ratio == 1 || d <= 2 ? 1 : pow(ratio, Float(d) / Float(d - 2)) }
        let ft = Self.freqs(axes[0], theta: 10000 * ntk(1, axes[0]))
        let fh = Self.freqs(axes[1], theta: 10000 * ntk(4, axes[1]))
        let fw = Self.freqs(axes[2], theta: 10000 * ntk(4, axes[2]))
        var angles = [Float]()
        angles.reserveCapacity(h * w * 64)
        for i in 0 ..< h {
            for j in 0 ..< w {
                angles += ft.map { _ in 0 }
                angles += fh.map { Float(i) * $0 }
                angles += fw.map { Float(j) * $0 }
            }
        }
        let a = MLXArray(angles, [h * w, ft.count + fh.count + fw.count])
        let c = (cos(a).asType(dtype), sin(a).asType(dtype))
        ropeCache[key] = c
        return c
    }

    private func lineRope(_ len: Int, dim: Int) -> (MLXArray, MLXArray) {
        let key = "1d\(len)x\(dim)"
        if let c = ropeCache[key] { return c }
        let f = Self.freqs(dim, theta: 10000)
        let a = MLXArray((0 ..< len).flatMap { p in f.map { Float(p) * $0 } }, [len, f.count])
        let c = (cos(a).asType(dtype), sin(a).asType(dtype))
        ropeCache[key] = c
        return c
    }

    // MARK: forward

    /// LLMAdapter: turns the Qwen3 hidden states into the T5-like context the DiT was trained on.
    private func adapter(_ source: MLXArray, ids: MLXArray) -> MLXArray {
        var x = w["llm_adapter.embed.weight"]![ids].expandedDimensions(axis: 0)   // [1, n, 1024]
        if source.dim(0) > 1 { x = broadcast(x, to: [source.dim(0), x.dim(1), x.dim(2)]) }
        let qPE = lineRope(x.dim(1), dim: 64), kPE = lineRope(source.dim(1), dim: 64)
        var i = 0
        while w["llm_adapter.blocks.\(i).norm_self_attn.weight"] != nil {
            let p = "llm_adapter.blocks.\(i)"
            var h = rms(x, p + ".norm_self_attn")
            x = x + attention(h, context: nil, p + ".self_attn", heads: 16, headDim: 64, ropeQ: qPE, ropeK: qPE, out: "o_proj")
            h = rms(x, p + ".norm_cross_attn")
            x = x + attention(h, context: source, p + ".cross_attn", heads: 16, headDim: 64, ropeQ: qPE, ropeK: kPE,
                              out: "o_proj")
            h = rms(x, p + ".norm_mlp")
            x = x + linear(gelu(linear(h, p + ".mlp.0")), p + ".mlp.2")
            i += 1
        }
        return rms(linear(x, "llm_adapter.out_proj"), "llm_adapter.norm")
    }

    /// Run the forward pass through `MLX.compile`. Opt-in (WAIGEN_COMPILE=1): it saves ~3.5 % per forward
    /// pass once traced, but cond and uncond have different text lengths, so every image traces it twice
    /// (and again for hi-res) and a whole image ends up no faster (docs/PERFORMANCE.md).
    public var compileForward = ProcessInfo.processInfo.environment["WAIGEN_COMPILE"] == "1"
    private var compiled: [Bool: @Sendable ([MLXArray]) -> [MLXArray]] = [:]

    /// x [N, 16, H, W] (latent), timesteps [N], context [N, L, 1024] (Qwen3 states), adapter token ids and
    /// weights [n] -> prediction [N, 16, H, W]. Inputs in any float type; output float32.
    public func callAsFunction(_ xIn: MLXArray, timesteps: MLXArray, context ctxIn: MLXArray,
                               tokenIDs: MLXArray?, tokenWeights: MLXArray?) -> MLXArray {
        guard compileForward else { return forward(xIn, timesteps: timesteps, context: ctxIn, tokenIDs: tokenIDs, tokenWeights: tokenWeights) }
        let hasTokens = tokenIDs != nil
        // RoPE tables are built (and cached) outside the traced function
        _ = imageRope(h: (xIn.dim(2) + 1) / patch, w: (xIn.dim(3) + 1) / patch)
        if hasTokens, let ids = tokenIDs { _ = lineRope(ids.dim(0), dim: 64); _ = lineRope(ctxIn.dim(1), dim: 64) }
        let f = compiled[hasTokens] ?? {
            let f = compile { [unowned self] a in
                [self.forward(a[0], timesteps: a[1], context: a[2], tokenIDs: hasTokens ? a[3] : nil,
                              tokenWeights: hasTokens ? a[4] : nil)]
            }
            compiled[hasTokens] = f
            return f
        }()
        var inputs = [xIn.asType(dtype), timesteps.asType(.float32), ctxIn.asType(dtype)]
        if let tokenIDs { inputs += [tokenIDs.asType(.int32), (tokenWeights ?? MLXArray.ones([tokenIDs.dim(0)])).asType(dtype)] }
        return f(inputs)[0]
    }

    /// Seconds per forward pass (the model runs twice per sampling step: cond + uncond) at width x height
    /// pixels, compiled or not, plus the share of the LLM adapter.
    public func benchmark(width: Int, height: Int, batch: Int = 1, runs: Int = 4, compiled useCompile: Bool)
        -> (forward: Double, adapter: Double)
    {
        let x = MLXRandom.normal([batch, 16, height / 8, width / 8])
        let t = MLXArray(Array(repeating: Float(0.5), count: batch))
        let ctx = MLXRandom.normal([batch, 120, 1024])
        let ids = MLXArray((0 ..< 120).map { Int32($0 * 37 % 32000) }), wts = MLXArray.ones([120])
        let saved = compileForward
        compileForward = useCompile
        defer { compileForward = saved }
        eval(self(x, timesteps: t, context: ctx, tokenIDs: ids, tokenWeights: wts))
        var t0 = Date()
        for _ in 0 ..< runs { eval(self(x, timesteps: t, context: ctx, tokenIDs: ids, tokenWeights: wts)) }
        let fwd = Date().timeIntervalSince(t0) / Double(runs * batch)   // per sample
        eval(adapter(ctx.asType(dtype), ids: ids))
        t0 = Date()
        for _ in 0 ..< runs { eval(adapter(ctx.asType(dtype), ids: ids)) }
        return (fwd, Date().timeIntervalSince(t0) / Double(runs))
    }

    private func forward(_ xIn: MLXArray, timesteps: MLXArray, context ctxIn: MLXArray,
                         tokenIDs: MLXArray?, tokenWeights: MLXArray?) -> MLXArray {
        let (n, c, hh, ww) = (xIn.dim(0), xIn.dim(1), xIn.dim(2), xIn.dim(3))
        let (ph, pw) = ((patch - hh % patch) % patch, (patch - ww % patch) % patch)
        var x = xIn.asType(dtype)
        if ph > 0 || pw > 0 { x = padded(x, widths: [[0, 0], [0, 0], [0, ph], [0, pw]]) }
        let (h, wd) = ((hh + ph) / patch, (ww + pw) / patch)
        // padding-mask channel, then patchify: [N, C+1, h, 2, w, 2] -> [N, h*w, (C+1)*2*2]
        x = concatenated([x, MLXArray.zeros([n, 1, hh + ph, ww + pw], dtype: dtype)], axis: 1)
        x = x.reshaped(n, c + 1, h, patch, wd, patch).transposed(0, 2, 4, 1, 3, 5).reshaped(n, h * wd, (c + 1) * patch * patch)
        x = linear(x, "x_embedder.proj.1")

        // timestep: [cos | sin] sinusoids, 2048 dims
        let half = hidden / 2
        let freqs = exp(-log(Float(10000)) * MLXArray((0 ..< half).map { Float($0) }) / Float(half))
        let args = timesteps.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        let tProj = concatenated([cos(args), sin(args)], axis: -1).asType(dtype)
        let temb = linear(silu(linear(tProj, "t_embedder.1.linear_1")), "t_embedder.1.linear_2")   // [N, 6144]
        let et = rms(tProj, "t_embedding_norm")

        var context = ctxIn.asType(dtype)
        if let tokenIDs {
            var adapted = adapter(context, ids: tokenIDs.asType(.int32))
            if let tokenWeights { adapted = adapted * tokenWeights.asType(dtype).reshaped(1, -1, 1) }
            let len = adapted.dim(1)
            if len < 512 {
                adapted = concatenated([adapted, MLXArray.zeros([adapted.dim(0), 512 - len, adapted.dim(2)], dtype: dtype)], axis: 1)
            } else if len > 512 {
                adapted = adapted[0..., 0 ..< 512]
            }
            context = adapted
        }

        let pe = imageRope(h: h, w: wd)
        for i in 0 ..< layers {
            let p = "blocks.\(i)"
            let (n1, g1) = modulation(x, p + ".adaln_modulation_self_attn", et: et, temb: temb, chunks: 3)
            x = x + attention(n1, context: nil, p + ".self_attn", heads: heads, headDim: headDim, ropeQ: pe, ropeK: pe,
                              out: "output_proj") * g1!
            let (n2, g2) = modulation(x, p + ".adaln_modulation_cross_attn", et: et, temb: temb, chunks: 3)
            x = x + attention(n2, context: context, p + ".cross_attn", heads: heads, headDim: headDim, ropeQ: nil, ropeK: nil,
                              out: "output_proj") * g2!
            let (n3, g3) = modulation(x, p + ".adaln_modulation_mlp", et: et, temb: temb, chunks: 3)
            x = x + linear(gelu(linear(n3, p + ".mlp.layer1")), p + ".mlp.layer2") * g3!
        }
        let (nf, _) = modulation(x, "final_layer.adaln_modulation", et: et, temb: temb, chunks: 2)
        x = linear(nf, "final_layer.linear")   // [N, h*w, 2*2*16], features ordered (ph, pw, C)
        x = x.reshaped(n, h, wd, patch, patch, outChannels).transposed(0, 5, 1, 3, 2, 4)
            .reshaped(n, outChannels, h * patch, wd * patch)
        return x[0..., 0..., 0 ..< hh, 0 ..< ww].asType(.float32)
    }

    // MARK: stable-diffusion.cpp hook layout

    /// One forward pass on raw stable-diffusion.cpp buffers (shapes in C order: x [N, C, H, W], context
    /// [N, L, D]); writes the prediction (same shape as x) to `output`.
    public func run(x: UnsafePointer<Float>, shape s: [Int], timesteps: UnsafePointer<Float>, nTimesteps: Int,
                    context: UnsafePointer<Float>, contextShape cs: [Int], tokenIDs: UnsafePointer<Int32>?,
                    tokenWeights: UnsafePointer<Float>?, nTokens: Int, into output: UnsafeMutablePointer<Float>) -> Bool {
        guard s.count == 4, s[1] == outChannels, cs.count == 3, nTimesteps == s[0] else { return false }
        let xa = MLXArray(UnsafeBufferPointer(start: x, count: s.reduce(1, *)), s)
        let ta = MLXArray(UnsafeBufferPointer(start: timesteps, count: nTimesteps), [nTimesteps])
        let ca = MLXArray(UnsafeBufferPointer(start: context, count: cs.reduce(1, *)), cs)
        let ids = tokenIDs.map { MLXArray(UnsafeBufferPointer(start: $0, count: nTokens), [nTokens]) }
        let wts = tokenWeights.map { MLXArray(UnsafeBufferPointer(start: $0, count: nTokens), [nTokens]) }
        let out = self(xa, timesteps: ta, context: ca, tokenIDs: nTokens > 0 ? ids : nil, tokenWeights: nTokens > 0 ? wts : nil)
        let values = out.asArray(Float.self)
        values.withUnsafeBufferPointer { output.update(from: $0.baseAddress!, count: $0.count) }
        return true
    }
}
