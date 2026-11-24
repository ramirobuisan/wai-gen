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

    public struct Unsupported: LocalizedError {
        public let errorDescription: String?
    }

    /// Loads the DiT from an AIO checkpoint (keys under `model.diffusion_model.`) or a bare DiT file.
    public init(url: URL, dtype: DType = .bfloat16) throws {
        self.dtype = dtype
        let prefix = "model.diffusion_model."
        var weights: [String: MLXArray] = [:]
        let all = try loadArrays(url: url)
        let hasPrefix = all.keys.contains { $0.hasPrefix(prefix) }
        for (k, v) in all where !hasPrefix || k.hasPrefix(prefix) {
            weights[hasPrefix ? String(k.dropFirst(prefix.count)) : k] = v.asType(dtype)
        }
        guard let qkv = weights["layers.0.attention.qkv.weight"], let qn = weights["layers.0.attention.q_norm.weight"],
              let xe = weights["x_embedder.weight"], weights["cap_embedder.1.weight"] != nil,
              weights["cap_pad_token"] != nil, weights["x_pad_token"] != nil
        else { throw Unsupported(errorDescription: "\(url.lastPathComponent) isn't a Z-Image NextDiT") }
        dim = xe.dim(0)
        headDim = qn.dim(0)
        heads = dim / headDim
        guard qkv.dim(0) == 3 * dim, axes.reduce(0, +) == headDim, xe.dim(1) == inChannels * patch * patch else {
            throw Unsupported(errorDescription: "\(url.lastPathComponent): unexpected NextDiT shapes")
        }
        var n = 0
        while weights["layers.\(n).attention.qkv.weight"] != nil { n += 1 }
        var r = 0
        while weights["noise_refiner.\(r).attention.qkv.weight"] != nil { r += 1 }
        guard n > 0, r > 0, weights["context_refiner.\(r - 1).attention.qkv.weight"] != nil else {
            throw Unsupported(errorDescription: "\(url.lastPathComponent): missing NextDiT blocks")
        }
        layers = n
        refinerLayers = r
        w = weights
        eval(Array(w.values))
    }

    // MARK: building blocks

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let y = matmul(x, w[name + ".weight"]!.T)
        if let b = w[name + ".bias"] { return y + b }
        return y
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

    /// x [1, 16, H, W] (latent, sampler space), sigma (1 = pure noise), context [l, 2560] or [1, l, 2560] (the
    /// Qwen3-4B conditioning) -> [1, 16, H, W] float32, negated: denoised = x − σ·out.
    public func callAsFunction(_ xIn: MLXArray, sigma: Float, context: MLXArray) -> MLXArray {
        let (n, c, hh, ww) = (xIn.dim(0), xIn.dim(1), xIn.dim(2), xIn.dim(3))
        precondition(n == 1, "ZImageDiT runs one image at a time")
        precondition(c == inChannels, "expected a \(inChannels)-channel latent, got \(c)")

        // time: t = 1 − σ, ×1000, 256-d sinusoid (cos, sin), MLP 256 → 1024 → 256
        let t = (1 - sigma) * timeScale
        let half = 128
        let freqs = (0 ..< half).map { expf(-logf(10000) * Float($0) / Float(half)) }
        let args = MLXArray(freqs.map { t * $0 }).reshaped(1, half)
        let tFreq = concatenated([cos(args), sin(args)], axis: 1).asType(dtype)
        let temb = linear(silu(linear(tFreq, "t_embedder.mlp.0")), "t_embedder.mlp.2")   // [1, 256]

        // caption: RMSNorm + Linear, pad with cap_pad_token
        var cap = context.asType(dtype)
        if cap.ndim == 2 { cap = cap.expandedDimensions(axis: 0) }
        cap = linear(rms(cap, "cap_embedder.0"), "cap_embedder.1")
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

        let tables = ropeTables(capLen: capLen, h: h, w: wd)
        for i in 0 ..< refinerLayers { cap = block(cap, "context_refiner.\(i)", rope: tables.cap, t: nil) }
        for i in 0 ..< refinerLayers { img = block(img, "noise_refiner.\(i)", rope: tables.img, t: temb) }

        var all = concatenated([cap, img], axis: 1)
        let full = (cos: concatenated([tables.cap.cos, tables.img.cos], axis: 0),
                    sin: concatenated([tables.cap.sin, tables.img.sin], axis: 0))
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
