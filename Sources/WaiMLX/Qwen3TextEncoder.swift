import Foundation
import MLX
import MLXNN

/// Qwen3 as a text encoder (WAI-ANIMA's Qwen3-0.6B), in MLX: token ids in, the final normed hidden states
/// out, like stable-diffusion.cpp's `LLMRunner` for `QWEN3` (causal attention, grouped KV heads, RMSNorm on
/// q and k, NEOX rotary embeddings with base 1e6). Tokenization stays in stable-diffusion.cpp, which
/// calls this through its text-encoder hook. bf16 (default): 0.01 s per prompt, cosine 0.99986 to
/// stable-diffusion.cpp's output; float32: 0.02 s, 0.9999997. The images look the same either way
/// (docs/PERFORMANCE.md, round 5); bf16 weights with float32 math was 0.04 s (the weights get converted every call).
public final class Qwen3TextEncoder {
    private let w: [String: MLXArray]
    private let dtype: DType
    public let layers: Int
    public let hidden: Int
    public let heads: Int
    public let kvHeads: Int
    public let headDim: Int
    public var ropeBase: Float = 1_000_000
    public var eps: Float = 1e-6
    /// Index (0-based) of the last layer to run, like ComfyUI's `layer_idx`: that layer's output is returned
    /// and the final norm is skipped (Qwen3-4B conditioning uses layer −2, the second-to-last, un-normed).
    /// nil (the default) runs every layer and applies the final norm, as WAI-ANIMA does.
    public var outputLayerIndex: Int? = nil

    public struct Unsupported: LocalizedError {
        public let errorDescription: String?
    }

    public init(url: URL, dtype: DType = .bfloat16, keyPrefix: String? = nil) throws {
        var weights: [String: MLXArray] = [:]
        for (k, v) in try loadArrays(url: url) {
            // AIO files (z-image-anime) keep all three models in one file: take only the keys of this model
            if let keyPrefix, !k.hasPrefix(keyPrefix) { continue }
            var key = keyPrefix.map { String(k.dropFirst($0.count)) } ?? k
            for prefix in ["text_encoders.llm.", "model."] where key.hasPrefix(prefix) { key = String(key.dropFirst(prefix.count)) }
            weights[key] = v.asType(dtype)
        }
        guard let embed = weights["embed_tokens.weight"], let q = weights["layers.0.self_attn.q_proj.weight"],
              let k = weights["layers.0.self_attn.k_proj.weight"], let qn = weights["layers.0.self_attn.q_norm.weight"]
        else { throw Unsupported(errorDescription: "\(url.lastPathComponent) isn't a Qwen3 model") }
        var n = 0
        while weights["layers.\(n).self_attn.q_proj.weight"] != nil { n += 1 }
        layers = n
        hidden = embed.dim(1)
        headDim = qn.dim(0)
        heads = q.dim(0) / headDim
        kvHeads = k.dim(0) / headDim
        w = weights
        self.dtype = dtype
        eval(Array(w.values))
    }

    private func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        matmul(x, w[name + ".weight"]!.T)
    }

    /// token ids -> [n, hidden] float32
    public func callAsFunction(_ ids: [Int32]) -> MLXArray {
        let n = ids.count
        var x = w["embed_tokens.weight"]![MLXArray(ids)][.newAxis]   // [1, n, hidden]
        let scale = 1 / Float(headDim).squareRoot()
        let runCount = min((outputLayerIndex ?? layers - 1) + 1, layers)   // layers to run (incl. outputLayerIndex)
        for i in 0 ..< runCount {
            let p = "layers.\(i)"
            let h = MLXFast.rmsNorm(x, weight: w[p + ".input_layernorm.weight"]!, eps: eps)
            var q = linear(h, p + ".self_attn.q_proj").reshaped(1, n, heads, headDim)
            var k = linear(h, p + ".self_attn.k_proj").reshaped(1, n, kvHeads, headDim)
            let v = linear(h, p + ".self_attn.v_proj").reshaped(1, n, kvHeads, headDim).transposed(0, 2, 1, 3)
            q = MLXFast.rmsNorm(q, weight: w[p + ".self_attn.q_norm.weight"]!, eps: eps).transposed(0, 2, 1, 3)
            k = MLXFast.rmsNorm(k, weight: w[p + ".self_attn.k_norm.weight"]!, eps: eps).transposed(0, 2, 1, 3)
            q = MLXFast.RoPE(q, dimensions: headDim, traditional: false, base: ropeBase, scale: 1, offset: 0)
            k = MLXFast.RoPE(k, dimensions: headDim, traditional: false, base: ropeBase, scale: 1, offset: 0)
            let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .causal)
            x = x + linear(o.transposed(0, 2, 1, 3).reshaped(1, n, heads * headDim), p + ".self_attn.o_proj")
            let h2 = MLXFast.rmsNorm(x, weight: w[p + ".post_attention_layernorm.weight"]!, eps: eps)
            x = x + linear(silu(linear(h2, p + ".mlp.gate_proj")) * linear(h2, p + ".mlp.up_proj"), p + ".mlp.down_proj")
        }
        if outputLayerIndex != nil { return x[0].asType(.float32) }   // hidden states of `last`, no final norm
        return MLXFast.rmsNorm(x, weight: w["norm.weight"]!, eps: eps)[0].asType(.float32)
    }

    /// For stable-diffusion.cpp's text-encoder hook: writes n_tokens x hidden floats (row-major) to `output`.
    public func encode(ids: UnsafePointer<Int32>, count: Int, into output: UnsafeMutablePointer<Float>, hidden: Int) -> Bool {
        guard hidden == self.hidden, count > 0 else { return false }
        let states = self(Array(UnsafeBufferPointer(start: ids, count: count)))
        eval(states)
        states.asArray(Float.self).withUnsafeBufferPointer { buf in
            output.update(from: buf.baseAddress!, count: count * hidden)
        }
        return true
    }
}
