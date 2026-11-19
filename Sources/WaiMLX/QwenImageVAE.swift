import Foundation
import MLX
import MLXNN

/// The Qwen-Image VAE (the Wan 2.1 video VAE) that WAI-ANIMA uses, for single images, in MLX.
///
/// The checkpoint is a causal 3D VAE. For one frame with empty causal caches every 3x3x3 convolution
/// sees two zero frames in front of the image, so only the last temporal slice of its kernel matters,
/// and the temporal convolutions of the down/upsample blocks are skipped on the first frame. What is
/// left is a 2D VAE, which is what this runs (NHWC). Same math as stable-diffusion.cpp's WanVAE and
/// the reference `wan/modules/vae.py`; the latent is the raw mean (no mean/std normalisation, the
/// caller's model applies that).
public final class QwenImageVAE {
    private var w: [String: MLXArray] = [:]
    private let encoderBlocks: Int
    private let decoderBlocks: Int
    public let latentChannels: Int
    public let dtype: DType

    public init(url: URL, dtype: DType = .bfloat16) throws {
        self.dtype = dtype
        var w: [String: MLXArray] = [:]
        for (key, value) in try loadArrays(url: url) {
            if key.contains(".time_conv.") { continue }   // temporal-only, unused for one frame
            var a = value
            if key.hasSuffix(".weight"), a.ndim == 5 {
                // [O, I, T, KH, KW]: keep the last temporal slice -> MLX layout [O, KH, KW, I]
                a = a[0..., 0..., a.dim(2) - 1].transposed(0, 2, 3, 1)
            } else if key.hasSuffix(".weight"), a.ndim == 4 {
                a = a.transposed(0, 2, 3, 1)   // 2D conv [O, I, KH, KW] -> [O, KH, KW, I]
            } else if key.hasSuffix(".gamma") {
                a = a.reshaped(-1)
            }
            w[key] = a.asType(dtype)
        }
        func count(_ prefix: String) -> Int {
            var n = 0
            while w.keys.contains(where: { $0.hasPrefix("\(prefix).\(n).") }) { n += 1 }
            return n
        }
        encoderBlocks = count("encoder.downsamples")
        decoderBlocks = count("decoder.upsamples")
        guard encoderBlocks > 0, decoderBlocks > 0, let c = w["conv2.weight"], w["decoder.conv1.weight"] != nil,
              w["encoder.conv1.weight"] != nil, w["conv1.weight"] != nil
        else {
            throw NSError(domain: "QwenImageVAE", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "\(url.lastPathComponent) isn't a Wan 2.1 / Qwen-Image VAE"])
        }
        latentChannels = c.dim(0)
        self.w = w
        eval(Array(w.values))
    }

    // MARK: layers

    private func conv(_ x: MLXArray, _ name: String, stride: Int = 1, padding: Int? = nil) -> MLXArray {
        let weight = w[name + ".weight"]!
        var y = conv2d(x, weight, stride: .init(stride), padding: .init(padding ?? weight.dim(1) / 2))
        if let b = w[name + ".bias"] { y = y + b }
        return y
    }

    /// Wan's RMS_norm: F.normalize over channels * sqrt(C) * gamma == x / rms(x) * gamma
    private func norm(_ x: MLXArray, _ name: String) -> MLXArray {
        MLXFast.rmsNorm(x, weight: w[name + ".gamma"]!, eps: 1e-12)
    }

    private func residual(_ x: MLXArray, _ p: String) -> MLXArray {
        var h = conv(silu(norm(x, p + ".residual.0")), p + ".residual.2")
        h = conv(silu(norm(h, p + ".residual.3")), p + ".residual.6")
        let shortcut = w[p + ".shortcut.weight"] != nil ? conv(x, p + ".shortcut") : x
        return shortcut + h
    }

    /// single-head self-attention over all pixels (fused, no [HW x HW] matrix in memory)
    private func attention(_ x: MLXArray, _ p: String) -> MLXArray {
        let (b, h, wd, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let qkv = split(conv(norm(x, p + ".norm"), p + ".to_qkv"), parts: 3, axis: -1)
        let y = MLXFast.scaledDotProductAttention(
            queries: qkv[0].reshaped(b, 1, h * wd, c), keys: qkv[1].reshaped(b, 1, h * wd, c),
            values: qkv[2].reshaped(b, 1, h * wd, c), scale: 1 / Float(c).squareRoot(), mask: nil)
        return x + conv(y.reshaped(b, h, wd, c), p + ".proj")
    }

    private func downsample(_ x: MLXArray, _ p: String) -> MLXArray {
        // ZeroPad2d((0, 1, 0, 1)) + 3x3 conv, stride 2
        conv(padded(x, widths: [[0, 0], [0, 1], [0, 1], [0, 0]]), p + ".resample.1", stride: 2, padding: 0)
    }

    private func upsample(_ x: MLXArray, _ p: String) -> MLXArray {
        let (b, h, wd, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let up = broadcast(x.reshaped(b, h, 1, wd, 1, c), to: [b, h, 2, wd, 2, c]).reshaped(b, h * 2, wd * 2, c)
        return conv(up, p + ".resample.1")
    }

    private func middle(_ x: MLXArray, _ p: String) -> MLXArray {
        residual(attention(residual(x, p + ".0"), p + ".1"), p + ".2")
    }

    // MARK: encode / decode

    /// image NHWC in -1...1 -> latent NHWC (the encoder's mean)
    public func encode(_ image: MLXArray) -> MLXArray {
        var x = conv(image.asType(dtype), "encoder.conv1")
        for i in 0 ..< encoderBlocks {
            let p = "encoder.downsamples.\(i)"
            x = w[p + ".resample.1.weight"] != nil ? downsample(x, p) : residual(x, p)
        }
        x = middle(x, "encoder.middle")
        x = conv(silu(norm(x, "encoder.head.0")), "encoder.head.2")
        x = conv(x, "conv1")
        return x[.ellipsis, 0 ..< latentChannels]   // mu; the second half is log-variance
    }

    /// latent NHWC -> image NHWC in -1...1
    public func decode(_ latent: MLXArray) -> MLXArray {
        var x = conv(latent.asType(dtype), "conv2")
        x = middle(conv(x, "decoder.conv1"), "decoder.middle")
        for i in 0 ..< decoderBlocks {
            let p = "decoder.upsamples.\(i)"
            x = w[p + ".resample.1.weight"] != nil ? upsample(x, p) : residual(x, p)
        }
        x = conv(silu(norm(x, "decoder.head.0")), "decoder.head.2")
        return clip(x, min: -1, max: 1)
    }

    // MARK: stable-diffusion.cpp hook layout (NCHW float32)

    /// image [N, 3, H, W] in 0...1 -> latent [N, C, H/8, W/8]
    public func encode(nchw input: UnsafePointer<Float>, n: Int, channels: Int, height: Int, width: Int,
                       into output: UnsafeMutablePointer<Float>) -> Bool {
        guard channels == 3, height % 8 == 0, width % 8 == 0 else { return false }
        let x = MLXArray(UnsafeBufferPointer(start: input, count: n * channels * height * width),
                         [n, channels, height, width])
            .transposed(0, 2, 3, 1) * 2 - 1
        let z = encode(x).asType(.float32).transposed(0, 3, 1, 2)
        return copy(z, to: output)
    }

    /// latent [N, C, h, w] -> image [N, 3, 8h, 8w] in 0...1
    public func decode(nchw input: UnsafePointer<Float>, n: Int, channels: Int, height: Int, width: Int,
                       into output: UnsafeMutablePointer<Float>) -> Bool {
        guard channels == latentChannels else { return false }
        let z = MLXArray(UnsafeBufferPointer(start: input, count: n * channels * height * width),
                         [n, channels, height, width])
            .transposed(0, 2, 3, 1)
        let image = ((decode(z).asType(.float32) + 1) * 0.5).transposed(0, 3, 1, 2)
        return copy(image, to: output)
    }

    private func copy(_ a: MLXArray, to output: UnsafeMutablePointer<Float>) -> Bool {
        let values = a.asArray(Float.self)   // evaluates; contiguous in the transposed (NCHW) order
        values.withUnsafeBufferPointer { output.update(from: $0.baseAddress!, count: $0.count) }
        Memory.clearCache()
        return true
    }
}
