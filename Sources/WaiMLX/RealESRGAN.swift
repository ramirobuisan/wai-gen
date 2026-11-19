import Foundation
import MLX
import MLXNN

/// ESRGAN / Real-ESRGAN upscalers with the RRDBNet architecture (x4, e.g. RealESRGAN_x4plus,
/// RealESRGAN_x4plus_anime_6B, most "4x-..." ESRGAN models), in MLX. Reads `.pth` (via
/// `TorchCheckpoint`, no pickle code is run) or `.safetensors`.
public final class RealESRGAN {
    public let scale = 4
    public let blocks: Int
    private let w: [String: MLXArray]
    private let dtype: DType

    public struct Unsupported: LocalizedError {
        public let errorDescription: String?
    }

    /// The scale (4) when `url` is a model this class can run, nil otherwise (other architectures such as
    /// DAT / SwinIR / HAT, e.g. 4x-UltraSharpV2). Safetensors are checked from their header only.
    public static func probe(_ url: URL) -> Int? {
        if url.pathExtension.lowercased() == "safetensors" {
            guard let h = try? FileHandle(forReadingFrom: url) else { return nil }
            defer { try? h.close() }
            guard let lenData = try? h.read(upToCount: 8), lenData.count == 8 else { return nil }
            let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }
            guard len < 100_000_000, let header = try? h.read(upToCount: Int(len)),
                  let json = try? JSONSerialization.jsonObject(with: header) as? [String: Any] else { return nil }
            let keys = Set(json.keys.map { k in
                k.hasPrefix("params_ema.") ? String(k.dropFirst(11)) : k.hasPrefix("params.") ? String(k.dropFirst(7)) : k
            })
            return ["conv_first.weight", "conv_up1.weight", "conv_up2.weight", "conv_last.weight", "body.0.rdb1.conv1.weight"]
                .allSatisfy(keys.contains) ? 4 : nil
        }
        return (try? RealESRGAN(url: url)) != nil ? 4 : nil
    }

    public init(url: URL, dtype: DType = .float16) throws {
        var raw = url.pathExtension.lowercased() == "pth" ? try TorchCheckpoint.load(url: url) : try loadArrays(url: url)
        // BasicSR files keep the weights under "params_ema" (preferred) or "params"
        for prefix in ["params_ema.", "params."] where raw.keys.contains(where: { $0.hasPrefix(prefix) }) {
            raw = Dictionary(uniqueKeysWithValues: raw.compactMap { k, v in
                k.hasPrefix(prefix) ? (String(k.dropFirst(prefix.count)), v) : nil
            })
            break
        }
        // old ESRGAN key names (model.0 / model.1.sub.N.RDB1 / ...) aren't mapped: only BasicSR RRDBNet
        guard let first = raw["conv_first.weight"], first.ndim == 4, first.dim(1) == 3,
              raw["conv_up1.weight"] != nil, raw["conv_up2.weight"] != nil, raw["conv_last.weight"] != nil
        else {
            throw Unsupported(errorDescription: "\(url.lastPathComponent) isn't an x4 RRDBNet (ESRGAN) model")
        }
        var weights: [String: MLXArray] = [:]
        for (k, v) in raw {
            weights[k] = (k.hasSuffix(".weight") && v.ndim == 4 ? v.transposed(0, 2, 3, 1) : v).asType(dtype)
        }
        var n = 0
        while weights["body.\(n).rdb1.conv1.weight"] != nil { n += 1 }
        guard n > 0 else { throw Unsupported(errorDescription: "\(url.lastPathComponent): no RRDB blocks found") }
        blocks = n
        w = weights
        self.dtype = dtype
        eval(Array(w.values))
    }

    private func conv(_ x: MLXArray, _ name: String) -> MLXArray {
        conv2d(x, w[name + ".weight"]!, padding: 1) + w[name + ".bias"]!
    }

    private func lrelu(_ x: MLXArray) -> MLXArray { leakyRelu(x, negativeSlope: 0.2) }

    private func denseBlock(_ x: MLXArray, _ p: String) -> MLXArray {
        let x1 = lrelu(conv(x, p + ".conv1"))
        let x2 = lrelu(conv(concatenated([x, x1], axis: -1), p + ".conv2"))
        let x3 = lrelu(conv(concatenated([x, x1, x2], axis: -1), p + ".conv3"))
        let x4 = lrelu(conv(concatenated([x, x1, x2, x3], axis: -1), p + ".conv4"))
        let x5 = conv(concatenated([x, x1, x2, x3, x4], axis: -1), p + ".conv5")
        return x5 * 0.2 + x
    }

    private func up2(_ x: MLXArray) -> MLXArray {
        let (b, h, wd, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return broadcast(x.reshaped(b, h, 1, wd, 1, c), to: [b, h, 2, wd, 2, c]).reshaped(b, h * 2, wd * 2, c)
    }

    /// NHWC in 0...1 -> NHWC (x4) in 0...1
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let feat = conv(x.asType(dtype), "conv_first")
        var body = feat
        for i in 0 ..< blocks {
            let p = "body.\(i)"
            body = denseBlock(denseBlock(denseBlock(body, p + ".rdb1"), p + ".rdb2"), p + ".rdb3") * 0.2 + body
        }
        var y = feat + conv(body, "conv_body")
        y = lrelu(conv(up2(y), "conv_up1"))
        y = lrelu(conv(up2(y), "conv_up2"))
        return clip(conv(lrelu(conv(y, "conv_hr")), "conv_last"), min: 0, max: 1)
    }

    /// RGB8 [H, W, 3] -> RGB8 [4H, 4W, 3]. Works in tiles (`tile` px of input plus `pad` px of context on
    /// each side) so big images don't need one huge activation.
    public func upscale(_ image: MLXArray, tile: Int = 640, pad: Int = 24) -> MLXArray {
        let (h, wd) = (image.dim(0), image.dim(1))
        let x = image.asType(.float32) / 255
        var rows: [MLXArray] = []
        for y0 in stride(from: 0, to: h, by: tile) {
            var row: [MLXArray] = []
            for x0 in stride(from: 0, to: wd, by: tile) {
                let (y1, x1) = (min(h, y0 + tile), min(wd, x0 + tile))
                let (ys, xs) = (max(0, y0 - pad), max(0, x0 - pad))
                let (ye, xe) = (min(h, y1 + pad), min(wd, x1 + pad))
                let out = self(x[ys ..< ye, xs ..< xe].expandedDimensions(axis: 0))[0]
                let cropped = out[((y0 - ys) * scale) ..< ((y1 - ys) * scale), ((x0 - xs) * scale) ..< ((x1 - xs) * scale)]
                let tile8 = (cropped.asType(.float32) * 255).round().asType(.uint8)
                eval(tile8)
                row.append(tile8)
            }
            rows.append(concatenated(row, axis: 1))
        }
        let result = concatenated(rows, axis: 0)
        eval(result)
        Memory.clearCache()
        return result
    }
}
