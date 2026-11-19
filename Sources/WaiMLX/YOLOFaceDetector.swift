import Foundation
import MLX
import MLXNN

/// YOLOv8 detector (e.g. ADetailer's face_yolov8m) in MLX, ported from stable-diffusion.cpp's
/// `model/detector/yolov8.h` and the pre/post-processing of its `detailer.cpp`: letterbox to 640 px
/// (bilinear, grey 114 padding), the network, DFL box decoding, confidence filter and per-class NMS.
/// Reads the same `.safetensors` (batch norms already folded into the convolutions).
public final class YOLOFaceDetector {
    private let w: [String: MLXArray]
    private let dtype: DType
    private var repeats: [Int: Int] = [:]
    public let regMax: Int
    public let numClasses: Int
    public var inputSize = 640
    public var nmsThreshold: Float = 0.45
    public var maxDetections = 100

    public struct Unsupported: LocalizedError {
        public let errorDescription: String?
    }

    public init(url: URL, dtype: DType = .float16) throws {
        var weights: [String: MLXArray] = [:]
        for (k, v) in try loadArrays(url: url) {
            weights[k] = (v.ndim == 4 ? v.transposed(0, 2, 3, 1) : v).asType(dtype)   // OIHW -> OHWI
        }
        guard let box = weights["model.22.cv2.0.2.weight"], let cls = weights["model.22.cv3.0.2.weight"],
              weights["model.0.conv.weight"] != nil, weights["model.9.cv2.conv.weight"] != nil
        else { throw Unsupported(errorDescription: "\(url.lastPathComponent) isn't a YOLOv8 detection model") }
        regMax = box.dim(0) / 4
        numClasses = cls.dim(0)
        for layer in [2, 4, 6, 8, 12, 15, 18, 21] {
            var n = 0
            while weights["model.\(layer).m.\(n).cv1.conv.weight"] != nil { n += 1 }
            repeats[layer] = n
        }
        w = weights
        self.dtype = dtype
        eval(Array(w.values))
    }

    // MARK: network (NHWC)

    private func conv(_ x: MLXArray, _ name: String, stride: Int = 1) -> MLXArray {
        let weight = w[name + ".weight"]!
        let k = weight.dim(1)
        return conv2d(x, weight, stride: .init(stride), padding: .init(k / 2)) + w[name + ".bias"]!
    }

    /// Conv + SiLU ("Conv" block, BN folded)
    private func cbs(_ x: MLXArray, _ name: String, stride: Int = 1) -> MLXArray {
        silu(conv(x, name + ".conv", stride: stride))
    }

    private func c2f(_ x: MLXArray, _ layer: Int, shortcut: Bool) -> MLXArray {
        let p = "model.\(layer)"
        let y = cbs(x, p + ".cv1")
        let hidden = y.dim(3) / 2
        var parts = [y[0..., 0..., 0..., ..<hidden], y[0..., 0..., 0..., hidden...]]
        var last = parts[1]
        for i in 0 ..< (repeats[layer] ?? 0) {
            let out = cbs(cbs(last, p + ".m.\(i).cv1"), p + ".m.\(i).cv2")
            last = shortcut ? last + out : out
            parts.append(last)
        }
        return cbs(concatenated(parts, axis: -1), p + ".cv2")
    }

    /// 5x5 max pool, stride 1, padding 2 (padding never wins: -inf)
    private func maxPool5(_ x: MLXArray) -> MLXArray {
        let (h, wd) = (x.dim(1), x.dim(2))
        let ninf = MLXArray(-Float.infinity).asType(x.dtype)
        var p = padded(x, widths: [0, 2, 0, 0], mode: .constant, value: ninf)
        var m = p[0..., 0 ..< h, 0..., 0...]
        for d in 1 ..< 5 { m = maximum(m, p[0..., d ..< d + h, 0..., 0...]) }
        p = padded(m, widths: [0, 0, 2, 0], mode: .constant, value: ninf)
        m = p[0..., 0..., 0 ..< wd, 0...]
        for d in 1 ..< 5 { m = maximum(m, p[0..., 0..., d ..< d + wd, 0...]) }
        return m
    }

    private func sppf(_ x: MLXArray) -> MLXArray {
        let a = cbs(x, "model.9.cv1")
        let b = maxPool5(a), c = maxPool5(b), d = maxPool5(c)
        return cbs(concatenated([a, b, c, d], axis: -1), "model.9.cv2")
    }

    private func up2(_ x: MLXArray) -> MLXArray {
        let (b, h, wd, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return broadcast(x.reshaped(b, h, 1, wd, 1, c), to: [b, h, 2, wd, 2, c]).reshaped(b, h * 2, wd * 2, c)
    }

    private func head(_ x: MLXArray, _ i: Int) -> MLXArray {
        let box = conv(cbs(cbs(x, "model.22.cv2.\(i).0"), "model.22.cv2.\(i).1"), "model.22.cv2.\(i).2")
        let cls = conv(cbs(cbs(x, "model.22.cv3.\(i).0"), "model.22.cv3.\(i).1"), "model.22.cv3.\(i).2")
        let y = concatenated([box, cls], axis: -1)
        return y.reshaped(y.dim(1) * y.dim(2), y.dim(3))   // anchors row by row
    }

    /// [1, S, S, 3] in 0...1 -> [anchors, 4 * regMax + classes] raw outputs
    func forward(_ x: MLXArray) -> MLXArray {
        let x0 = cbs(x.asType(dtype), "model.0", stride: 2)
        let x1 = cbs(x0, "model.1", stride: 2)
        let x2 = c2f(x1, 2, shortcut: true)
        let x3 = cbs(x2, "model.3", stride: 2)
        let x4 = c2f(x3, 4, shortcut: true)
        let x5 = cbs(x4, "model.5", stride: 2)
        let x6 = c2f(x5, 6, shortcut: true)
        let x7 = cbs(x6, "model.7", stride: 2)
        let x8 = c2f(x7, 8, shortcut: true)
        let x9 = sppf(x8)
        let x12 = c2f(concatenated([up2(x9), x6], axis: -1), 12, shortcut: false)
        let x15 = c2f(concatenated([up2(x12), x4], axis: -1), 15, shortcut: false)
        let x16 = cbs(x15, "model.16", stride: 2)
        let x18 = c2f(concatenated([x16, x12], axis: -1), 18, shortcut: false)
        let x19 = cbs(x18, "model.19", stride: 2)
        let x21 = c2f(concatenated([x19, x9], axis: -1), 21, shortcut: false)
        return concatenated([head(x15, 0), head(x18, 1), head(x21, 2)], axis: 0).asType(.float32)
    }

    // MARK: detection

    /// Faces (or whatever the model detects) in an RGB8 image, `width` x `height`: boxes in pixels,
    /// highest confidence first, after NMS. Same results as stable-diffusion.cpp's `sd_adetailer_detect`.
    public func detect(rgb: Data, width: Int, height: Int, confidence: Float = 0.3) -> [FaceBox] {
        let s = inputSize
        let scale = min(Float(s) / Float(width), Float(s) / Float(height))
        let rw = max(1, Int((Float(width) * scale).rounded())), rh = max(1, Int((Float(height) * scale).rounded()))
        let padX = (s - rw) / 2, padY = (s - rh) / 2

        // letterbox: bilinear at half-pixel centres, edges clamped, grey (114) around
        var input = [Float](repeating: 114 / 255, count: s * s * 3)
        rgb.withUnsafeBytes { raw in
            let px = raw.bindMemory(to: UInt8.self)
            for y in 0 ..< rh {
                let sy = (Float(y) + 0.5) / scale - 0.5
                let y0 = Int(sy.rounded(.down)), fy = sy - Float(y0)
                let ya = clampi(y0, 0, height - 1) * width, yb = clampi(y0 + 1, 0, height - 1) * width
                for x in 0 ..< rw {
                    let sx = (Float(x) + 0.5) / scale - 0.5
                    let x0 = Int(sx.rounded(.down)), fx = sx - Float(x0)
                    let xa = clampi(x0, 0, width - 1), xb = clampi(x0 + 1, 0, width - 1)
                    let o = ((y + padY) * s + x + padX) * 3
                    for c in 0 ..< 3 {
                        let a = Float(px[(ya + xa) * 3 + c]) * (1 - fx) + Float(px[(ya + xb) * 3 + c]) * fx
                        let b = Float(px[(yb + xa) * 3 + c]) * (1 - fx) + Float(px[(yb + xb) * 3 + c]) * fx
                        input[o + c] = (a * (1 - fy) + b * fy) / 255
                    }
                }
            }
        }

        let raw = forward(MLXArray(input, [1, s, s, 3]))
        // DFL: softmax over the regMax bins of each side, expectation = distance in strides
        let anchors = raw.dim(0)
        let dist = matmul(softmax(raw[0..., ..<(4 * regMax)].reshaped(anchors, 4, regMax), axis: -1),
                          MLXArray((0 ..< regMax).map(Float.init)))
        let scores = sigmoid(raw[0..., (4 * regMax)...])
        let best = scores.max(axis: -1), cls = scores.argMax(axis: -1)
        eval(dist, best, cls)
        let d = dist.asArray(Float.self), conf = best.asArray(Float.self), ids = cls.asArray(Int32.self)

        var candidates: [(box: FaceBox, cls: Int32)] = []
        var offset = 0
        for stride in [8, 16, 32] {
            let grid = s / stride
            for local in 0 ..< grid * grid {
                let a = offset + local
                guard conf[a] >= confidence else { continue }
                let cx = (Float(local % grid) + 0.5) * Float(stride), cy = (Float(local / grid) + 0.5) * Float(stride)
                let st = Float(stride)
                let x1 = min(max((cx - d[a * 4] * st - Float(padX)) / scale, 0), Float(width))
                let y1 = min(max((cy - d[a * 4 + 1] * st - Float(padY)) / scale, 0), Float(height))
                let x2 = min(max((cx + d[a * 4 + 2] * st - Float(padX)) / scale, 0), Float(width))
                let y2 = min(max((cy + d[a * 4 + 3] * st - Float(padY)) / scale, 0), Float(height))
                if x2 > x1, y2 > y1 {
                    candidates.append((FaceBox(x1: x1, y1: y1, x2: x2, y2: y2, confidence: conf[a]), ids[a]))
                }
            }
            offset += grid * grid
        }
        candidates.sort { $0.box.confidence > $1.box.confidence }
        var kept: [(box: FaceBox, cls: Int32)] = []
        for c in candidates {
            if kept.contains(where: { $0.cls == c.cls && iou($0.box, c.box) > nmsThreshold }) { continue }
            kept.append(c)
            if kept.count >= maxDetections { break }
        }
        return kept.map(\.box)
    }

    private func iou(_ a: FaceBox, _ b: FaceBox) -> Float {
        let inter = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
        let ua = max(0, a.x2 - a.x1) * max(0, a.y2 - a.y1) + max(0, b.x2 - b.x1) * max(0, b.y2 - b.y1) - inter
        return ua > 0 ? inter / ua : 0
    }
}
