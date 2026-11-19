import Foundation
import MLX
import MLXRandom

/// A detected face (pixels, top-left origin).
public struct FaceBox: Sendable {
    public var x1, y1, x2, y2, confidence: Float
    public init(x1: Float, y1: Float, x2: Float, y2: Float, confidence: Float) {
        (self.x1, self.y1, self.x2, self.y2, self.confidence) = (x1, y1, x2, y2, confidence)
    }
}

/// Settings of the ADetailer-style pass (defaults = stable-diffusion.cpp's detailer as WAI Gen uses it).
public struct FaceRepairSettings: Sendable {
    public var strength: Float = 0.4
    public var inpaintSize = 1024
    public var padding = 32
    public var dilate = 4
    public var maskBlur = 4
    public init() {}
}

/// The geometry of stable-diffusion.cpp's ADetailer ("inpaint only masked", detailer.cpp ported line by line),
/// with the repaint supplied by the caller: each box is dilated, padded and expanded to a square crop, the
/// crop is resized to `inpaintSize`, `repaint(crop, mask, faceIndex)` repaints it (RGB8 and a 0/1 mask, both
/// `inpaintSize` square, returns RGB8 of the same size), and the result is resized back and blended in
/// through a Gaussian-feathered mask.
public enum FaceRepairGeometry {
    public static func apply(
        rgb: Data, width: Int, height: Int, faces: [FaceBox], settings s: FaceRepairSettings,
        repaint: (_ crop: [UInt8], _ mask: [Float], _ index: Int) throws -> [UInt8]
    ) rethrows -> Data {
        var image = [UInt8](rgb)
        let size = s.inpaintSize
        for (i, face) in faces.enumerated() {
            // bbox_mask + morphology_mask(dilate): a box dilated by kernel `dilate` (before = k/2, after = k-before-1)
            let bx1 = clampi(Int(face.x1.rounded(.down)), 0, width), by1 = clampi(Int(face.y1.rounded(.down)), 0, height)
            let bx2 = clampi(Int(face.x2.rounded(.up)), 0, width), by2 = clampi(Int(face.y2.rounded(.up)), 0, height)
            guard bx2 > bx1, by2 > by1 else { continue }
            let before = s.dilate / 2, after = s.dilate - before - 1
            let mask = Rect(x1: max(0, bx1 - after), y1: max(0, by1 - after),
                            x2: min(width, bx2 + before), y2: min(height, by2 + before))
            let crop = expandCrop(mask, width: width, height: height, padding: s.padding, targetW: size, targetH: size)

            // resize_crop_image (bilinear, half-pixel centres) and resize_crop_mask (nearest)
            let local = resizeCrop(image, width: width, height: height, crop: crop, toW: size, toH: size)
            var localMask = [Float](repeating: 0, count: size * size)
            let sx = Float(crop.w) / Float(size), sy = Float(crop.h) / Float(size)
            for y in 0 ..< size {
                let srcY = clampi(crop.y1 + Int((Float(y) + 0.5) * sy), 0, height - 1)
                guard srcY >= mask.y1, srcY < mask.y2 else { continue }
                for x in 0 ..< size {
                    let srcX = clampi(crop.x1 + Int((Float(x) + 0.5) * sx), 0, width - 1)
                    if srcX >= mask.x1, srcX < mask.x2 { localMask[y * size + x] = 1 }
                }
            }

            let generated = try repaint(local, localMask, i)

            // gaussian_blur_mask(mask, blur) -> feather; composite_crop (bilinear from the generated crop)
            let feather = featherMask(mask, width: width, height: height, radius: s.maskBlur)
            let gx = Float(size) / Float(crop.w), gy = Float(size) / Float(crop.h)
            for y in crop.y1 ..< crop.y2 {
                let srcY = (Float(y - crop.y1) + 0.5) * gy - 0.5
                for x in crop.x1 ..< crop.x2 {
                    let alpha = Float(feather[y * width + x]) / 255
                    if alpha <= 0 { continue }
                    let srcX = (Float(x - crop.x1) + 0.5) * gx - 0.5
                    for c in 0 ..< 3 {
                        let g = bilinear(generated, size, size, srcX, srcY, c) * 255
                        let o = Float(image[(y * width + x) * 3 + c])
                        image[(y * width + x) * 3 + c] = UInt8(min(max(o * (1 - alpha) + g * alpha, 0), 255) + 0.5)
                    }
                }
            }
        }
        return Data(image)
    }

    /// Side of the largest square crop the faces need (box + dilation + padding on both sides), to pick
    /// a repaint size that doesn't throw away detail.
    public static func largestCrop(_ faces: [FaceBox], settings s: FaceRepairSettings) -> Int {
        faces.map { Int((max($0.x2 - $0.x1, $0.y2 - $0.y1)).rounded(.up)) + s.dilate + 2 * s.padding }.max() ?? 0
    }
}

extension WaiSDXL {
    /// Repaint each face like stable-diffusion.cpp's ADetailer (`FaceRepairGeometry`) with this MLX engine:
    /// masked img2img at `strength`, keeping the original latent outside the box at every step.
    /// `rgb`: RGB8, `width` x `height`. `p`: prompt, negative, cfg, sampler, scheduler, steps, seed
    /// (face i uses seed + i). `progress(face, step, steps)`.
    public func repairFaces(
        rgb: Data, width: Int, height: Int, faces: [FaceBox], _ p: SDXLParameters,
        settings s: FaceRepairSettings = FaceRepairSettings(),
        isCancelled: () -> Bool = { false },
        progress: (Int, Int, Int) -> Void = { _, _, _ in }
    ) throws -> Data {
        let size = s.inpaintSize
        var cond: (MLXArray, MLXArray)?
        return try FaceRepairGeometry.apply(rgb: rgb, width: width, height: height, faces: faces, settings: s) {
            local, localMask, i in
            if cond == nil { cond = encodeText(p) }
            var pi = p
            pi.seed = p.seed &+ UInt64(i)
            let repainted = try repaint(
                MLXArray(local, [size, size, 3]), mask: MLXArray(localMask, [size, size]), pi, strength: s.strength,
                cond: cond!, isCancelled: isCancelled, progress: { progress(i, $0, $1) })
            return repainted.asArray(UInt8.self)
        }
    }

    /// img2img on `image` (uint8 [H, W, 3], multiples of 8) that only changes the masked part: after
    /// every step the prediction outside `mask` (float [H, W]) is replaced by the original latent, as in
    /// stable-diffusion.cpp's denoise mask. Runs the last steps*strength steps of a `p.steps` schedule.
    func repaint(
        _ image: MLXArray, mask: MLXArray, _ p: SDXLParameters, strength: Float, cond: (MLXArray, MLXArray),
        isCancelled: () -> Bool, progress: (Int, Int) -> Void
    ) throws -> MLXArray {
        defer { Memory.clearCache() }
        MLXRandom.seed(p.seed)
        let (h, w) = (image.dim(0), image.dim(1))
        let src = image.asType(.float32)[.newAxis] / 127.5 - 1
        var (x0, _) = vae.encode(src.asType(vaeDType))
        x0 = x0.asType(.float32)
        // latent mask: nearest-max downsample by 8 (max over each 8x8 block)
        let m = mask.reshaped(1, h / 8, 8, w / 8, 8).max(axes: [2, 4]).reshaped(1, h / 8, w / 8, 1)
        let full = schedule.sigmas(steps: p.steps, scheduler: p.scheduler)
        let tEnc = max(1, Int(Float(p.steps) * strength))
        let sigmas = Array(full.suffix(tEnc + 1))
        let x = x0 + MLXRandom.normal(x0.shape) * sigmas[0]
        let cache = p.stepCache.map { StepCache(threshold: $0, sigmas: sigmas) }
        let base = denoiser(context: cond.0, pooled: cond.1, cfg: p.cfg, width: w, height: h, cache: cache)
        let masked: Denoiser = { x, sigma in base(x, sigma) * m + x0 * (1 - m) }
        guard let latent = sample(p.sampler, denoise: masked, x: x, sigmas: sigmas, isCancelled: isCancelled,
                                  preview: nil, progress: progress)
        else { throw WaiCancelled() }
        let img = vae.decode(latent.asType(vaeDType)).asType(.float32)
        let out = (clip(img / 2 + 0.5, min: 0, max: 1) * 255).round().asType(.uint8)[0]
        eval(out)
        return out
    }
}

// MARK: - stable-diffusion.cpp detailer geometry (detailer.cpp), ported

struct Rect {
    var x1, y1, x2, y2: Int
    var w: Int { x2 - x1 }
    var h: Int { y2 - y1 }
}

@inline(__always) func clampi(_ v: Int, _ lo: Int, _ hi: Int) -> Int { min(max(v, lo), hi) }

/// expand_crop: pad, grow to the target aspect around the centre, keep inside the image.
func expandCrop(_ r: Rect, width: Int, height: Int, padding: Int, targetW: Int, targetH: Int) -> Rect {
    var c = Rect(x1: max(0, r.x1 - padding), y1: max(0, r.y1 - padding),
                 x2: min(width, r.x2 + padding), y2: min(height, r.y2 + padding))
    let aspect = Float(targetW) / Float(targetH)
    var dw = c.w, dh = c.h
    if Float(c.w) / Float(c.h) < aspect { dw = Int((Float(c.h) * aspect).rounded(.up)) } else {
        dh = Int((Float(c.w) / aspect).rounded(.up))
    }
    dw = min(dw, width)
    dh = min(dh, height)
    let cx = (c.x1 + c.x2) / 2, cy = (c.y1 + c.y2) / 2
    c.x1 = clampi(cx - dw / 2, 0, width - dw)
    c.y1 = clampi(cy - dh / 2, 0, height - dh)
    c.x2 = c.x1 + dw
    c.y2 = c.y1 + dh
    return c
}

/// image_channel + bilinear_channel: edge-clamped bilinear sample in 0...1.
@inline(__always) func bilinear(_ img: [UInt8], _ w: Int, _ h: Int, _ x: Float, _ y: Float, _ c: Int) -> Float {
    let x0 = Int(x.rounded(.down)), y0 = Int(y.rounded(.down))
    let wx = x - Float(x0), wy = y - Float(y0)
    func px(_ xx: Int, _ yy: Int) -> Float { Float(img[(clampi(yy, 0, h - 1) * w + clampi(xx, 0, w - 1)) * 3 + c]) / 255 }
    let a = px(x0, y0) * (1 - wx) + px(x0 + 1, y0) * wx
    let b = px(x0, y0 + 1) * (1 - wx) + px(x0 + 1, y0 + 1) * wx
    return a * (1 - wy) + b * wy
}

/// resize_crop_image: crop resized with bilinear sampling at half-pixel centres.
func resizeCrop(_ img: [UInt8], width: Int, height: Int, crop: Rect, toW: Int, toH: Int) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: toW * toH * 3)
    let sx = Float(crop.w) / Float(toW), sy = Float(crop.h) / Float(toH)
    for y in 0 ..< toH {
        let srcY = Float(crop.y1) + (Float(y) + 0.5) * sy - 0.5
        for x in 0 ..< toW {
            let srcX = Float(crop.x1) + (Float(x) + 0.5) * sx - 0.5
            for c in 0 ..< 3 {
                out[(y * toW + x) * 3 + c] = UInt8(min(max(bilinear(img, width, height, srcX, srcY, c) * 255, 0), 255) + 0.5)
            }
        }
    }
    return out
}

/// gaussian_blur_mask of a filled rectangle (sigma = radius, kernel radius ceil(2.5 sigma), edges
/// clamped, separable, rounded to uint8), computed only where it can be non-zero.
func featherMask(_ r: Rect, width: Int, height: Int, radius: Int) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: width * height)
    guard radius > 0 else {
        for y in r.y1 ..< r.y2 { for x in r.x1 ..< r.x2 { out[y * width + x] = 255 } }
        return out
    }
    let sigma = Float(radius)
    let kr = max(1, Int((2.5 * sigma).rounded(.up)))
    var kernel = (-kr ... kr).map { exp(-Float($0 * $0) / (2 * sigma * sigma)) }
    let sum = kernel.reduce(0, +)
    kernel = kernel.map { $0 / sum }
    // the source is 255 inside the rectangle; with clamped edges a 1-D pass is a function of one coordinate
    func pass(_ v: Int, _ lo: Int, _ hi: Int, _ n: Int) -> Float {
        var s: Float = 0
        for k in -kr ... kr where (lo ..< hi).contains(clampi(v + k, 0, n - 1)) { s += kernel[k + kr] }
        return s
    }
    let ys = max(0, r.y1 - kr) ..< min(height, r.y2 + kr), xs = max(0, r.x1 - kr) ..< min(width, r.x2 + kr)
    let colH = xs.map { pass($0, r.x1, r.x2, width) * 255 }
    for y in ys {
        let wy = pass(y, r.y1, r.y2, height)
        for (j, x) in xs.enumerated() {
            out[y * width + x] = UInt8(min(max(colH[j] * wy, 0), 255) + 0.5)
        }
    }
    return out
}
