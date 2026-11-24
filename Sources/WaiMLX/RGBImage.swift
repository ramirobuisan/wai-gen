import CoreGraphics
import Foundation
import ImageIO
import MLX

/// One RGB8 image (interleaved rows).
public struct RGBImage: Sendable {
    public let width: Int
    public let height: Int
    public let pixels: Data

    public init(width: Int, height: Int, pixels: Data) {
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    /// From a uint8 [H, W, 3] array.
    public init(_ array: MLXArray) {
        self.init(width: array.dim(1), height: array.dim(0), pixels: array.asData(access: .copy).data)
    }

    /// uint8 [H, W, 3]
    public var array: MLXArray { MLXArray(pixels, [height, width, 3], type: UInt8.self) }

    /// The `width` × `height` region at (`x`, `y`), clamped to the image.
    public func cropped(x: Int, y: Int, width w: Int, height h: Int) -> RGBImage {
        let x0 = max(0, min(x, width - 1)), y0 = max(0, min(y, height - 1))
        let cw = min(w, width - x0), ch = min(h, height - y0)
        if x0 == 0, y0 == 0, cw == width, ch == height { return self }
        var out = Data(count: cw * ch * 3)
        out.withUnsafeMutableBytes { dst in
            pixels.withUnsafeBytes { src in
                for row in 0 ..< ch {
                    let from = ((y0 + row) * width + x0) * 3
                    dst.baseAddress!.advanced(by: row * cw * 3).copyMemory(from: src.baseAddress!.advanced(by: from), byteCount: cw * 3)
                }
            }
        }
        return RGBImage(width: cw, height: ch, pixels: out)
    }

    /// High-quality resample (Core Graphics) to an exact size.
    public func resized(width w: Int, height h: Int) -> RGBImage {
        if w == width, h == height { return self }
        guard let provider = CGDataProvider(data: pixels as CFData),
            let src = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24,
                              bytesPerRow: width * 3, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                              bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider, decode: nil,
                              shouldInterpolate: true, intent: .defaultIntent)
        else { return self }
        var rgba = Data(count: w * h * 4)
        rgba.withUnsafeMutableBytes { buf in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
            else { return }
            ctx.interpolationQuality = .high
            ctx.draw(src, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
        var rgb = Data(count: w * h * 3)
        rgba.withUnsafeBytes { s in
            rgb.withUnsafeMutableBytes { d in
                let sp = s.bindMemory(to: UInt8.self)
                let dp = d.bindMemory(to: UInt8.self)
                for i in 0 ..< w * h { dp[i * 3] = sp[i * 4]; dp[i * 3 + 1] = sp[i * 4 + 1]; dp[i * 3 + 2] = sp[i * 4 + 2] }
            }
        }
        return RGBImage(width: w, height: h, pixels: rgb)
    }

    /// Any image file as RGB8 (sRGB).
    public static func load(_ url: URL) -> RGBImage? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cg = CGImageSourceCreateImageAtIndex(src, 0, nil)
        else { return nil }
        let w = cg.width, h = cg.height
        var rgba = Data(count: w * h * 4)
        let ok = rgba.withUnsafeMutableBytes { buf -> Bool in
            guard let ctx = CGContext(data: buf.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue | CGBitmapInfo.byteOrder32Big.rawValue)
            else { return false }
            ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { return nil }
        var rgb = Data(count: w * h * 3)
        rgba.withUnsafeBytes { s in
            rgb.withUnsafeMutableBytes { d in
                let sp = s.bindMemory(to: UInt8.self)
                let dp = d.bindMemory(to: UInt8.self)
                for i in 0 ..< w * h { dp[i * 3] = sp[i * 4]; dp[i * 3 + 1] = sp[i * 4 + 1]; dp[i * 3 + 2] = sp[i * 4 + 2] }
            }
        }
        return RGBImage(width: w, height: h, pixels: rgb)
    }
}
