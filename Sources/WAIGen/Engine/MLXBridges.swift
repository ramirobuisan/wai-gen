import Foundation
import MLX
import WaiMLX
import WaiSDCpp

/// Runs stable-diffusion.cpp's single-image VAE encode/decode in MLX (WAI-ANIMA's Qwen-Image VAE:
/// ~0.3-0.7 s instead of 45-135 s in sd.cpp's Metal backend).
final class MLXVAEBridge: SDCppExternalVAE {
    let vae: QwenImageVAE
    init(_ vae: QwenImageVAE) { self.vae = vae }

    func encode(_ input: UnsafePointer<Float>, shape: [Int], into output: UnsafeMutablePointer<Float>, outputShape: [Int]) -> Bool {
        guard outputShape[1] == vae.latentChannels else { return false }
        return vae.encode(nchw: input, n: shape[0], channels: shape[1], height: shape[2], width: shape[3], into: output)
    }

    func decode(_ input: UnsafePointer<Float>, shape: [Int], into output: UnsafeMutablePointer<Float>, outputShape: [Int]) -> Bool {
        guard outputShape[1] == 3 else { return false }
        return vae.decode(nchw: input, n: shape[0], channels: shape[1], height: shape[2], width: shape[3], into: output)
    }
}

/// An upscaler model: MLX for ESRGAN (RRDBNet) files, ~14x faster than stable-diffusion.cpp;
/// stable-diffusion.cpp for anything else.
final class ModelUpscaler {
    let url: URL
    private let mlx: RealESRGAN?
    private let sd: SDCppUpscaler?
    var runsInMLX: Bool { mlx != nil }

    init(url: URL) throws {
        self.url = url
        if let m = try? RealESRGAN(url: url) {
            mlx = m
            sd = nil
        } else {
            mlx = nil
            sd = try SDCppUpscaler(model: url)
        }
    }

    func upscale(_ img: RGBImage) throws -> RGBImage {
        if let mlx {
            let out = mlx.upscale(MLXArray(img.pixels, [img.height, img.width, 3], type: UInt8.self))
            return RGBImage(width: out.dim(1), height: out.dim(0), pixels: out.asData(access: .copy).data)
        }
        return try sd!.upscale(img)
    }
}
