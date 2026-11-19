import Foundation
import MLX
import WaiMLX

/// An ESRGAN-family upscaler model (RRDBNet), in MLX.
final class ModelUpscaler {
    let url: URL
    private let model: RealESRGAN

    init(url: URL) throws {
        self.url = url
        model = try RealESRGAN(url: url)
    }

    func upscale(_ img: RGBImage) throws -> RGBImage {
        RGBImage(model.upscale(img.array))
    }
}
