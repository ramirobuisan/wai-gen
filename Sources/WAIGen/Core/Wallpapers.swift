import Foundation

/// Screens to make wallpapers for: the generation size (the screen's aspect ratio at ~1 MP, multiples of 64) and the
/// screen's own resolution, which "Save as … wallpaper" reaches (upscaler, scale to cover, centre crop).
struct Wallpaper: Identifiable, Sendable {
    let device: String
    let width: Int, height: Int              // generation size
    let screenWidth: Int, screenHeight: Int  // the screen
    var id: String { device }

    static let all: [Wallpaper] = [
        Wallpaper(device: "iPhone Pro Max", width: 704, height: 1536, screenWidth: 1320, screenHeight: 2868),
        Wallpaper(device: "MacBook Pro 14\"", width: 1280, height: 832, screenWidth: 3024, screenHeight: 1964),
    ]

    static func matching(width: Int, height: Int) -> Wallpaper? { all.first { $0.width == width && $0.height == height } }

    /// The size an image of `w` × `h` is scaled to so it covers the screen (then cropped to it).
    func coverSize(_ w: Int, _ h: Int) -> (Int, Int) {
        let s = max(Double(screenWidth) / Double(w), Double(screenHeight) / Double(h))
        return (max(screenWidth, Int((Double(w) * s).rounded(.up))), max(screenHeight, Int((Double(h) * s).rounded(.up))))
    }
}
