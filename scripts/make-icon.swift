// Draws WAI Gen's app icon and writes Resources/AppIcon.icns (plus the 1024 px PNG).
// Usage: swift scripts/make-icon.swift            (run from the repository root)
import AppKit
import CoreGraphics

let size: CGFloat = 1024

func color(_ hex: UInt32, _ alpha: CGFloat = 1) -> CGColor {
    CGColor(srgbRed: CGFloat(hex >> 16 & 0xFF) / 255, green: CGFloat(hex >> 8 & 0xFF) / 255,
            blue: CGFloat(hex & 0xFF) / 255, alpha: alpha)
}

/// Four-point sparkle with concave sides, centred on `c`.
func sparkle(_ c: CGPoint, _ r: CGFloat, pinch: CGFloat = 0.13) -> CGPath {
    let k = r * pinch
    let p = CGMutablePath()
    p.move(to: CGPoint(x: c.x, y: c.y + r))
    p.addQuadCurve(to: CGPoint(x: c.x + r, y: c.y), control: CGPoint(x: c.x + k, y: c.y + k))
    p.addQuadCurve(to: CGPoint(x: c.x, y: c.y - r), control: CGPoint(x: c.x + k, y: c.y - k))
    p.addQuadCurve(to: CGPoint(x: c.x - r, y: c.y), control: CGPoint(x: c.x - k, y: c.y - k))
    p.addQuadCurve(to: CGPoint(x: c.x, y: c.y + r), control: CGPoint(x: c.x - k, y: c.y + k))
    p.closeSubpath()
    return p
}

func linear(_ ctx: CGContext, _ colors: [CGColor], from: CGPoint, to: CGPoint) {
    let g = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: colors as CFArray, locations: nil)!
    ctx.drawLinearGradient(g, start: from, end: to, options: [.drawsBeforeStartLocation, .drawsAfterEndLocation])
}

func radial(_ ctx: CGContext, _ colors: [CGColor], at c: CGPoint, radius: CGFloat) {
    let g = CGGradient(colorsSpace: CGColorSpace(name: CGColorSpace.sRGB), colors: colors as CFArray, locations: nil)!
    ctx.drawRadialGradient(g, startCenter: c, startRadius: 0, endCenter: c, endRadius: radius, options: [])
}

let ctx = CGContext(data: nil, width: Int(size), height: Int(size), bitsPerComponent: 8, bytesPerRow: 0,
                    space: CGColorSpace(name: CGColorSpace.sRGB)!,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
ctx.interpolationQuality = .high

// --- body: macOS icon grid (824 px rounded square, 100 px margin) with a soft drop shadow
let body = CGRect(x: 100, y: 100, width: 824, height: 824)
let bodyPath = CGPath(roundedRect: body, cornerWidth: 185, cornerHeight: 185, transform: nil)
ctx.saveGState()
ctx.setShadow(offset: CGSize(width: 0, height: -12), blur: 28, color: color(0x000000, 0.35))
ctx.addPath(bodyPath)
ctx.setFillColor(color(0x24145E))
ctx.fillPath()
ctx.restoreGState()

ctx.saveGState()
ctx.addPath(bodyPath)
ctx.clip()
// night sky: violet top -> deep indigo bottom, pink glow low right, lighter haze top left
linear(ctx, [color(0x6A3BE0), color(0x3A1F9E), color(0x1B1148)], from: CGPoint(x: 300, y: 924), to: CGPoint(x: 724, y: 100))
radial(ctx, [color(0xFF5FA8, 0.75), color(0xFF5FA8, 0)], at: CGPoint(x: 780, y: 230), radius: 460)
radial(ctx, [color(0xB9A6FF, 0.35), color(0xB9A6FF, 0)], at: CGPoint(x: 230, y: 860), radius: 380)

// --- picture card (portrait, like the images the app makes), tilted a little
ctx.saveGState()
ctx.translateBy(x: 470, y: 535)
ctx.rotate(by: 7 * .pi / 180)
let card = CGRect(x: -185, y: -270, width: 370, height: 540)
let cardPath = CGPath(roundedRect: card, cornerWidth: 46, cornerHeight: 46, transform: nil)
ctx.saveGState()
ctx.setShadow(offset: CGSize(width: 0, height: -14), blur: 34, color: color(0x0B0630, 0.55))
ctx.addPath(cardPath)
ctx.setFillColor(color(0xFFFFFF))
ctx.fillPath()
ctx.restoreGState()
// the "image": pastel dusk sky, sun, two mountain ridges
ctx.saveGState()
let inner = card.insetBy(dx: 22, dy: 22)
ctx.addPath(CGPath(roundedRect: inner, cornerWidth: 28, cornerHeight: 28, transform: nil))
ctx.clip()
linear(ctx, [color(0xFFD1E6), color(0xC9B8FF), color(0x8C7CF0)], from: CGPoint(x: 0, y: inner.minY), to: CGPoint(x: 0, y: inner.maxY))
ctx.setFillColor(color(0xFFF4D6))
ctx.fillEllipse(in: CGRect(x: -95, y: 30, width: 120, height: 120))
let back = CGMutablePath()
back.move(to: CGPoint(x: inner.minX, y: -40))
back.addLine(to: CGPoint(x: -70, y: 70))
back.addLine(to: CGPoint(x: 20, y: -10))
back.addLine(to: CGPoint(x: 95, y: 55))
back.addLine(to: CGPoint(x: inner.maxX, y: -30))
back.addLine(to: CGPoint(x: inner.maxX, y: inner.minY))
back.addLine(to: CGPoint(x: inner.minX, y: inner.minY))
back.closeSubpath()
ctx.addPath(back)
ctx.setFillColor(color(0x6B5BD6))
ctx.fillPath()
let front = CGMutablePath()
front.move(to: CGPoint(x: inner.minX, y: -150))
front.addLine(to: CGPoint(x: -40, y: -60))
front.addLine(to: CGPoint(x: 60, y: -140))
front.addLine(to: CGPoint(x: inner.maxX, y: -95))
front.addLine(to: CGPoint(x: inner.maxX, y: inner.minY))
front.addLine(to: CGPoint(x: inner.minX, y: inner.minY))
front.closeSubpath()
ctx.addPath(front)
ctx.setFillColor(color(0x3E2F9C))
ctx.fillPath()
ctx.restoreGState()
ctx.restoreGState()

// --- the big sparkle over the card's corner, glowing
let star = sparkle(CGPoint(x: 690, y: 330), 205)
ctx.saveGState()
ctx.setShadow(offset: .zero, blur: 60, color: color(0xFFE3F3, 0.95))
ctx.addPath(star)
ctx.setFillColor(color(0xFFFFFF))
ctx.fillPath()
ctx.restoreGState()
ctx.saveGState()
ctx.addPath(star)
ctx.clip()
linear(ctx, [color(0xFFFFFF), color(0xFFF0F8), color(0xFFC6E2)], from: CGPoint(x: 600, y: 520), to: CGPoint(x: 780, y: 140))
ctx.restoreGState()

// small sparkles
for (c, r) in [(CGPoint(x: 800, y: 560), CGFloat(62)), (CGPoint(x: 255, y: 250), CGFloat(44))] {
    ctx.saveGState()
    ctx.setShadow(offset: .zero, blur: 22, color: color(0xFFFFFF, 0.9))
    ctx.addPath(sparkle(c, r))
    ctx.setFillColor(color(0xFFFFFF, 0.95))
    ctx.fillPath()
    ctx.restoreGState()
}

// subtle top highlight on the body
radial(ctx, [color(0xFFFFFF, 0.12), color(0xFFFFFF, 0)], at: CGPoint(x: 512, y: 980), radius: 520)
ctx.restoreGState()

// --- write the PNG, the iconset and the icns
let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let resources = root.appendingPathComponent("Resources", isDirectory: true)
try FileManager.default.createDirectory(at: resources, withIntermediateDirectories: true)
let master = resources.appendingPathComponent("AppIcon-1024.png")
let rep = NSBitmapImageRep(cgImage: ctx.makeImage()!)
try rep.representation(using: .png, properties: [:])!.write(to: master)

let iconset = FileManager.default.temporaryDirectory.appendingPathComponent("AppIcon.iconset", isDirectory: true)
try? FileManager.default.removeItem(at: iconset)
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
let source = NSImage(contentsOf: master)!
for base in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let px = base * scale
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: px, pixelsHigh: px, bitsPerSample: 8,
                                      samplesPerPixel: 4, hasAlpha: true, isPlanar: false, colorSpaceName: .deviceRGB,
                                      bytesPerRow: 0, bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
        NSGraphicsContext.current?.imageInterpolation = .high
        source.draw(in: NSRect(x: 0, y: 0, width: px, height: px))
        NSGraphicsContext.restoreGraphicsState()
        let name = scale == 1 ? "icon_\(base)x\(base).png" : "icon_\(base)x\(base)@2x.png"
        try bitmap.representation(using: .png, properties: [:])!.write(to: iconset.appendingPathComponent(name))
    }
}
let iconutil = Process()
iconutil.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
iconutil.arguments = ["-c", "icns", iconset.path, "-o", resources.appendingPathComponent("AppIcon.icns").path]
try iconutil.run()
iconutil.waitUntilExit()
print(iconutil.terminationStatus == 0 ? "wrote Resources/AppIcon.icns and Resources/AppIcon-1024.png" : "iconutil failed")
