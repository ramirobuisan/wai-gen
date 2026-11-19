import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

/// What produced an image; stored inside the PNG so the gallery and "use settings" work later.
struct GenerationRecord: Codable, Hashable {
    var modelID: String
    var modelName: String
    var engine: String
    var positive: String          // final prompt sent to the engine
    var negative: String
    var characters: [String]
    var userPrompt: String
    var width: Int
    var height: Int
    var steps: Int
    var cfg: Double
    var sampler: String
    var scheduler: String
    var seed: Int64
    var hiresScale: Double?
    var hiresSteps: Int?
    var hiresDenoise: Double?
    var seconds: Double
    var date: Date
    var faceRepairStrength: Double? = nil
    var faceRepairConfidence: Double? = nil
    var faceRepairPrompt: String? = nil
    var faceRepairSeconds: Double? = nil
    var loras: [LoRARecord]? = nil
    var hiresUpscaler: String? = nil
    /// set on images made with the gallery's Upscale action
    var upscaledFrom: String? = nil
    var upscaleDetailPass: Bool? = nil
    /// true when made with the step cache (EasyCache) on
    var stepCache: Bool? = nil

    /// A1111-style "parameters" text (with ADetailer's key names), readable by other tools.
    var parametersText: String {
        let loraTags = (loras ?? []).map {
            " <lora:\((($0.path as NSString).lastPathComponent as NSString).deletingPathExtension):\($0.strength)>"
        }.joined()
        var s = "\(positive)\(loraTags)\nNegative prompt: \(negative)\n"
        s += "Steps: \(steps), Sampler: \(sampler), Schedule type: \(scheduler), CFG scale: \(cfg), Seed: \(seed), "
        s += "Size: \(width)x\(height), Model: \(modelName)"
        if let hs = hiresScale {
            s += ", Hires upscale: \(hs), Hires steps: \(hiresSteps ?? 0), Denoising strength: \(hiresDenoise ?? 0)"
            if let u = hiresUpscaler { s += ", Hires upscaler: \((u as NSString).deletingPathExtension)" }
        }
        if let fs = faceRepairStrength {
            s += ", ADetailer model: face_yolov8m.pt, ADetailer confidence: \(faceRepairConfidence ?? 0.3)"
            s += ", ADetailer denoising strength: \(fs)"
            if let fp = faceRepairPrompt { s += ", ADetailer prompt: \"\(fp.replacingOccurrences(of: "\"", with: "\\\""))\"" }
        }
        if stepCache == true { s += ", Step cache: EasyCache" }
        if let src = upscaledFrom { s += ", Upscaled from: \(src)\(upscaleDetailPass == false ? " (upscaler only)" : "")" }
        s += ", Engine: \(engine), Version: WAI Gen"
        return s
    }
}

struct LoRARecord: Codable, Hashable {
    var path: String       // relative to models/loras
    var strength: Double
}

struct GalleryItem: Identifiable, Hashable {
    let url: URL
    let record: GenerationRecord?
    var id: URL { url }
}

enum ImageFiles {
    static func cgImage(rgb: Data, width: Int, height: Int) -> CGImage? {
        guard let provider = CGDataProvider(data: rgb as CFData) else { return nil }
        return CGImage(
            width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3,
            space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: 0),
            provider: provider, decode: nil, shouldInterpolate: true, intent: .defaultIntent)
    }

    /// Save as PNG with the record in the PNG description (JSON) and comment (A1111 text).
    static func save(_ image: CGImage, record: GenerationRecord) throws -> URL {
        let day = DateFormatter()
        day.dateFormat = "yyyy-MM-dd"
        let time = DateFormatter()
        time.dateFormat = "HHmmss"
        let dir = AppPaths.output.appendingPathComponent(day.string(from: record.date), isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var url = dir.appendingPathComponent("\(time.string(from: record.date))_\(record.seed).png")
        var n = 1
        while FileManager.default.fileExists(atPath: url.path) {
            url = dir.appendingPathComponent("\(time.string(from: record.date))_\(record.seed)_\(n).png")
            n += 1
        }
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw CocoaError(.fileWriteUnknown) }
        let json = String(decoding: try JSONEncoder().encode(record), as: UTF8.self)
        let props: [CFString: Any] = [
            kCGImagePropertyPNGDictionary: [
                kCGImagePropertyPNGDescription: json,
                kCGImagePropertyPNGComment: record.parametersText,
                kCGImagePropertyPNGSoftware: "WAI Gen",
            ],
        ]
        CGImageDestinationAddImage(dest, image, props as CFDictionary)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
        // ImageIO drops the PNG comment; add the "parameters" text chunk that A1111/Civitai read
        try? insertTextChunk(into: url, keyword: "parameters", text: record.parametersText)
        return url
    }

    /// Insert an uncompressed UTF-8 iTXt chunk before IEND.
    static func insertTextChunk(into url: URL, keyword: String, text: String) throws {
        var png = try Data(contentsOf: url)
        guard png.count > 20, let iend = png.range(of: Data("IEND".utf8), options: .backwards) else { return }
        var body = Data("iTXt".utf8)
        body.append(Data(keyword.utf8))
        body.append(contentsOf: [0, 0, 0, 0, 0])   // NUL, compression flag, method, language NUL, translated NUL
        body.append(Data(text.utf8))
        var chunk = Data()
        withUnsafeBytes(of: UInt32(body.count - 4).bigEndian) { chunk.append(contentsOf: $0) }
        chunk.append(body)
        withUnsafeBytes(of: crc32(body).bigEndian) { chunk.append(contentsOf: $0) }
        png.insert(contentsOf: chunk, at: iend.lowerBound - 4)
        try png.write(to: url, options: .atomic)
    }

    private static let crcTable: [UInt32] = (0 ..< 256).map { n -> UInt32 in
        var c = UInt32(n)
        for _ in 0 ..< 8 { c = c & 1 != 0 ? 0xEDB8_8320 ^ (c >> 1) : c >> 1 }
        return c
    }

    static func crc32(_ data: Data) -> UInt32 {
        var c: UInt32 = 0xFFFF_FFFF
        for b in data { c = crcTable[Int((c ^ UInt32(b)) & 0xFF)] ^ (c >> 8) }
        return c ^ 0xFFFF_FFFF
    }

    static func readRecord(_ url: URL) -> GenerationRecord? {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let props = CGImageSourceCopyPropertiesAtIndex(src, 0, nil) as? [CFString: Any],
            let png = props[kCGImagePropertyPNGDictionary] as? [CFString: Any],
            let json = png[kCGImagePropertyPNGDescription] as? String
        else { return nil }
        return try? JSONDecoder().decode(GenerationRecord.self, from: Data(json.utf8))
    }

    /// Most recent images in the output folder (newest first).
    static func recentImages(limit: Int = 300) -> [GalleryItem] {
        let root = AppPaths.output
        guard let e = FileManager.default.enumerator(
            at: root, includingPropertiesForKeys: [.contentModificationDateKey], options: [.skipsHiddenFiles])
        else { return [] }
        var files = [(URL, Date)]()
        for case let url as URL in e where url.pathExtension.lowercased() == "png" {
            let d = (try? url.resourceValues(forKeys: [.contentModificationDateKey]).contentModificationDate) ?? .distantPast
            files.append((url, d))
        }
        return files.sorted { $0.1 > $1.1 }.prefix(limit).map { GalleryItem(url: $0.0, record: readRecord($0.0)) }
    }

    /// Small cached thumbnails for the gallery strip.
    static let thumbCache = NSCache<NSURL, NSImage>()

    static func thumbnail(_ url: URL, maxPixel: Int = 256) -> NSImage? {
        if let t = thumbCache.object(forKey: url as NSURL) { return t }
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
            let cg = CGImageSourceCreateThumbnailAtIndex(src, 0, [
                kCGImageSourceCreateThumbnailFromImageAlways: true,
                kCGImageSourceThumbnailMaxPixelSize: maxPixel,
            ] as CFDictionary)
        else { return nil }
        let img = NSImage(cgImage: cg, size: NSSize(width: cg.width, height: cg.height))
        thumbCache.setObject(img, forKey: url as NSURL)
        return img
    }
}
