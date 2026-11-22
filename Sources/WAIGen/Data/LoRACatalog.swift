import Foundation

/// A LoRA file in the models/loras folder, with what its safetensors metadata says about it.
struct LoRAFile: Identifiable, Hashable {
    let relativePath: String     // inside models/loras
    let baseModel: String?       // e.g. "SDXL", "Illustrious", "Anima"
    let triggerTags: [String]    // most frequent training tags
    /// a preview image next to the file (`<name>.preview.jpeg`, as ComfyUI's LoRA managers save them, or `<name>.png`)
    var preview: URL? = nil
    var id: String { relativePath }
    var name: String { ((relativePath as NSString).lastPathComponent as NSString).deletingPathExtension }

    /// Rough family check so the inspector can flag LoRAs made for other model types.
    func fits(_ engine: EngineKind) -> Bool? {
        guard let b = baseModel?.lowercased() else { return nil }
        switch engine {
        case .sdxl: return b.contains("sdxl") || b.contains("illustrious") || b.contains("noob") || b.contains("pony")
        case .anima: return b.contains("anima") || b.contains("cosmos")
        }
    }
}

/// One LoRA slot in the settings.
struct LoRASelection: Codable, Hashable, Identifiable {
    var relativePath: String
    var strength: Double = 1
    var enabled = true
    var id: String { relativePath }
    var name: String { ((relativePath as NSString).lastPathComponent as NSString).deletingPathExtension }
}

enum LoRACatalog {
    static var folder: URL { AppPaths.models.appendingPathComponent("loras", isDirectory: true) }

    static func scan() -> [LoRAFile] {
        let root = folder
        guard let e = FileManager.default.enumerator(at: root, includingPropertiesForKeys: nil, options: [.skipsHiddenFiles])
        else { return [] }
        var out = [LoRAFile]()
        for case let url as URL in e where url.pathExtension.lowercased() == "safetensors" {
            let rel = url.path.replacingOccurrences(of: root.path + "/", with: "")
            let meta = metadata(url)
            out.append(LoRAFile(relativePath: rel, baseModel: baseModel(meta), triggerTags: topTags(meta),
                                preview: previewImage(for: url)))
        }
        return out.sorted { $0.relativePath.localizedStandardCompare($1.relativePath) == .orderedAscending }
    }

    /// `x.safetensors` -> the first of x.preview.jpeg / .jpg / .png / .webp, x.jpeg / .jpg / .png / .webp that exists.
    static func previewImage(for url: URL) -> URL? {
        let base = url.deletingPathExtension().path
        for suffix in [".preview.jpeg", ".preview.jpg", ".preview.png", ".preview.webp", ".jpeg", ".jpg", ".png", ".webp"] {
            let candidate = URL(fileURLWithPath: base + suffix)
            if FileManager.default.fileExists(atPath: candidate.path) { return candidate }
        }
        return nil
    }

    /// The "__metadata__" block of a safetensors header (no tensor data is read).
    static func metadata(_ url: URL) -> [String: String] {
        guard let h = try? FileHandle(forReadingFrom: url) else { return [:] }
        defer { try? h.close() }
        guard let lenData = try? h.read(upToCount: 8), lenData.count == 8 else { return [:] }
        let len = lenData.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) }.littleEndian
        guard len > 0, len < 100_000_000, let header = try? h.read(upToCount: Int(len)),
            let json = try? JSONSerialization.jsonObject(with: header) as? [String: Any],
            let meta = json["__metadata__"] as? [String: String]
        else { return [:] }
        return meta
    }

    static func baseModel(_ m: [String: String]) -> String? {
        if let a = m["modelspec.architecture"], !a.isEmpty {
            let l = a.lowercased()
            if l.contains("xl") { return "SDXL" }
            return a.split(separator: "/").first.map(String.init)
        }
        if let v = m["ss_base_model_version"], !v.isEmpty {
            return v.lowercased().contains("sdxl") ? "SDXL" : v
        }
        if let n = m["ss_sd_model_name"]?.lowercased() {
            if n.contains("illustrious") || n.contains("noob") || n.contains("xl") { return "SDXL" }
            if n.contains("anima") { return "Anima" }
        }
        return nil
    }

    static func topTags(_ m: [String: String], limit: Int = 8) -> [String] {
        guard let raw = m["ss_tag_frequency"], let data = raw.data(using: .utf8),
            let sets = try? JSONSerialization.jsonObject(with: data) as? [String: [String: Int]]
        else { return [] }
        var counts = [String: Int]()
        for (_, tags) in sets { for (t, c) in tags { counts[t.trimmingCharacters(in: .whitespaces), default: 0] += c } }
        return counts.sorted { $0.value > $1.value }.prefix(limit).map(\.key).filter { !$0.isEmpty }
    }
}
