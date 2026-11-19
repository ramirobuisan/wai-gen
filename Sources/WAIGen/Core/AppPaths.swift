import Foundation

/// Folders the app reads and writes. Model and output folders are user settings.
enum AppPaths {
    static let modelsKey = "modelsFolder"
    static let outputKey = "outputFolder"

    static var support: URL {
        let url = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("WAI Gen", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Character lists, thumbnails and the tag list.
    static var data: URL {
        let url = support.appendingPathComponent("data", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    static var wildcards: URL {
        let url = support.appendingPathComponent("wildcards", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// Where the WAI model files live (same layout as ComfyUI's models folder).
    static var models: URL {
        get {
            if let p = UserDefaults.standard.string(forKey: modelsKey) { return URL(fileURLWithPath: p) }
            // first existing ComfyUI-style models folder (change it in Settings)
            let home = FileManager.default.homeDirectoryForCurrentUser
            let candidates = ["ComfyUI-Shared/models", "ComfyUI/models", "Documents/ComfyUI/models"].map {
                home.appendingPathComponent($0, isDirectory: true)
            }
            return candidates.first { FileManager.default.fileExists(atPath: $0.path) } ?? candidates[0]
        }
        set { UserDefaults.standard.set(newValue.path, forKey: modelsKey) }
    }

    static var output: URL {
        get {
            if let p = UserDefaults.standard.string(forKey: outputKey) { return URL(fileURLWithPath: p) }
            return FileManager.default.urls(for: .picturesDirectory, in: .userDomainMask)[0]
                .appendingPathComponent("WAI Gen", isDirectory: true)
        }
        set { UserDefaults.standard.set(newValue.path, forKey: outputKey) }
    }

    /// YOLOv8 face detector for face repair (converted with scripts/convert_yolov8_safe.py).
    static var faceDetector: URL { models.appendingPathComponent("detectors/face_yolov8m.safetensors") }

    /// Character Select SAA's data folder, used to seed our data without downloading.
    static var saaData: URL {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let candidates = ["Developer/character_select_stand_alone_app/data", "character_select_stand_alone_app/data",
                          "Documents/character_select_stand_alone_app/data"].map { home.appendingPathComponent($0) }
        return candidates.first { FileManager.default.fileExists(atPath: $0.path) } ?? candidates[0]
    }

    static let tagListName = "danbooru_e621_merged.csv"
    static let tagListURL =
        "https://raw.githubusercontent.com/DraconicDragon/dbr-e621-lists-archive/refs/heads/main/tag-lists/danbooru_e621_merged/danbooru_e621_merged_2026-04-01_pt20-ia-dd-ed-spc.csv"

    /// Make sure `name` exists in the data folder: copy it from SAA if present, else download.
    static func ensureDataFile(_ name: String, url: String) async throws -> URL {
        let dest = data.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: dest.path) { return dest }
        let saa = saaData.appendingPathComponent(name)
        if FileManager.default.fileExists(atPath: saa.path) {
            try FileManager.default.copyItem(at: saa, to: dest)
            return dest
        }
        guard let remote = URL(string: url) else { throw URLError(.badURL) }
        let (tmp, response) = try await URLSession.shared.download(from: remote)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            throw URLError(.badServerResponse)
        }
        try? FileManager.default.removeItem(at: dest)
        try FileManager.default.moveItem(at: tmp, to: dest)
        return dest
    }
}
