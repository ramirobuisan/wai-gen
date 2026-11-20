import AppKit
import CryptoKit
import Foundation
import Observation

struct CharacterEntry: Identifiable, Hashable {
    let tag: String        // unescaped Danbooru tag, e.g. "rebecca (cyberpunk)"
    let localName: String  // name from the dataset (Chinese)
    var id: String { tag }
    /// how the tag list (danbooru_e621_merged.csv) writes it
    var tagKey: String { tag.lowercased().replacingOccurrences(of: " ", with: "_") }
    var thumbKey: String { Insecure.MD5.hash(data: Data(escapeTag(tag).utf8)).map { String(format: "%02x", $0) }.joined() }
}

/// Character lists from SAA's dataset: names, thumbnails (gzip+base64 WebP keyed by md5 of the
/// escaped tag) and tag-assist extras.
@MainActor @Observable
final class CharacterStore {
    enum State: Equatable { case idle, loading(String), ready, failed(String) }
    enum Sort: String, CaseIterable, Identifiable {
        case popularity, name
        var id: String { rawValue }
        var label: String { self == .popularity ? "Popularity" : "Name" }
    }

    private(set) var list: CharacterListID?
    private(set) var state: State = .idle
    private(set) var characters: [CharacterEntry] = []
    private(set) var favorites: Set<String> = Set(UserDefaults.standard.stringArray(forKey: "favoriteCharacters") ?? [])
    /// Danbooru posts per character tag (from the tag list, once it has loaded)
    private(set) var popularity: [String: Int] = [:]
    var sort: Sort = Sort(rawValue: UserDefaults.standard.string(forKey: "characterSort") ?? "") ?? .popularity {
        didSet {
            UserDefaults.standard.set(sort.rawValue, forKey: "characterSort")
            order()
        }
    }
    /// the tag list, for popularity (set by AppModel)
    @ObservationIgnored weak var tagIndex: TagIndex?

    @ObservationIgnored private var thumbs: [String: String] = [:]
    @ObservationIgnored private var tagAssist: [String: String] = [:]
    @ObservationIgnored private let cache = NSCache<NSString, NSImage>()
    @ObservationIgnored private var loadTask: Task<Void, Never>?

    func load(_ id: CharacterListID) {
        guard id != list || state != .ready else { return }
        list = id
        loadTask?.cancel()
        state = .loading("Loading \(id.label) characters…")
        loadTask = Task {
            do {
                var urls = [URL]()
                for f in id.files {
                    if !FileManager.default.fileExists(atPath: AppPaths.data.appendingPathComponent(f.name).path) {
                        state = .loading("Downloading \(id.label) character list…")
                    }
                    urls.append(try await AppPaths.ensureDataFile(f.name, url: f.url))
                }
                let (chars, assist, thumbs) = try await Task.detached(priority: .userInitiated) {
                    (try Self.parseCSV(urls[0]), try Self.parseStringMap(urls[1]), try Self.parseStringMap(urls[2]))
                }.value
                guard !Task.isCancelled, list == id else { return }
                self.characters = chars
                self.tagAssist = assist
                self.thumbs = thumbs
                self.cache.removeAllObjects()
                self.order()
                self.state = .ready
                self.refreshPopularity()
            } catch {
                if !Task.isCancelled { state = .failed(error.localizedDescription) }
            }
        }
    }

    func toggleFavorite(_ tag: String) {
        if favorites.contains(tag) { favorites.remove(tag) } else { favorites.insert(tag) }
        UserDefaults.standard.set(Array(favorites).sorted(), forKey: "favoriteCharacters")
    }

    func posts(_ entry: CharacterEntry) -> Int? { popularity[entry.tagKey] }

    /// Looks up every character's post count in the tag list (call again once the tag list has loaded).
    func refreshPopularity() {
        guard let index = tagIndex, index.isLoaded, !characters.isEmpty else { return }
        let keys = Set(characters.map(\.tagKey))
        Task {
            let counts = await Task.detached(priority: .utility) { index.postCounts(for: keys) }.value
            guard keys == Set(characters.map(\.tagKey)) else { return }   // the list changed meanwhile
            popularity = counts
            order()
        }
    }

    private func order() {
        switch sort {
        case .name:
            characters.sort { nameSortKey($0.tag).localizedStandardCompare(nameSortKey($1.tag)) == .orderedAscending }
        case .popularity:
            let p = popularity
            characters.sort {
                let a = p[$0.tagKey] ?? 0, b = p[$1.tagKey] ?? 0
                return a != b ? a > b : $0.tag.localizedStandardCompare($1.tag) == .orderedAscending
            }
        }
    }

    /// Extra tags SAA's "tag assist" adds for a character (e.g. gender for ambiguous names).
    func assistTags(for tag: String) -> String? {
        guard let v = tagAssist[tag], !v.isEmpty else { return nil }
        return v
    }

    func search(_ query: String) -> [CharacterEntry] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return characters }
        return characters.filter { $0.tag.lowercased().contains(q) || $0.localName.contains(q) }
    }

    /// Thumbnail for a character, decoded on demand and cached.
    func thumbnail(for entry: CharacterEntry) -> NSImage? {
        let key = entry.thumbKey
        if let img = cache.object(forKey: key as NSString) { return img }
        guard let b64 = thumbs[key], let gz = Data(base64Encoded: b64), let webp = gunzip(gz),
            let img = NSImage(data: webp)
        else { return nil }
        cache.setObject(img, forKey: key as NSString)
        return img
    }

    // MARK: parsing

    nonisolated static func parseCSV(_ url: URL) throws -> [CharacterEntry] {
        let text = try String(contentsOf: url, encoding: .utf8)
        var out = [CharacterEntry]()
        for line in text.split(whereSeparator: \.isNewline) {
            var line = String(line)
            if line.hasPrefix("\u{FEFF}") { line.removeFirst() }
            // "local name,tag" — the tag is the last field and may itself contain commas only if quoted
            let fields = splitCSVLine(line)
            guard fields.count >= 2 else { continue }
            let tag = fields[fields.count - 1].trimmingCharacters(in: .whitespaces)
            let name = fields[0 ..< fields.count - 1].joined(separator: ",")
            if !tag.isEmpty { out.append(CharacterEntry(tag: tag, localName: name)) }
        }
        return out
    }

    nonisolated static func parseStringMap(_ url: URL) throws -> [String: String] {
        let obj = try JSONSerialization.jsonObject(with: Data(contentsOf: url, options: .mappedIfSafe))
        return (obj as? [String: Any])?.compactMapValues { $0 as? String } ?? [:]
    }
}

/// For sorting by name: ".sin" sorts as "sin", not before "a".
func nameSortKey(_ name: String) -> Substring { name.drop { !$0.isLetter && !$0.isNumber } }

func splitCSVLine(_ line: String) -> [String] {
    var fields = [String]()
    var current = ""
    var quoted = false
    for ch in line {
        if ch == "\"" { quoted.toggle() } else if ch == ",", !quoted {
            fields.append(current)
            current = ""
        } else { current.append(ch) }
    }
    fields.append(current)
    return fields
}

/// Decompress a gzip member (RFC 1952) using Foundation's raw-DEFLATE decoder.
func gunzip(_ data: Data) -> Data? {
    let b = [UInt8](data)
    guard b.count > 18, b[0] == 0x1F, b[1] == 0x8B, b[2] == 8 else { return nil }
    let flags = b[3]
    var i = 10
    if flags & 4 != 0 { guard i + 2 <= b.count else { return nil }; i += 2 + Int(b[i]) + Int(b[i + 1]) << 8 }
    if flags & 8 != 0 { while i < b.count, b[i] != 0 { i += 1 }; i += 1 }
    if flags & 16 != 0 { while i < b.count, b[i] != 0 { i += 1 }; i += 1 }
    if flags & 2 != 0 { i += 2 }
    guard i < b.count - 8 else { return nil }
    let deflate = Data(b[i ..< (b.count - 8)])
    return try? (deflate as NSData).decompressed(using: .zlib) as Data
}
