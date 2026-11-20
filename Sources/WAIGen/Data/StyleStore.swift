import AppKit
import Foundation
import Observation

/// An artist style from ThetaCursed's Illustrious & NoobAI Style Explorer (MIT,
/// github.com/ThetaCursed/Illustrious-NoobAI-Style-Explorer): the artist's Danbooru tag, a sample made with
/// Illustrious XL from one fixed prompt (so only the style changes), and how many of their works Danbooru has.
struct StyleEntry: Identifiable, Hashable, Sendable {
    let id: String
    let tag: String           // unescaped Danbooru artist tag, e.g. "hammer (sunset beach)"
    let works: Int
    let folder: Int
    /// higher = more distinctive (nil for artists the explorer didn't score)
    let uniqueness: Double?
}

/// The style list (16,006 artists, one 2.3 MB file downloaded once) and its samples, each downloaded the first
/// time it's shown and kept in Application Support/WAI Gen/data/styles.
@MainActor @Observable
final class StyleStore {
    enum State: Equatable { case idle, loading, ready, failed(String) }
    enum Sort: String, CaseIterable, Identifiable {
        case works, distinctive, name, shuffled
        var id: String { rawValue }
        var label: String {
            switch self {
            case .works: "Popularity"
            case .distinctive: "Most distinctive"
            case .name: "Name"
            case .shuffled: "Shuffled"
            }
        }
    }

    /// The explorer's files at a fixed commit, so the list and the samples always match.
    nonisolated static let source = "https://raw.githubusercontent.com/ThetaCursed/Illustrious-NoobAI-Style-Explorer/"
        + "ef220b46f2b67f5e7f82291e92a6a8ebac9fb781"
    nonisolated static let page = URL(string: "https://github.com/ThetaCursed/Illustrious-NoobAI-Style-Explorer")!
    nonisolated static let listFile = "illustrious_styles.js"
    private static let favoritesKey = "favoriteStyles"
    private static let sortKey = "styleSort"

    private(set) var state: State = .idle
    private(set) var styles: [StyleEntry] = []
    private(set) var favorites: Set<String> = Set(UserDefaults.standard.stringArray(forKey: StyleStore.favoritesKey) ?? [])
    var sort: Sort = Sort(rawValue: UserDefaults.standard.string(forKey: StyleStore.sortKey) ?? "") ?? .works {
        didSet {
            UserDefaults.standard.set(sort.rawValue, forKey: Self.sortKey)
            if sort == .shuffled { shuffleSeed = UInt64.random(in: 1 ... .max) }
            order()
        }
    }
    @ObservationIgnored private var shuffleSeed = UInt64.random(in: 1 ... .max)
    @ObservationIgnored private let cache: NSCache<NSString, NSImage> = {
        let c = NSCache<NSString, NSImage>()
        c.countLimit = 500
        return c
    }()

    func loadIfNeeded() {
        switch state {
        case .idle, .failed: break
        case .loading, .ready: return
        }
        state = .loading
        Task {
            do {
                let url = try await AppPaths.ensureDataFile(Self.listFile, url: Self.source + "/app/data.js")
                do {
                    styles = try await Task.detached(priority: .userInitiated) { try Self.parse(url) }.value
                } catch {
                    try? FileManager.default.removeItem(at: url)   // a broken download: fetch it again next time
                    throw error
                }
                order()
                state = .ready
            } catch {
                state = .failed(error.localizedDescription)
            }
        }
    }

    func shuffle() {
        shuffleSeed = UInt64.random(in: 1 ... .max)
        if sort == .shuffled { order() } else { sort = .shuffled }
    }

    func search(_ query: String) -> [StyleEntry] {
        let q = query.trimmingCharacters(in: .whitespaces).lowercased()
        guard !q.isEmpty else { return styles }
        return styles.filter { $0.tag.lowercased().contains(q) }
    }

    func toggleFavorite(_ tag: String) {
        if favorites.contains(tag) { favorites.remove(tag) } else { favorites.insert(tag) }
        UserDefaults.standard.set(Array(favorites).sorted(), forKey: Self.favoritesKey)
    }

    func cachedSample(_ entry: StyleEntry) -> NSImage? { cache.object(forKey: entry.id as NSString) }

    /// The style's sample: from memory, from disk, or downloaded (cancelled with the calling task, e.g. when
    /// the cell scrolls away).
    func sample(_ entry: StyleEntry) async -> NSImage? {
        if let img = cache.object(forKey: entry.id as NSString) { return img }
        guard let data = await Self.sampleData(entry), let img = NSImage(data: data) else { return nil }
        cache.setObject(img, forKey: entry.id as NSString)
        return img
    }

    private func order() {
        switch sort {
        case .works: styles.sort { ($0.works, $1.tag) > ($1.works, $0.tag) }
        case .distinctive: styles.sort { ($0.uniqueness ?? -1, $1.tag) > ($1.uniqueness ?? -1, $0.tag) }
        case .name: styles.sort { nameSortKey($0.tag).localizedStandardCompare(nameSortKey($1.tag)) == .orderedAscending }
        case .shuffled:
            var rng = SeededRNG(state: shuffleSeed)
            styles.sort { $0.id < $1.id }
            styles.shuffle(using: &rng)
        }
    }

    // MARK: files

    nonisolated static var sampleFolder: URL {
        let url = AppPaths.data.appendingPathComponent("styles", isDirectory: true)
        try? FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    /// At most six downloads at a time; queued ones that are cancelled never start.
    nonisolated private static let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.httpMaximumConnectionsPerHost = 6
        config.urlCache = nil
        config.timeoutIntervalForRequest = 30
        return URLSession(configuration: config)
    }()

    nonisolated static func sampleData(_ entry: StyleEntry) async -> Data? {
        let file = sampleFolder.appendingPathComponent("\(entry.id).webp")
        if let data = try? Data(contentsOf: file) { return data }
        guard let url = URL(string: "\(source)/images/\(entry.folder)/\(entry.id).webp"),
            let (data, response) = try? await session.data(from: url),
            (response as? HTTPURLResponse)?.statusCode == 200, !data.isEmpty
        else { return nil }
        try? data.write(to: file, options: .atomic)
        return data
    }

    /// data.js is `const galleryData = [ {id, name, post_count, p, uniqueness_score}, … ];`
    nonisolated static func parse(_ url: URL) throws -> [StyleEntry] {
        struct Raw: Decodable {
            let id: String
            let name: String
            let post_count: Int?
            let p: Int
            let uniqueness_score: Double?
        }
        let text = try Data(contentsOf: url)
        guard let start = text.firstIndex(of: UInt8(ascii: "[")), let end = text.lastIndex(of: UInt8(ascii: "]")),
            start < end
        else { throw CocoaError(.fileReadCorruptFile) }
        let raws = try JSONDecoder().decode([Raw].self, from: text[start ... end])
        var seen = Set<String>()
        return raws.compactMap { r in
            let tag = r.name.replacingOccurrences(of: "\\(", with: "(").replacingOccurrences(of: "\\)", with: ")")
                .trimmingCharacters(in: .whitespaces)
            guard !tag.isEmpty, seen.insert(r.id).inserted else { return nil }
            return StyleEntry(id: r.id, tag: tag, works: r.post_count ?? 0, folder: r.p, uniqueness: r.uniqueness_score)
        }
    }
}
