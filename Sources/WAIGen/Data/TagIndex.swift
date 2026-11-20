import Foundation

struct TagSuggestion: Identifiable, Hashable {
    let tag: String        // display/insert form: spaces instead of underscores
    let count: Int
    let category: Int
    let matchedAlias: String?
    var id: String { tag }
}

/// Danbooru/e621 tag list for autocomplete (same file SAA uses), sorted by post count.
final class TagIndex: @unchecked Sendable {
    private struct Entry {
        let key: String     // lowercase, underscores
        let display: String
        let count: Int
        let category: Int
        let aliases: [String]
    }

    private var entries: [Entry] = []
    private let lock = NSLock()
    private(set) var isLoaded = false

    func load(from url: URL) throws {
        let text = try String(contentsOf: url, encoding: .utf8)
        var list = [Entry]()
        list.reserveCapacity(330_000)
        for line in text.split(whereSeparator: \.isNewline) {
            let f = splitCSVLine(String(line))
            guard f.count >= 3, let count = Int(f[2]) else { continue }
            let key = f[0].lowercased()
            let aliases = f.count > 3 && !f[3].isEmpty ? f[3].lowercased().split(separator: ",").map(String.init) : []
            list.append(Entry(key: key, display: f[0].replacingOccurrences(of: "_", with: " "), count: count,
                              category: Int(f[1]) ?? 0, aliases: aliases))
        }
        list.sort { $0.count > $1.count }
        lock.lock()
        entries = list
        isLoaded = true
        lock.unlock()
    }

    /// Danbooru post counts for these tags (lowercase, underscores), found by name or alias.
    func postCounts(for keys: Set<String>) -> [String: Int] {
        lock.lock()
        defer { lock.unlock() }
        var out = [String: Int]()
        for e in entries {   // most used first, so an alias keeps its most used tag's count
            if keys.contains(e.key), out[e.key] == nil { out[e.key] = e.count }
            for a in e.aliases where keys.contains(a) && out[a] == nil { out[a] = e.count }
        }
        return out
    }

    /// Up to `limit` tags starting with `prefix` (most used first), then alias matches.
    func complete(_ prefix: String, limit: Int = 8) -> [TagSuggestion] {
        let q = prefix.lowercased().trimmingCharacters(in: .whitespaces)
            .replacingOccurrences(of: "\\(", with: "(").replacingOccurrences(of: "\\)", with: ")")
            .replacingOccurrences(of: " ", with: "_")
        guard q.count >= 2 else { return [] }
        lock.lock()
        defer { lock.unlock() }
        var out = [TagSuggestion]()
        var seen = Set<String>()
        for e in entries where e.key.hasPrefix(q) {
            out.append(TagSuggestion(tag: e.display, count: e.count, category: e.category, matchedAlias: nil))
            seen.insert(e.key)
            if out.count >= limit { return out }
        }
        for e in entries where !seen.contains(e.key) {
            if let a = e.aliases.first(where: { $0.hasPrefix(q) }) {
                out.append(TagSuggestion(tag: e.display, count: e.count, category: e.category,
                                         matchedAlias: a.replacingOccurrences(of: "_", with: " ")))
                if out.count >= limit { break }
            }
        }
        return out
    }
}

/// `__name__` wildcards: replaced by a random line of `name.txt` in the wildcards folder.
enum Wildcards {
    static func available() -> [String] {
        let files = (try? FileManager.default.contentsOfDirectory(atPath: AppPaths.wildcards.path)) ?? []
        return files.filter { $0.hasSuffix(".txt") }.map { String($0.dropLast(4)) }.sorted()
    }

    static func expand(_ text: String, using rng: inout some RandomNumberGenerator, depth: Int = 0) -> String {
        guard depth < 5, text.contains("__") else { return text }
        var result = ""
        var rest = Substring(text)
        while let start = rest.range(of: "__") {
            result += rest[..<start.lowerBound]
            let after = rest[start.upperBound...]
            guard let end = after.range(of: "__") else {
                result += rest[start.lowerBound...]
                rest = ""
                break
            }
            let name = String(after[..<end.lowerBound])
            let file = AppPaths.wildcards.appendingPathComponent("\(name).txt")
            if let content = try? String(contentsOf: file, encoding: .utf8) {
                let lines = content.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }
                    .filter { !$0.isEmpty && !$0.hasPrefix("#") }
                result += lines.randomElement(using: &rng).map { expand($0, using: &rng, depth: depth + 1) } ?? ""
            } else {
                result += "__\(name)__"
            }
            rest = after[end.upperBound...]
        }
        return result + rest
    }

    /// Copy SAA's example wildcards on first run so the feature has something to show.
    static func seedFromSAA() {
        let dest = AppPaths.wildcards
        guard (try? FileManager.default.contentsOfDirectory(atPath: dest.path))?.isEmpty ?? true else { return }
        let src = AppPaths.saaData.appendingPathComponent("wildcards")
        for f in (try? FileManager.default.contentsOfDirectory(atPath: src.path)) ?? [] where f.hasSuffix(".txt") {
            try? FileManager.default.copyItem(at: src.appendingPathComponent(f), to: dest.appendingPathComponent(f))
        }
    }
}
