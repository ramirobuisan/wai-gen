import Foundation

// WAI-ANIMA's prompt handling, ported from stable-diffusion.cpp so the same prompt gives the same tokens:
// `(tag:1.2)` attention parsing (core/util.cpp), the Qwen2 byte-level BPE tokenizer (qwen2_tokenizer.cpp,
// bpe_tokenizer.cpp) and the T5 SentencePiece-unigram tokenizer (t5_unigram_tokenizer.cpp). The vocabularies
// (Resources/qwen2_merges.txt, Resources/t5_vocab.json) are the ones stable-diffusion.cpp embeds.

/// A1111-style prompt weights: "(word)" x1.1, "[word]" /1.1, "(word:1.3)", "\(" literals, BREAK.
/// Returns (text, weight) runs with neighbours of equal weight merged.
public enum PromptAttention {
    private static let token = try! NSRegularExpression(
        pattern: #"\\\(|\\\)|\\\[|\\\]|\\\\|\\|\(|\[|\)|\]|\bBREAK\b|[^\\()\[\]:B]{1,1024}|:|\bB"#)
    private static let breakRE = try! NSRegularExpression(pattern: #"\s*\bBREAK\b\s*"#)

    public static func parse(_ text: String) -> [(String, Float)] {
        var res: [(String, Float)] = []
        var round: [Int] = [], square: [Int] = []
        func multiply(_ start: Int, _ m: Float) { for p in start ..< res.count { res[p].1 *= m } }

        /// "+1.2)" -> (1.2, length incl. ")"), or nil
        func lexWeight(_ s: Substring) -> (Float, Int)? {
            var end = s.startIndex
            if end < s.endIndex, s[end] == "+" || s[end] == "-" { end = s.index(after: end) }
            while end < s.endIndex, s[end].isASCII, s[end].isNumber || s[end] == "." { end = s.index(after: end) }
            guard end < s.endIndex, s[end] == ")" else { return nil }
            let number = String(s[s.startIndex ..< end])
            // strtof must consume the whole number: rejects ".", "+." and "1.2.3"
            guard !number.isEmpty, let v = strtofExact(number), v.isFinite else { return nil }
            return (v, s.distance(from: s.startIndex, to: end) + 1)
        }

        var remaining = Substring(text)
        while true {
            let ns = String(remaining) as NSString
            guard let m = token.firstMatch(in: ns as String, range: NSRange(location: 0, length: ns.length)) else { break }
            let piece = ns.substring(with: m.range)
            let suffix = Substring(ns.substring(from: m.range.location + m.range.length))
            if piece == ":", let (w, len) = lexWeight(suffix) {
                if let start = round.popLast() { multiply(start, w) }
                remaining = suffix.dropFirst(len)
                continue
            }
            if piece == "(" {
                round.append(res.count)
            } else if piece == "[" {
                square.append(res.count)
            } else if piece == ")", let start = round.popLast() {
                multiply(start, 1.1)
            } else if piece == "]", let start = square.popLast() {
                multiply(start, 1 / Float(1.1))
            } else if piece == "\\(" {
                res.append(("(", 1))
            } else if breakRE.firstMatch(in: piece, range: NSRange(location: 0, length: (piece as NSString).length)) != nil {
                res.append(("BREAK", -1))
            } else {
                res.append((piece, 1))
            }
            remaining = suffix
        }
        for p in round { multiply(p, 1.1) }
        for p in square { multiply(p, 1 / Float(1.1)) }
        if res.isEmpty { res.append(("", 1)) }
        var i = 0
        while i + 1 < res.count {
            if res[i].1 == res[i + 1].1 {
                res[i].0 += res[i + 1].0
                res.remove(at: i + 1)
            } else {
                i += 1
            }
        }
        return res
    }

    private static func strtofExact(_ s: String) -> Float? {
        s.withCString { p -> Float? in
            var end: UnsafeMutablePointer<CChar>?
            let v = strtof(p, &end)
            guard let end, end - UnsafeMutablePointer(mutating: p) == strlen(p) else { return nil }
            return v
        }
    }
}

/// splitWithSpecialTokens (tokenize_util.cpp): pieces of `text` with the special tokens cut out on their own.
func splitWithSpecialTokens(_ text: String, _ specials: [String]) -> [String] {
    let bytes = Array(text.utf8)
    var result: [String] = []
    var pos = 0
    let specialBytes = specials.map { Array($0.utf8) }
    func find(_ needle: [UInt8], from: Int) -> Int? {
        guard !needle.isEmpty, needle.count <= bytes.count else { return nil }
        var i = from
        while i + needle.count <= bytes.count {
            if bytes[i] == needle[0], Array(bytes[i ..< i + needle.count]) == needle { return i }
            i += 1
        }
        return nil
    }
    while pos < bytes.count {
        var next = bytes.count
        var matched: [UInt8] = []
        for t in specialBytes {
            if let p = find(t, from: pos), p < next || (p == next && t.count > matched.count) {
                next = p
                matched = t
            }
        }
        if next > pos { result.append(String(decoding: bytes[pos ..< next], as: UTF8.self)) }
        if matched.isEmpty { break }
        result.append(String(decoding: matched, as: UTF8.self))
        pos = next + matched.count
    }
    return result
}

/// Qwen2 byte-level BPE (vocabulary = the 256 byte symbols + every merge + the special tokens).
public final class Qwen2Tokenizer {
    public static let padID: Int32 = 151643
    static let specials = [
        "<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|object_ref_start|>", "<|object_ref_end|>", "<|box_start|>",
        "<|box_end|>", "<|quad_start|>", "<|quad_end|>", "<|vision_start|>", "<|vision_end|>", "<|vision_pad|>",
        "<|image_pad|>", "<|video_pad|>", "<tool_call>", "</tool_call>", "<|fim_prefix|>", "<|fim_middle|>",
        "<|fim_suffix|>", "<|fim_pad|>", "<|repo_name|>", "<|file_sep|>", "<tool_response>", "</tool_response>",
        "<think>", "</think>", "<|boi_token|>", "<|bor_token|>", "<|eor_token|>", "<|bot_token|>", "<|tms_token|>",
    ]
    private var byteEncoder = [String](repeating: "", count: 256)
    private var encoder: [String: Int32] = [:]
    private var ranks: [String: Int] = [:]   // "a b" -> rank
    private let split = try! NSRegularExpression(
        pattern: #"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"#)

    public init(merges url: URL) throws {
        // bytes_to_unicode, in stable-diffusion.cpp's order (it decides the first 256 ids)
        var order: [(Int, Int)] = []
        var used = Set<Int>()
        for b in Array(33 ... 126) + Array(161 ... 172) + Array(174 ... 255) {
            used.insert(b)
            order.append((b, b))
        }
        var n = 0
        for b in 0 ..< 256 where !used.contains(b) {
            order.append((b, 256 + n))
            n += 1
        }
        var tokens: [String] = []
        for (b, u) in order {
            let s = String(Character(Unicode.Scalar(UInt32(u))!))
            byteEncoder[b] = s
            tokens.append(s)
        }
        let text = try String(contentsOf: url, encoding: .utf8)
        var lines = text.components(separatedBy: "\n")
        lines.removeLast()   // split_utf32 drops whatever follows the last newline
        var merges: [(String, String)] = []
        merges.reserveCapacity(lines.count)
        for line in lines {
            guard let sp = line.firstIndex(of: " ") else { merges.append((line, "")); continue }
            merges.append((String(line[..<sp]), String(line[line.index(after: sp)...])))
        }
        for m in merges { tokens.append(m.0 + m.1) }
        tokens.append(contentsOf: Self.specials)
        encoder.reserveCapacity(tokens.count)
        for (i, t) in tokens.enumerated() { encoder[t] = Int32(i) }   // later duplicates win, like std::map
        ranks.reserveCapacity(merges.count)
        for (i, m) in merges.enumerated() { ranks[m.0 + " " + m.1] = i }
    }

    private func bpe(_ token: [String]) -> [String] {
        var word = token
        guard word.count > 1 else { return word }
        while true {
            var best: (rank: Int, i: Int)? = nil
            for i in 0 ..< word.count - 1 {
                if let r = ranks[word[i] + " " + word[i + 1]], best == nil || r < best!.rank { best = (r, i) }
            }
            guard let b = best else { break }
            let first = word[b.i], second = word[b.i + 1]
            var merged: [String] = []
            var i = 0
            while i < word.count {
                if i < word.count - 1, word[i] == first, word[i + 1] == second {
                    merged.append(first + second)
                    i += 2
                } else {
                    merged.append(word[i])
                    i += 1
                }
            }
            word = merged
            if word.count == 1 { break }
        }
        return word
    }

    public func encode(_ text: String) -> [Int32] {
        var ids: [Int32] = []
        for piece in splitWithSpecialTokens(text, Self.specials) {
            if Self.specials.contains(piece) {
                ids.append(encoder[piece]!)
                continue
            }
            let ns = piece as NSString
            for m in split.matches(in: piece, range: NSRange(location: 0, length: ns.length)) where m.range.length > 0 {
                let tok = ns.substring(with: m.range)
                let symbols = tok.utf8.map { byteEncoder[Int($0)] }
                for p in bpe(symbols) { ids.append(encoder[p] ?? Self.padID) }
            }
        }
        return ids
    }
}

/// T5 SentencePiece unigram (Viterbi over the vocabulary's scores), as stable-diffusion.cpp runs it:
/// runs of spaces collapsed, Metaspace pre-tokenizer, unknown characters scored min - 10.
public final class T5UnigramTokenizer {
    public static let eosID: Int32 = 1
    static let pad = "<pad>", eos = "</s>", unk = "<unk>"
    private var pieces: [[UInt8]: Int32] = [:]
    private var scores: [Float] = []
    private var maxPieceBytes = 0
    private var minScore: Float = .greatestFiniteMagnitude
    private let unkID: Int32
    private let replacement: String
    private let addPrefixSpace: Bool

    public init(vocab url: URL) throws {
        guard let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any],
              let vocab = json["vocab"] as? [[Any]] else { throw CocoaError(.fileReadCorruptFile) }
        unkID = Int32((json["unk_id"] as? Int) ?? 2)
        replacement = (json["replacement"] as? String) ?? "\u{2581}"
        addPrefixSpace = (json["add_prefix_space"] as? Bool) ?? true
        scores.reserveCapacity(vocab.count)
        for (i, item) in vocab.enumerated() {
            var piece = (item[0] as? String) ?? ""
            if piece.isEmpty { piece = "<empty_token>" }
            let score = Float((item[1] as? NSNumber)?.doubleValue ?? 0)
            scores.append(score)
            minScore = min(minScore, score)
            let bytes = Array(piece.utf8)
            pieces[bytes] = Int32(i)
            maxPieceBytes = max(maxPieceBytes, bytes.count)
        }
    }

    private func pretokenize(_ input: String) -> String {
        // std::getline(ss, token, ' ') semantics: a trailing delimiter gives no empty last token
        var parts = input.components(separatedBy: " ")
        if input.isEmpty { parts = [] } else if input.hasSuffix(" ") { parts.removeLast() }
        var out = addPrefixSpace ? replacement : ""
        for (i, p) in parts.enumerated() { out += (i == 0 ? "" : replacement) + p }
        return out
    }

    private func viterbi(_ s: [UInt8]) -> [Int32] {
        guard !s.isEmpty else { return [] }
        let size = s.count
        let unkScore = minScore - 10
        var bestScore = [Float](repeating: 0, count: size + 1)
        var startsAt = [Int](repeating: -1, count: size + 1)
        var ids = [Int32](repeating: -1, count: size + 1)
        var start = 0
        while start < size {
            let lead = Int(s[start]) >> 4
            let charLen = min([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4][lead], size - start)
            let base = bestScore[start]
            var single = false
            var end = start + 1
            while end <= min(size, start + maxPieceBytes) {
                if let id = pieces[Array(s[start ..< end])] {
                    let cand = scores[Int(id)] + base
                    if startsAt[end] == -1 || cand > bestScore[end] {
                        bestScore[end] = cand
                        startsAt[end] = start
                        ids[end] = id
                    }
                    if end - start == charLen { single = true }
                }
                end += 1
            }
            if !single {
                let e = start + charLen
                let cand = unkScore + base
                if startsAt[e] == -1 || cand > bestScore[e] {
                    bestScore[e] = cand
                    startsAt[e] = start
                    ids[e] = unkID
                }
            }
            start += charLen
        }
        var out: [Int32] = []
        var e = size
        while e > 0 {
            out.append(ids[e])
            e = startsAt[e]
        }
        return out.reversed()
    }

    /// Token ids of `input` (no end-of-sequence token).
    public func encode(_ input: String) -> [Int32] {
        // normalize: runs of two or more spaces become one
        var normalized = input
        while normalized.contains("  ") { normalized = normalized.replacingOccurrences(of: "  ", with: " ") }
        guard !normalized.isEmpty else { return [] }
        var parts = splitWithSpecialTokens(normalized, [Self.pad, Self.eos, Self.unk])
        if parts.isEmpty { parts = [normalized] }
        var ids: [Int32] = []
        for p in parts {
            switch p {
            case Self.unk: ids.append(unkID)
            case Self.eos: ids.append(Self.eosID)
            case Self.pad: ids.append(0)
            default: ids.append(contentsOf: viterbi(Array(pretokenize(p).utf8)))
            }
        }
        return ids
    }
}

/// What WAI-ANIMA's conditioner feeds its models for one prompt: Qwen3 token ids (uniform weights) and
/// the T5 ids + per-token weights for the LLM adapter (ending with EOS).
public struct AnimaTokens: Sendable, Hashable {
    public var qwen: [Int32]
    public var t5: [Int32]
    public var t5Weights: [Float]
}

public final class AnimaTokenizer {
    public let qwen: Qwen2Tokenizer
    let t5: T5UnigramTokenizer

    public init() throws {
        guard let merges = Bundle.module.url(forResource: "qwen2_merges", withExtension: "txt", subdirectory: "Resources"),
              let vocab = Bundle.module.url(forResource: "t5_vocab", withExtension: "json", subdirectory: "Resources")
        else { throw CocoaError(.fileNoSuchFile) }
        qwen = try Qwen2Tokenizer(merges: merges)
        t5 = try T5UnigramTokenizer(vocab: vocab)
    }

    public func tokenize(_ text: String) -> AnimaTokens {
        let parsed = PromptAttention.parse(text)
        var q: [Int32] = []
        for (t, _) in parsed { q.append(contentsOf: qwen.encode(t)) }
        if q.isEmpty { q = [Qwen2Tokenizer.padID] }
        var ids: [Int32] = [], weights: [Float] = []
        for (t, w) in parsed {
            let tk = t5.encode(t)
            ids.append(contentsOf: tk)
            weights.append(contentsOf: [Float](repeating: w, count: tk.count))
        }
        ids.append(T5UnigramTokenizer.eosID)
        weights.append(1)
        return AnimaTokens(qwen: q, t5: ids, t5Weights: weights)
    }
}
