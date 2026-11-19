// Prompt parsing and SDXL text conditioning, matching ComfyUI's behaviour (comfy/sd1_clip.py):
// - `(text:1.2)` sets a weight, `(text)` multiplies by 1.1 per nesting level
// - `\(` and `\)` are literal parentheses (SAA writes character tags like `rebecca \(cyberpunk\)`)
// - prompts are split into 75-token chunks, each wrapped as BOS + tokens + EOS and padded to 77
// - weighted tokens are interpolated against the empty-prompt encoding
// - SDXL uses the penultimate hidden layer of CLIP-L and CLIP-G (concatenated, 2048 wide) and
//   the projected pooled CLIP-G output of the first chunk

import Foundation
import MLX

/// One piece of a prompt with its attention weight.
public struct WeightedText: Equatable {
    public var text: String
    public var weight: Float
}

public enum PromptParser {
    private static let openEsc = "\u{0}\u{1}"
    private static let closeEsc = "\u{0}\u{2}"

    /// Split a prompt into (text, weight) pieces with ComfyUI semantics.
    public static func parse(_ prompt: String) -> [WeightedText] {
        let escaped = prompt
            .replacingOccurrences(of: "\\(", with: openEsc)
            .replacingOccurrences(of: "\\)", with: closeEsc)
        return weights(escaped, current: 1.0).map {
            WeightedText(
                text: $0.text.replacingOccurrences(of: openEsc, with: "(")
                    .replacingOccurrences(of: closeEsc, with: ")"),
                weight: $0.weight)
        }
    }

    /// Top-level split into plain runs and balanced `( ... )` groups (comfy `parse_parentheses`).
    static func parentheses(_ s: String) -> [String] {
        var result = [String]()
        var current = ""
        var level = 0
        for ch in s {
            if ch == "(" {
                if level == 0 {
                    if !current.isEmpty { result.append(current) }
                    current = "("
                } else {
                    current.append(ch)
                }
                level += 1
            } else if ch == ")" {
                level -= 1
                if level == 0 {
                    current.append(")")
                    result.append(current)
                    current = ""
                } else {
                    current.append(ch)
                }
            } else {
                current.append(ch)
            }
        }
        if !current.isEmpty { result.append(current) }
        return result
    }

    static func weights(_ s: String, current: Float) -> [WeightedText] {
        var out = [WeightedText]()
        for part in parentheses(s) {
            if part.count >= 2, part.first == "(", part.last == ")" {
                var inner = String(part.dropFirst().dropLast())
                var weight = current * 1.1
                if let colon = inner.lastIndex(of: ":"), colon != inner.startIndex,
                    let w = Float(inner[inner.index(after: colon)...].trimmingCharacters(in: .whitespaces))
                {
                    weight = w
                    inner = String(inner[..<colon])
                }
                out += weights(inner, current: weight)
            } else {
                out.append(WeightedText(text: part, weight: current))
            }
        }
        return out
    }
}

extension CLIPTokenizer {
    /// BPE token ids for `text` without BOS/EOS and without truncation.
    func wordTokens(_ text: String) -> [Int] {
        let clean = text.lowercased().replacing(#/\s+/#, with: " ")
        let words = clean.matches(of: pattern).map { String($0.description) }
        return words.flatMap { bpe(text: $0) }.compactMap { vocabulary[$0] }
    }
}

/// A prompt tokenised into 77-token chunks with per-token weights.
struct TokenizedPrompt {
    var chunks: [[Int]]       // each 77 long, padded with EOS (CLIP-L padding)
    var weights: [[Float]]    // per token
    var hasWeights: Bool { weights.contains { $0.contains { $0 != 1 } } }

    static let contentLength = 75

    init(_ prompt: String, tokenizer: CLIPTokenizer) {
        // tokens grouped per word so chunk breaks fall between words where possible
        var words = [[(Int, Float)]]()
        for piece in PromptParser.parse(prompt) {
            for word in piece.text.replacingOccurrences(of: "\n", with: " ").split(separator: " ") {
                let ids = tokenizer.wordTokens(String(word))
                if !ids.isEmpty { words.append(ids.map { ($0, piece.weight) }) }
            }
        }

        var chunkTokens = [[(Int, Float)]]()
        var current = [(Int, Float)]()
        for var word in words {
            while !word.isEmpty {
                let room = Self.contentLength - current.count
                if word.count <= room {
                    current += word
                    word = []
                } else if word.count > Self.contentLength / 2 || current.isEmpty {
                    // very long word: split it across chunks (ComfyUI does the same)
                    current += word.prefix(room)
                    word = Array(word.dropFirst(room))
                    chunkTokens.append(current)
                    current = []
                } else {
                    chunkTokens.append(current)
                    current = []
                }
            }
        }
        if !current.isEmpty || chunkTokens.isEmpty { chunkTokens.append(current) }

        let bos = tokenizer.bosToken
        let eos = tokenizer.eosToken
        chunks = chunkTokens.map { c in
            [bos] + c.map(\.0) + [eos] + Array(repeating: eos, count: Self.contentLength - c.count)
        }
        weights = chunkTokens.map { c in
            [1] + c.map(\.1) + [1] + Array(repeating: 1, count: Self.contentLength - c.count)
        }
    }
}

/// SDXL conditioning for one prompt: cross-attention context [1, 77*chunks, 2048] and pooled [1, 1280].
struct SDXLConditioning {
    var context: MLXArray
    var pooled: MLXArray
}

func encodeSDXL(
    _ prompt: String, tokenizer: CLIPTokenizer, clipL: CLIPTextModel, clipG: CLIPTextModel
) -> SDXLConditioning {
    let tp = TokenizedPrompt(prompt, tokenizer: tokenizer)
    let eos = tokenizer.eosToken
    let empty = TokenizedPrompt("", tokenizer: tokenizer).chunks[0]

    func run(_ model: CLIPTextModel, padWithZero: Bool) -> (hidden: MLXArray, pooled: MLXArray) {
        // CLIP-G pads with 0 instead of EOS; the first EOS stays (pooled output is taken there)
        func padFix(_ c: [Int]) -> [Int] {
            guard padWithZero, let first = c.firstIndex(of: eos) else { return c }
            return c.enumerated().map { i, t in i > first && t == eos ? 0 : t }
        }
        var batch = tp.chunks.map(padFix)
        if tp.hasWeights { batch.append(padFix(empty)) }
        let ids = MLXArray(batch.flatMap { $0.map(Int32.init) }).reshaped(batch.count, 77)
        let out = model(ids)
        var hidden = out.hiddenStates[out.hiddenStates.count - 2]   // penultimate layer
        if tp.hasWeights {
            let n = tp.chunks.count
            let zEmpty = hidden[n ..< (n + 1)]
            let w = MLXArray(tp.weights.flatMap { $0 }).reshaped(n, 77, 1).asType(hidden.dtype)
            hidden = (hidden[0 ..< n] - zEmpty) * w + zEmpty
        }
        let n = tp.chunks.count
        let seq = hidden[0 ..< n].reshaped(1, n * 77, hidden.dim(-1))
        return (seq, out.pooledOutput[0 ..< 1])
    }

    let l = run(clipL, padWithZero: false)
    let g = run(clipG, padWithZero: true)
    return SDXLConditioning(context: concatenated([l.hidden, g.hidden], axis: -1), pooled: g.pooled)
}

/// Make cond/uncond the same sequence length by repeating the shorter (ComfyUI repeats to the LCM).
func matchContextLength(_ a: MLXArray, _ b: MLXArray) -> (MLXArray, MLXArray) {
    let la = a.dim(1) / 77
    let lb = b.dim(1) / 77
    if la == lb { return (a, b) }
    func gcd(_ x: Int, _ y: Int) -> Int { y == 0 ? x : gcd(y, x % y) }
    let lcm = la / gcd(la, lb) * lb
    return (tiled(a, repetitions: [1, lcm / la, 1]), tiled(b, repetitions: [1, lcm / lb, 1]))
}
