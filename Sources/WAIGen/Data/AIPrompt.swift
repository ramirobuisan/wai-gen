import Foundation

/// Turns a plain-language scene description into Danbooru tags with a local OpenAI-compatible
/// server (MLX-Serve by default).
enum AIPrompt {
    static let urlKey = "aiPromptBaseURL"
    static let modelKey = "aiPromptModel"
    static let defaultURL = "http://127.0.0.1:11234/v1"

    static var baseURL: String {
        get { UserDefaults.standard.string(forKey: urlKey) ?? defaultURL }
        set { UserDefaults.standard.set(newValue, forKey: urlKey) }
    }

    /// Models the server had in memory at the last `models()` call.
    nonisolated(unsafe) static var loadedModels: Set<String> = []

    static var model: String? {
        get { UserDefaults.standard.string(forKey: modelKey) }
        set { UserDefaults.standard.set(newValue, forKey: modelKey) }
    }

    /// What the AI writes: Danbooru tags (WAI v16, NoobAI-XL, WAI-ANIMA) or plain sentences (Z-Image, whose Qwen3
    /// text encoder reads natural language).
    enum Style: String, CaseIterable, Identifiable {
        case tags, sentences
        var id: String { rawValue }
        var label: String { self == .tags ? "Tags" : "Sentences" }
        static func `default`(for engine: EngineKind) -> Style { engine == .zimage ? .sentences : .tags }
    }

    static let sentencePrompt = """
        You write prompts for an anime image model whose text encoder understands plain English.
        Turn the user's scene description into 2 to 4 concrete, visual English sentences describing one
        anime-style illustration: the subject, pose and expression, clothing, setting, lighting, mood and the
        camera framing. Do not add character names the user didn't give, quality words (masterpiece, best quality),
        headings, lists or any explanation. Reply with the sentences only.
        """

    static let systemPrompt = """
        You write prompts for an anime image model that was trained on Danbooru tags.
        Turn the user's scene description into ONE line of 8 to 20 English Danbooru tags separated by commas:
        framing and composition, pose, expression, clothing, setting, lighting and mood.
        Use lowercase tags with spaces instead of underscores. Do not include character names,
        quality tags (masterpiece, best quality) or any explanation. Reply with the tag line only.
        """

    struct ModelList: Decodable {
        struct M: Decodable { let id: String; let capabilities: [String]?; let loaded: Bool?; let state: String? }
        let data: [M]
    }

    /// Chat models on the server, the ones already in memory first (MLX-Serve reports `loaded`/`state`),
    /// so the default pick doesn't make the server load a second large model.
    static func models() async throws -> [String] {
        guard let url = URL(string: baseURL.trimmingCharacters(in: .whitespaces) + "/models") else { throw URLError(.badURL) }
        var req = URLRequest(url: url)
        req.timeoutInterval = 8
        let (data, _) = try await URLSession.shared.data(for: req)
        // MLX-Serve also lists image models; keep chat models (servers without capabilities list everything)
        let chat = try JSONDecoder().decode(ModelList.self, from: data).data
            .filter { $0.capabilities?.contains("chat") ?? true }
        loadedModels = Set(chat.filter { $0.loaded == true || $0.state == "loaded" }.map(\.id))
        return chat.filter { loadedModels.contains($0.id) }.map(\.id) + chat.filter { !loadedModels.contains($0.id) }.map(\.id)
    }

    /// Returns a tag line ready for the prompt (parentheses escaped for the weight syntax), or plain sentences.
    static func write(for brief: String, model: String, style: Style = .tags) async throws -> String {
        guard let url = URL(string: baseURL.trimmingCharacters(in: .whitespaces) + "/chat/completions") else {
            throw URLError(.badURL)
        }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.timeoutInterval = 120
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let body: [String: Any] = [
            "model": model,
            "messages": [
                ["role": "system", "content": style == .tags ? systemPrompt : sentencePrompt],
                ["role": "user", "content": brief + "\n/no_think"],
            ],
            "temperature": 0.7,
            "max_tokens": 400,
            "stream": false,
            "chat_template_kwargs": ["enable_thinking": false],
        ]
        req.httpBody = try JSONSerialization.data(withJSONObject: body)
        let (data, response) = try await URLSession.shared.data(for: req)
        if let http = response as? HTTPURLResponse, http.statusCode != 200 {
            let msg = String(data: data, encoding: .utf8)?.prefix(300) ?? ""
            throw NSError(domain: "AIPrompt", code: http.statusCode,
                          userInfo: [NSLocalizedDescriptionKey: "The server answered \(http.statusCode): \(msg)"])
        }
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let choices = json["choices"] as? [[String: Any]],
            let message = choices.first?["message"] as? [String: Any],
            let content = message["content"] as? String
        else { throw NSError(domain: "AIPrompt", code: 0, userInfo: [NSLocalizedDescriptionKey: "Unexpected reply from the server"]) }
        return style == .tags ? clean(content) : cleanSentences(content)
    }

    /// Strip reasoning blocks, fences, quotes and line breaks: one paragraph of sentences, text as written.
    static func cleanSentences(_ text: String) -> String {
        var t = text.replacing(#/(?s)<think>.*?</think>/#, with: "")
        t = t.replacingOccurrences(of: "```", with: "")
        return t.split(whereSeparator: \.isNewline)
            .map { $0.trimmingCharacters(in: .whitespaces).trimmingCharacters(in: CharacterSet(charactersIn: "\"'`")) }
            .filter { !$0.isEmpty }
            .joined(separator: " ")
    }

    /// Strip reasoning blocks and chatter, keep one comma-separated tag line.
    static func clean(_ text: String) -> String {
        var t = text.replacing(#/(?s)<think>.*?</think>/#, with: "")
        t = t.replacingOccurrences(of: "```", with: "")
        let line = t.split(whereSeparator: \.isNewline).map { $0.trimmingCharacters(in: .whitespaces) }
            .last { $0.contains(",") } ?? t.trimmingCharacters(in: .whitespacesAndNewlines)
        let tags = line.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines).trimmingCharacters(in: CharacterSet(charactersIn: ".\"'`")) }
            .filter { !$0.isEmpty }
            .map { $0.replacingOccurrences(of: "_", with: " ") }
            .map(escapeTag)
        var seen = Set<String>()
        return tags.filter { seen.insert($0.lowercased()).inserted }.joined(separator: ", ")
    }
}
