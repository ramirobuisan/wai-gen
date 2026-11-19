import SwiftUI

/// Popover: describe a scene, get Danbooru tags from the local LLM, insert them into the prompt.
struct AIPromptView: View {
    @Environment(AppModel.self) private var model
    @State private var brief = ""
    @State private var result = ""
    @State private var models: [String] = []
    @State private var selected = AIPrompt.model ?? ""
    @State private var busy = false
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("AI prompt").font(.headline)
            TextField("Describe the scene", text: $brief, prompt: Text("She sits on a rooftop at night, smiling, city lights behind her"),
                      axis: .vertical)
                .lineLimit(2 ... 5)
                .textFieldStyle(.roundedBorder)
                .onSubmit(run)
            HStack {
                Picker("Model", selection: $selected) {
                    if models.isEmpty { Text(selected.isEmpty ? "No models" : selected).tag(selected) }
                    ForEach(models, id: \.self) { Text($0).tag($0) }
                }
                .labelsHidden()
                .frame(maxWidth: 300)
                Spacer()
                if busy { ProgressView().controlSize(.small) }
                Button("Write tags", action: run)
                    .keyboardShortcut(.return, modifiers: .command)
                    .disabled(busy || brief.trimmingCharacters(in: .whitespaces).isEmpty || selected.isEmpty)
            }
            if let error {
                Text(error).font(.callout).foregroundStyle(.red).fixedSize(horizontal: false, vertical: true)
            }
            if !result.isEmpty {
                Text(result)
                    .font(.callout)
                    .textSelection(.enabled)
                    .padding(8)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 6))
                HStack {
                    Spacer()
                    Button("Append to prompt") {
                        let p = model.settings.prompt.trimmingCharacters(in: .whitespacesAndNewlines)
                        model.settings.prompt = p.isEmpty ? result : p + (p.hasSuffix(",") ? " " : ", ") + result
                    }
                    Button("Replace prompt") { model.settings.prompt = result }
                        .buttonStyle(.borderedProminent)
                }
            }
            Text("Uses \(AIPrompt.baseURL) (change it in Settings)")
                .font(.caption).foregroundStyle(.tertiary)
        }
        .padding(14)
        .frame(width: 460)
        .task { await loadModels() }
        .onChange(of: selected) { _, v in AIPrompt.model = v }
    }

    private func loadModels() async {
        do {
            let list = try await AIPrompt.models()
            models = list
            if selected.isEmpty || !list.contains(selected) {
                // a model the server already has loaded, else a Qwen model, else the first one
                selected = list.first { AIPrompt.loadedModels.contains($0) }
                    ?? list.first { $0.localizedCaseInsensitiveContains("qwen") } ?? list.first ?? ""
            }
            error = nil
        } catch {
            self.error = "Couldn't reach the AI server at \(AIPrompt.baseURL). Is MLX-Serve running?"
        }
    }

    private func run() {
        let text = brief.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty, !selected.isEmpty, !busy else { return }
        busy = true
        error = nil
        Task {
            do {
                result = try await AIPrompt.tags(for: text, model: selected)
                if result.isEmpty { error = "The model didn't return any tags. Try again or pick another model." }
            } catch {
                self.error = error.localizedDescription
            }
            busy = false
        }
    }
}
