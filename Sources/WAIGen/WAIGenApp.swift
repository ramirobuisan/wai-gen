import AppKit
import SwiftUI

@main
struct WAIGenApp: App {
    @State private var model = AppModel()

    init() {
        // also behave like a normal app when launched as a bare executable during development
        NSApplication.shared.setActivationPolicy(.regular)
    }

    var body: some Scene {
        Window("WAI Gen", id: "main") {
            ContentView()
                .environment(model)
                .frame(minWidth: 1080, minHeight: 700)
                .task {
                    model.start()
                    NSApplication.shared.activate()
                }
        }
        .defaultSize(width: 1440, height: 920)
        .commands {
            CommandGroup(after: .newItem) {
                Button("Generate") { model.generate() }
                    .keyboardShortcut(.return, modifiers: .command)
                Button("Stop") { model.cancel() }
                    .keyboardShortcut(".", modifiers: .command)
                    .disabled(!model.isBusy)
            }
            CommandGroup(after: .sidebar) {
                Button(model.showInspector ? "Hide Inspector" : "Show Inspector") { model.showInspector.toggle() }
                    .keyboardShortcut("i", modifiers: [.command, .option])
            }
            CommandMenu("Model") {
                ForEach(Array(ModelCatalog.presets.enumerated()), id: \.element.id) { i, p in
                    Button(p.name) { model.selectModel(p) }
                        .keyboardShortcut(KeyEquivalent(Character("\(i + 1)")), modifiers: .command)
                }
                Divider()
                Button("Reset to Preset") { model.resetToPreset() }
                Divider()
                Menu("Add LoRA") {
                    ForEach(model.loraFiles) { f in
                        Button(f.name) { model.addLoRA(f) }
                            .disabled(model.settings.loras.contains { $0.relativePath == f.relativePath })
                    }
                    if model.loraFiles.isEmpty { Text("No LoRAs in the loras folder") }
                    Divider()
                    Button("Refresh List") { model.refreshLoRAs() }
                }
                Button("Remove All LoRAs") { model.settings.loras.removeAll() }
                    .disabled(model.settings.loras.isEmpty)
            }
            CommandMenu("Image") {
                Button("Upscale with Detail Pass") { if let s = model.selection { model.upscale(s, detail: true) } }
                    .keyboardShortcut("u", modifiers: .command)
                    .disabled(model.selection == nil)
                Button("Upscale ×2 (Upscaler Only)") { if let s = model.selection { model.upscale(s, detail: false) } }
                    .keyboardShortcut("u", modifiers: [.command, .shift])
                    .disabled(model.selection == nil)
                Divider()
                Button("Use Prompt and Settings") { if let s = model.selection { model.reuse(s, seed: false) } }
                    .disabled(model.selection?.record == nil)
                Button("Use Prompt, Settings and Seed") { if let s = model.selection { model.reuse(s, seed: true) } }
                    .disabled(model.selection?.record == nil)
                Divider()
                Button("Show in Finder") { if let s = model.selection { NSWorkspace.shared.activateFileViewerSelecting([s.url]) } }
                    .disabled(model.selection == nil)
                // no ⌘⌫ shortcut: it would fire while editing the prompt
                Button("Move to Trash") { if let s = model.selection { model.moveToTrash(s) } }
                    .disabled(model.selection == nil || model.isBusy)
            }
        }

        Settings {
            SettingsView()
                .environment(model)
        }
    }
}

struct ContentView: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var model = model
        NavigationSplitView {
            CharacterSidebar()
                .navigationSplitViewColumnWidth(min: 220, ideal: 270, max: 380)
        } detail: {
            CanvasView()
        }
        .inspector(isPresented: $model.showInspector) {
            InspectorView()
                .inspectorColumnWidth(min: 260, ideal: 300, max: 400)
        }
        .navigationTitle("WAI Gen")
        .navigationSubtitle(model.preset.name)
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                if model.queueCount > 0 {
                    Menu {
                        if let r = model.running { Text("Now: seed \(r.seed)") }
                        ForEach(model.pending) { Text("Waiting: seed \($0.seed)") }
                        if !model.pending.isEmpty {
                            Divider()
                            Button("Clear waiting images") { model.clearQueue() }
                        }
                    } label: {
                        Label("Queue \(model.queueCount)", systemImage: "square.stack.3d.up")
                            .labelStyle(.titleAndIcon)
                    }
                    .help("Images in the queue")
                }
                if model.isBusy {
                    Button { model.cancel() } label: { Label("Stop", systemImage: "stop.fill") }
                        .help("Stop (⌘.)")
                }
                Button { model.generate() } label: {
                    Label("Generate", systemImage: "sparkles").labelStyle(.titleAndIcon)
                }
                .buttonStyle(.borderedProminent)
                .help("Generate (⌘↩)")
                Button { model.showInspector.toggle() } label: {
                    Label("Inspector", systemImage: "sidebar.right")
                }
                .help("Show or hide the inspector (⌥⌘I)")
            }
        }
        .alert("Something went wrong", isPresented: Binding(
            get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }
}

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @State private var modelsPath = AppPaths.models.path
    @State private var outputPath = AppPaths.output.path
    @State private var aiURL = AIPrompt.baseURL

    var body: some View {
        Form {
            Section("Folders") {
                LabeledContent("Models") {
                    HStack {
                        Text(modelsPath).lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                        Button("Choose…") { choose { AppPaths.models = $0; modelsPath = $0.path } }
                    }
                }
                Text("Same layout as ComfyUI: checkpoints/, diffusion_models/, text_encoders/, vae/")
                    .font(.caption).foregroundStyle(.secondary)
                LabeledContent("Images") {
                    HStack {
                        Text(outputPath).lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                        Button("Choose…") { choose { AppPaths.output = $0; outputPath = $0.path } }
                    }
                }
            }
            Section("AI prompt") {
                TextField("Server", text: $aiURL, prompt: Text(AIPrompt.defaultURL))
                    .onSubmit { AIPrompt.baseURL = aiURL.isEmpty ? AIPrompt.defaultURL : aiURL }
                Text("Any OpenAI-compatible server. MLX-Serve listens on 127.0.0.1:11234 by default.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Data") {
                Button("Open wildcards folder") { NSWorkspace.shared.open(AppPaths.wildcards) }
                Button("Open character data folder") { NSWorkspace.shared.open(AppPaths.data) }
            }
            Section("Presets") {
                ForEach(ModelCatalog.presets) { p in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(p.name)
                        Text(p.source).font(.caption).foregroundStyle(.secondary)
                    }
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 560, height: 460)
    }

    private func choose(_ done: (URL) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { done(url) }
    }
}
