import AppKit
import SwiftUI

@main
struct WAIGenApp: App {
    @State private var model = AppModel()
    @AppStorage("showPromptPanel") private var showPrompt = true
    @AppStorage("showGallery") private var showGallery = true

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
            CommandGroup(after: .appInfo) {
                CheckForUpdatesButton(updater: model.updater)
                SetupButton()
            }
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
                Button(showPrompt ? "Hide Prompt" : "Show Prompt") { withAnimation { showPrompt.toggle() } }
                    .keyboardShortcut("p", modifiers: [.command, .option])
                Button(showGallery ? "Hide Gallery" : "Show Gallery") { withAnimation { showGallery.toggle() } }
                    .keyboardShortcut("g", modifiers: [.command, .option])
                Divider()
                Button("Zoom In") { model.zoom.zoomIn() }
                    .keyboardShortcut("+", modifiers: .command)
                Button("Zoom Out") { model.zoom.zoomOut() }
                    .keyboardShortcut("-", modifiers: .command)
                Button("Actual Size") { model.zoom.actualSize() }
                    .keyboardShortcut("0", modifiers: .command)
                Button("Zoom to Fit") { model.zoom.fit() }
                    .keyboardShortcut("9", modifiers: .command)
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
                Divider()
                Button("Load All Models") { model.preloadModels() }
                Button("Unload All Models") { model.unloadModels() }
                    .disabled(model.isBusy)
            }
            CommandMenu("Image") {
                Button("Upscale with Detail Pass") { if let s = model.selection { model.upscale(s, detail: true) } }
                    .keyboardShortcut("u", modifiers: .command)
                    .disabled(model.selection == nil)
                Button("Upscale ×2 (Upscaler Only)") { if let s = model.selection { model.upscale(s, detail: false) } }
                    .keyboardShortcut("u", modifiers: [.command, .shift])
                    .disabled(model.selection == nil)
                ForEach(Wallpaper.all) { wall in
                    Button("Save as \(wall.device) Wallpaper") { if let s = model.selection { model.saveWallpaper(s, wall) } }
                        .disabled(model.selection == nil || model.settings.hiresUpscaler.isEmpty)
                }
                Divider()
                Button("Use Prompt and Settings") { if let s = model.selection { model.reuse(s, seed: false) } }
                    .disabled(model.selection?.record == nil)
                Button("Use Prompt, Settings and Seed") { if let s = model.selection { model.reuse(s, seed: true) } }
                    .disabled(model.selection?.record == nil)
                Divider()
                Button("Copy Image") { if let s = model.selection { model.copyImage(s) } }
                    .keyboardShortcut("c", modifiers: [.command, .shift])
                    .disabled(model.selection == nil)
                Button("Show in Finder") { if let s = model.selection { NSWorkspace.shared.activateFileViewerSelecting([s.url]) } }
                    .disabled(model.selection == nil)
                // ⌘⌫ as in Finder; while the prompt or a field is being edited it deletes text there instead
                Button("Move to Trash") { model.trashSelectionOrDeleteText() }
                    .keyboardShortcut(.delete, modifiers: .command)
                    .disabled(!(model.selection.map(model.canTrash) ?? false))
            }
        }

        Window("Software Update", id: SoftwareUpdateView.windowID) {
            SoftwareUpdateView()
                .environment(model)
        }
        .windowResizability(.contentSize)
        .windowStyle(.hiddenTitleBar)
        .defaultPosition(.center)
        .commandsRemoved()

        Window("Welcome to WAI Gen", id: SetupView.windowID) {
            SetupView()
                .environment(model)
        }
        .windowResizability(.contentSize)
        .windowStyle(.hiddenTitleBar)
        .defaultPosition(.center)
        .commandsRemoved()

        Settings {
            SettingsView()
                .environment(model)
        }
    }
}

struct ContentView: View {
    @Environment(AppModel.self) private var model
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        @Bindable var model = model
        NavigationSplitView {
            LibrarySidebar()
                .navigationSplitViewColumnWidth(min: 220, ideal: 270, max: 380)
        } detail: {
            CanvasView()
                .overlay(alignment: .topLeading) {
                    if let item = model.hoverPreview {
                        HoverPreviewCard(item: item)
                            .padding(12)
                            .allowsHitTesting(false)
                            .transition(.opacity)
                    }
                }
                .animation(.easeOut(duration: 0.12), value: model.hoverPreview?.id)
        }
        .inspector(isPresented: $model.showInspector) {
            InspectorView()
                .inspectorColumnWidth(min: 260, ideal: 300, max: 400)
        }
        .navigationTitle("WAI Gen")
        .navigationSubtitle(model.preset.name)
        .background(WindowTracker { model.mainWindow = $0 })
        .toolbar {
            ToolbarItemGroup(placement: .primaryAction) {
                if let update = model.updater.available {
                    Button { openWindow(id: SoftwareUpdateView.windowID) } label: {
                        Label("Update to \(update.version)", systemImage: "arrow.down.circle")
                    }
                    .help("WAI Gen \(update.version) is available: see what's new and install it")
                }
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
        // an automatic check found a new version: show it (as LocalType does)
        .onChange(of: model.updater.announced) { _, _ in openWindow(id: SoftwareUpdateView.windowID) }
        // first launch: the setup window, when the app isn't in Applications or no model is there yet
        .task {
            guard !UserDefaults.standard.bool(forKey: SetupView.shownKey) else { return }
            UserDefaults.standard.set(true, forKey: SetupView.shownKey)
            if SetupView.needsAttention(model) { openWindow(id: SetupView.windowID) }
        }
        .alert("Something went wrong", isPresented: Binding(
            get: { model.errorMessage != nil }, set: { if !$0 { model.errorMessage = nil } })) {
            Button("OK") { model.errorMessage = nil }
        } message: {
            Text(model.errorMessage ?? "")
        }
    }
}

/// Tells the model which window is the main one (its undo stack; ← / → only browse there, not in Settings).
private struct WindowTracker: NSViewRepresentable {
    let found: (NSWindow?) -> Void

    final class Tracker: NSView {
        var found: (NSWindow?) -> Void = { _ in }
        override func viewDidMoveToWindow() {
            super.viewDidMoveToWindow()
            found(window)
        }
    }

    func makeNSView(context: Context) -> Tracker {
        let v = Tracker()
        v.found = found
        return v
    }

    func updateNSView(_ v: Tracker, context: Context) {}
}

struct SettingsView: View {
    @Environment(AppModel.self) private var model
    @State private var modelsPath = AppPaths.models.path
    @State private var outputPath = AppPaths.output.path
    @State private var aiURL = AIPrompt.baseURL
    @AppStorage(GenerationService.keepLoadedKey) private var keepAllLoaded = false
    @AppStorage(GenerationService.preloadKey) private var preloadAtLaunch = true
    @AppStorage(GenerationService.zimageLowMemoryKey) private var zimageLowMemory = GenerationService.zimageLowMemory
    @AppStorage(AppModel.idleMinutesKey) private var idleMinutes = 5

    var body: some View {
        Form {
            Section("Folders") {
                LabeledContent("Models") {
                    HStack {
                        Text(modelsPath).lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                        Button("Choose…") { choose { AppPaths.models = $0; modelsPath = $0.path; model.refreshLoRAs(); model.refreshUpscalers() } }
                    }
                }
                Text("Same layout as ComfyUI: checkpoints/, diffusion_models/, text_encoders/, vae/")
                    .font(.caption).foregroundStyle(.secondary)
                LabeledContent("Images") {
                    HStack {
                        Text(outputPath).lineLimit(1).truncationMode(.middle).foregroundStyle(.secondary)
                        Button("Choose…") { choose { AppPaths.output = $0; outputPath = $0.path; model.reloadGallery() } }
                    }
                }
            }
            UpdateSettingsSection()
            GeneratorSettingsSections()
            Section("AI prompt") {
                TextField("Server", text: $aiURL, prompt: Text(AIPrompt.defaultURL))
                    .onChange(of: aiURL) { _, v in   // saved as you type, no Return needed
                        let url = v.trimmingCharacters(in: .whitespaces)
                        AIPrompt.baseURL = url.isEmpty ? AIPrompt.defaultURL : url
                    }
                Text("Any OpenAI-compatible server. MLX-Serve listens on 127.0.0.1:11234 by default.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Memory") {
                Toggle("Load the model, upscaler and face repair in advance", isOn: $preloadAtLaunch)
                Text("When the app opens and when you pick another model, so the first image doesn't wait for them.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Keep every image model loaded", isOn: $keepAllLoaded)
                Text("Off: only the selected model stays in memory. On: switching between WAI v16 and WAI-ANIMA is instant, at the cost of more memory.")
                    .font(.caption).foregroundStyle(.secondary)
                Toggle("Z-Image: use less memory (8-bit)", isOn: $zimageLowMemory)
                    .onChange(of: zimageLowMemory) { _, _ in if !model.isBusy { model.unloadModels() } }
                Text("11 GB instead of 20 GB for Z-Image Anime, about 20 % slower, nearly the same image. Applies the next time it loads.")
                    .font(.caption).foregroundStyle(.secondary)
                Picker("Free the memory after", selection: $idleMinutes) {
                    Text("5 minutes idle").tag(5)
                    Text("15 minutes idle").tag(15)
                    Text("30 minutes idle").tag(30)
                    Text("1 hour idle").tag(60)
                    Text("Never").tag(0)
                }
                Text("Idle = nothing generating. They load again when you come back to the app or press Generate.")
                    .font(.caption).foregroundStyle(.secondary)
            }
            Section("Data") {
                Button("Open wildcards folder") { NSWorkspace.shared.open(AppPaths.wildcards) }
                Button("Open data folder (characters, styles, tags)") { NSWorkspace.shared.open(AppPaths.data) }
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
        .frame(width: 600, height: 720)
    }

    private func choose(_ done: (URL) -> Void) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        if panel.runModal() == .OK, let url = panel.url { done(url) }
    }
}
