import SwiftUI

struct CanvasView: View {
    @Environment(AppModel.self) private var model
    @AppStorage("showPromptPanel") private var showPrompt = true
    @AppStorage("showGallery") private var showGallery = true

    var body: some View {
        VStack(spacing: 0) {
            ImageStage()
                .overlay(alignment: .bottomTrailing) {
                    // hide / show the prompt and the gallery (also View ▸ ⌥⌘P / ⌥⌘G)
                    let visible = showPrompt || showGallery
                    Button {
                        withAnimation(.easeInOut(duration: 0.2)) {
                            showPrompt = !visible
                            showGallery = !visible
                        }
                    } label: {
                        Label(visible ? "Hide Prompt and Gallery" : "Show Prompt and Gallery",
                              systemImage: visible ? "chevron.down" : "chevron.up")
                            .labelStyle(.iconOnly)
                            .frame(width: 22, height: 18)
                    }
                    .buttonStyle(.borderless)
                    .padding(4)
                    .background(.regularMaterial, in: Capsule())
                    .padding(10)
                    .help(visible ? "Hide the prompt and gallery" : "Show the prompt and gallery")
                }
            if showPrompt {
                Divider()
                PromptPanel()
                    .padding(12)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
            if showGallery {
                Divider()
                GalleryStrip()
                    .frame(height: 92)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
            }
        }
    }
}

struct ImageStage: View {
    @Environment(AppModel.self) private var model
    @State private var showInfo = false

    var body: some View {
        ZStack {
            Color(nsColor: .underPageBackgroundColor)
                .contentShape(Rectangle())
                // clicking the canvas ends text editing, so ← / → browse the gallery
                .onTapGesture { NSApp.keyWindow?.makeFirstResponder(nil) }
            if let preview = model.livePreview {
                Image(nsImage: preview)
                    .resizable()
                    .interpolation(.high)
                    .scaledToFit()
                    .padding(12)
                    .overlay(alignment: .topLeading) {
                        Label("Live preview", systemImage: "eye")
                            .font(.caption)
                            .padding(.horizontal, 8).padding(.vertical, 3)
                            .background(.regularMaterial, in: Capsule())
                            .padding(20)
                    }
            } else if let image = model.displayedImage {
                // pinch / ⌘+ ⌘- to zoom, double-click for 100 %, drag out when fitted
                ZoomableImage(image: image, fileURL: model.selection?.url, zoom: model.zoom)
            } else if !model.isBusy {
                ContentUnavailableView {
                    Label("Pick a character", systemImage: "person.crop.rectangle.stack")
                } description: {
                    Text("Choose characters on the left, write the scene, then press Generate (⌘↩).")
                }
            }
            if let p = model.progress {
                VStack {
                    Spacer()
                    ProgressCard(info: p) { model.cancel() }
                        .padding(16)
                }
            }
        }
        .overlay(alignment: .topTrailing) {
            if let item = model.selection {
                HStack(spacing: 2) {
                    UpscaleMenu(item: item) {
                        Label("Upscale", systemImage: "arrow.up.left.and.arrow.down.right").labelStyle(.iconOnly)
                    }
                    .menuStyle(.borderlessButton)
                    .menuIndicator(.hidden)
                    .fixedSize()
                    .help("Upscale this image")
                    Button { showInfo.toggle() } label: { Label("Image Info", systemImage: "info.circle").labelStyle(.iconOnly) }
                        .help("Image info")
                        .popover(isPresented: $showInfo, arrowEdge: .bottom) { ImageInfoView(item: item) }
                    Button { NSWorkspace.shared.activateFileViewerSelecting([item.url]) } label: {
                        Label("Show in Finder", systemImage: "folder").labelStyle(.iconOnly)
                    }
                    .help("Show in Finder")
                    Button { model.copyImage(item) } label: {
                        Label("Copy Image", systemImage: "doc.on.doc").labelStyle(.iconOnly)
                    }
                    .help("Copy image (⇧⌘C)")
                    Button { model.moveToTrash(item) } label: {
                        Label("Move to Trash", systemImage: "trash").labelStyle(.iconOnly)
                    }
                    .help(model.canTrash(item) ? "Move to Trash (⌘⌫)" : "Waiting to be upscaled: can't move it to the Trash yet")
                    .disabled(!model.canTrash(item))
                }
                .buttonStyle(.borderless)
                .padding(6)
                .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 8))
                .padding(10)
            }
        }
    }
}

struct ProgressCard: View {
    let info: ProgressInfo
    let stop: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text(info.title).font(.callout.weight(.medium))
                    Spacer()
                    Text(info.detail).font(.caption).foregroundStyle(.secondary)
                }
                if let f = info.fraction {
                    ProgressView(value: f)
                } else {
                    ProgressView().progressViewStyle(.linear)
                }
            }
            Button(action: stop) { Image(systemName: "stop.fill") }
                .buttonStyle(.borderless)
                .help("Stop (⌘.)")
        }
        .padding(12)
        .frame(maxWidth: 420)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 10))
    }
}

struct PromptPanel: View {
    @Environment(AppModel.self) private var model
    @State private var completion: CompletionController?
    @State private var showAI = false

    var body: some View {
        @Bindable var model = model
        VStack(alignment: .leading, spacing: 10) {
            VStack(alignment: .leading, spacing: 4) {
                Text("Characters").font(.caption).foregroundStyle(.secondary)
                FlowLayout(spacing: 6) {
                    ForEach(model.settings.characters, id: \.self) { tag in
                        CharacterChip(tag: tag) { model.toggleCharacter(tag) }
                    }
                    if model.settings.characters.isEmpty {
                        Text("None — click characters in the sidebar to add them")
                            .font(.callout).foregroundStyle(.tertiary)
                    }
                }
            }
            if !model.settings.artists.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    Text("Styles").font(.caption).foregroundStyle(.secondary)
                    FlowLayout(spacing: 6) {
                        ForEach(model.settings.artists, id: \.self) { tag in
                            CharacterChip(tag: tag) { model.toggleStyle(tag) }
                        }
                    }
                }
            }
            PromptField(title: "Prompt", text: $model.settings.prompt,
                        placeholder: "1girl, solo, cowboy shot, night city, neon lights",
                        completion: completion, minHeight: 64)
                .overlay(alignment: .topTrailing) {
                    Button { showAI.toggle() } label: {
                        Label("AI", systemImage: "wand.and.stars").font(.caption)
                    }
                    .buttonStyle(.borderless)
                    .help("Write tags from a description with your local LLM")
                    .popover(isPresented: $showAI, arrowEdge: .top) { AIPromptView() }
                }
            PromptField(title: "Negative", text: $model.settings.negative,
                        placeholder: "bad quality, worst quality", completion: nil, minHeight: 36)
        }
        .onAppear { if completion == nil { completion = CompletionController(index: model.tags) } }
    }
}

struct CharacterChip: View {
    let tag: String
    let remove: () -> Void

    var body: some View {
        HStack(spacing: 4) {
            Text(tag)
            Button(action: remove) {
                Label("Remove \(tag)", systemImage: "xmark").labelStyle(.iconOnly).font(.caption2.weight(.bold))
            }
                .buttonStyle(.borderless)
                .help("Remove")
        }
        .font(.callout)
        .padding(.horizontal, 8).padding(.vertical, 3)
        .background(Color.accentColor.opacity(0.15), in: Capsule())
        .foregroundStyle(Color.accentColor)
    }
}

struct GalleryStrip: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        if model.gallery.isEmpty {
            Text("Your images will appear here")
                .foregroundStyle(.tertiary)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            ScrollViewReader { scroller in
            ScrollView(.horizontal) {
                LazyHStack(spacing: 8) {
                    ForEach(model.gallery) { item in
                        GalleryThumb(item: item, selected: model.selection == item)
                            .id(item.id)
                            // drag a thumbnail out to Finder or another app: the PNG file itself
                            .onDrag { NSItemProvider(contentsOf: item.url) ?? NSItemProvider() }
                            .onTapGesture {
                                // end text editing so ← / → browse the gallery
                                NSApp.keyWindow?.makeFirstResponder(nil)
                                model.show(item)
                            }
                            .contextMenu {
                                UpscaleMenu(item: item) { Text("Upscale") }
                                Divider()
                                if let r = item.record {
                                    Button("Use prompt and settings") { model.reuse(item, seed: false) }
                                    Button("Use prompt, settings and seed") { model.reuse(item, seed: true) }
                                    Divider()
                                }
                                Button("Copy Image") { model.copyImage(item) }
                                Button("Show in Finder") { NSWorkspace.shared.activateFileViewerSelecting([item.url]) }
                                Button("Move to Trash", role: .destructive) { model.moveToTrash(item) }
                                    .disabled(!model.canTrash(item))
                            }
                    }
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
            }
            .onChange(of: model.selection) { _, item in
                if let item { withAnimation(.easeOut(duration: 0.15)) { scroller.scrollTo(item.id) } }
            }
            }
        }
    }
}

struct GalleryThumb: View {
    let item: GalleryItem
    let selected: Bool
    @State private var image: NSImage?

    var body: some View {
        Group {
            if let image { Image(nsImage: image).resizable().scaledToFill() } else { Rectangle().fill(.quaternary) }
        }
        .frame(width: 58, height: 76)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(selected ? Color.accentColor : .clear, lineWidth: 2))
        .task(id: item.id) {
            let url = item.url
            image = await Task.detached(priority: .utility) { ImageFiles.thumbnail(url) }.value
        }
    }
}

/// "Upscale" menu for a saved image: detail pass at the hi-res scale, or upscaler-only x2.
struct UpscaleMenu<Label: View>: View {
    @Environment(AppModel.self) private var model
    let item: GalleryItem
    @ViewBuilder let label: () -> Label

    var body: some View {
        Menu {
            let detail = model.upscaleTarget(item, detail: true)
            let quick = model.upscaleTarget(item, detail: false)
            let scale = String(format: "%.2g", model.settings.hiresScale)
            Button(detail.map { "Upscale ×\(scale) with detail pass  (\($0.0) × \($0.1))" } ?? "Upscale with detail pass") {
                model.upscale(item, detail: true)
            }
            .disabled(detail.map { $0.0 * $0.1 > 4_500_000 } ?? true)
            Button(quick.map { "Upscale ×2, upscaler only  (\($0.0) × \($0.1))" } ?? "Upscale ×2, upscaler only") {
                model.upscale(item, detail: false)
            }
            .disabled(model.settings.hiresUpscaler.isEmpty)
            Divider()
            ForEach(Wallpaper.all) { wall in
                Button("\(wall.device) wallpaper  (\(wall.screenWidth) × \(wall.screenHeight))") { model.saveWallpaper(item, wall) }
                    .disabled(model.settings.hiresUpscaler.isEmpty)
            }
            Divider()
            Text("Uses the hi-res scale, steps, denoise and upscaler from the inspector; wallpapers are upscaled and cropped to the screen")
        } label: {
            label()
        }
    }
}

/// Wrapping horizontal layout for chips.
struct FlowLayout: Layout {
    var spacing: CGFloat = 6

    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let width = proposal.width ?? .infinity
        var x: CGFloat = 0, y: CGFloat = 0, rowHeight: CGFloat = 0, maxX: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x > 0, x + s.width > width { x = 0; y += rowHeight + spacing; rowHeight = 0 }
            x += s.width + spacing
            maxX = max(maxX, x - spacing)
            rowHeight = max(rowHeight, s.height)
        }
        return CGSize(width: min(maxX, width), height: y + rowHeight)
    }

    func placeSubviews(in bounds: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = bounds.minX, y = bounds.minY, rowHeight: CGFloat = 0
        for v in subviews {
            let s = v.sizeThatFits(.unspecified)
            if x > bounds.minX, x + s.width > bounds.maxX { x = bounds.minX; y += rowHeight + spacing; rowHeight = 0 }
            v.place(at: CGPoint(x: x, y: y), proposal: ProposedViewSize(s))
            x += s.width + spacing
            rowHeight = max(rowHeight, s.height)
        }
    }
}
