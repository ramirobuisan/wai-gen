import AppKit
import SwiftUI

/// Puts WAI Gen in Applications: updates replace the app in place, which needs it there (not in Downloads or a
/// read-only place macOS runs it from).
@MainActor
enum AppInstaller {
    static var bundle: URL { Bundle.main.bundleURL }
    /// macOS runs a downloaded app from a temporary read-only copy until it's moved with Finder.
    static var isTranslocated: Bool { bundle.path.contains("/AppTranslocation/") }
    static var applicationsFolders: [URL] {
        [URL(fileURLWithPath: "/Applications"),
         FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Applications")]
    }
    static var isInApplications: Bool {
        let path = bundle.resolvingSymlinksInPath().path
        return applicationsFolders.contains { path.hasPrefix($0.resolvingSymlinksInPath().path + "/") }
    }
    /// /Applications when this user can write there, else ~/Applications.
    static var destinationFolder: URL {
        FileManager.default.isWritableFile(atPath: "/Applications") ? applicationsFolders[0] : applicationsFolders[1]
    }
    static var destination: URL { destinationFolder.appendingPathComponent(bundle.lastPathComponent) }

    /// Copies this app into Applications (replacing the copy there, if any), moves this copy to the Trash and
    /// reopens WAI Gen from Applications.
    static func moveToApplications() throws {
        let fm = FileManager.default
        guard !isTranslocated else {
            throw Updater.UpdateError("macOS is running WAI Gen from a temporary copy. Quit it, drag WAI Gen.app into "
                                      + "Applications in Finder and open it from there.")
        }
        let target = destination
        if NSRunningApplication.runningApplications(withBundleIdentifier: Bundle.main.bundleIdentifier ?? "")
            .contains(where: { $0.bundleURL?.standardizedFileURL == target.standardizedFileURL }) {
            throw Updater.UpdateError("The WAI Gen in \(target.deletingLastPathComponent().path) is open. Quit it first.")
        }
        try fm.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        // copy next to the target, then swap it in (a rename on the same volume)
        let staging = target.deletingLastPathComponent().appendingPathComponent(".WAI Gen install \(UUID().uuidString).app")
        try fm.copyItem(at: bundle, to: staging)
        do {
            if fm.fileExists(atPath: target.path) {
                _ = try fm.replaceItemAt(target, withItemAt: staging, backupItemName: nil, options: [])
            } else {
                try fm.moveItem(at: staging, to: target)
            }
        } catch {
            try? fm.removeItem(at: staging)
            throw error
        }
        try? fm.trashItem(at: bundle, resultingItemURL: nil)   // this process keeps running until it quits
        try Updater.relaunch(target)
    }
}

/// "Welcome to WAI Gen": where the app is, the models folder, and which models, upscaler and face detector are
/// there, with where to get each missing file. Opens by itself at the first launch when something is missing,
/// and from WAI Gen → Setup….
struct SetupView: View {
    static let windowID = "setup"
    static let shownKey = "setupShown"

    @Environment(AppModel.self) private var model
    @Environment(\.dismissWindow) private var dismissWindow
    @State private var modelsPath = AppPaths.models.path
    @State private var refresh = 0
    @State private var confirmReplace = false
    @State private var installError: String?

    /// At the first launch the window opens by itself only when something needs doing.
    static func needsAttention(_ model: AppModel) -> Bool {
        if !AppInstaller.isInApplications { return true }
        if model.isRemote { return false }
        return ModelCatalog.presets.allSatisfy { !model.missingFiles($0).isEmpty }
    }

    var body: some View {
        let _ = refresh   // re-reads the files when it changes
        VStack(spacing: 0) {
            ScrollView {
                VStack(spacing: 18) {
                    header
                    group("Install") {
                        appRow
                        Divider().padding(.leading, 56)
                        foldersRow
                    }
                    group(model.isRemote ? "Models on \(model.generatorName)" : "Models") {
                        ForEach(Array(ModelCatalog.presets.enumerated()), id: \.element.id) { i, p in
                            if i > 0 { Divider().padding(.leading, 56) }
                            presetRow(p, index: i)
                        }
                    }
                    group("Extras") {
                        upscalerRow
                        Divider().padding(.leading, 56)
                        faceRow
                    }
                }
                .padding(24)
            }
            Divider()
            HStack {
                Text("Open this again from WAI Gen → Setup…").font(.caption).foregroundStyle(.secondary)
                Spacer()
                Button("Check Again") { recheck() }
                Button("Done") { dismissWindow(id: Self.windowID) }
                    .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 24).padding(.vertical, 14)
        }
        .frame(width: 640, height: 760)
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in recheck() }
        .confirmationDialog("Replace the WAI Gen in Applications?", isPresented: $confirmReplace) {
            Button("Replace") { move() }
        } message: {
            Text("There's already a WAI Gen in \(AppInstaller.destinationFolder.path). This copy replaces it; your "
                 + "models, images and settings stay.")
        }
        .alert("Couldn't move WAI Gen", isPresented: Binding(get: { installError != nil },
                                                             set: { if !$0 { installError = nil } })) {
            Button("OK") { installError = nil }
        } message: {
            Text(installError ?? "")
        }
    }

    private var header: some View {
        VStack(spacing: 8) {
            Image(nsImage: NSApplication.shared.applicationIconImage).resizable().frame(width: 84, height: 84)
            Text("Welcome to WAI Gen").font(.system(size: 26, weight: .bold))
            Text("Anime images generated on this Mac, entirely in MLX. Put the models in a folder with ComfyUI's "
                 + "layout and pick the ones you want: each model is optional.")
                .multilineTextAlignment(.center).foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
                .frame(maxWidth: 480)
        }
        .padding(.top, 8)
    }

    // MARK: Rows

    @ViewBuilder private var appRow: some View {
        if AppInstaller.isInApplications {
            row("app.badge.checkmark", .blue, "WAI Gen is in Applications",
                "Updates install in place (\(Self.abbreviated(AppInstaller.bundle.deletingLastPathComponent())))",
                ok: true) { EmptyView() }
        } else {
            row("app.badge", .blue, "Move WAI Gen to Applications",
                "It's running from \(Self.short(AppInstaller.bundle.deletingLastPathComponent())). In Applications it can "
                    + "update itself.", ok: false) {
                Button("Move to Applications") {
                    if FileManager.default.fileExists(atPath: AppInstaller.destination.path) { confirmReplace = true } else { move() }
                }
            }
        }
    }

    @ViewBuilder private var foldersRow: some View {
        if model.isRemote {
            row("network", .teal, "Generating on \(model.generatorName)",
                "Images are made on that Mac with its models; this Mac needs none. Change it in Settings → Generator.",
                ok: model.remoteInfo != nil) { EmptyView() }
        } else {
            let exists = FileManager.default.fileExists(atPath: modelsPath)
            row("folder.fill", .orange, "Models folder",
                exists ? Self.abbreviated(URL(fileURLWithPath: modelsPath)) + "\nSame layout as ComfyUI: checkpoints/, "
                    + "diffusion_models/, text_encoders/, vae/, upscale_models/, detectors/, loras/"
                    : "Choose the folder with your models (ComfyUI's models folder works as it is).",
                ok: exists) {
                HStack(spacing: 6) {
                    if exists {
                        Button("Show") { NSWorkspace.shared.open(URL(fileURLWithPath: modelsPath)) }
                    }
                    Button("Choose…") { chooseModelsFolder() }
                }
            }
        }
    }

    private func presetRow(_ p: ModelPreset, index: Int) -> some View {
        let missing = model.missingFiles(p)
        let colors: [Color] = [.pink, .purple, .indigo, .blue, .mint]
        let detail: String
        if missing.isEmpty {
            detail = "Ready"
        } else if model.isRemote {
            detail = missing.joined(separator: "\n")
        } else {
            detail = "Missing " + missing.map { "\($0)" }.joined(separator: ", ")
        }
        return row("photo.artframe", colors[index % colors.count], p.name, detail, ok: missing.isEmpty) {
            if !missing.isEmpty && !model.isRemote { getIt(missing) }
        }
    }

    private var upscalerRow: some View {
        let name = model.settings.hiresUpscaler
        let ok = model.upscalerAvailable(name)
        return row("arrow.up.left.and.arrow.down.right", .green, "Upscaler",
                   ok ? "\(name): hi-res fix and Upscale use it"
                      : "Optional, for hi-res fix and Upscale: \(name) (18 MB) in upscale_models/",
                   ok: ok) {
            if !ok && !model.isRemote { getIt(["upscale_models/\(name)"]) }
        }
    }

    private var faceRow: some View {
        let ok = model.faceDetectorAvailable
        return row("face.smiling", .yellow, "Face repair",
                   ok ? "face_yolov8m finds the faces to repaint"
                      : "Optional: face_yolov8m.pt converted with scripts/convert_yolov8_safe.py, in detectors/",
                   ok: ok) {
            if !ok && !model.isRemote { getIt(["detectors/face_yolov8m.safetensors"], label: "How To") }
        }
    }

    /// "Get It": the file's download page (a menu when the files come from different pages).
    @ViewBuilder private func getIt(_ files: [String], label: String = "Get It") -> some View {
        let pages = files.compactMap { f in ModelCatalog.sources[f].map { (file: f, site: $0.site, url: $0.url) } }
        if Set(pages.map(\.url)).count > 1 {
            Menu(label) {
                ForEach(pages, id: \.file) { page in
                    Button("\((page.file as NSString).lastPathComponent) (\(page.site))") { NSWorkspace.shared.open(page.url) }
                }
            }
            .fixedSize()
        } else if let page = pages.first {
            Button(label) { NSWorkspace.shared.open(page.url) }
                .help("Opens \(page.site): \(page.url.absoluteString)")
        }
    }

    // MARK: Layout

    private func row<Control: View>(_ symbol: String, _ color: Color, _ title: String, _ detail: String,
                                    ok: Bool?, @ViewBuilder control: () -> Control) -> some View {
        HStack(alignment: .center, spacing: 12) {
            Image(systemName: symbol)
                .font(.system(size: 14, weight: .semibold)).foregroundStyle(.white)
                .frame(width: 30, height: 30)
                .background(color.gradient, in: RoundedRectangle(cornerRadius: 7))
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(title).fontWeight(.medium)
                    if let ok {
                        Image(systemName: ok ? "checkmark.circle.fill" : "exclamationmark.circle.fill")
                            .foregroundStyle(ok ? .green : .orange)
                    }
                }
                Text(detail).font(.callout).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                    .textSelection(.enabled)
            }
            Spacer(minLength: 8)
            control()
        }
        .padding(.vertical, 9).padding(.horizontal, 14)
    }

    private func group<Content: View>(_ title: String, @ViewBuilder _ content: () -> Content) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(title.uppercased()).font(.caption.weight(.semibold)).foregroundStyle(.secondary).padding(.leading, 6)
            VStack(spacing: 0) { content() }
                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
                .overlay(RoundedRectangle(cornerRadius: 10).strokeBorder(.separator))
        }
    }

    // MARK: Actions

    private func recheck() {
        modelsPath = AppPaths.models.path
        model.refreshUpscalers()
        model.refreshLoRAs()
        refresh += 1
    }

    private func move() {
        do { try AppInstaller.moveToApplications() } catch { installError = error.localizedDescription }
    }

    private func chooseModelsFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Use Folder"
        panel.directoryURL = URL(fileURLWithPath: modelsPath)
        guard panel.runModal() == .OK, let url = panel.url else { return }
        AppPaths.models = url
        modelsPath = url.path
        recheck()
    }

    static func abbreviated(_ url: URL) -> String {
        (url.path as NSString).abbreviatingWithTildeInPath
    }

    /// A folder as people know it: ~/Downloads, or …/parent/folder when the path is long.
    static func short(_ url: URL) -> String {
        let full = abbreviated(url)
        guard full.count > 40 else { return full }
        return "…/" + url.pathComponents.suffix(2).joined(separator: "/")
    }
}

/// App menu → Setup…
struct SetupButton: View {
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        Button("Setup…") { openWindow(id: SetupView.windowID) }
    }
}
