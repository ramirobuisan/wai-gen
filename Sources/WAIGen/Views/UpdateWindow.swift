import AppKit
import SwiftUI

/// The Software Update window: opens by itself when an automatic check finds a new version, and from
/// WAI Gen → Check for Updates… and the toolbar's "Update to …" button. Shows what's new, downloads, checks and
/// installs the update, then WAI Gen reopens. The channel and automatic checks are in Settings → Updates.
struct SoftwareUpdateView: View {
    static let windowID = "update"
    @Environment(AppModel.self) private var model
    @Environment(\.dismissWindow) private var dismissWindow

    private var updater: Updater { model.updater }
    private var release: Updater.Release? { updater.available ?? updater.installing }
    /// Installing waits until nothing is generating here or for another Mac.
    private var busy: Bool { model.queueCount > 0 || model.servingClient != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(alignment: .center, spacing: 14) {
                Image(nsImage: NSApplication.shared.applicationIconImage)
                    .resizable().frame(width: 64, height: 64)
                VStack(alignment: .leading, spacing: 4) {
                    Text(heading).font(.title2.weight(.semibold))
                    Text(subheading).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
                }
            }
            if let release, !release.notes.isEmpty {
                ScrollView {
                    Text(Self.notes(release.notes))
                        .textSelection(.enabled)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(12)
                }
                .frame(height: 220)
                .background(Color(nsColor: .textBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).strokeBorder(.separator))
            }
            HStack(spacing: 10) {
                if let release {
                    Button("View on GitHub") { NSWorkspace.shared.open(release.page) }
                }
                Spacer()
                status
                buttons
            }
        }
        .padding(22)
        .frame(width: 540)
    }

    private var heading: String {
        switch updater.state {
        case .checking: "Checking for updates…"
        case .upToDate: "WAI Gen is up to date"
        case .failed: release == nil ? "Couldn't check for updates" : "Couldn't install the update"
        case .idle: "Software Update"
        case .available, .downloading, .installing:
            "WAI Gen \(release?.version ?? "")\(release?.prerelease == true ? " (beta)" : "") is available"
        }
    }

    private var subheading: String {
        let current = Updater.currentVersion
        switch updater.state {
        case .checking: return "You have version \(current)."
        case .upToDate:
            return "You have version \(current), the newest \(updater.channel == .beta ? "on the Beta channel" : "release")."
        case .failed(let message): return message
        case .idle:
            let last = updater.lastCheck.map { " Last checked \($0.formatted(.relative(presentation: .named)))." } ?? ""
            return "You have version \(current).\(last)"
        case .available, .downloading, .installing:
            return "You have \(current). The update installs in place and WAI Gen reopens by itself; your models, "
                + "images and settings stay."
        }
    }

    @ViewBuilder private var status: some View {
        switch updater.state {
        case .checking:
            ProgressView().controlSize(.small)
        case .downloading:
            if let f = updater.downloadProgress {
                ProgressView(value: f).frame(width: 120)
                Text("\(Int(f * 100)) %").monospacedDigit().foregroundStyle(.secondary)
            } else {
                ProgressView().controlSize(.small)
                Text("Downloading…").foregroundStyle(.secondary)
            }
        case .installing:
            ProgressView().controlSize(.small)
            Text("Installing…").foregroundStyle(.secondary)
        case .available:
            if busy {
                Text(model.servingClient != nil ? "Waiting for the other Mac's image" : "Waiting for the queue")
                    .font(.callout).foregroundStyle(.secondary)
            }
        default:
            EmptyView()
        }
    }

    @ViewBuilder private var buttons: some View {
        switch updater.state {
        case .available(let release):
            Button("Later") { dismissWindow(id: Self.windowID) }
                .keyboardShortcut(.cancelAction)
            Button("Install and Relaunch") { Task { await updater.install(release) } }
                .keyboardShortcut(.defaultAction)
                .disabled(busy)
        case .downloading, .installing:
            EmptyView()
        case .failed:
            Button("Close") { dismissWindow(id: Self.windowID) }
                .keyboardShortcut(.cancelAction)
            Button("Try Again") { Task { await updater.check() } }
                .keyboardShortcut(.defaultAction)
        case .idle:
            Button("Close") { dismissWindow(id: Self.windowID) }
                .keyboardShortcut(.cancelAction)
            Button("Check Now") { Task { await updater.check() } }
                .keyboardShortcut(.defaultAction)
        case .checking, .upToDate:
            Button("OK") { dismissWindow(id: Self.windowID) }
                .keyboardShortcut(.defaultAction)
        }
    }

    /// Release notes are Markdown: bold, code and links render; list dashes and line breaks stay as written.
    static func notes(_ markdown: String) -> AttributedString {
        let options = AttributedString.MarkdownParsingOptions(interpretedSyntax: .inlineOnlyPreservingWhitespace)
        return (try? AttributedString(markdown: markdown, options: options)) ?? AttributedString(markdown)
    }
}
