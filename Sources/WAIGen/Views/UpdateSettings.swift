import SwiftUI

/// Settings → Updates: the channel, automatic checks, and installing a new version from GitHub.
struct UpdateSettingsSection: View {
    @Environment(AppModel.self) private var model

    var body: some View {
        @Bindable var updater = model.updater
        Section("Updates") {
            LabeledContent("Version", value: Updater.currentVersion)
            Picker("Channel", selection: $updater.channel) {
                ForEach(Updater.Channel.allCases) { Text($0.label).tag($0) }
            }
            Toggle("Check for updates automatically", isOn: $updater.automatic)
            HStack {
                Button("Check Now") { Task { await updater.check() } }
                    .disabled([.checking, .downloading, .installing].contains(updater.state))
                Spacer()
                status
            }
            if let release = updater.available {
                VStack(alignment: .leading, spacing: 6) {
                    Text("WAI Gen \(release.version)\(release.prerelease ? " (beta)" : "") is available")
                        .font(.headline)
                    if !release.notes.isEmpty {
                        ScrollView {
                            Text(release.notes).font(.caption).textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                        }
                        .frame(maxHeight: 140)
                    }
                    HStack {
                        Button("Install and Relaunch") { Task { await updater.install(release) } }
                            .buttonStyle(.borderedProminent)
                            .disabled(busy)
                        Link("Release page", destination: release.page)
                        Spacer()
                        if busy {
                            Text(model.servingClient != nil ? "After the other Mac's image" : "After the queued images")
                                .font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
            Text("Updates come from the GitHub releases of \(Updater.repository). The download is checked "
                 + "(size, SHA-256, signature) before it replaces the app; your models, images and settings stay. "
                 + "macOS asks for Local Network access again after an update.")
                .font(.caption).foregroundStyle(.secondary)
        }
    }

    private var busy: Bool { model.queueCount > 0 || model.servingClient != nil }

    @ViewBuilder private var status: some View {
        switch model.updater.state {
        case .idle:
            if let last = model.updater.lastCheck {
                Text("Checked \(last.formatted(.relative(presentation: .named)))").font(.caption).foregroundStyle(.secondary)
            }
        case .checking: ProgressView().controlSize(.small)
        case .upToDate:
            Label("Up to date", systemImage: "checkmark.circle.fill").font(.caption).foregroundStyle(.green)
        case .available: EmptyView()
        case .downloading:
            HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Downloading…").font(.caption) }
        case .installing:
            HStack(spacing: 6) { ProgressView().controlSize(.small); Text("Installing…").font(.caption) }
        case .failed(let message):
            Text(message).font(.caption).foregroundStyle(.orange).fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
    }
}

/// App menu → Check for Updates…: opens Settings and checks.
struct CheckForUpdatesButton: View {
    @Environment(\.openSettings) private var openSettings
    let updater: Updater

    var body: some View {
        Button("Check for Updates…") {
            openSettings()
            Task { await updater.check() }
        }
    }
}
