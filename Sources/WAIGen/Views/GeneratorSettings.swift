import SwiftUI
import WaiRemote

/// Settings: generate on another Mac (found with Bonjour on the same network, or typed: a Tailscale name or
/// address), and share this Mac as a generator for others.
struct GeneratorSettingsSections: View {
    @Environment(AppModel.self) private var model
    @State private var found: [RemoteEndpoint] = []
    @State private var browser: RemoteBrowser?
    /// "local", "service:<name>" or "address"
    @State private var choice = "local"
    @State private var address = ""
    @State private var code = ""
    @State private var sharing = UserDefaults.standard.bool(forKey: AppModel.shareKey)
    @State private var shareCode = AppModel.shareCode

    var body: some View {
        Section("Generator") {
            Picker("Generate on", selection: $choice) {
                Text("This Mac").tag("local")
                ForEach(services, id: \.self) { name in Text(name).tag("service:" + name) }
                Text("Another address (e.g. Tailscale)…").tag("address")
            }
            if choice == "address" {
                TextField("Address", text: $address, prompt: Text("my-mac.tailnet.ts.net or 100.x.y.z"))
            }
            if choice != "local" {
                TextField("Pairing code", text: $code, prompt: Text("shown in the other Mac's Settings"))
                HStack {
                    Button(isCurrentChoice ? "Reconnect" : "Use this Mac") { apply() }
                        .disabled(code.isEmpty || (choice == "address" && RemoteEndpoint.address(address) == nil))
                    Spacer()
                    connectionStatus
                }
            } else if model.isRemote {
                Button("Generate on this Mac again") { model.setGenerator(nil) }
            }
            Text("The other Mac needs WAI Gen open with “Let other Macs generate on this Mac” on. Images are saved here; "
                 + "the models, LoRAs and upscalers are the other Mac's. Macs on the same Wi-Fi appear in the list; over "
                 + "Tailscale type the other Mac's Tailscale name or 100.x address.")
                .font(.caption).foregroundStyle(.secondary)
        }
        Section("Share this Mac") {
            Toggle("Let other Macs generate on this Mac", isOn: $sharing)
                .onChange(of: sharing) { _, on in model.setSharing(on) }
            if sharing {
                LabeledContent("Pairing code") {
                    HStack {
                        Text(shareCode).font(.title3.monospaced()).textSelection(.enabled)
                        Button("New code") {
                            model.setSharing(true, newCode: true)
                            shareCode = AppModel.shareCode
                        }
                    }
                }
                LabeledContent("Addresses") {
                    VStack(alignment: .trailing, spacing: 2) {
                        ForEach(LocalAddresses.current(), id: \.address) { a in
                            Text("\(a.kind): \(a.address)").textSelection(.enabled)
                        }
                        Text("port \(String(RemoteProtocol.defaultPort))").foregroundStyle(.secondary)
                    }
                }
                if let status = model.sharingStatus {
                    Label(status == "listening" ? "Waiting for other Macs" : status,
                          systemImage: status == "listening" ? "dot.radiowaves.left.and.right" : "exclamationmark.triangle")
                        .font(.caption).foregroundStyle(status == "listening" ? Color.secondary : Color.orange)
                }
                Text("Only Macs on your local network or your Tailscale network can connect, and only with the code. "
                     + "Over Wi-Fi the traffic isn't encrypted (prompts and images); over Tailscale it is.")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
        .onAppear(perform: appear)
        .onDisappear { browser?.stop() }
    }

    private var services: [String] {
        var names = found.map(\.name)
        if let s = model.generator?.endpoint.service, !names.contains(s) { names.append(s) }   // saved but not seen yet
        return names.sorted()
    }

    private var isCurrentChoice: Bool {
        guard let g = model.generator else { return false }
        return choice == tag(for: g.endpoint)
    }

    @ViewBuilder private var connectionStatus: some View {
        if isCurrentChoice {
            if let info = model.remoteInfo {
                Label("Connected: \(info.presets.count) model\(info.presets.count == 1 ? "" : "s"), \(info.loras.count) LoRA\(info.loras.count == 1 ? "" : "s")",
                      systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green).font(.caption)
            } else if let s = model.remoteStatus {
                Text(s).font(.caption).foregroundStyle(s.hasPrefix("Connecting") ? Color.secondary : Color.orange)
                    .lineLimit(2)
            }
        }
    }

    private func tag(for e: RemoteEndpoint) -> String {
        if let s = e.service { return "service:" + s }
        return "address"
    }

    private func appear() {
        if let g = model.generator {
            choice = tag(for: g.endpoint)
            code = g.code
            if g.endpoint.service == nil, let host = g.endpoint.host {
                address = g.endpoint.port.map { $0 == Int(RemoteProtocol.defaultPort) ? host : "\(host):\($0)" } ?? host
            }
        }
        let b = RemoteBrowser()
        let me = Host.current().localizedName
        b.onChange = { list in
            Task { @MainActor in found = list.filter { $0.name != me || !sharing } }
        }
        b.start()
        browser = b
    }

    private func apply() {
        let endpoint: RemoteEndpoint?
        if choice.hasPrefix("service:") {
            let name = String(choice.dropFirst("service:".count))
            endpoint = RemoteEndpoint(name: name, service: name)
        } else {
            endpoint = RemoteEndpoint.address(address)
        }
        guard let endpoint else { return }
        model.setGenerator(AppModel.GeneratorChoice(endpoint: endpoint, code: code.trimmingCharacters(in: .whitespaces)))
    }
}
