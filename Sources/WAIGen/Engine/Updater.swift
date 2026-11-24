import AppKit
import CryptoKit
import Foundation
import Observation
import Security

/// Updates from the GitHub releases of ramirobuisan/wai-gen. The Stable channel follows releases; Beta also takes
/// pre-releases. The download (the release's WAI-Gen.zip) must match GitHub's size and SHA-256, be WAI Gen (same
/// bundle id, the release's version) and pass `codesign --verify`; then it replaces this app in place and the app
/// reopens. Models, images and settings live outside the app, so they stay. When this copy is signed with a
/// certificate this Mac has (not ad hoc), the update is signed with it too, so macOS keeps its permissions.
@MainActor @Observable
final class Updater {
    enum Channel: String, CaseIterable, Identifiable {
        case stable, beta
        var id: String { rawValue }
        var label: String { self == .stable ? "Stable (releases)" : "Beta (pre-releases too)" }
    }

    struct Release: Equatable {
        let version: String
        let name: String
        let notes: String
        let prerelease: Bool
        let page: URL
        let asset: URL
        let size: Int
        let sha256: String?
    }

    enum State: Equatable {
        case idle
        case checking
        case upToDate
        case available(Release)
        case downloading
        case installing
        case failed(String)
    }

    nonisolated static let repository = "ramirobuisan/wai-gen"
    nonisolated static let assetName = "WAI-Gen.zip"
    static let channelKey = "updateChannel"
    static let automaticKey = "updateAutomatically"
    static let lastCheckKey = "updateLastCheck"

    private(set) var state: State = .idle
    /// The release being downloaded or installed (`state` no longer carries it then).
    private(set) var installing: Release?
    /// 0...1 while downloading, when the size is known.
    private(set) var downloadProgress: Double?
    /// Goes up when an automatic check finds a new version: the main window opens the Software Update window.
    private(set) var announced = 0
    private(set) var lastCheck: Date? = UserDefaults.standard.object(forKey: Updater.lastCheckKey) as? Date
    var channel: Channel = Channel(rawValue: UserDefaults.standard.string(forKey: Updater.channelKey) ?? "") ?? .stable {
        didSet {
            UserDefaults.standard.set(channel.rawValue, forKey: Self.channelKey)
            if oldValue != channel { Task { await check() } }
        }
    }
    var automatic: Bool = UserDefaults.standard.object(forKey: Updater.automaticKey) as? Bool ?? true {
        didSet { UserDefaults.standard.set(automatic, forKey: Self.automaticKey) }
    }
    @ObservationIgnored private var timer: Task<Void, Never>?

    nonisolated static var currentVersion: String {
        Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "0"
    }

    var available: Release? {
        if case .available(let r) = state { return r }
        return nil
    }

    /// At launch and every few hours while the app stays open: checks when automatic updates are on and the last
    /// check is older than a day.
    func startAutomaticChecks() {
        timer?.cancel()
        timer = Task { [weak self] in
            while !Task.isCancelled {
                if let self, self.automatic, Date().timeIntervalSince(self.lastCheck ?? .distantPast) > 20 * 3600 {
                    await self.check(automatic: true)
                }
                try? await Task.sleep(for: .seconds(6 * 3600))
            }
        }
    }

    /// `automatic`: a background check; a newer version found then announces itself (`announced`).
    func check(automatic: Bool = false) async {
        switch state {
        case .checking, .downloading, .installing: return
        default: break
        }
        state = .checking
        do {
            let releases = try await Self.fetchReleases()
            lastCheck = Date()
            UserDefaults.standard.set(lastCheck, forKey: Self.lastCheckKey)
            let newest = releases
                .filter { channel == .beta || !$0.prerelease }
                .max { Self.isNewer($1.version, than: $0.version) }
            if let newest, Self.isNewer(newest.version, than: Self.currentVersion) {
                state = .available(newest)
                if automatic { announced += 1 }
            } else {
                state = .upToDate
            }
        } catch {
            state = .failed("Couldn't check for updates: \(error.localizedDescription)")
        }
    }

    /// Downloads, checks and installs `release`, then reopens the app. The caller makes sure nothing is generating.
    func install(_ release: Release) async {
        installing = release
        downloadProgress = nil
        state = .downloading
        do {
            let zip = try await Self.download(release) { fraction in
                Task { @MainActor [weak self] in self?.downloadProgress = fraction }
            }
            state = .installing
            let target = Bundle.main.bundleURL
            try await Task.detached { try Self.replace(target, withAppIn: zip, version: release.version) }.value
            try Self.relaunch(target)
        } catch {
            installing = nil
            state = .failed("Couldn't install \(release.version): \(error.localizedDescription)")
        }
    }

    // MARK: - GitHub

    private struct GitHubRelease: Decodable {
        struct Asset: Decodable {
            let name: String
            let browser_download_url: URL
            let size: Int
            let digest: String?
        }
        let tag_name: String
        let name: String?
        let body: String?
        let draft: Bool
        let prerelease: Bool
        let html_url: URL
        let assets: [Asset]
    }

    /// The releases with a WAI-Gen.zip, newest first. `WAIGEN_UPDATE_URL` points somewhere else (for testing).
    nonisolated static func fetchReleases() async throws -> [Release] {
        let env = ProcessInfo.processInfo.environment["WAIGEN_UPDATE_URL"]
        let url = env.flatMap(URL.init(string:))
            ?? URL(string: "https://api.github.com/repos/\(repository)/releases?per_page=30")!
        var request = URLRequest(url: url, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 20)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("WAI-Gen/\(currentVersion)", forHTTPHeaderField: "User-Agent")
        let (data, response) = try await URLSession.shared.data(for: request)
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else {
            throw UpdateError(status == 403 || status == 429 ? "GitHub is limiting requests, try again later"
                                                             : "GitHub answered \(status)")
        }
        return try JSONDecoder().decode([GitHubRelease].self, from: data).compactMap { r in
            guard !r.draft, let asset = r.assets.first(where: { $0.name == assetName }) else { return nil }
            let version = r.tag_name.hasPrefix("v") ? String(r.tag_name.dropFirst()) : r.tag_name
            let sha = asset.digest.flatMap { $0.hasPrefix("sha256:") ? String($0.dropFirst(7)).lowercased() : nil }
            return Release(version: version, name: r.name ?? r.tag_name, notes: r.body ?? "", prerelease: r.prerelease,
                           page: r.html_url, asset: asset.browser_download_url, size: asset.size, sha256: sha)
        }
    }

    /// "1.10.0" > "1.9.2"; "1.0.0" > "1.0.0-beta.2" > "1.0.0-beta.1"; a leading "v" is ignored.
    nonisolated static func isNewer(_ a: String, than b: String) -> Bool {
        func parts(_ v: String) -> ([Int], String?) {
            let v = v.hasPrefix("v") ? String(v.dropFirst()) : v
            let halves = v.split(separator: "-", maxSplits: 1)
            let core = (halves.first ?? "").split(separator: ".").map { Int($0) ?? 0 }
            return (core, halves.count > 1 ? String(halves[1]) : nil)
        }
        let (ca, pa) = parts(a), (cb, pb) = parts(b)
        for i in 0..<max(ca.count, cb.count) {
            let x = i < ca.count ? ca[i] : 0, y = i < cb.count ? cb[i] : 0
            if x != y { return x > y }
        }
        switch (pa, pb) {
        case (nil, nil): return false
        case (nil, _): return true
        case (_, nil): return false
        case let (x?, y?): return x.compare(y, options: .numeric) == .orderedDescending
        }
    }

    // MARK: - Install

    struct UpdateError: LocalizedError {
        let message: String
        init(_ message: String) { self.message = message }
        var errorDescription: String? { message }
    }

    /// Reports the download's progress (0...1) to `progress`.
    private final class ProgressReporter: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        let report: @Sendable (Double) -> Void
        var observation: NSKeyValueObservation?
        init(_ report: @escaping @Sendable (Double) -> Void) { self.report = report }
        func urlSession(_ session: URLSession, didCreateTask task: URLSessionTask) {
            observation = task.progress.observe(\.fractionCompleted) { [report] p, _ in report(p.fractionCompleted) }
        }
    }

    nonisolated private static func download(_ release: Release,
                                             progress: @escaping @Sendable (Double) -> Void) async throws -> URL {
        var request = URLRequest(url: release.asset, cachePolicy: .reloadIgnoringLocalCacheData, timeoutInterval: 60)
        request.setValue("application/octet-stream", forHTTPHeaderField: "Accept")
        let reporter = ProgressReporter(progress)
        let (file, response) = try await URLSession.shared.download(for: request, delegate: reporter)
        reporter.observation = nil
        let status = (response as? HTTPURLResponse)?.statusCode ?? 0
        guard status == 200 else { throw UpdateError("the download answered \(status)") }
        let zip = FileManager.default.temporaryDirectory.appendingPathComponent("WAI-Gen-\(release.version)-\(UUID().uuidString).zip")
        try FileManager.default.moveItem(at: file, to: zip)
        let data = try Data(contentsOf: zip, options: .mappedIfSafe)
        guard data.count == release.size else { throw UpdateError("the download is incomplete") }
        if let expected = release.sha256 {
            let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
            guard actual == expected else { throw UpdateError("the download doesn't match GitHub's checksum") }
        }
        return zip
    }

    /// Unzips the release next to the app, checks it and swaps it in (an atomic rename on APFS).
    nonisolated static func replace(_ target: URL, withAppIn zip: URL, version: String) throws {
        let fm = FileManager.default
        defer { try? fm.removeItem(at: zip) }
        let parent = target.deletingLastPathComponent()
        guard fm.isWritableFile(atPath: parent.path), !target.path.contains("/AppTranslocation/") else {
            throw UpdateError("WAI Gen can't replace itself in \(parent.path). Move it to Applications (or "
                              + "~/Applications) and open it from there.")
        }
        // unpack on the same volume as the app, so the swap is a rename
        let staging = parent.appendingPathComponent(".WAI Gen update \(UUID().uuidString)")
        try fm.createDirectory(at: staging, withIntermediateDirectories: true)
        defer { try? fm.removeItem(at: staging) }
        try run("/usr/bin/ditto", ["-x", "-k", zip.path, staging.path])
        let app = staging.appendingPathComponent("WAI Gen.app")
        guard let info = NSDictionary(contentsOf: app.appendingPathComponent("Contents/Info.plist")),
              info["CFBundleIdentifier"] as? String == Bundle.main.bundleIdentifier else {
            throw UpdateError("the download isn't WAI Gen")
        }
        guard info["CFBundleShortVersionString"] as? String == version else {
            throw UpdateError("the download is version \(info["CFBundleShortVersionString"] ?? "?"), not \(version)")
        }
        try run("/usr/bin/codesign", ["--verify", "--deep", "--strict", app.path])
        // sign it like the running copy, so macOS keeps this Mac's permissions (Local Network); the release
        // signature stays if that fails
        if let identity = runningSigningIdentity() {
            do { try run("/usr/bin/codesign", ["--force", "--deep", "--sign", identity, app.path]) } catch {
                NSLog("WAI Gen update: re-signing with this Mac's certificate failed, keeping the release signature")
            }
        }
        _ = try fm.replaceItemAt(target, withItemAt: app, backupItemName: nil, options: [])
    }

    /// SHA-1 of the certificate this copy is signed with, when it's a real certificate (not ad hoc) whose private
    /// key is in this Mac's keychain; nil otherwise.
    nonisolated static func runningSigningIdentity() -> String? {
        var code: SecStaticCode?
        guard SecStaticCodeCreateWithPath(Bundle.main.bundleURL as CFURL, [], &code) == errSecSuccess, let code
        else { return nil }
        var info: CFDictionary?
        guard SecCodeCopySigningInformation(code, SecCSFlags(rawValue: kSecCSSigningInformation), &info) == errSecSuccess,
              let dict = info as? [String: Any],
              let leaf = (dict[kSecCodeInfoCertificates as String] as? [SecCertificate])?.first
        else { return nil }
        let sha1 = Insecure.SHA1.hash(data: SecCertificateCopyData(leaf) as Data)
            .map { String(format: "%02X", $0) }.joined()
        // only if this Mac can actually sign with it
        let out = Pipe()
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/security")
        p.arguments = ["find-identity", "-v", "-p", "codesigning"]
        p.standardOutput = out
        p.standardError = FileHandle.nullDevice
        guard (try? p.run()) != nil else { return nil }
        let listing = String(decoding: out.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        p.waitUntilExit()
        return listing.contains(sha1) ? sha1 : nil
    }

    nonisolated private static func run(_ tool: String, _ arguments: [String]) throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: tool)
        p.arguments = arguments
        let err = Pipe()
        p.standardError = err
        p.standardOutput = FileHandle.nullDevice
        try p.run()
        p.waitUntilExit()
        guard p.terminationStatus == 0 else {
            let message = String(decoding: err.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            throw UpdateError("\(URL(fileURLWithPath: tool).lastPathComponent) failed: "
                              + message.trimmingCharacters(in: .whitespacesAndNewlines))
        }
    }

    /// Quits and opens the (new) app at `target` once this process has gone.
    static func relaunch(_ target: URL) throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/bin/sh")
        p.arguments = ["-c", "while kill -0 \(ProcessInfo.processInfo.processIdentifier) 2>/dev/null; do sleep 0.2; done; "
                       + "exec /usr/bin/open \"$0\"", target.path]
        try p.run()
        NSApplication.shared.terminate(nil)
    }
}
