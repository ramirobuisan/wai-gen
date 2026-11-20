import SwiftUI

/// The sidebar: characters or artist styles.
struct LibrarySidebar: View {
    enum Tab: String { case characters, styles }
    @Environment(AppModel.self) private var model
    @AppStorage("sidebarTab") private var tab: Tab = .characters
    @State private var characterQuery = ""
    @State private var styleQuery = ""

    var body: some View {
        Group {
            switch tab {
            case .characters: CharacterSidebar(query: characterQuery)
            case .styles: StyleSidebar(query: styleQuery)
            }
        }
        // one search field for both lists (a field declared inside each list doesn't come back after switching)
        .searchable(text: tab == .characters ? $characterQuery : $styleQuery, placement: .sidebar, prompt: searchPrompt)
        .safeAreaInset(edge: .top, spacing: 0) {
            Picker("Show", selection: $tab) {
                Text("Characters").tag(Tab.characters)
                Text("Styles").tag(Tab.styles)
            }
            .pickerStyle(.segmented)
            .labelsHidden()
            .padding(.horizontal, 10)
            .padding(.vertical, 6)
        }
    }

    private var searchPrompt: String {
        switch tab {
        case .characters:
            let n = model.characters.characters.count
            return n == 0 ? "Search characters" : "Search \(n.formatted()) characters"
        case .styles:
            let n = model.styles.styles.count
            return n == 0 ? "Search styles" : "Search \(n.formatted()) artists"
        }
    }
}

/// Artist styles with a sample each (same prompt, only the style changes); a click adds the artist's tag.
struct StyleSidebar: View {
    @Environment(AppModel.self) private var model
    /// the sidebar's search text (LibrarySidebar owns the field)
    let query: String

    var body: some View {
        @Bindable var store = model.styles
        let results = store.search(query)
        let favorites = results.filter { store.favorites.contains($0.tag) }

        // a List (two styles a row) rather than a grid in a ScrollView: the sidebar's search field only attaches
        // to a List, and List rows load lazily
        List {
            switch store.state {
            case .loading, .idle:
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Loading styles…").foregroundStyle(.secondary)
                }
            case .failed(let message):
                Label("Couldn't load styles: \(message)", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.secondary)
                Button("Try again") { store.loadIfNeeded() }
            case .ready:
                if !favorites.isEmpty {
                    Section("Favorites") { rows(favorites) }
                }
                Section(query.isEmpty ? "\(results.count.formatted()) styles"
                        : "\(results.count.formatted()) match\(results.count == 1 ? "" : "es")") {
                    rows(results)
                }
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .bottom) {
            VStack(alignment: .leading, spacing: 6) {
                Divider()
                Text(model.preset.engine == .anima
                     ? "Samples made with Illustrious XL. On WAI-ANIMA the artist goes in as @name, as Anima expects, "
                       + "and may look different."
                     : "Samples made with Illustrious XL, the base of WAI v16. Click one to add the artist to the prompt.")
                    .font(.caption2).foregroundStyle(.secondary)
                    .lineLimit(3)   // not fixedSize: a narrow width while the sidebar lays out would make it very tall
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 10)
                HStack {
                    Picker("Sort", selection: $store.sort) {
                        ForEach(StyleStore.Sort.allCases) { Text($0.label).tag($0) }
                    }
                    .pickerStyle(.menu)
                    .help("Popularity: the artist's number of works on Danbooru")
                    Button { store.shuffle() } label: { Image(systemName: "shuffle") }
                        .buttonStyle(.borderless)
                        .help("Shuffle the styles")
                }
                .padding(.horizontal, 10)
                Link("From ThetaCursed's Style Explorer (MIT)", destination: StyleStore.page)
                    .font(.caption2)
                    .padding(.horizontal, 10)
                    .padding(.bottom, 8)
            }
            .background(.bar)
        }
        .onAppear { store.loadIfNeeded() }
    }

    @ViewBuilder private func rows(_ list: [StyleEntry]) -> some View {
        ForEach(Array(stride(from: 0, to: list.count, by: 2)), id: \.self) { i in
            HStack(alignment: .top, spacing: 8) {
                StyleCell(entry: list[i])
                if i + 1 < list.count {
                    StyleCell(entry: list[i + 1])
                } else {
                    Color.clear.frame(maxWidth: .infinity)
                }
            }
            .listRowSeparator(.hidden)
            .listRowInsets(EdgeInsets(top: 4, leading: 8, bottom: 4, trailing: 8))
        }
    }
}

struct StyleCell: View {
    @Environment(AppModel.self) private var model
    let entry: StyleEntry
    @State private var image: NSImage?

    var body: some View {
        let selected = model.settings.artists.contains(entry.tag)
        let favorite = model.styles.favorites.contains(entry.tag)
        VStack(alignment: .leading, spacing: 2) {
            Rectangle().fill(.quaternary)
                .aspectRatio(520.0 / 760.0, contentMode: .fit)
                .overlay {
                    if let image { Image(nsImage: image).resizable().scaledToFill() }
                }
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay {
                    RoundedRectangle(cornerRadius: 6)
                        .strokeBorder(selected ? Color.accentColor : Color.clear, lineWidth: 3)
                }
                .overlay(alignment: .topTrailing) {
                    if selected {
                        Image(systemName: "checkmark.circle.fill")
                            .font(.title3)
                            .foregroundStyle(.white, Color.accentColor)
                            .padding(5)
                    }
                }
            Text(entry.tag).font(.caption).lineLimit(1).truncationMode(.middle)
            HStack(spacing: 3) {
                Text("\(entry.works.formatted()) works")
                if favorite { Image(systemName: "star.fill").foregroundStyle(.yellow) }
            }
            .font(.caption2).foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .contentShape(Rectangle())
        .onTapGesture { model.toggleStyle(entry.tag) }
        .task(id: entry.id) {
            image = model.styles.cachedSample(entry)
            if image == nil { image = await model.styles.sample(entry) }
        }
        .contextMenu {
            Button(selected ? "Remove from prompt" : "Add to prompt") { model.toggleStyle(entry.tag) }
            Button(favorite ? "Remove from favorites" : "Add to favorites") { model.styles.toggleFavorite(entry.tag) }
            Divider()
            Button("Copy tag") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(model.styleTag(entry.tag), forType: .string)
            }
        }
        .help(selected ? "\(entry.tag): click to remove from the prompt" : "\(entry.tag): click to add to the prompt")
    }
}
