import SwiftUI

struct CharacterSidebar: View {
    @Environment(AppModel.self) private var model
    /// the sidebar's search text (LibrarySidebar owns the field)
    let query: String

    var body: some View {
        let store = model.characters
        let results = store.search(query)
        let favorites = results.filter { store.favorites.contains($0.tag) }

        List {
            switch store.state {
            case .loading(let message):
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text(message).foregroundStyle(.secondary)
                }
            case .failed(let message):
                Label("Couldn't load characters: \(message)", systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.secondary)
                Button("Try again") { store.load(model.characterList) }
            default:
                EmptyView()
            }

            if !favorites.isEmpty {
                Section("Favorites") {
                    ForEach(favorites) { CharacterRow(entry: $0) }
                }
            }
            if store.state == .ready {
                Section(query.isEmpty ? "\(results.count.formatted()) characters"
                        : "\(results.count.formatted()) match\(results.count == 1 ? "" : "es")") {
                    ForEach(results) { CharacterRow(entry: $0) }
                }
            }
        }
        .listStyle(.sidebar)
        .safeAreaInset(edge: .bottom) {
            @Bindable var store = model.characters
            VStack(alignment: .leading, spacing: 4) {
                Divider()
                Picker("Sort", selection: $store.sort) {
                    ForEach(CharacterStore.Sort.allCases) { Text($0.label).tag($0) }
                }
                .pickerStyle(.menu)
                .padding(.horizontal, 10)
                .help("Popularity: the character's number of posts on Danbooru")
                Picker("Character list", selection: Binding(
                    get: { model.characterList }, set: { model.characterList = $0 })) {
                    ForEach(CharacterListID.allCases) { Text($0.label).tag($0) }
                }
                .pickerStyle(.menu)
                .padding(.horizontal, 10)
                .padding(.bottom, 8)
                .help("Thumbnails and tags were made with this model")
            }
            .background(.bar)
        }
    }
}

struct CharacterRow: View {
    @Environment(AppModel.self) private var model
    let entry: CharacterEntry
    @State private var thumb: NSImage?

    var body: some View {
        let selected = model.settings.characters.contains(entry.tag)
        HStack(spacing: 8) {
            Group {
                if let thumb {
                    Image(nsImage: thumb).resizable().scaledToFill()
                } else {
                    Rectangle().fill(.quaternary)
                }
            }
            .frame(width: 34, height: 44)
            .clipShape(RoundedRectangle(cornerRadius: 4))
            .hoverPreview(thumb, id: "character:" + entry.tag, title: entry.tag, subtitle: entry.localName)

            VStack(alignment: .leading, spacing: 1) {
                Text(entry.tag).lineLimit(1)
                Group {
                    if let posts = model.characters.posts(entry) {
                        Text("\(posts.formatted()) posts · \(entry.localName)")
                    } else {
                        Text(entry.localName)
                    }
                }
                .font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
            Spacer(minLength: 0)
            if model.characters.favorites.contains(entry.tag) {
                Image(systemName: "star.fill").font(.caption).foregroundStyle(.yellow)
            }
            if selected {
                Image(systemName: "checkmark.circle.fill").foregroundStyle(Color.accentColor)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { model.toggleCharacter(entry.tag) }
        .task(id: entry.id) { thumb = model.characters.thumbnail(for: entry) }
        .contextMenu {
            Button(selected ? "Remove from prompt" : "Add to prompt") { model.toggleCharacter(entry.tag) }
            Button(model.characters.favorites.contains(entry.tag) ? "Remove from favorites" : "Add to favorites") {
                model.characters.toggleFavorite(entry.tag)
            }
            Divider()
            Button("Copy tag") {
                NSPasteboard.general.clearContents()
                NSPasteboard.general.setString(escapeTag(entry.tag), forType: .string)
            }
        }
        .help(selected ? "Click to remove from the prompt" : "Click to add to the prompt")
    }
}
