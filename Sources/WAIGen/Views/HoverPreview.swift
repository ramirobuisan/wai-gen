import SwiftUI

/// A bigger look at a sidebar thumbnail, shown over the canvas (beside the sidebar) while the pointer is on it.
struct HoverPreviewItem: Equatable {
    let id: String
    let image: NSImage
    let title: String
    let subtitle: String?

    static func == (a: Self, b: Self) -> Bool { a.id == b.id && a.image === b.image }
}

extension View {
    /// Shows `image` bigger beside the sidebar once the pointer has rested on this view for a moment
    /// (at once when another thumbnail's preview is already showing).
    func hoverPreview(_ image: NSImage?, id: String, title: String, subtitle: String? = nil) -> some View {
        modifier(HoverPreview(image: image, id: id, title: title, subtitle: subtitle))
    }
}

private struct HoverPreview: ViewModifier {
    @Environment(AppModel.self) private var model
    let image: NSImage?
    let id: String
    let title: String
    let subtitle: String?
    @State private var pending: Task<Void, Never>?

    /// long enough that moving the pointer down the list doesn't flash previews
    private static let delay = Duration.milliseconds(300)

    func body(content: Content) -> some View {
        content
            .onHover { inside in
                pending?.cancel()
                if inside, let image {
                    let item = HoverPreviewItem(id: id, image: image, title: title, subtitle: subtitle)
                    if model.hoverPreview != nil {
                        model.hoverPreview = item
                    } else {
                        pending = Task {
                            try? await Task.sleep(for: Self.delay)
                            if !Task.isCancelled { model.hoverPreview = item }
                        }
                    }
                } else if model.hoverPreview?.id == id {
                    model.hoverPreview = nil
                }
            }
            .onDisappear {
                pending?.cancel()
                if model.hoverPreview?.id == id { model.hoverPreview = nil }
            }
    }
}

struct HoverPreviewCard: View {
    let item: HoverPreviewItem
    private let height: CGFloat = 440

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Image(nsImage: item.image)
                .resizable()
                .interpolation(.high)
                .aspectRatio(contentMode: .fit)
                .frame(height: height)
                .clipShape(RoundedRectangle(cornerRadius: 8))
            Text(item.title).font(.headline).lineLimit(1)
            if let subtitle = item.subtitle {
                Text(subtitle).font(.caption).foregroundStyle(.secondary).lineLimit(1)
            }
        }
        .padding(10)
        .frame(width: height * item.image.size.width / max(1, item.image.size.height) + 20, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 12))
        .shadow(radius: 12, y: 4)
    }
}
