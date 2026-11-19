import AppKit
import SwiftUI

/// Tag autocomplete state for one prompt editor.
@MainActor @Observable
final class CompletionController {
    var suggestions: [TagSuggestion] = []
    var selected = 0
    @ObservationIgnored weak var textView: NSTextView?
    @ObservationIgnored private var tokenRange: NSRange?
    @ObservationIgnored let index: TagIndex

    init(index: TagIndex) { self.index = index }

    /// Find the tag being typed (text since the last comma/newline) and look it up.
    func update(for tv: NSTextView) {
        textView = tv
        let ns = tv.string as NSString
        let cursor = tv.selectedRange().location
        guard tv.selectedRange().length == 0, cursor <= ns.length, index.isLoaded else { return clear() }
        var start = cursor
        while start > 0 {
            let c = ns.character(at: start - 1)
            if c == 44 /* , */ || c == 10 /* \n */ { break }
            start -= 1
        }
        var range = NSRange(location: start, length: cursor - start)
        var token = ns.substring(with: range)
        let lead = token.prefix { $0 == " " }.count
        range.location += lead
        range.length -= lead
        token = String(token.dropFirst(lead))
        guard token.count >= 2, !token.contains("__"), !token.hasSuffix(" ") else { return clear() }
        // strip a leading "(" of a weight group so "(long ha" still completes
        let core = token.trimmingCharacters(in: CharacterSet(charactersIn: "("))
        let results = index.complete(core)
        // nothing to suggest once the typed text is already a complete tag
        if results.contains(where: { $0.tag.caseInsensitiveCompare(core) == .orderedSame }) { return clear() }
        tokenRange = NSRange(location: range.location + (token.count - core.count), length: core.utf16.count)
        suggestions = results
        selected = 0
    }

    func move(_ delta: Int) {
        guard !suggestions.isEmpty else { return }
        selected = (selected + delta + suggestions.count) % suggestions.count
    }

    func accept(_ suggestion: TagSuggestion? = nil) {
        guard let tv = textView, let range = tokenRange else { return clear() }
        let s = suggestion ?? suggestions[min(selected, suggestions.count - 1)]
        let ns = tv.string as NSString
        let after = range.location + range.length
        let nextChar = after < ns.length ? ns.character(at: after) : 0
        let insert = escapeTag(s.tag) + (nextChar == 44 ? "" : ", ")
        tv.insertText(insert, replacementRange: range)
        clear()
    }

    func clear() {
        if !suggestions.isEmpty { suggestions = [] }
        tokenRange = nil
    }
}

/// Plain-text prompt editor (NSTextView) with optional tag autocomplete.
struct PromptTextView: NSViewRepresentable {
    @Binding var text: String
    var completion: CompletionController?

    func makeCoordinator() -> Coordinator { Coordinator(self) }

    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSTextView.scrollableTextView()
        scroll.drawsBackground = false
        scroll.hasVerticalScroller = true
        scroll.autohidesScrollers = true
        let tv = scroll.documentView as! NSTextView
        tv.delegate = context.coordinator
        tv.isRichText = false
        tv.importsGraphics = false
        tv.allowsUndo = true
        tv.font = .systemFont(ofSize: 13)
        tv.textColor = .labelColor
        tv.drawsBackground = false
        tv.textContainerInset = NSSize(width: 2, height: 4)
        tv.isAutomaticQuoteSubstitutionEnabled = false
        tv.isAutomaticDashSubstitutionEnabled = false
        tv.isAutomaticTextReplacementEnabled = false
        tv.isContinuousSpellCheckingEnabled = false
        tv.string = text
        return scroll
    }

    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let tv = scroll.documentView as? NSTextView else { return }
        if tv.string != text {
            tv.string = text
            completion?.clear()
        }
    }

    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: PromptTextView
        init(_ parent: PromptTextView) { self.parent = parent }

        func textDidChange(_ notification: Notification) {
            guard let tv = notification.object as? NSTextView else { return }
            parent.text = tv.string
            MainActor.assumeIsolated { parent.completion?.update(for: tv) }
        }

        func textView(_ textView: NSTextView, doCommandBy selector: Selector) -> Bool {
            MainActor.assumeIsolated {
                guard let c = parent.completion, !c.suggestions.isEmpty else { return false }
                switch selector {
                case #selector(NSResponder.moveDown(_:)): c.move(1)
                case #selector(NSResponder.moveUp(_:)): c.move(-1)
                case #selector(NSResponder.insertTab(_:)), #selector(NSResponder.insertNewline(_:)): c.accept()
                case #selector(NSResponder.cancelOperation(_:)): c.clear()
                default: return false
                }
                return true
            }
        }
    }
}

/// Editor + placeholder + suggestion list, styled like a text field.
struct PromptField: View {
    let title: String
    @Binding var text: String
    var placeholder: String
    var completion: CompletionController?
    var minHeight: CGFloat = 56

    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            ZStack(alignment: .topLeading) {
                PromptTextView(text: $text, completion: completion)
                    .frame(height: minHeight)
                if text.isEmpty {
                    Text(placeholder).foregroundStyle(.tertiary).padding(.horizontal, 7).padding(.vertical, 4)
                        .allowsHitTesting(false)
                }
            }
            .padding(4)
            .background(.background.opacity(0.6), in: RoundedRectangle(cornerRadius: 6))
            .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.separator))
            if let c = completion, !c.suggestions.isEmpty {
                SuggestionList(controller: c)
            }
        }
    }
}

struct SuggestionList: View {
    let controller: CompletionController

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            ForEach(Array(controller.suggestions.enumerated()), id: \.element.id) { i, s in
                HStack(spacing: 6) {
                    Circle().fill(color(for: s.category)).frame(width: 6, height: 6)
                    Text(s.tag)
                    if let a = s.matchedAlias { Text("← \(a)").foregroundStyle(.secondary) }
                    Spacer()
                    Text(s.count.formatted(.number.notation(.compactName))).foregroundStyle(.secondary).monospacedDigit()
                }
                .font(.callout)
                .padding(.horizontal, 8).padding(.vertical, 3)
                .background(i == controller.selected ? Color.accentColor.opacity(0.18) : .clear,
                            in: RoundedRectangle(cornerRadius: 4))
                .contentShape(Rectangle())
                .onTapGesture { controller.accept(s) }
            }
            Text("↑↓ to choose · Tab or Return to insert · Esc to close")
                .font(.caption2).foregroundStyle(.tertiary).padding(.horizontal, 8).padding(.top, 2)
        }
        .padding(4)
        .frame(maxWidth: 380, alignment: .leading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 6))
        .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(.separator))
    }

    // Danbooru categories: 0 general, 1 artist, 3 copyright, 4 character, 5 meta
    func color(for category: Int) -> Color {
        switch category {
        case 1: .red
        case 3: .purple
        case 4: .green
        case 5: .orange
        default: .blue
        }
    }
}
