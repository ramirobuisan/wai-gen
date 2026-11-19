import AppKit
import SwiftUI

/// Lets menu commands reach the image stage (View ▸ Zoom In / Zoom Out / Actual Size / Zoom to Fit).
@MainActor
final class StageZoom {
    fileprivate weak var view: StageScrollView?
    func zoomIn() { view?.zoom(by: 1.25) }
    func zoomOut() { view?.zoom(by: 0.8) }
    func actualSize() { view?.setZoom(1) }
    func fit() { view?.fitImage() }
}

/// The finished image in a magnifying scroll view: pinch or ⌘+ / ⌘- to zoom, two-finger scroll or drag
/// to pan, double-click to switch between fit and 100 %. When it fits, dragging it drags the file out.
struct ZoomableImage: NSViewRepresentable {
    let image: NSImage
    let fileURL: URL?
    let zoom: StageZoom

    func makeNSView(context: Context) -> StageScrollView {
        let view = StageScrollView()
        zoom.view = view
        return view
    }

    func updateNSView(_ view: StageScrollView, context: Context) {
        zoom.view = view
        view.imageView.fileURL = fileURL
        if view.imageView.image !== image { view.show(image) }
    }
}

final class StageScrollView: NSScrollView {
    let imageView = StageImageView()
    private(set) var isFit = true
    private var settingZoom = false

    init() {
        super.init(frame: .zero)
        contentView = CenteringClipView()
        drawsBackground = false
        contentView.drawsBackground = false
        hasHorizontalScroller = true
        hasVerticalScroller = true
        autohidesScrollers = true
        scrollerStyle = .overlay
        allowsMagnification = true
        minMagnification = 0.02
        maxMagnification = 16
        documentView = imageView
        imageView.stage = self
        NotificationCenter.default.addObserver(self, selector: #selector(userZoomed), name: NSScrollView.didEndLiveMagnifyNotification, object: self)
    }

    required init?(coder: NSCoder) { fatalError() }

    func show(_ image: NSImage) {
        imageView.image = image
        imageView.frame = NSRect(origin: .zero, size: image.size)
        fitImage()
    }

    /// Magnification at which the whole image fits (with a small margin).
    private var fitMagnification: CGFloat {
        let size = imageView.frame.size
        let avail = NSSize(width: max(bounds.width - 24, 1), height: max(bounds.height - 24, 1))
        guard size.width > 0, size.height > 0 else { return 1 }
        return min(avail.width / size.width, avail.height / size.height)
    }

    func fitImage() {
        isFit = true
        apply(fitMagnification, centeredAt: NSPoint(x: imageView.frame.midX, y: imageView.frame.midY))
    }

    func setZoom(_ m: CGFloat, centeredAt point: NSPoint? = nil) {
        isFit = false
        let visible = contentView.documentVisibleRect
        apply(m, centeredAt: point ?? NSPoint(x: visible.midX, y: visible.midY))
    }

    func zoom(by factor: CGFloat) {
        setZoom(min(max(magnification * factor, minMagnification), maxMagnification))
    }

    func toggleZoom(at point: NSPoint) {
        if isFit, fitMagnification < 0.95 { setZoom(1, centeredAt: point) } else { fitImage() }
    }

    private func apply(_ m: CGFloat, centeredAt point: NSPoint) {
        settingZoom = true
        setMagnification(m, centeredAt: point)
        settingZoom = false
        imageView.needsDisplay = true
    }

    @objc private func userZoomed() {
        if !settingZoom { isFit = false }
        imageView.needsDisplay = true
    }

    override func setFrameSize(_ newSize: NSSize) {
        super.setFrameSize(newSize)
        if isFit { fitImage() }   // keep fitting while the window or panels change size
    }
}

/// Centres the image when it is smaller than the visible area.
final class CenteringClipView: NSClipView {
    override func constrainBoundsRect(_ proposed: NSRect) -> NSRect {
        var r = super.constrainBoundsRect(proposed)
        guard let doc = documentView else { return r }
        if r.width > doc.frame.width { r.origin.x = (doc.frame.width - r.width) / 2 }
        if r.height > doc.frame.height { r.origin.y = (doc.frame.height - r.height) / 2 }
        return r
    }
}

final class StageImageView: NSView, NSDraggingSource {
    var image: NSImage? { didSet { needsDisplay = true } }
    var fileURL: URL?
    weak var stage: StageScrollView?
    private var dragStart: NSPoint?
    private var panLast: NSPoint?

    override var isFlipped: Bool { true }

    override func draw(_ dirtyRect: NSRect) {
        guard let image else { return }
        // smooth when shrunk, sharp pixels from 200 % up
        NSGraphicsContext.current?.imageInterpolation = (stage?.magnification ?? 1) >= 2 ? .none : .high
        image.draw(in: bounds, from: .zero, operation: .sourceOver, fraction: 1, respectFlipped: true, hints: nil)
    }

    override func mouseDown(with event: NSEvent) {
        window?.makeFirstResponder(nil)   // end text editing so ← / → browse the gallery
        if event.clickCount == 2 {
            stage?.toggleZoom(at: convert(event.locationInWindow, from: nil))
            return
        }
        dragStart = event.locationInWindow
        panLast = event.locationInWindow
    }

    override func mouseDragged(with event: NSEvent) {
        guard let stage else { return }
        if stage.isFit {
            // fitted: drag the image file out (to Finder, another app, ...)
            guard let start = dragStart, let url = fileURL, let image,
                  hypot(event.locationInWindow.x - start.x, event.locationInWindow.y - start.y) > 4 else { return }
            dragStart = nil
            let item = NSDraggingItem(pasteboardWriter: url as NSURL)
            item.setDraggingFrame(bounds, contents: image)
            beginDraggingSession(with: [item], event: event, source: self)
            return
        }
        // zoomed in: drag to pan
        guard let last = panLast else { return }
        let p = event.locationInWindow
        panLast = p
        let clip = stage.contentView
        var origin = clip.bounds.origin
        origin.x -= (p.x - last.x) / stage.magnification
        origin.y += (p.y - last.y) / stage.magnification   // document is flipped
        clip.scroll(to: clip.constrainBoundsRect(NSRect(origin: origin, size: clip.bounds.size)).origin)
        stage.reflectScrolledClipView(clip)
    }

    override func mouseUp(with event: NSEvent) {
        dragStart = nil
        panLast = nil
    }

    func draggingSession(_ session: NSDraggingSession, sourceOperationMaskFor context: NSDraggingContext) -> NSDragOperation {
        .copy
    }
}
