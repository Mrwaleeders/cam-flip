import AVFoundation
import SwiftUI

struct PreviewWindow: View {
    @EnvironmentObject private var model: AppModel

    var body: some View {
        VStack(spacing: 0) {
            SampleBufferView(pipeline: model.pipeline)
                .background(Color.black)
            HStack {
                Text("This is exactly what the other people in your call see.")
                Spacer()
                Text(model.captureInfo)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(8)
        }
        .frame(minWidth: 480, minHeight: 300)
        .onAppear { model.previewOpen = true }
        .onDisappear {
            model.pipeline.previewLayer = nil
            model.previewOpen = false
        }
    }
}

final class SampleBufferHostView: NSView {
    let displayLayer = AVSampleBufferDisplayLayer()

    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        wantsLayer = true
        displayLayer.videoGravity = .resizeAspect
        displayLayer.backgroundColor = NSColor.black.cgColor
        layer = displayLayer
    }

    required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }
}

struct SampleBufferView: NSViewRepresentable {
    let pipeline: CameraPipeline

    func makeNSView(context: Context) -> SampleBufferHostView {
        let view = SampleBufferHostView()
        pipeline.previewLayer = view.displayLayer
        return view
    }

    func updateNSView(_ nsView: SampleBufferHostView, context: Context) {}

    static func dismantleNSView(_ nsView: SampleBufferHostView, coordinator: ()) {
        nsView.displayLayer.sampleBufferRenderer.flush()
    }
}
