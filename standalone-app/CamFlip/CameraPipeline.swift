import AVFoundation
import CoreMedia
import Foundation

/// Captures the physical camera at full quality, runs the frame processor, and feeds the
/// result to the virtual camera and the preview window.
final class CameraPipeline: NSObject, AVCaptureVideoDataOutputSampleBufferDelegate {
    let processor = FrameProcessor()
    let sink = VirtualCameraSink()

    private let session = AVCaptureSession()
    private let output = AVCaptureVideoDataOutput()
    private let sessionQueue = DispatchQueue(label: "com.mrwaleeders.CamFlip.session")
    private let videoQueue = DispatchQueue(label: "com.mrwaleeders.CamFlip.video", qos: .userInteractive)

    private var currentInput: AVCaptureDeviceInput?
    private var desiredSize = (width: Int32(1920), height: Int32(1080))

    private let previewLock = NSLock()
    private var _previewLayer: AVSampleBufferDisplayLayer?
    var previewLayer: AVSampleBufferDisplayLayer? {
        get { previewLock.lock(); defer { previewLock.unlock() }; return _previewLayer }
        set { previewLock.lock(); _previewLayer = newValue; previewLock.unlock() }
    }

    /// Called on the main queue with a short description such as "1920×1080 · 30 fps".
    var onCaptureInfo: ((String) -> Void)?
    /// Called on the main queue when the virtual camera connection changes.
    var onSinkConnected: ((Bool) -> Void)?

    private(set) var isRunning = false
    private var lastConnectAttempt = Date.distantPast

    override init() {
        super.init()
        output.alwaysDiscardsLateVideoFrames = true
        // BGRA straight from the camera's hardware converter, so the no-effects path is a
        // true pass-through and the GPU path needs no extra conversion.
        output.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        output.setSampleBufferDelegate(self, queue: videoQueue)
    }

    static func availableCameras() -> [AVCaptureDevice] {
        AVCaptureDevice.DiscoverySession(
            deviceTypes: [.builtInWideAngleCamera, .external, .continuityCamera],
            mediaType: .video,
            position: .unspecified
        ).devices.filter { device in
            // Never capture our own virtual camera (that would loop).
            device.uniqueID != CamFlipShared.deviceUUID.uuidString &&
                !device.localizedName.localizedCaseInsensitiveContains("Cam Flip")
        }
    }

    func start(cameraID: String?) {
        sessionQueue.async { [self] in
            configure(cameraID: cameraID)
            if !session.isRunning { session.startRunning() }
            isRunning = true
            connectSinkIfNeeded(force: true)
        }
    }

    func stop() {
        sessionQueue.async { [self] in
            if session.isRunning { session.stopRunning() }
            isRunning = false
            sink.disconnect()
            DispatchQueue.main.async { self.onSinkConnected?(false) }
        }
    }

    /// The call app picked a format on the virtual camera; capture at that size if the
    /// camera supports it, so frames pass through without scaling.
    func setDesiredFormat(index: Int) {
        guard CamFlipShared.formats.indices.contains(index) else { return }
        let size = CamFlipShared.formats[index]
        sessionQueue.async { [self] in
            guard desiredSize != size else { return }
            desiredSize = size
            if let device = currentInput?.device { applyBestFormat(to: device) }
        }
    }

    // MARK: Session

    private func configure(cameraID: String?) {
        let cameras = Self.availableCameras()
        guard let device = cameras.first(where: { $0.uniqueID == cameraID })
            ?? cameras.first(where: { $0.deviceType == .builtInWideAngleCamera })
            ?? cameras.first else { return }

        if currentInput?.device.uniqueID == device.uniqueID {
            applyBestFormat(to: device)
            return
        }

        session.beginConfiguration()
        if let currentInput { session.removeInput(currentInput) }
        if let input = try? AVCaptureDeviceInput(device: device), session.canAddInput(input) {
            session.addInput(input)
            currentInput = input
        }
        if !session.outputs.contains(output), session.canAddOutput(output) {
            session.addOutput(output)
        }
        session.commitConfiguration()
        applyBestFormat(to: device)
    }

    private func applyBestFormat(to device: AVCaptureDevice) {
        let target = desiredSize
        let candidates = device.formats.filter { format in
            format.videoSupportedFrameRateRanges.contains { $0.maxFrameRate >= Double(CamFlipShared.frameRate) - 0.5 }
        }
        func dims(_ f: AVCaptureDevice.Format) -> CMVideoDimensions {
            CMVideoFormatDescriptionGetDimensions(f.formatDescription)
        }
        let exact = candidates.first { dims($0).width == target.width && dims($0).height == target.height }
        let larger = candidates
            .filter { dims($0).width >= target.width && dims($0).height >= target.height }
            .min { dims($0).width * dims($0).height < dims($1).width * dims($1).height }
        let largest = candidates.max { dims($0).width * dims($0).height < dims($1).width * dims($1).height }
        guard let format = exact ?? larger ?? largest else { return }

        do {
            try device.lockForConfiguration()
            device.activeFormat = format
            let frameDuration = CMTime(value: 1, timescale: CamFlipShared.frameRate)
            if format.videoSupportedFrameRateRanges.contains(where: {
                $0.minFrameDuration <= frameDuration && frameDuration <= $0.maxFrameDuration
            }) {
                device.activeVideoMinFrameDuration = frameDuration
                device.activeVideoMaxFrameDuration = frameDuration
            }
            device.unlockForConfiguration()
        } catch {
            return
        }

        let d = dims(format)
        let info = "\(device.localizedName) · \(d.width)×\(d.height) · \(CamFlipShared.frameRate) fps"
        DispatchQueue.main.async { self.onCaptureInfo?(info) }
    }

    private func connectSinkIfNeeded(force: Bool = false) {
        guard !sink.isConnected else { return }
        guard force || Date().timeIntervalSince(lastConnectAttempt) > 1 else { return }
        lastConnectAttempt = Date()
        let connected = sink.connect()
        DispatchQueue.main.async { self.onSinkConnected?(connected) }
    }

    // MARK: Frames

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer, from connection: AVCaptureConnection) {
        guard let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }

        if !sink.isConnected {
            sessionQueue.async { self.connectSinkIfNeeded() }
        }

        let processed = processor.process(pixelBuffer)
        guard let outBuffer = makeSampleBuffer(processed, timing: sampleBuffer) else { return }

        sink.send(outBuffer)

        if let layer = previewLayer {
            let renderer = layer.sampleBufferRenderer
            if renderer.status == .failed { renderer.flush() }
            renderer.enqueue(outBuffer)
        }
    }

    private func makeSampleBuffer(_ pixelBuffer: CVPixelBuffer, timing source: CMSampleBuffer) -> CMSampleBuffer? {
        var formatDescription: CMFormatDescription?
        guard CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault, imageBuffer: pixelBuffer, formatDescriptionOut: &formatDescription
        ) == noErr, let formatDescription else { return nil }

        var timing = CMSampleTimingInfo(
            duration: CMSampleBufferGetDuration(source),
            presentationTimeStamp: CMSampleBufferGetPresentationTimeStamp(source),
            decodeTimeStamp: .invalid
        )
        var sampleBuffer: CMSampleBuffer?
        guard CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: formatDescription,
            sampleTiming: &timing,
            sampleBufferOut: &sampleBuffer
        ) == noErr, let sampleBuffer else { return nil }

        if let attachments = CMSampleBufferGetSampleAttachmentsArray(sampleBuffer, createIfNecessary: true),
           CFArrayGetCount(attachments) > 0 {
            let dict = unsafeBitCast(CFArrayGetValueAtIndex(attachments, 0), to: CFMutableDictionary.self)
            CFDictionarySetValue(
                dict,
                Unmanaged.passUnretained(kCMSampleAttachmentKey_DisplayImmediately).toOpaque(),
                Unmanaged.passUnretained(kCFBooleanTrue).toOpaque()
            )
        }
        return sampleBuffer
    }
}
