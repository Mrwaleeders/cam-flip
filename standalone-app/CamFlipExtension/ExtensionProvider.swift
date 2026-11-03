import CoreImage
import CoreMediaIO
import CoreText
import CoreVideo
import Foundation
import IOKit.audio
import os.log

private let log = Logger(subsystem: "com.mrwaleeders.CamFlip.Extension", category: "camera")

// MARK: - Device

/// The virtual camera. It has two streams:
/// - a **source** stream that Teams, Zoom and Meet read from, and
/// - a **sink** stream that the Cam Flip app writes processed frames into.
///
/// Frames from the sink are forwarded to the source untouched (same IOSurface, no copy,
/// no recompression) whenever their size matches the format the call app picked.
final class ExtensionDeviceSource: NSObject, CMIOExtensionDeviceSource {
    private(set) var device: CMIOExtensionDevice!

    private var sourceStreamSource: StreamSource!
    private var sinkStreamSource: StreamSource!

    private let queue = DispatchQueue(label: "com.mrwaleeders.CamFlip.Extension.device", qos: .userInteractive)

    private var sourceStarted = false
    private var sinkStarted = false
    private var sinkClient: CMIOExtensionClient?
    private var lastSinkFrameNanos: UInt64 = 0
    private var placeholderTimer: DispatchSourceTimer?

    // Only used when a frame's size differs from the format the call app picked.
    private lazy var ciContext = CIContext(options: [.cacheIntermediates: false, .name: "CamFlipExtension"])
    private var scalePool: CVPixelBufferPool?
    private var scalePoolSize = (width: 0, height: 0)
    private var placeholder: CVPixelBuffer?

    init(localizedName: String) {
        super.init()
        device = CMIOExtensionDevice(
            localizedName: localizedName,
            deviceID: CamFlipShared.deviceUUID,
            legacyDeviceID: nil,
            source: self
        )

        let formats: [CMIOExtensionStreamFormat] = CamFlipShared.formats.compactMap { size in
            var description: CMFormatDescription?
            let status = CMVideoFormatDescriptionCreate(
                allocator: kCFAllocatorDefault,
                codecType: kCVPixelFormatType_32BGRA,
                width: size.width,
                height: size.height,
                extensions: nil,
                formatDescriptionOut: &description
            )
            guard status == noErr, let description else { return nil }
            let frameDuration = CMTime(value: 1, timescale: CamFlipShared.frameRate)
            return CMIOExtensionStreamFormat(
                formatDescription: description,
                maxFrameDuration: frameDuration,
                minFrameDuration: frameDuration,
                validFrameDurations: nil
            )
        }

        sourceStreamSource = StreamSource(
            localizedName: "\(localizedName) Output",
            streamID: CamFlipShared.sourceStreamUUID,
            direction: .source,
            formats: formats,
            owner: self
        )
        sinkStreamSource = StreamSource(
            localizedName: "\(localizedName) Input",
            streamID: CamFlipShared.sinkStreamUUID,
            direction: .sink,
            formats: formats,
            owner: self
        )

        do {
            // Order matters: the app looks for the sink as the second stream.
            try device.addStream(sourceStreamSource.stream)
            try device.addStream(sinkStreamSource.stream)
        } catch {
            fatalError("Cam Flip: failed to add streams: \(error.localizedDescription)")
        }

        observeDarwin(CamFlipShared.notifyQuery) { [weak self] in
            self?.queue.async { self?.postState() }
        }
    }

    // MARK: CMIOExtensionDeviceSource

    var availableProperties: Set<CMIOExtensionProperty> {
        [.deviceTransportType, .deviceModel]
    }

    func deviceProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionDeviceProperties {
        let deviceProperties = CMIOExtensionDeviceProperties(dictionary: [:])
        if properties.contains(.deviceTransportType) {
            deviceProperties.transportType = kIOAudioDeviceTransportTypeVirtual
        }
        if properties.contains(.deviceModel) {
            deviceProperties.model = "Cam Flip"
        }
        return deviceProperties
    }

    func setDeviceProperties(_ deviceProperties: CMIOExtensionDeviceProperties) throws {}

    // MARK: Source stream (read by call apps)

    fileprivate func startSource() {
        queue.async {
            self.sourceStarted = true
            self.postState()
            self.startPlaceholderTimer()
        }
    }

    fileprivate func stopSource() {
        queue.async {
            self.sourceStarted = false
            self.placeholderTimer?.cancel()
            self.placeholderTimer = nil
            self.postState()
        }
    }

    fileprivate func activeFormatChanged() {
        queue.async { self.postState() }
    }

    private func postState() {
        postDarwin(sourceStarted ? CamFlipShared.notifyClientStarted : CamFlipShared.notifyClientStopped)
        postDarwin(CamFlipShared.notifyFormat(sourceStreamSource.activeFormatIndex))
    }

    private var activeSize: (width: Int, height: Int) {
        let index = min(max(sourceStreamSource.activeFormatIndex, 0), CamFlipShared.formats.count - 1)
        let size = CamFlipShared.formats[index]
        return (Int(size.width), Int(size.height))
    }

    /// Shows a calm placeholder when the app isn't sending frames, so the call app never
    /// freezes on a stale image.
    private func startPlaceholderTimer() {
        placeholderTimer?.cancel()
        let timer = DispatchSource.makeTimerSource(flags: .strict, queue: queue)
        let interval = 1.0 / Double(CamFlipShared.frameRate)
        timer.schedule(deadline: .now(), repeating: interval, leeway: .milliseconds(2))
        timer.setEventHandler { [weak self] in
            guard let self, self.sourceStarted else { return }
            let now = Self.hostNanos()
            if now &- self.lastSinkFrameNanos > 500_000_000 {
                if let buffer = self.placeholderBuffer() {
                    self.send(buffer)
                }
            }
        }
        timer.resume()
        placeholderTimer = timer
    }

    // MARK: Sink stream (written by the Cam Flip app)

    fileprivate func setSinkClient(_ client: CMIOExtensionClient) {
        queue.async { self.sinkClient = client }
    }

    fileprivate func startSink() {
        queue.async {
            self.sinkStarted = true
            self.consumeNext()
        }
    }

    fileprivate func stopSink() {
        queue.async {
            self.sinkStarted = false
        }
    }

    private func consumeNext() {
        guard sinkStarted, let client = sinkClient else { return }
        sinkStreamSource.stream.consumeSampleBuffer(from: client) { [weak self] sampleBuffer, sequenceNumber, _, hasMore, _ in
            guard let self else { return }
            self.queue.async {
                guard let sampleBuffer else {
                    // Nothing queued yet: check again shortly.
                    self.queue.asyncAfter(deadline: .now() + .milliseconds(3)) { self.consumeNext() }
                    return
                }
                let now = Self.hostNanos()
                self.lastSinkFrameNanos = now
                if self.sourceStarted, let pixelBuffer = CMSampleBufferGetImageBuffer(sampleBuffer) {
                    self.forward(pixelBuffer)
                }
                let output = CMIOExtensionScheduledOutput(sequenceNumber: sequenceNumber, hostTimeInNanoseconds: now)
                self.sinkStreamSource.stream.notifyScheduledOutputChanged(output)
                if hasMore {
                    self.consumeNext()
                } else {
                    self.queue.asyncAfter(deadline: .now() + .milliseconds(1)) { self.consumeNext() }
                }
            }
        }
    }

    private func forward(_ pixelBuffer: CVPixelBuffer) {
        let target = activeSize
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)
        if width == target.width, height == target.height,
           CVPixelBufferGetPixelFormatType(pixelBuffer) == kCVPixelFormatType_32BGRA {
            // Pass-through: same buffer, no copy, no quality change.
            send(pixelBuffer)
        } else if let scaled = scale(pixelBuffer, to: target) {
            send(scaled)
        }
    }

    private func send(_ pixelBuffer: CVPixelBuffer) {
        var formatDescription: CMFormatDescription?
        guard CMVideoFormatDescriptionCreateForImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescriptionOut: &formatDescription
        ) == noErr, let formatDescription else { return }

        var timing = CMSampleTimingInfo(
            duration: CMTime(value: 1, timescale: CamFlipShared.frameRate),
            presentationTimeStamp: CMClockGetTime(CMClockGetHostTimeClock()),
            decodeTimeStamp: .invalid
        )
        var sampleBuffer: CMSampleBuffer?
        guard CMSampleBufferCreateReadyWithImageBuffer(
            allocator: kCFAllocatorDefault,
            imageBuffer: pixelBuffer,
            formatDescription: formatDescription,
            sampleTiming: &timing,
            sampleBufferOut: &sampleBuffer
        ) == noErr, let sampleBuffer else { return }

        sourceStreamSource.stream.send(
            sampleBuffer,
            discontinuity: [],
            hostTimeInNanoseconds: UInt64(timing.presentationTimeStamp.seconds * Double(NSEC_PER_SEC))
        )
    }

    // MARK: Helpers

    /// Aspect-fill scale on the GPU. Only used when the app's frame size differs from what
    /// the call app asked for (for example while the physical camera is switching formats).
    private func scale(_ pixelBuffer: CVPixelBuffer, to size: (width: Int, height: Int)) -> CVPixelBuffer? {
        guard let output = makeBuffer(size) else { return nil }
        let image = CIImage(cvPixelBuffer: pixelBuffer)
        let sx = CGFloat(size.width) / image.extent.width
        let sy = CGFloat(size.height) / image.extent.height
        let s = max(sx, sy)
        let scaledWidth = image.extent.width * s
        let scaledHeight = image.extent.height * s
        let transform = CGAffineTransform(translationX: (CGFloat(size.width) - scaledWidth) / 2,
                                          y: (CGFloat(size.height) - scaledHeight) / 2)
            .scaledBy(x: s, y: s)
        let scaled = image
            .applyingFilter("CILanczosScaleTransform", parameters: [kCIInputScaleKey: s, kCIInputAspectRatioKey: 1.0])
            .transformed(by: CGAffineTransform(translationX: transform.tx, y: transform.ty))
        let bounds = CGRect(x: 0, y: 0, width: size.width, height: size.height)
        ciContext.render(scaled.cropped(to: bounds), to: output, bounds: bounds,
                         colorSpace: image.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB))
        CVBufferPropagateAttachments(pixelBuffer, output)
        return output
    }

    private func makeBuffer(_ size: (width: Int, height: Int)) -> CVPixelBuffer? {
        if scalePool == nil || scalePoolSize.width != size.width || scalePoolSize.height != size.height {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: size.width,
                kCVPixelBufferHeightKey as String: size.height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var pool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &pool)
            scalePool = pool
            scalePoolSize = size
            placeholder = nil
        }
        guard let scalePool else { return nil }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, scalePool, &buffer)
        return buffer
    }

    private func placeholderBuffer() -> CVPixelBuffer? {
        let size = activeSize
        if let placeholder, CVPixelBufferGetWidth(placeholder) == size.width,
           CVPixelBufferGetHeight(placeholder) == size.height {
            return placeholder
        }
        guard let buffer = makeBuffer(size) else { return nil }
        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: size.width,
            height: size.height,
            bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
        ) else { return nil }

        context.setFillColor(CGColor(red: 0.09, green: 0.09, blue: 0.10, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))

        let fontSize = CGFloat(size.height) / 24
        let font = CTFontCreateWithName("Helvetica Neue" as CFString, fontSize, nil)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): font,
            NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 0.75, alpha: 1),
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: "Open the Cam Flip app to start your camera", attributes: attributes)
        )
        let lineWidth = CTLineGetTypographicBounds(line, nil, nil, nil)
        context.textPosition = CGPoint(x: (CGFloat(size.width) - CGFloat(lineWidth)) / 2,
                                       y: CGFloat(size.height) / 2 - fontSize / 3)
        CTLineDraw(line, context)

        placeholder = buffer
        return buffer
    }

    private static func hostNanos() -> UInt64 {
        UInt64(CMClockGetTime(CMClockGetHostTimeClock()).seconds * Double(NSEC_PER_SEC))
    }
}

// MARK: - Darwin notifications

private func postDarwin(_ name: String) {
    CFNotificationCenterPostNotification(
        CFNotificationCenterGetDarwinNotifyCenter(),
        CFNotificationName(name as CFString),
        nil, nil, true
    )
}

private var darwinHandlers: [String: () -> Void] = [:]

private func observeDarwin(_ name: String, handler: @escaping () -> Void) {
    darwinHandlers[name] = handler
    CFNotificationCenterAddObserver(
        CFNotificationCenterGetDarwinNotifyCenter(),
        nil,
        { _, _, name, _, _ in
            guard let name = name?.rawValue as String? else { return }
            darwinHandlers[name]?()
        },
        name as CFString,
        nil,
        .deliverImmediately
    )
}

// MARK: - Streams

final class StreamSource: NSObject, CMIOExtensionStreamSource {
    private(set) var stream: CMIOExtensionStream!
    private let streamFormats: [CMIOExtensionStreamFormat]
    private let isSink: Bool
    private unowned let owner: ExtensionDeviceSource

    private var _activeFormatIndex = 0
    var activeFormatIndex: Int {
        get { _activeFormatIndex }
        set {
            guard newValue >= 0, newValue < streamFormats.count else { return }
            _activeFormatIndex = newValue
            if !isSink { owner.activeFormatChanged() }
        }
    }

    init(localizedName: String,
         streamID: UUID,
         direction: CMIOExtensionStream.Direction,
         formats: [CMIOExtensionStreamFormat],
         owner: ExtensionDeviceSource) {
        self.streamFormats = formats
        self.isSink = direction == .sink
        self.owner = owner
        super.init()
        stream = CMIOExtensionStream(
            localizedName: localizedName,
            streamID: streamID,
            direction: direction,
            clockType: .hostTime,
            source: self
        )
    }

    var formats: [CMIOExtensionStreamFormat] { streamFormats }

    var availableProperties: Set<CMIOExtensionProperty> {
        if isSink {
            return [.streamActiveFormatIndex, .streamFrameDuration,
                    .streamSinkBufferQueueSize, .streamSinkBuffersRequiredForStartup]
        }
        return [.streamActiveFormatIndex, .streamFrameDuration]
    }

    func streamProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionStreamProperties {
        let streamProperties = CMIOExtensionStreamProperties(dictionary: [:])
        if properties.contains(.streamActiveFormatIndex) {
            streamProperties.activeFormatIndex = activeFormatIndex
        }
        if properties.contains(.streamFrameDuration) {
            streamProperties.frameDuration = CMTime(value: 1, timescale: CamFlipShared.frameRate)
        }
        if isSink {
            if properties.contains(.streamSinkBufferQueueSize) {
                streamProperties.sinkBufferQueueSize = 2
            }
            if properties.contains(.streamSinkBuffersRequiredForStartup) {
                streamProperties.sinkBuffersRequiredForStartup = 1
            }
        }
        return streamProperties
    }

    func setStreamProperties(_ streamProperties: CMIOExtensionStreamProperties) throws {
        if let index = streamProperties.activeFormatIndex {
            activeFormatIndex = index
        }
    }

    func authorizedToStartStream(for client: CMIOExtensionClient) -> Bool {
        if isSink { owner.setSinkClient(client) }
        return true
    }

    func startStream() throws {
        isSink ? owner.startSink() : owner.startSource()
    }

    func stopStream() throws {
        isSink ? owner.stopSink() : owner.stopSource()
    }
}

// MARK: - Provider

final class ExtensionProviderSource: NSObject, CMIOExtensionProviderSource {
    private(set) var provider: CMIOExtensionProvider!
    private var deviceSource: ExtensionDeviceSource!

    init(clientQueue: DispatchQueue?) {
        super.init()
        provider = CMIOExtensionProvider(source: self, clientQueue: clientQueue)
        deviceSource = ExtensionDeviceSource(localizedName: CamFlipShared.deviceName)
        do {
            try provider.addDevice(deviceSource.device)
        } catch {
            fatalError("Cam Flip: failed to add device: \(error.localizedDescription)")
        }
        log.info("Cam Flip camera extension started")
    }

    func connect(to client: CMIOExtensionClient) throws {}

    func disconnect(from client: CMIOExtensionClient) {}

    var availableProperties: Set<CMIOExtensionProperty> { [.providerManufacturer] }

    func providerProperties(forProperties properties: Set<CMIOExtensionProperty>) throws -> CMIOExtensionProviderProperties {
        let providerProperties = CMIOExtensionProviderProperties(dictionary: [:])
        if properties.contains(.providerManufacturer) {
            providerProperties.manufacturer = "Cam Flip"
        }
        return providerProperties
    }

    func setProviderProperties(_ providerProperties: CMIOExtensionProviderProperties) throws {}
}
