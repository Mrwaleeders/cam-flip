import CoreImage
import CoreVideo
import Foundation
import Metal
import Vision

struct ProcessingSettings {
    var mirror = true
    var eyeContact = true
    /// 0...1. 1 fully cancels the measured downward glance (within safe limits).
    var strength: Double = 0.7
    /// Pupil position (relative to the eye) while looking straight into the lens.
    var baseline: Double = 0
}

/// One eye's warp parameters, in Core Image pixel coordinates (origin bottom-left).
struct EyeWarp {
    var center: CGPoint
    var radius: CGSize
    var shift: CGVector

    static let zero = EyeWarp(center: .zero, radius: CGSize(width: 1, height: 1), shift: .zero)

    var vectors: (CIVector, CIVector) {
        (CIVector(x: center.x, y: center.y, z: radius.width, w: radius.height),
         CIVector(x: shift.dx, y: shift.dy))
    }
}

/// GPU frame pipeline: gaze correction, then mirroring, rendered into a new IOSurface-backed
/// buffer at the camera's native size. With both effects off, the camera buffer is returned
/// unchanged.
final class FrameProcessor {
    private let context: CIContext
    private let warpKernel: CIWarpKernel?

    private var pool: CVPixelBufferPool?
    private var poolSize = (width: 0, height: 0)

    private let settingsLock = NSLock()
    private var _settings = ProcessingSettings()
    var settings: ProcessingSettings {
        get { settingsLock.lock(); defer { settingsLock.unlock() }; return _settings }
        set { settingsLock.lock(); _settings = newValue; settingsLock.unlock() }
    }

    private let faceRequest: VNDetectFaceLandmarksRequest = {
        let request = VNDetectFaceLandmarksRequest()
        request.constellation = .constellation76Points
        return request
    }()
    private let sequenceHandler = VNSequenceRequestHandler()

    // Smoothed state, so the correction never jitters between frames.
    private var smoothedLeft: EyeWarp?
    private var smoothedRight: EyeWarp?

    // Calibration
    private var calibrationSamples: [Double] = []
    private var isCalibrating = false
    var onCalibrated: ((Double) -> Void)?
    var onFaceChange: ((Bool) -> Void)?
    private var faceFound = false

    var kernelLoaded: Bool { warpKernel != nil }

    init() {
        if let device = MTLCreateSystemDefaultDevice() {
            context = CIContext(mtlDevice: device, options: [.cacheIntermediates: false, .name: "CamFlip"])
        } else {
            context = CIContext(options: [.cacheIntermediates: false])
        }
        warpKernel = FrameProcessor.loadKernel(bundle: .main)
    }

    static func loadKernel(bundle: Bundle) -> CIWarpKernel? {
        guard let url = bundle.url(forResource: "default", withExtension: "metallib"),
              let data = try? Data(contentsOf: url) else { return nil }
        return try? CIWarpKernel(functionName: "gazeWarp", fromMetalLibraryData: data)
    }

    func startCalibration() {
        settingsLock.lock()
        calibrationSamples = []
        isCalibrating = true
        settingsLock.unlock()
    }

    var calibrating: Bool {
        settingsLock.lock(); defer { settingsLock.unlock() }; return isCalibrating
    }

    // MARK: Pipeline

    func process(_ pixelBuffer: CVPixelBuffer) -> CVPixelBuffer {
        let s = settings
        let width = CVPixelBufferGetWidth(pixelBuffer)
        let height = CVPixelBufferGetHeight(pixelBuffer)

        var eyes: (EyeWarp, EyeWarp)?
        if s.eyeContact || calibrating {
            eyes = analyzeGaze(pixelBuffer, size: CGSize(width: width, height: height), settings: s)
        }
        let warpActive: Bool = {
            guard s.eyeContact, warpKernel != nil, let eyes else { return false }
            return hypot(eyes.0.shift.dx, eyes.0.shift.dy) > 0.05 || hypot(eyes.1.shift.dx, eyes.1.shift.dy) > 0.05
        }()

        if !s.mirror && !warpActive {
            return pixelBuffer
        }

        let source = CIImage(cvPixelBuffer: pixelBuffer)
        let extent = source.extent
        var image = source

        if warpActive, let eyes, let warpKernel {
            let (eyeA, shiftA) = eyes.0.vectors
            let (eyeB, shiftB) = eyes.1.vectors
            let margin = ceil(max(hypot(eyes.0.shift.dx, eyes.0.shift.dy), hypot(eyes.1.shift.dx, eyes.1.shift.dy))) + 2
            if let warped = warpKernel.apply(
                extent: extent,
                roiCallback: { _, rect in rect.insetBy(dx: -margin, dy: -margin) },
                image: source.clampedToExtent(),
                arguments: [eyeA, shiftA, eyeB, shiftB]
            ) {
                image = warped.cropped(to: extent)
            }
        }

        if s.mirror {
            // Exact horizontal flip: every output pixel lands on a source pixel centre,
            // so there is no resampling and no softening.
            image = image.transformed(by: CGAffineTransform(scaleX: -1, y: 1).translatedBy(x: -extent.width, y: 0))
        }

        guard let output = makeBuffer(width: width, height: height) else { return pixelBuffer }
        let colorSpace = source.colorSpace ?? CGColorSpace(name: CGColorSpace.sRGB)!
        context.render(image, to: output, bounds: CGRect(x: 0, y: 0, width: width, height: height), colorSpace: colorSpace)
        CVBufferPropagateAttachments(pixelBuffer, output)
        return output
    }

    private func makeBuffer(width: Int, height: Int) -> CVPixelBuffer? {
        if pool == nil || poolSize.width != width || poolSize.height != height {
            let attributes: [String: Any] = [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
                kCVPixelBufferIOSurfacePropertiesKey as String: [:] as [String: Any],
                kCVPixelBufferMetalCompatibilityKey as String: true,
            ]
            var newPool: CVPixelBufferPool?
            CVPixelBufferPoolCreate(kCFAllocatorDefault, nil, attributes as CFDictionary, &newPool)
            pool = newPool
            poolSize = (width, height)
        }
        guard let pool else { return nil }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(kCFAllocatorDefault, pool, &buffer)
        return buffer
    }

    // MARK: Gaze

    private struct EyeMeasure {
        var center: CGPoint
        var width: CGFloat
        var height: CGFloat
        var pupil: CGPoint
    }

    private func measure(eye: [CGPoint], pupil: [CGPoint]) -> EyeMeasure? {
        guard eye.count >= 4, let pupilPoint = pupil.first else { return nil }
        let xs = eye.map(\.x), ys = eye.map(\.y)
        let minX = xs.min()!, maxX = xs.max()!, minY = ys.min()!, maxY = ys.max()!
        return EyeMeasure(
            center: CGPoint(x: (minX + maxX) / 2, y: (minY + maxY) / 2),
            width: maxX - minX,
            height: maxY - minY,
            pupil: pupilPoint
        )
    }

    /// Finds the eyes, measures how far the pupils sit below where they sit when you look into
    /// the lens, and returns a warp that moves them back up. Large glances (e.g. at the
    /// keyboard) fade out on purpose, because correcting those looks unnatural.
    private func analyzeGaze(_ pixelBuffer: CVPixelBuffer, size: CGSize, settings s: ProcessingSettings) -> (EyeWarp, EyeWarp)? {
        try? sequenceHandler.perform([faceRequest], on: pixelBuffer, orientation: .up)

        guard let face = faceRequest.results?.max(by: { $0.boundingBox.width < $1.boundingBox.width }),
              let landmarks = face.landmarks,
              let leftEye = landmarks.leftEye, let rightEye = landmarks.rightEye,
              let leftPupil = landmarks.leftPupil, let rightPupil = landmarks.rightPupil,
              let left = measure(eye: leftEye.pointsInImage(imageSize: size), pupil: leftPupil.pointsInImage(imageSize: size)),
              let right = measure(eye: rightEye.pointsInImage(imageSize: size), pupil: rightPupil.pointsInImage(imageSize: size))
        else {
            setFaceFound(false)
            return decay()
        }
        setFaceFound(true)

        // "Up" for the face, perpendicular to the line between the eyes (handles head tilt).
        var across = CGVector(dx: right.center.x - left.center.x, dy: right.center.y - left.center.y)
        let acrossLength = max(hypot(across.dx, across.dy), 1)
        across = CGVector(dx: across.dx / acrossLength, dy: across.dy / acrossLength)
        var up = CGVector(dx: -across.dy, dy: across.dx)
        if up.dy < 0 { up = CGVector(dx: -up.dx, dy: -up.dy) }

        func offset(_ m: EyeMeasure) -> Double {
            let dx = m.pupil.x - m.center.x, dy = m.pupil.y - m.center.y
            return Double((dx * up.dx + dy * up.dy) / max(m.height, 1))
        }
        let currentOffset = (offset(left) + offset(right)) / 2
        let blinking = (left.height / max(left.width, 1)) < 0.12 || (right.height / max(right.width, 1)) < 0.12

        if !blinking { recordCalibration(currentOffset) }

        // Positive when looking further down than during calibration.
        let down = s.baseline - currentOffset
        let fadeStart = 0.45, fadeEnd = 0.75
        var factor = 1.0
        if down > fadeStart { factor = max(0, 1 - (down - fadeStart) / (fadeEnd - fadeStart)) }
        if blinking || !s.eyeContact { factor = 0 }

        func warp(for m: EyeMeasure) -> EyeWarp {
            let radius = CGSize(width: m.width * 0.6, height: max(m.height, m.width * 0.3) * 1.1)
            var amount = CGFloat(max(0, down) * factor * s.strength) * m.height
            amount = min(amount, radius.height * 0.38) // stay well below the fold-over limit
            return EyeWarp(center: m.center, radius: radius,
                           shift: CGVector(dx: up.dx * amount, dy: up.dy * amount))
        }

        smoothedLeft = smooth(smoothedLeft, warp(for: left))
        smoothedRight = smooth(smoothedRight, warp(for: right))
        return (smoothedLeft!, smoothedRight!)
    }

    private func smooth(_ previous: EyeWarp?, _ next: EyeWarp) -> EyeWarp {
        guard let p = previous else { return next }
        let a: CGFloat = 0.5  // position follows quickly
        let b: CGFloat = 0.3  // shift eases in and out
        func mix(_ x: CGFloat, _ y: CGFloat, _ t: CGFloat) -> CGFloat { x + (y - x) * t }
        return EyeWarp(
            center: CGPoint(x: mix(p.center.x, next.center.x, a), y: mix(p.center.y, next.center.y, a)),
            radius: CGSize(width: mix(p.radius.width, next.radius.width, a), height: mix(p.radius.height, next.radius.height, a)),
            shift: CGVector(dx: mix(p.shift.dx, next.shift.dx, b), dy: mix(p.shift.dy, next.shift.dy, b))
        )
    }

    /// Face lost: ease the correction back to zero instead of snapping.
    private func decay() -> (EyeWarp, EyeWarp)? {
        guard var l = smoothedLeft, var r = smoothedRight else { return nil }
        l.shift = CGVector(dx: l.shift.dx * 0.6, dy: l.shift.dy * 0.6)
        r.shift = CGVector(dx: r.shift.dx * 0.6, dy: r.shift.dy * 0.6)
        if hypot(l.shift.dx, l.shift.dy) < 0.05 && hypot(r.shift.dx, r.shift.dy) < 0.05 {
            smoothedLeft = nil
            smoothedRight = nil
            return nil
        }
        smoothedLeft = l
        smoothedRight = r
        return (l, r)
    }

    private func recordCalibration(_ value: Double) {
        settingsLock.lock()
        guard isCalibrating else { settingsLock.unlock(); return }
        calibrationSamples.append(value)
        var finished: Double?
        if calibrationSamples.count >= 30 {
            let sorted = calibrationSamples.sorted()
            finished = sorted[sorted.count / 2]
            isCalibrating = false
            _settings.baseline = finished!
        }
        settingsLock.unlock()
        if let finished { onCalibrated?(finished) }
    }

    private func setFaceFound(_ found: Bool) {
        guard found != faceFound else { return }
        faceFound = found
        onFaceChange?(found)
    }
}
