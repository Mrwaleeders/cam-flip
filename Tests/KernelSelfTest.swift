// Headless check of the GPU pipeline pieces, run in CI against the built app:
//   swift Tests/KernelSelfTest.swift "path/to/CamFlip.app/Contents/Resources/default.metallib"
//
// 1. The eye-contact warp kernel loads from the app's Metal library.
// 2. The warp moves content inside the eye region upward by the requested amount.
// 3. Pixels outside the eye regions stay bit-for-bit identical (no quality loss).
// 4. The mirror transform is an exact pixel flip (no resampling blur).
import CoreImage
import Foundation

func fail(_ message: String) -> Never {
    print("FAIL: \(message)")
    exit(1)
}

guard CommandLine.arguments.count > 1 else { fail("pass the metallib path") }
let libraryURL = URL(fileURLWithPath: CommandLine.arguments[1])
guard let data = try? Data(contentsOf: libraryURL) else { fail("cannot read \(libraryURL.path)") }

let kernel: CIWarpKernel
do {
    kernel = try CIWarpKernel(functionName: "gazeWarp", fromMetalLibraryData: data)
} catch {
    fail("kernel did not load: \(error)")
}
print("ok: gazeWarp kernel loaded")

let context = CIContext(options: [.workingColorSpace: NSNull(), .outputColorSpace: NSNull(), .cacheIntermediates: false])
let width = 256, height = 256
let rowBytes = width * 4

func render(_ image: CIImage) -> [UInt8] {
    var out = [UInt8](repeating: 0, count: rowBytes * height)
    context.render(image, toBitmap: &out, rowBytes: rowBytes,
                   bounds: CGRect(x: 0, y: 0, width: width, height: height),
                   format: .RGBA8, colorSpace: nil)
    return out
}

func image(from bytes: [UInt8]) -> CIImage {
    CIImage(bitmapData: Data(bytes), bytesPerRow: rowBytes,
            size: CGSize(width: width, height: height), format: .RGBA8, colorSpace: nil)
}

// Random noise image: any blur or resampling would change it.
var noise = [UInt8](repeating: 255, count: rowBytes * height)
var rng = SystemRandomNumberGenerator()
for i in 0..<noise.count where i % 4 != 3 { noise[i] = UInt8.random(in: 0...255, using: &rng) }
let noiseImage = image(from: noise)
let extent = noiseImage.extent

// --- Mirror
let mirrored = render(noiseImage.transformed(by: CGAffineTransform(scaleX: -1, y: 1).translatedBy(x: -extent.width, y: 0)))
for y in 0..<height {
    for x in 0..<width {
        for c in 0..<4 where mirrored[y * rowBytes + x * 4 + c] != noise[y * rowBytes + (width - 1 - x) * 4 + c] {
            fail("mirror is not exact at (\(x), \(y))")
        }
    }
}
print("ok: mirror is an exact pixel flip")

// --- Warp
// Eye A centred at (128, 128) with radius (40, 30), shifted up by 8 px. Eye B is far away with no shift.
func warp(_ input: CIImage, shift: CGFloat) -> CIImage {
    kernel.apply(
        extent: extent,
        roiCallback: { _, r in r.insetBy(dx: -12, dy: -12) },
        image: input.clampedToExtent(),
        arguments: [CIVector(x: 128, y: 128, z: 40, w: 30), CIVector(x: 0, y: shift),
                    CIVector(x: -1000, y: -1000, z: 1, w: 1), CIVector(x: 0, y: 0)]
    )!.cropped(to: extent)
}

// Outside the eye region nothing may change.
let warpedNoise = render(warp(noiseImage, shift: 8))
for y in 0..<height {
    for x in 0..<width {
        let dx = (Double(x) + 0.5 - 128) / 40, dy = (Double(y) + 0.5 - 128) / 30
        if (dx * dx + dy * dy).squareRoot() < 1.02 { continue }
        // Bitmap row 0 is the top; Core Image y = height - 1 - row.
        let row = height - 1 - y
        for c in 0..<4 where warpedNoise[row * rowBytes + x * 4 + c] != noise[row * rowBytes + x * 4 + c] {
            fail("warp changed a pixel outside the eye at (\(x), \(y))")
        }
    }
}
print("ok: pixels outside the eyes are untouched")

// A single bright row at y = 124 (Core Image coords) should move up by ~8 px at the centre.
var stripe = [UInt8](repeating: 0, count: rowBytes * height)
for i in stride(from: 3, to: stripe.count, by: 4) { stripe[i] = 255 }
let stripeRow = height - 1 - 124
for x in 0..<width { for c in 0..<3 { stripe[stripeRow * rowBytes + x * 4 + c] = 255 } }
let warpedStripe = render(warp(image(from: stripe), shift: 8))
var brightestY = -1, brightest = -1
for y in 100..<160 {
    let row = height - 1 - y
    let v = Int(warpedStripe[row * rowBytes + 128 * 4])
    if v > brightest { brightest = v; brightestY = y }
}
guard abs(brightestY - 132) <= 1 else { fail("expected the stripe at y≈132 after warp, found \(brightestY)") }
print("ok: warp moved the eye region up by 8 px (stripe now at y=\(brightestY))")
print("ALL CHECKS PASSED")
