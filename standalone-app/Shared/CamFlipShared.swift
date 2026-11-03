import Foundation

/// Values shared by the menu bar app and the camera extension.
enum CamFlipShared {
    /// Name that Teams, Zoom and Meet show in their camera list.
    static let deviceName = "Cam Flip Camera"

    /// Fixed identity of the virtual camera. CoreMediaIO reports this as the device UID,
    /// which is how the app finds the extension's device.
    static let deviceUUID = UUID(uuidString: "8B2F6C1E-5A3D-4F7B-9C0E-2D4A6B8C1F35")!
    static let sourceStreamUUID = UUID(uuidString: "3E1D5F7A-9B2C-4D6E-8F01-A2B3C4D5E6F7")!
    static let sinkStreamUUID = UUID(uuidString: "7C9E1A3B-5D7F-4B2D-9E4F-B6A8C0D2E4F6")!

    /// Formats the virtual camera offers. The first one is the default.
    /// The app captures the physical camera at whichever of these the call app picks,
    /// so frames pass through at full resolution with no scaling.
    static let formats: [(width: Int32, height: Int32)] = [
        (1920, 1080),
        (1280, 720),
        (3840, 2160),
    ]
    static let frameRate: Int32 = 30

    // Darwin notifications between the extension and the app. The extension runs in its
    // own sandboxed process, so these carry simple state changes.
    static let notifyPrefix = "com.mrwaleeders.CamFlip"
    static let notifyClientStarted = notifyPrefix + ".clientStarted"
    static let notifyClientStopped = notifyPrefix + ".clientStopped"
    static let notifyQuery = notifyPrefix + ".query"
    static func notifyFormat(_ index: Int) -> String { notifyPrefix + ".format.\(index)" }
}
