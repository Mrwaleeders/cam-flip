import AVFoundation
import Combine
import Foundation

enum RunMode: String, CaseIterable, Identifiable {
    case auto
    case always

    var id: String { rawValue }
    var title: String {
        switch self {
        case .auto: return "Only while a call uses it"
        case .always: return "Always"
        }
    }
}

/// App state for the menu bar UI. Used on the main thread only.
final class AppModel: ObservableObject {
    private let defaults = UserDefaults.standard
    let pipeline = CameraPipeline()
    private let installer = ExtensionInstaller()

    @Published var mirror: Bool { didSet { defaults.set(mirror, forKey: "mirror"); pushSettings() } }
    @Published var eyeContact: Bool { didSet { defaults.set(eyeContact, forKey: "eyeContact"); pushSettings() } }
    @Published var strength: Double { didSet { defaults.set(strength, forKey: "strength"); pushSettings() } }
    @Published var runMode: RunMode { didSet { defaults.set(runMode.rawValue, forKey: "runMode"); updateCamera() } }
    @Published var cameraID: String {
        didSet {
            defaults.set(cameraID, forKey: "cameraID")
            if pipeline.isRunning { pipeline.start(cameraID: cameraID) }
        }
    }
    @Published var previewOpen = false { didSet { updateCamera() } }

    @Published private(set) var cameras: [AVCaptureDevice] = []
    @Published private(set) var captureInfo = "Camera off"
    @Published private(set) var extensionStatus = ""
    @Published private(set) var callActive = false
    @Published private(set) var sinkConnected = false
    @Published private(set) var cameraRunning = false
    @Published private(set) var faceFound = false
    @Published private(set) var isCalibrating = false
    @Published private(set) var calibrated: Bool
    @Published private(set) var cameraDenied = false

    private var baseline: Double
    private var stopWorkItem: DispatchWorkItem?
    private var observers: [NSObjectProtocol] = []

    init() {
        defaults.register(defaults: ["mirror": true, "eyeContact": true, "strength": 0.7, "runMode": RunMode.auto.rawValue])
        mirror = defaults.bool(forKey: "mirror")
        eyeContact = defaults.bool(forKey: "eyeContact")
        strength = defaults.double(forKey: "strength")
        runMode = RunMode(rawValue: defaults.string(forKey: "runMode") ?? "") ?? .auto
        cameraID = defaults.string(forKey: "cameraID") ?? ""
        baseline = defaults.double(forKey: "baseline")
        calibrated = defaults.object(forKey: "baseline") != nil

        pushSettings()
        wirePipeline()
        refreshCameras()
        observeExtension()

        installer.onStatus = { [weak self] in self?.extensionStatus = $0 }
        installer.onInstalled = { [weak self] in self?.queryExtension() }
        installer.install()

        let center = NotificationCenter.default
        observers.append(center.addObserver(forName: .AVCaptureDeviceWasConnected, object: nil, queue: .main) { [weak self] _ in
            self?.refreshCameras()
        })
        observers.append(center.addObserver(forName: .AVCaptureDeviceWasDisconnected, object: nil, queue: .main) { [weak self] _ in
            self?.refreshCameras()
        })

        AVCaptureDevice.requestAccess(for: .video) { granted in
            DispatchQueue.main.async {
                self.cameraDenied = !granted
                self.updateCamera()
            }
        }
    }

    // MARK: Actions

    func calibrate() {
        isCalibrating = true
        pipeline.processor.startCalibration()
        updateCamera()
    }

    func reinstallExtension() { installer.install() }
    func removeExtension() { installer.uninstall() }

    var statusLine: String {
        if cameraDenied { return "Camera access is off. Allow it in System Settings › Privacy & Security › Camera." }
        if !cameraRunning { return callActive ? "Starting camera…" : "Idle. Pick “Cam Flip Camera” in Teams, Zoom or Meet." }
        if !sinkConnected { return "Camera on, waiting for the Cam Flip Camera…" }
        return callActive ? "Live in your call" : "Camera on"
    }

    // MARK: Wiring

    private func pushSettings() {
        var s = pipeline.processor.settings
        s.mirror = mirror
        s.eyeContact = eyeContact
        s.strength = strength
        s.baseline = baseline
        pipeline.processor.settings = s
    }

    private func wirePipeline() {
        pipeline.onCaptureInfo = { [weak self] in self?.captureInfo = $0 }
        pipeline.onSinkConnected = { [weak self] in self?.sinkConnected = $0 }
        pipeline.processor.onFaceChange = { [weak self] found in
            DispatchQueue.main.async { self?.faceFound = found }
        }
        pipeline.processor.onCalibrated = { [weak self] value in
            DispatchQueue.main.async {
                guard let self else { return }
                self.baseline = value
                self.defaults.set(value, forKey: "baseline")
                self.calibrated = true
                self.isCalibrating = false
                self.pushSettings()
                self.updateCamera()
            }
        }
    }

    private func refreshCameras() {
        cameras = CameraPipeline.availableCameras()
        if !cameras.contains(where: { $0.uniqueID == cameraID }) {
            cameraID = cameras.first(where: { $0.deviceType == .builtInWideAngleCamera })?.uniqueID
                ?? cameras.first?.uniqueID ?? ""
        }
    }

    /// Runs the physical camera only when something needs it, so the green light is off
    /// outside calls (in "auto" mode).
    private func updateCamera() {
        let needed = !cameraDenied && (runMode == .always || callActive || previewOpen || isCalibrating)
        if needed {
            stopWorkItem?.cancel()
            stopWorkItem = nil
            if !cameraRunning {
                pipeline.start(cameraID: cameraID.isEmpty ? nil : cameraID)
                cameraRunning = true
            }
        } else if cameraRunning, stopWorkItem == nil {
            // Short grace period: call apps often stop and restart the camera when switching formats.
            let item = DispatchWorkItem { [weak self] in
                guard let self else { return }
                self.stopWorkItem = nil
                self.pipeline.stop()
                self.cameraRunning = false
                self.captureInfo = "Camera off"
            }
            stopWorkItem = item
            DispatchQueue.main.asyncAfter(deadline: .now() + 3, execute: item)
        }
    }

    // MARK: Extension state (Darwin notifications)

    private func observeExtension() {
        observeDarwin(CamFlipShared.notifyClientStarted) { [weak self] in
            self?.callActive = true
            self?.updateCamera()
        }
        observeDarwin(CamFlipShared.notifyClientStopped) { [weak self] in
            self?.callActive = false
            self?.updateCamera()
        }
        for index in CamFlipShared.formats.indices {
            observeDarwin(CamFlipShared.notifyFormat(index)) { [weak self] in
                self?.pipeline.setDesiredFormat(index: index)
            }
        }
        queryExtension()
    }

    private func queryExtension() {
        CFNotificationCenterPostNotification(
            CFNotificationCenterGetDarwinNotifyCenter(),
            CFNotificationName(CamFlipShared.notifyQuery as CFString),
            nil, nil, true
        )
    }
}

// MARK: - Darwin notification helper

private var darwinHandlers: [String: () -> Void] = [:]

private func observeDarwin(_ name: String, handler: @escaping () -> Void) {
    darwinHandlers[name] = handler
    CFNotificationCenterAddObserver(
        CFNotificationCenterGetDarwinNotifyCenter(),
        nil,
        { _, _, name, _, _ in
            guard let key = name?.rawValue as String? else { return }
            DispatchQueue.main.async { darwinHandlers[key]?() }
        },
        name as CFString,
        nil,
        .deliverImmediately
    )
}
