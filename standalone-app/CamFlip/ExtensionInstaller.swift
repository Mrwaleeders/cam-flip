import Foundation
import SystemExtensions

/// Installs, updates and removes the camera extension through the system.
final class ExtensionInstaller: NSObject, OSSystemExtensionRequestDelegate {
    /// Called on the main queue with a status line for the menu.
    var onStatus: ((String) -> Void)?
    /// Called on the main queue when the extension is known to be installed.
    var onInstalled: (() -> Void)?

    private var extensionID: String {
        (Bundle.main.bundleIdentifier ?? "com.mrwaleeders.CamFlip") + ".Extension"
    }

    func install() {
        guard Bundle.main.bundlePath.hasPrefix("/Applications/") else {
            report("Move Cam Flip to the Applications folder first, then open it again.")
            return
        }
        let request = OSSystemExtensionRequest.activationRequest(forExtensionWithIdentifier: extensionID, queue: .main)
        request.delegate = self
        OSSystemExtensionManager.shared.submitRequest(request)
    }

    func uninstall() {
        let request = OSSystemExtensionRequest.deactivationRequest(forExtensionWithIdentifier: extensionID, queue: .main)
        request.delegate = self
        OSSystemExtensionManager.shared.submitRequest(request)
    }

    // MARK: OSSystemExtensionRequestDelegate

    func request(_ request: OSSystemExtensionRequest,
                 actionForReplacingExtension existing: OSSystemExtensionProperties,
                 withExtension ext: OSSystemExtensionProperties) -> OSSystemExtensionRequest.ReplacementAction {
        // Reinstalling the same version would restart the camera mid-call for nothing.
        if existing.bundleVersion == ext.bundleVersion && existing.bundleShortVersion == ext.bundleShortVersion {
            return .cancel
        }
        return .replace
    }

    func requestNeedsUserApproval(_ request: OSSystemExtensionRequest) {
        report("Approve it in System Settings › General › Login Items & Extensions › Camera Extensions.")
    }

    func request(_ request: OSSystemExtensionRequest, didFinishWithResult result: OSSystemExtensionRequest.Result) {
        switch result {
        case .completed:
            report("Cam Flip Camera is installed.")
            onInstalled?()
        case .willCompleteAfterReboot:
            report("Restart your Mac to finish installing.")
        @unknown default:
            report("Install finished.")
        }
    }

    func request(_ request: OSSystemExtensionRequest, didFailWithError error: Error) {
        let nsError = error as NSError
        if nsError.domain == OSSystemExtensionErrorDomain,
           nsError.code == OSSystemExtensionError.requestCanceled.rawValue {
            // Same version already installed.
            report("Cam Flip Camera is installed.")
            onInstalled?()
            return
        }
        report("Couldn't install the camera: \(error.localizedDescription)")
    }

    private func report(_ message: String) {
        DispatchQueue.main.async { self.onStatus?(message) }
    }
}
