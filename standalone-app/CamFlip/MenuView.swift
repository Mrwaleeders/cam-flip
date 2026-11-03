import AppKit
import SwiftUI

struct MenuView: View {
    @EnvironmentObject private var model: AppModel
    @Environment(\.openWindow) private var openWindow

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Circle()
                    .fill(model.callActive && model.sinkConnected ? Color.green : Color.secondary.opacity(0.5))
                    .frame(width: 8, height: 8)
                Text("Cam Flip").font(.headline)
                Spacer()
            }
            Text(model.statusLine)
                .font(.callout)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if model.cameraRunning {
                Text(model.captureInfo).font(.caption).foregroundStyle(.secondary)
            }

            Divider()

            Toggle("Mirror my video for everyone", isOn: $model.mirror)
                .toggleStyle(.switch)

            Toggle("Eye contact", isOn: $model.eyeContact)
                .toggleStyle(.switch)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Strength")
                    Spacer()
                    Text("\(Int((model.strength * 100).rounded()))%")
                        .monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Slider(value: $model.strength, in: 0...1)
            }
            .disabled(!model.eyeContact)

            HStack {
                Button(model.isCalibrating ? "Look into the camera…" : "Calibrate eye contact") {
                    model.calibrate()
                }
                .disabled(model.isCalibrating)
                Spacer()
                if model.isCalibrating {
                    ProgressView().controlSize(.small)
                } else {
                    Text(model.calibrated ? "Calibrated" : "Not calibrated")
                        .font(.caption)
                        .foregroundStyle(model.calibrated ? Color.secondary : Color.orange)
                }
            }
            .disabled(!model.eyeContact)
            if model.cameraRunning && model.eyeContact && !model.faceFound {
                Text("No face found").font(.caption).foregroundStyle(.orange)
            }

            Divider()

            Picker("Camera", selection: $model.cameraID) {
                ForEach(model.cameras, id: \.uniqueID) { camera in
                    Text(camera.localizedName).tag(camera.uniqueID)
                }
            }
            Picker("Run camera", selection: $model.runMode) {
                ForEach(RunMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }

            Divider()

            Button("Show preview (what others see)") {
                openWindow(id: "preview")
                NSApp.activate(ignoringOtherApps: true)
            }

            if !model.extensionStatus.isEmpty {
                Text(model.extensionStatus)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack {
                Button("Reinstall camera") { model.reinstallExtension() }
                Button("Remove camera") { model.removeExtension() }
            }
            .controlSize(.small)

            Divider()

            Button("Quit Cam Flip") { NSApp.terminate(nil) }
        }
        .padding(16)
        .frame(width: 320)
    }
}
