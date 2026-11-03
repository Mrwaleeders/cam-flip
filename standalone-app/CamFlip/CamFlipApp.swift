import SwiftUI

@main
struct CamFlipApp: App {
    @StateObject private var model = AppModel()

    var body: some Scene {
        MenuBarExtra {
            MenuView()
                .environmentObject(model)
        } label: {
            Image(systemName: model.callActive ? "camera.fill" : "camera")
        }
        .menuBarExtraStyle(.window)

        Window("Cam Flip Preview", id: "preview") {
            PreviewWindow()
                .environmentObject(model)
        }
        .defaultSize(width: 720, height: 460)
    }
}
