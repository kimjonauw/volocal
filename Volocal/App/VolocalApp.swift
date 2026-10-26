import SwiftUI

@main
struct VolocalApp: App {
    @StateObject private var modelManager = UnifiedModelManager()
    @StateObject private var metrics = SystemMetrics()
    @StateObject private var pipeline = VoicePipeline()
    @Environment(\.scenePhase) private var scenePhase
    // Choose before ModelLoadingView exists: the lab must not compete with loaded models.
    private enum LaunchRoute { case choose, conversation, qwenLab }
    @State private var route: LaunchRoute = .choose

    var body: some Scene {
        WindowGroup {
            if route == .choose {
                NavigationStack {
                    Form {
                        Section("Voice experiment") {
                            Button("Try Qwen3-TTS voice lab") { route = .qwenLab }
                            Text("Audition delivery styles locally before connecting the conversational LLM. Conversation models remain unloaded.")
                        }
                        Section {
                            Button("Continue normal Volocal") { route = .conversation }
                            Text("PocketTTS and Persona speech-to-speech are unchanged. Relaunch the app to return to this chooser.")
                        }
                    }
                    .navigationTitle("Volocal")
                }
            } else if route == .qwenLab {
                QwenVoiceLabView { route = .choose }
            } else if !modelManager.hasCompletedOnboarding {
                OnboardingView().environmentObject(modelManager)
            } else if !pipeline.isReady {
                ModelLoadingView()
                    .environmentObject(pipeline)
                    .environmentObject(modelManager)
                    .environmentObject(metrics)
            } else {
                ContentView()
                    .environmentObject(modelManager)
                    .environmentObject(metrics)
                    .environmentObject(pipeline)
                    .overlay { MetricsOverlay().environmentObject(metrics) }
            }
        }
        .onChange(of: scenePhase) { _, newPhase in
            switch newPhase {
            case .active:
                metrics.startMonitoring(); pipeline.setSceneActive(true)
            case .inactive, .background:
                metrics.stopMonitoring(); pipeline.setSceneActive(false)
            @unknown default: break
            }
        }
    }
}
