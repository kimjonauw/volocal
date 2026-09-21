import SwiftUI

@main
struct VolocalApp: App {
    @StateObject private var modelManager = UnifiedModelManager()
    @StateObject private var metrics = SystemMetrics()
    @StateObject private var pipeline = VoicePipeline()
    @Environment(\.scenePhase) private var scenePhase

    var body: some Scene {
        WindowGroup {
            if !modelManager.hasCompletedOnboarding {
                OnboardingView()
                    .environmentObject(modelManager)
            } else if !pipeline.isReady {
                ModelLoadingView()
                    .environmentObject(pipeline)
                    .task {
                        pipeline.metrics = metrics
                        metrics.startMonitoring()
                        await pipeline.configure(
                            llmModelPath: modelManager.llmModelPath,
                            displayName: modelManager.selectedLLM.displayName,
                            stt: modelManager.selectedSTT,
                            tts: modelManager.selectedTTS
                        )
                    }
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
                metrics.startMonitoring()
            case .inactive, .background:
                metrics.stopMonitoring()
            @unknown default:
                break
            }
        }
    }
}
