import SwiftUI

struct ModelLoadingView: View {
    @EnvironmentObject var pipeline: VoicePipeline
    @EnvironmentObject var modelManager: UnifiedModelManager
    @EnvironmentObject var metrics: SystemMetrics

    @State private var showLLMPicker = false
    @State private var showVoicePicker = false

    var body: some View {
        VStack(spacing: 24) {
            Spacer()

            if pipeline.currentError == nil {
                ProgressView()
                    .scaleEffect(1.5)
            }

            Text(pipeline.loadingStatus ?? (pipeline.currentError == nil ? "Preparing..." : "Could not load"))
                .font(.headline)

            Text("Loading models into memory")
                .font(.subheadline)
                .foregroundStyle(.secondary)

            if let error = pipeline.currentError {
                Text(error)
                    .font(.caption)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
                    .multilineTextAlignment(.center)

                Button("Try again") {
                    Task { await load() }
                }
                .buttonStyle(.borderedProminent)

                Button("Change language model") {
                    showLLMPicker = true
                }

                Button("Change speech or voice") {
                    showVoicePicker = true
                }

                Button("Back to downloads") {
                    modelManager.reopenSetup()
                }
                .font(.subheadline)
            }

            Spacer()
        }
        .frame(maxWidth: .infinity)
        .sheet(isPresented: $showLLMPicker) {
            LLMPickerView { _ in
                pipeline.invalidateForReload()
                Task { await load() }
            }
            .environmentObject(modelManager)
        }
        .sheet(isPresented: $showVoicePicker) {
            VoiceEnginePickerView(
                onEnginesChanged: {
                    pipeline.invalidateForReload()
                    Task { await load() }
                },
                onVoiceChanged: { pipeline.setTTSVoice($0) }
            )
            .environmentObject(modelManager)
        }
        .task {
            await load()
        }
    }

    private func load() async {
        pipeline.metrics = metrics
        metrics.startMonitoring()
        await pipeline.configure(
            llmModelPath: modelManager.llmModelPath,
            displayName: modelManager.selectedLLM.displayName,
            stt: modelManager.selectedSTT,
            tts: modelManager.selectedTTS,
            ttsVoice: modelManager.selectedTTSVoice
        )
    }
}
