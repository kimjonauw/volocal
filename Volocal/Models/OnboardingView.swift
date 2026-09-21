import SwiftUI

/// Onboarding screen showing all model download status with per-model progress.
struct OnboardingView: View {
    @EnvironmentObject var modelManager: UnifiedModelManager
    @State private var isDownloading = false
    @State private var showLLMPicker = false
    @State private var showVoicePicker = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 32) {
                Spacer()

                // App icon area
                VStack(spacing: 12) {
                    Image(systemName: "waveform.circle.fill")
                        .font(.system(size: 80))
                        .foregroundStyle(.blue)

                    Text("Volocal")
                        .font(.largeTitle.bold())

                    Text("On-device voice AI")
                        .font(.title3)
                        .foregroundStyle(.secondary)
                }

                // Model status cards
                VStack(spacing: 12) {
                    ForEach(ModelRegistry.ModelType.allCases) { type in
                        ModelStatusCard(
                            type: type,
                            state: modelManager.modelStates[type] ?? .notDownloaded,
                            subtitle: subtitle(for: type),
                            onRetry: {
                                Task { await modelManager.retryModel(type) }
                            }
                        )
                    }

                    Button("Choose a different GGUF…") {
                        showLLMPicker = true
                    }
                    .font(.subheadline)

                    Button("Change speech or voice engine…") {
                        showVoicePicker = true
                    }
                    .font(.subheadline)
                }
                .padding(.horizontal)

                // Wi-Fi recommendation
                Text("Recommended: Wi-Fi for first download. STT + TTS ~1 GB, plus whatever GGUF you pick.")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                // Download button
                Button {
                    isDownloading = true
                    Task {
                        await modelManager.downloadAllModels()
                        isDownloading = false
                    }
                } label: {
                    HStack {
                        if isDownloading {
                            ProgressView()
                                .tint(.white)
                                .padding(.trailing, 4)
                        }
                        Text(isDownloading ? "Downloading..." : "Download All Models")
                    }
                    .font(.headline)
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 14)
                }
                .buttonStyle(.borderedProminent)
                .disabled(isDownloading || modelManager.allModelsReady)
                .padding(.horizontal)

                if let error = modelManager.error {
                    Text(error)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.horizontal)
                }

                Text("Speech + voice stay local. The language model is any GGUF you pull from Hugging Face.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .padding(.horizontal)

                Spacer().frame(height: 20)
            }
            .navigationTitle("")
            .sheet(isPresented: $showLLMPicker) {
                LLMPickerView()
                    .environmentObject(modelManager)
            }
            .sheet(isPresented: $showVoicePicker) {
                VoiceEnginePickerView()
                    .environmentObject(modelManager)
            }
        }
    }

    private func subtitle(for type: ModelRegistry.ModelType) -> String {
        switch type {
        case .llm: return modelManager.selectedLLM.displayName
        case .stt: return modelManager.selectedSTT.displayName
        case .tts: return modelManager.selectedTTS.displayName
        }
    }
}

// MARK: - Model Status Card

struct ModelStatusCard: View {
    let type: ModelRegistry.ModelType
    let state: UnifiedModelManager.ModelState
    var subtitle: String?
    var onRetry: (() -> Void)?

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: type.icon)
                .font(.title3)
                .frame(width: 32)
                .foregroundStyle(state.isReady ? .green : .secondary)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(type.displayName)
                        .font(.body.weight(.medium))
                    Spacer()
                    statusLabel
                }

                if let subtitle {
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                }

                if case .downloading(let progress) = state {
                    ProgressView(value: progress)
                        .tint(.blue)
                }
            }

            if state.isReady {
                Image(systemName: "checkmark.circle.fill")
                    .foregroundStyle(.green)
            } else if case .error = state {
                Button {
                    onRetry?()
                } label: {
                    Image(systemName: "arrow.clockwise.circle.fill")
                        .foregroundStyle(.orange)
                }
            }
        }
        .padding()
        .background(Color(.systemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12))
    }

    @ViewBuilder
    private var statusLabel: some View {
        switch state {
        case .notDownloaded:
            Text(type.sizeDescription)
                .font(.caption)
                .foregroundStyle(.secondary)
        case .downloading(let progress):
            Text("\(Int(progress * 100))%")
                .font(.caption.monospacedDigit())
                .foregroundStyle(.blue)
        case .downloaded:
            Text("Ready")
                .font(.caption)
                .foregroundStyle(.green)
        case .error(let msg):
            Text(msg)
                .font(.caption)
                .foregroundStyle(.red)
                .lineLimit(1)
        }
    }
}
