import SwiftUI

struct VoiceEnginePickerView: View {
    @EnvironmentObject var modelManager: UnifiedModelManager
    @Environment(\.dismiss) private var dismiss

    var onEnginesChanged: (() -> Void)?
    var onVoiceChanged: ((String) -> Void)?
    var onExpressionsChanged: ((Bool) -> Void)?

    @State private var busyLabel: String?
    @State private var localError: String?

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("Speech recognition and voice stay on-device. Changing an engine downloads its weights from Hugging Face if they are not already here.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if let localError {
                    Section {
                        Text(localError)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }

                if let busyLabel {
                    Section {
                        HStack {
                            ProgressView()
                            Text(busyLabel)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("Speech recognition") {
                    ForEach(STTEngine.allCases) { engine in
                        engineRow(
                            title: engine.displayName,
                            detail: engine.detail,
                            size: engine.sizeDescription,
                            selected: modelManager.selectedSTT == engine,
                            ready: engine.isDownloaded(in: FluidAudioCache.asrModelsRoot)
                        ) {
                            Task { await pickSTT(engine) }
                        }
                    }
                }

                Section("Text-to-speech engine") {
                    ForEach(TTSEngine.allCases) { engine in
                        engineRow(
                            title: engine.displayName,
                            detail: engine.detail,
                            size: engine.sizeDescription,
                            selected: modelManager.selectedTTS == engine,
                            ready: engine.isDownloaded()
                        ) {
                            Task { await pickTTS(engine) }
                        }
                    }
                }

                Section {
                    Toggle(isOn: Binding(
                        get: { modelManager.ttsExpressionsEnabled },
                        set: { newValue in
                            modelManager.setTTSExpressions(newValue)
                            onExpressionsChanged?(newValue)
                        }
                    )) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Laughs, gasps, whispers")
                                .foregroundStyle(.primary)
                            Text("[laugh] [gasp] [sigh] play as short cues. [whisper]…[/whisper] lowers the audio. PocketTTS streams; Supertonic-3 is the newer CoreML voice. Kokoro is gone — it crashed this iPhone.")
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                Section("Speaker") {
                    ForEach(modelManager.selectedTTS.voiceChoices) { voice in
                        Button {
                            modelManager.selectTTSVoice(voice.id)
                            onVoiceChanged?(voice.id)
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(voice.displayName)
                                        .foregroundStyle(.primary)
                                    Text(voice.detail)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                if modelManager.selectedTTSVoice == voice.id {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(Color.accentColor)
                                }
                            }
                        }
                    }
                }
            }
            .navigationTitle("Voice engines")
            .navigationBarTitleDisplayMode(.inline)
            .disabled(busyLabel != nil)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }

    private func pickSTT(_ engine: STTEngine) async {
        localError = nil
        let changed = modelManager.selectedSTT != engine
        let wasReady = engine.isDownloaded(in: FluidAudioCache.asrModelsRoot)
        modelManager.selectSTT(engine)
        if !engine.isDownloaded(in: FluidAudioCache.asrModelsRoot) {
            busyLabel = "Downloading \(engine.displayName)…"
            await modelManager.retryModel(.stt)
            busyLabel = nil
            guard engine.isDownloaded(in: FluidAudioCache.asrModelsRoot) else {
                localError = modelManager.error ?? "STT download failed."
                return
            }
        }
        if changed || !wasReady { onEnginesChanged?() }
    }

    private func pickTTS(_ engine: TTSEngine) async {
        localError = nil
        let changed = modelManager.selectedTTS != engine
        let wasReady = engine.isDownloaded()
        modelManager.selectTTS(engine)
        if !engine.isDownloaded() {
            busyLabel = "Downloading \(engine.displayName)…"
            await modelManager.retryModel(.tts)
            busyLabel = nil
            guard engine.isDownloaded() else {
                localError = modelManager.error ?? "TTS download failed."
                return
            }
        }
        if changed || !wasReady { onEnginesChanged?() }
    }

    private func engineRow(
        title: String,
        detail: String,
        size: String,
        selected: Bool,
        ready: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            HStack(alignment: .top, spacing: 12) {
                Image(systemName: selected ? "checkmark.circle.fill" : "circle")
                    .foregroundStyle(selected ? Color.accentColor : .secondary)
                    .padding(.top, 2)
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(title)
                            .foregroundStyle(.primary)
                        Spacer()
                        Text(ready ? "Ready" : size)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Text(detail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .buttonStyle(.plain)
    }
}
