import SwiftUI

struct VoiceEnginePickerView: View {
    @EnvironmentObject var modelManager: UnifiedModelManager
    @Environment(\.dismiss) private var dismiss

    var onEnginesChanged: (() -> Void)?

    var body: some View {
        NavigationStack {
            List {
                Section {
                    Text("Speech recognition and voice stay on-device. Changing an engine downloads its weights from Hugging Face if they are not already here.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
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
                            let changed = modelManager.selectedSTT != engine
                            modelManager.selectSTT(engine)
                            if changed { onEnginesChanged?() }
                        }
                    }
                }

                Section("Text-to-speech") {
                    ForEach(TTSEngine.allCases) { engine in
                        engineRow(
                            title: engine.displayName,
                            detail: engine.detail,
                            size: engine.sizeDescription,
                            selected: modelManager.selectedTTS == engine,
                            ready: engine.isDownloaded()
                        ) {
                            let changed = modelManager.selectedTTS != engine
                            modelManager.selectTTS(engine)
                            if changed { onEnginesChanged?() }
                        }
                    }
                }
            }
            .navigationTitle("Voice engines")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
        }
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
