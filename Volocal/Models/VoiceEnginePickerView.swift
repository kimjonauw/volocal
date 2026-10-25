import SwiftUI
import AVFoundation
import UniformTypeIdentifiers

struct VoiceEnginePickerView: View {
    @EnvironmentObject var modelManager: UnifiedModelManager
    @Environment(\.dismiss) private var dismiss

    var pipeline: VoicePipeline?
    var onEnginesChanged: (() -> Void)?
    var onVoiceChanged: ((String) -> Void)?
    var onExpressionsChanged: ((Bool) -> Void)?

    @State private var busyLabel: String?
    @State private var localError: String?
    @State private var previewingVoice: String?
    @State private var voiceListRefreshID = UUID()

    // Voice cloning state
    @State private var showRecorderSheet = false
    @State private var showFileImporter = false
    @State private var showNamePrompt = false
    @State private var pendingImportURL: URL?
    @State private var pendingVoiceName: String = ""

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

                if modelManager.selectedTTS == .pocketTts {
                    Section(
                        header: Text("Voice cloning"),
                        footer: Text("Record a 5–10s audio clip or import a WAV/M4A/MP3 file. Audio is encoded on-device using the Mimi Neural Audio Codec into conditioning weights for PocketTTS 100M.")
                    ) {
                        Button {
                            showRecorderSheet = true
                        } label: {
                            Label("Record voice clip (5–10s)", systemImage: "mic.badge.plus")
                        }

                        Button {
                            showFileImporter = true
                        } label: {
                            Label("Import audio file…", systemImage: "square.and.arrow.down")
                        }
                    }
                }

                let allVoices = modelManager.selectedTTS.voiceChoices
                let clonedVoices = allVoices.filter { TTSManager.isClonedVoice($0.id) }
                let presetVoices = allVoices.filter { !TTSManager.isClonedVoice($0.id) }

                if !clonedVoices.isEmpty {
                    Section("My Cloned Voices") {
                        ForEach(clonedVoices) { voice in
                            voiceRow(for: voice)
                                .swipeActions(edge: .trailing, allowsFullSwipe: false) {
                                    Button(role: .destructive) {
                                        deleteClonedVoice(voice.id)
                                    } label: {
                                        Label("Delete", systemImage: "trash")
                                    }
                                }
                        }
                    }
                }

                Section(clonedVoices.isEmpty ? "Speaker" : "Preset Voices") {
                    ForEach(presetVoices) { voice in
                        voiceRow(for: voice)
                    }
                }
                .id(voiceListRefreshID)
            }
            .navigationTitle("Voice engines")
            .navigationBarTitleDisplayMode(.inline)
            .disabled(busyLabel != nil)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        pipeline?.stopVoicePreview()
                        dismiss()
                    }
                }
            }
            .onDisappear {
                pipeline?.stopVoicePreview()
            }
            .sheet(isPresented: $showRecorderSheet) {
                VoiceClipRecorderSheet(pipeline: pipeline) { newVoice in
                    voiceListRefreshID = UUID()
                    modelManager.selectTTSVoice(newVoice)
                    onVoiceChanged?(newVoice)
                }
            }
            .fileImporter(
                isPresented: $showFileImporter,
                allowedContentTypes: [.audio],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    pendingImportURL = url
                    let base = url.deletingPathExtension().lastPathComponent
                    pendingVoiceName = base.prefix(1).capitalized + base.dropFirst()
                    showNamePrompt = true
                case .failure(let error):
                    localError = "Failed to select file: \(error.localizedDescription)"
                }
            }
            .alert("Name Cloned Voice", isPresented: $showNamePrompt) {
                TextField("Voice name (e.g. Rachel)", text: $pendingVoiceName)
                Button("Cancel", role: .cancel) {
                    pendingImportURL = nil
                }
                Button("Clone") {
                    if let url = pendingImportURL {
                        Task { await cloneFromAudioFile(url, name: pendingVoiceName) }
                    }
                }
            } message: {
                Text("Enter a unique name for this cloned voice.")
            }
        }
    }

    @ViewBuilder
    private func voiceRow(for voice: TTSVoiceChoice) -> some View {
        HStack(spacing: 12) {
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
            .contentShape(Rectangle())
            .onTapGesture {
                selectVoice(voice.id)
            }

            Button {
                togglePreview(for: voice)
            } label: {
                Image(systemName: previewingVoice == voice.id && (pipeline?.ttsManager.isSpeaking == true) ? "stop.circle.fill" : "play.circle")
                    .font(.title3)
                    .foregroundStyle(Color.accentColor)
                    .frame(width: 36, height: 36)
                    .contentShape(Rectangle())
            }
            .buttonStyle(.borderless)
            .accessibilityLabel("Preview \(voice.displayName) voice")
        }
        .accessibilityElement(children: .combine)
        .accessibilityAddTraits(modelManager.selectedTTSVoice == voice.id ? [.isSelected] : [])
    }

    private func selectVoice(_ voiceId: String) {
        pipeline?.stopVoicePreview()
        previewingVoice = nil
        modelManager.selectTTSVoice(voiceId)
        onVoiceChanged?(voiceId)
    }

    private func togglePreview(for voice: TTSVoiceChoice) {
        if previewingVoice == voice.id && pipeline?.ttsManager.isSpeaking == true {
            pipeline?.stopVoicePreview()
            previewingVoice = nil
        } else {
            previewingVoice = voice.id
            Task {
                await pipeline?.previewVoice(voice.id, displayName: voice.displayName)
            }
        }
    }

    private func deleteClonedVoice(_ voiceId: String) {
        pipeline?.deleteClonedVoice(voiceId)
        voiceListRefreshID = UUID()
        if modelManager.selectedTTSVoice == voiceId {
            let fallback = modelManager.selectedTTS.defaultVoice
            modelManager.selectTTSVoice(fallback)
            onVoiceChanged?(fallback)
        }
    }

    private func cloneFromAudioFile(_ url: URL, name: String) async {
        localError = nil
        let targetName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !targetName.isEmpty else {
            localError = "Please enter a voice name."
            return
        }

        busyLabel = "Cloning voice '\(targetName)' on device…"
        let accessing = url.startAccessingSecurityScopedResource()
        defer {
            if accessing { url.stopAccessingSecurityScopedResource() }
            pendingImportURL = nil
        }

        do {
            try await pipeline?.cloneVoice(from: url, as: targetName)
            let sanitized = targetName.lowercased().replacingOccurrences(of: " ", with: "_").filter { $0.isLetter || $0.isNumber || $0 == "_" }
            voiceListRefreshID = UUID()
            modelManager.selectTTSVoice(sanitized)
            onVoiceChanged?(sanitized)
            busyLabel = nil
        } catch {
            busyLabel = nil
            localError = "Voice cloning failed: \(error.localizedDescription)"
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
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
    }
}

// MARK: - Voice Clip Recorder Sheet

private struct VoiceClipRecorderSheet: View {
    let pipeline: VoicePipeline?
    let onCloned: (String) -> Void
    @Environment(\.dismiss) private var dismiss

    @StateObject private var recorder: VoiceClipRecorder
    @State private var voiceName: String = ""
    @State private var isCloning = false
    @State private var cloneError: String?

    init(pipeline: VoicePipeline?, onCloned: @escaping (String) -> Void) {
        self.pipeline = pipeline
        self.onCloned = onCloned
        _recorder = StateObject(wrappedValue: VoiceClipRecorder(sharedAudio: pipeline?.sharedAudio))
    }

    var body: some View {
        NavigationStack {
            VStack(spacing: 24) {
                if let cloneError {
                    Text(cloneError)
                        .font(.caption)
                        .foregroundStyle(.red)
                        .padding(.horizontal)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("Voice name")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    TextField("e.g. My Voice", text: $voiceName)
                        .textFieldStyle(.roundedBorder)
                        .autocorrectionDisabled()
                }
                .padding(.horizontal)

                Spacer()

                // Mic / Status Section
                VStack(spacing: 16) {
                    Text(statusMessage)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)

                    Button {
                        handleMicTap()
                    } label: {
                        ZStack {
                            Circle()
                                .fill(micColor)
                                .frame(width: 88, height: 88)

                            Image(systemName: micIcon)
                                .font(.system(size: 34))
                                .foregroundStyle(.white)
                        }
                    }
                    .disabled(isCloning)

                    if recorder.isRecording {
                        ProgressView(value: recorder.elapsedSeconds, total: 10.0)
                            .tint(.red)
                            .padding(.horizontal, 48)
                    }

                    if recorder.isFinished, let _ = recorder.recordedURL {
                        HStack(spacing: 16) {
                            Button {
                                if recorder.isPlayingPreview {
                                    recorder.stopRecordedPreview()
                                } else {
                                    recorder.playRecordedPreview()
                                }
                            } label: {
                                Label(recorder.isPlayingPreview ? "Stop" : "Listen", systemImage: recorder.isPlayingPreview ? "stop.fill" : "play.fill")
                            }
                            .buttonStyle(.bordered)

                            Button {
                                recorder.cleanupTempFile()
                            } label: {
                                Label("Re-record", systemImage: "arrow.counterclockwise")
                            }
                            .buttonStyle(.bordered)
                        }
                    }

                    Text("Tip: Speak clearly close to the mic in a quiet room for 5–8 seconds. Leading/trailing silence is trimmed and volume is normalized automatically.")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal, 24)
                }

                Spacer()

                // Clone button
                if isCloning {
                    VStack(spacing: 8) {
                        ProgressView()
                        Text("Encoding voice on Neural Engine…")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.bottom, 24)
                } else {
                    Button {
                        Task { await doClone() }
                    } label: {
                        Text("Create Cloned Voice")
                            .font(.headline)
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 14)
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(!canClone)
                    .padding(.horizontal)
                    .padding(.bottom, 24)
                }
            }
            .navigationTitle("Record Voice")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button("Cancel") {
                        recorder.cleanupTempFile()
                        dismiss()
                    }
                    .disabled(isCloning)
                }
            }
            .onDisappear {
                recorder.cleanupTempFile()
            }
        }
    }

    private var canClone: Bool {
        let name = voiceName.trimmingCharacters(in: .whitespacesAndNewlines)
        return !name.isEmpty && recorder.isFinished && recorder.elapsedSeconds >= 1.0 && !isCloning
    }

    private var statusMessage: String {
        if recorder.isRecording {
            return "Recording: \(String(format: "%.1f", recorder.elapsedSeconds))s / 10.0s\n(Speak naturally)"
        } else if recorder.isFinished {
            return "Sample recorded (\(String(format: "%.1f", recorder.elapsedSeconds))s). Ready to clone!"
        } else {
            return "Tap microphone and speak for 5 to 10 seconds."
        }
    }

    private var micColor: Color {
        if recorder.isRecording {
            return .red
        } else if recorder.isFinished {
            return .green
        } else {
            return Color.accentColor
        }
    }

    private var micIcon: String {
        if recorder.isRecording {
            return "stop.fill"
        } else if recorder.isFinished {
            return "checkmark"
        } else {
            return "mic.fill"
        }
    }

    private func handleMicTap() {
        if recorder.isRecording {
            recorder.stopRecording()
        } else if !recorder.isFinished {
            Task { await recorder.startRecording() }
        }
    }

    private func doClone() async {
        guard let url = recorder.recordedURL else { return }
        cloneError = nil
        isCloning = true
        let targetName = voiceName.trimmingCharacters(in: .whitespacesAndNewlines)

        do {
            try await pipeline?.cloneVoice(from: url, as: targetName)
            let sanitized = targetName.lowercased().replacingOccurrences(of: " ", with: "_").filter { $0.isLetter || $0.isNumber || $0 == "_" }
            recorder.cleanupTempFile()
            isCloning = false
            onCloned(sanitized)
            dismiss()
        } catch {
            isCloning = false
            cloneError = "Cloning failed: \(error.localizedDescription)"
        }
    }
}

// MARK: - Voice Clip Recorder Helper

@MainActor
private final class VoiceClipRecorder: ObservableObject {
    @Published var isRecording = false
    @Published var isFinished = false
    @Published var elapsedSeconds: Double = 0
    @Published var isPlayingPreview = false
    @Published var error: String?

    private var audioFile: AVAudioFile?
    private var tempURL: URL?
    private var timer: Timer?
    private var audioPlayer: AVAudioPlayer?
    private weak var sharedAudio: SharedAudioEngine?
    private var captureTask: Task<Void, Never>?

    init(sharedAudio: SharedAudioEngine?) {
        self.sharedAudio = sharedAudio
    }

    var recordedURL: URL? {
        tempURL
    }

    func startRecording() async {
        guard let sharedAudio else {
            error = "Audio engine unavailable"
            return
        }

        cleanupTempFile()
        error = nil
        isRecording = true
        isFinished = false
        elapsedSeconds = 0

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("cloning_\(UUID().uuidString).wav")
        self.tempURL = url

        let (stream, continuation) = AsyncStream<AVAudioPCMBuffer>.makeStream()
        sharedAudio.bridge.inputContinuation = continuation
        await sharedAudio.beginInputCapture()

        captureTask = Task { [weak self] in
            for await buffer in stream {
                guard let self else { break }
                if self.audioFile == nil {
                    do {
                        self.audioFile = try AVAudioFile(forWriting: url, settings: buffer.format.settings)
                    } catch {
                        Task { @MainActor in self.error = "File write failed: \(error.localizedDescription)" }
                        break
                    }
                }
                try? self.audioFile?.write(from: buffer)
            }
        }

        timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { [weak self] _ in
            Task { @MainActor [weak self] in
                guard let self, self.isRecording else { return }
                self.elapsedSeconds += 0.1
                if self.elapsedSeconds >= 10.0 {
                    self.stopRecording()
                }
            }
        }
    }

    func stopRecording() {
        guard isRecording else { return }
        timer?.invalidate()
        timer = nil
        isRecording = false
        isFinished = true
        captureTask?.cancel()
        captureTask = nil
        sharedAudio?.endInputCapture()
        audioFile = nil
    }

    func playRecordedPreview() {
        guard let url = tempURL else { return }
        do {
            audioPlayer = try AVAudioPlayer(contentsOf: url)
            audioPlayer?.play()
            isPlayingPreview = true
            Timer.scheduledTimer(withTimeInterval: 0.2, repeats: true) { [weak self] t in
                Task { @MainActor [weak self] in
                    guard let self else { t.invalidate(); return }
                    if self.audioPlayer?.isPlaying != true {
                        self.isPlayingPreview = false
                        t.invalidate()
                    }
                }
            }
        } catch {
            self.error = "Playback failed: \(error.localizedDescription)"
        }
    }

    func stopRecordedPreview() {
        audioPlayer?.stop()
        isPlayingPreview = false
    }

    func cleanupTempFile() {
        if let url = tempURL {
            try? FileManager.default.removeItem(at: url)
            tempURL = nil
        }
        audioFile = nil
        audioPlayer?.stop()
        audioPlayer = nil
        isFinished = false
        elapsedSeconds = 0
    }
}
