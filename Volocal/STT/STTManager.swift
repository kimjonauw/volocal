import Foundation
import AVFoundation
import FluidAudio
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "stt")

/// Wraps FluidAudio streaming ASR (Parakeet EOU or Nemotron) for live speech-to-text.
/// Uses SharedAudioEngine for mic input instead of creating its own AVAudioEngine.
@MainActor
final class STTManager: ObservableObject {
    @Published var transcript: String = ""
    @Published var isListening: Bool = false
    @Published var partialResult: String = ""
    @Published var error: String?
    @Published private(set) var engine: STTEngine = .parakeetEou320

    /// Called when a complete utterance is detected (EOU)
    var onUtteranceCompleted: ((String) -> Void)?

    /// Called when speech is first detected (partial result arrives)
    var onSpeechDetected: (() -> Void)?

    /// Shared audio engine — injected by VoicePipeline
    weak var sharedAudio: SharedAudioEngine?

    private var eouManager: StreamingEouAsrManager?
    private var nemotronManager: StreamingNemotronAsrManager?
    private var hasFiredSpeechDetected = false
    private var isStopping = false
    private var nemotronDebounceTask: Task<Void, Never>?

    /// Serial stream for backpressure — prevents unbounded Task spawning per audio buffer
    private var bufferContinuation: AsyncStream<AVAudioPCMBuffer>.Continuation?
    private var processingTask: Task<Void, Never>?

    init() {}

    var isReady: Bool { eouManager != nil || nemotronManager != nil }

    /// Download models from Hugging Face if needed and load into memory.
    func initialize(engine: STTEngine) async {
        stopListening()
        eouManager = nil
        nemotronManager = nil
        self.engine = engine
        error = nil

        do {
            let asrRoot = FluidAudioCache.asrModelsRoot
            switch engine {
            case .parakeetEou320, .parakeetEou160:
                try await loadParakeet(engine: engine, asrRoot: asrRoot)
            case .nemotron560:
                try await loadNemotron(asrRoot: asrRoot)
            }
            logger.info("STT ready: \(engine.displayName)")
        } catch {
            self.error = "STT init failed: \(error.localizedDescription)"
            logger.error("STT init failed: \(error.localizedDescription)")
        }
    }

    func startListening() {
        guard !isListening, isReady else {
            if !isReady { error = "STT not initialized" }
            return
        }
        guard let sharedAudio else {
            error = "No shared audio engine"
            return
        }

        isStopping = false

        let (stream, continuation) = AsyncStream.makeStream(of: AVAudioPCMBuffer.self)
        self.bufferContinuation = continuation

        let eou = eouManager
        let nemotron = nemotronManager
        processingTask = Task {
            for await buffer in stream {
                guard !Task.isCancelled else { break }
                do {
                    if let eou {
                        _ = try await eou.process(audioBuffer: buffer)
                    } else if let nemotron {
                        _ = try await nemotron.process(audioBuffer: buffer)
                    }
                } catch {
                    await MainActor.run {
                        self.error = "STT error: \(error.localizedDescription)"
                    }
                }
            }
        }

        sharedAudio.bridge.inputContinuation = continuation
        sharedAudio.beginInputCapture()

        isListening = true
        transcript = ""
        partialResult = ""
        hasFiredSpeechDetected = false
        error = nil
        logger.info("STT listening started (\(self.engine.displayName))")
    }

    func stopListening() {
        isStopping = true
        nemotronDebounceTask?.cancel()
        nemotronDebounceTask = nil

        sharedAudio?.endInputCapture()
        bufferContinuation?.finish()
        bufferContinuation = nil
        processingTask?.cancel()
        processingTask = nil

        isListening = false

        Task {
            _ = try? await eouManager?.finish()
            await eouManager?.reset()
            _ = try? await nemotronManager?.finish()
            await nemotronManager?.reset()
        }

        logger.info("STT listening stopped")
    }

    /// Reset ASR state for next utterance without stopping the mic.
    func resetForNextUtterance() {
        hasFiredSpeechDetected = false
        partialResult = ""
        nemotronDebounceTask?.cancel()
        nemotronDebounceTask = nil
        Task {
            await eouManager?.reset()
            await nemotronManager?.reset()
        }
    }

    /// Simulate a transcript for testing without a real microphone.
    func simulateTranscript(_ text: String) {
        transcript += text + "\n"
        partialResult = ""
        onUtteranceCompleted?(text)
    }

    // MARK: - Loaders

    private func loadParakeet(engine: STTEngine, asrRoot: URL) async throws {
        guard let chunk = engine.eouChunkSize else { return }
        let manager = StreamingEouAsrManager(chunkSize: chunk, eouDebounceMs: 300)

        await manager.setPartialCallback { [weak self] text in
            Task { @MainActor in
                guard let self, !self.isStopping else { return }
                self.partialResult = text
                self.considerSpeechDetected(text)
            }
        }

        await manager.setEouCallback { [weak self] text in
            Task { @MainActor in
                self?.emitUtterance(text)
            }
        }

        logger.info("Loading \(engine.displayName)…")
        try await manager.loadModels(to: asrRoot)
        self.eouManager = manager
    }

    private func loadNemotron(asrRoot: URL) async throws {
        let manager = StreamingNemotronAsrManager(requestedChunkSize: .ms560)

        await manager.setPartialCallback { [weak self] text in
            Task { @MainActor in
                guard let self, !self.isStopping else { return }
                self.partialResult = text
                self.considerSpeechDetected(text)
                self.scheduleNemotronUtterance(text)
            }
        }

        logger.info("Loading Nemotron 560…")
        try await manager.loadModels(to: asrRoot)
        self.nemotronManager = manager
    }

    private func considerSpeechDetected(_ text: String) {
        let wordCount = text.split(separator: " ").count
        if !hasFiredSpeechDetected && wordCount >= 2 {
            hasFiredSpeechDetected = true
            onSpeechDetected?()
        }
    }

    /// Nemotron has no EOU head — treat a ~900 ms stall in the partial as a turn.
    private func scheduleNemotronUtterance(_ text: String) {
        nemotronDebounceTask?.cancel()
        let snapshot = text
        nemotronDebounceTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(900))
            guard !Task.isCancelled, !self.isStopping else { return }
            guard self.partialResult == snapshot else { return }
            self.emitUtterance(snapshot)
        }
    }

    private func emitUtterance(_ text: String) {
        guard !isStopping else { return }
        let finalText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !finalText.isEmpty else { return }
        transcript = finalText
        partialResult = ""
        hasFiredSpeechDetected = false
        nemotronDebounceTask?.cancel()
        nemotronDebounceTask = nil
        onUtteranceCompleted?(finalText)
    }
}
