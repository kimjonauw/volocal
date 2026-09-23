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
    private var isStarting = false
    private var nemotronDebounceTask: Task<Void, Never>?
    private let turnDetector = MicTurnDetector()
    private var lastEmitAt = Date.distantPast
    private var lastEmittedNormalized = ""
    /// Skip `process` while `reset()` is in flight so CoreML is not reset mid-step,
    /// and while TTS is synthesizing so Parakeet does not share the ANE.
    private var asrPaused = false
    private var asrPauseDepth = 0
    private var asrResetEpoch = 0
    /// Partial already reads as a finished thought, so the energy detector may end sooner.
    private var eagerEndpoint = false

    /// Serial stream for backpressure — prevents unbounded Task spawning per audio buffer
    private var bufferContinuation: AsyncStream<AVAudioPCMBuffer>.Continuation?
    private var processingTask: Task<Void, Never>?

    init() {}

    var isReady: Bool { eouManager != nil || nemotronManager != nil }

    /// Download models from Hugging Face if needed and load into memory.
    func initialize(engine: STTEngine) async {
        isStarting = false
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

    func startListening() async {
        guard !isListening, !isStarting, isReady else {
            if !isReady { error = "STT not initialized" }
            return
        }
        guard let sharedAudio else {
            error = "No shared audio engine"
            return
        }

        isStopping = false
        isStarting = true
        error = nil

        let (stream, continuation) = AsyncStream.makeStream(
            of: AVAudioPCMBuffer.self,
            bufferingPolicy: .bufferingNewest(8)
        )
        self.bufferContinuation = continuation

        let eou = eouManager
        let nemotron = nemotronManager
        let detector = turnDetector
        processingTask = Task.detached(priority: .userInitiated) { [weak self] in
            for await buffer in stream {
                guard !Task.isCancelled else { break }
                let rms = SharedAudioEngine.rmsEnergy(buffer)
                let eager = await MainActor.run { self?.eagerEndpoint == true }
                let decision = detector.observe(rms: rms, eager: eager)
                if decision.endTurn {
                    await self?.forceEndOfTurn()
                }
                guard decision.feedAsr else { continue }
                let paused = await MainActor.run { self?.asrPaused == true }
                guard !paused else { continue }
                do {
                    if let eou {
                        _ = try await eou.process(audioBuffer: buffer)
                    } else if let nemotron {
                        _ = try await nemotron.process(audioBuffer: buffer)
                    }
                } catch {
                    await self?.reportSTTError(error)
                }
            }
        }

        sharedAudio.bridge.inputContinuation = continuation
        await sharedAudio.beginInputCapture()
        isStarting = false

        if Task.isCancelled || isStopping || sharedAudio.error != nil {
            if let captureError = sharedAudio.error, !isStopping {
                error = captureError
            }
            stopListening()
            return
        }

        isListening = true
        transcript = ""
        partialResult = ""
        hasFiredSpeechDetected = false
        lastEmittedNormalized = ""
        eagerEndpoint = false
        asrPaused = false
        asrPauseDepth = 0
        turnDetector.resetUtterance()
        logger.info("STT listening started (\(self.engine.displayName))")
    }

    func stopListening() {
        isStarting = false
        isStopping = true
        asrPaused = true
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
        eagerEndpoint = false
        nemotronDebounceTask?.cancel()
        nemotronDebounceTask = nil
        turnDetector.resetUtterance()
        asrPaused = true
        asrResetEpoch += 1
        let epoch = asrResetEpoch
        Task {
            await eouManager?.reset()
            await nemotronManager?.reset()
            if epoch == asrResetEpoch, asrPauseDepth == 0 {
                asrPaused = false
            }
        }
    }

    /// Keep Parakeet off the ANE while CoreML TTS is synthesizing.
    func pauseInference() {
        asrPauseDepth += 1
        asrPaused = true
    }

    func resumeInference() {
        asrPauseDepth = max(0, asrPauseDepth - 1)
        guard asrPauseDepth == 0, !isStopping else { return }
        asrPaused = false
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
        let manager = StreamingEouAsrManager(chunkSize: chunk, eouDebounceMs: 220)

        await manager.setPartialCallback { [weak self] text in
            Task { @MainActor in
                guard let self, !self.isStopping else { return }
                self.partialResult = text
                self.eagerEndpoint = UtteranceReadiness.looksComplete(text)
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
                self.eagerEndpoint = UtteranceReadiness.looksComplete(text)
                self.considerSpeechDetected(text)
                self.scheduleNemotronUtterance(text)
            }
        }

        logger.info("Loading Nemotron 560…")
        try await manager.loadModels(to: asrRoot)
        self.nemotronManager = manager
    }

    private func considerSpeechDetected(_ text: String) {
        let words = text.split(separator: " ").filter { $0.count >= 2 }
        let letters = text.filter(\.isLetter).count
        let looksLikeSpeech = words.count >= 2 || (words.count >= 1 && letters >= 4)
        guard looksLikeSpeech, !hasFiredSpeechDetected else { return }
        hasFiredSpeechDetected = true
        onSpeechDetected?()
    }

    private func forceEndOfTurn() {
        let text = partialResult.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        emitUtterance(text)
    }

    /// Nemotron has no EOU head — a finished phrase ends after ~320 ms, otherwise ~900 ms.
    private func scheduleNemotronUtterance(_ text: String) {
        nemotronDebounceTask?.cancel()
        let snapshot = text
        let waitMs = UtteranceReadiness.looksComplete(text) ? 320 : 900
        nemotronDebounceTask = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(waitMs))
            guard !Task.isCancelled, !self.isStopping else { return }
            guard self.partialResult == snapshot else { return }
            self.emitUtterance(snapshot)
        }
    }

    private func emitUtterance(_ text: String) {
        guard !isStopping else { return }
        let finalText = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !finalText.isEmpty else { return }
        let normalized = finalText.lowercased()
        if Date().timeIntervalSince(lastEmitAt) < 0.8, normalized == lastEmittedNormalized {
            return
        }
        lastEmitAt = Date()
        lastEmittedNormalized = normalized
        transcript = finalText
        partialResult = ""
        eagerEndpoint = false
        hasFiredSpeechDetected = false
        nemotronDebounceTask?.cancel()
        nemotronDebounceTask = nil
        turnDetector.resetUtterance()
        onUtteranceCompleted?(finalText)
    }

    private func reportSTTError(_ error: Error) {
        self.error = "STT error: \(error.localizedDescription)"
    }
}
