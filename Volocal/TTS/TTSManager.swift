import Foundation
import AVFoundation
import FluidAudio
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "tts")

/// Wraps FluidAudio TTS (streaming PocketTTS or batched Kokoro ANE).
/// Uses SharedAudioEngine for audio output instead of creating its own AVAudioEngine.
@MainActor
final class TTSManager: ObservableObject {
    @Published var isSpeaking: Bool = false
    @Published var selectedVoice: String = PocketTtsConstants.defaultVoice
    @Published var error: String?
    @Published private(set) var engineKind: TTSEngine = .pocketTts
    @Published var playbackPhase: PlaybackPhase = .idle

    enum PlaybackPhase: Equatable {
        case idle
        case synthesizing
        case playing
    }

    private var pocket: PocketTtsManager?
    private var kokoro: KokoroAneManager?
    private var speakTask: Task<Void, Never>?
    private var hasTrackedFirstInference = false
    var metrics: SystemMetrics?

    /// Shared audio engine — injected by VoicePipeline
    weak var sharedAudio: SharedAudioEngine?

    var voiceNames: [String] { engineKind.voiceNames }

    init() {}

    /// Initialize the selected TTS engine. Downloads CoreML models on first use,
    /// then runs a dummy generation to warm up.
    func initialize(engine: TTSEngine, voice: String? = nil) async {
        stop()
        pocket = nil
        kokoro = nil
        engineKind = engine
        selectedVoice = {
            if let voice, engine.voiceNames.contains(voice) { return voice }
            return engine.defaultVoice
        }()
        error = nil

        do {
            switch engine {
            case .pocketTts:
                let manager = PocketTtsManager(
                    defaultVoice: engine.defaultVoice,
                    language: .english,
                    placement: .gpu
                )
                try await manager.initialize()
                self.pocket = manager
                logger.info("TTS warmup: PocketTTS dummy generation…")
                let stream = try await manager.synthesizeStreaming(text: "Hi", voice: selectedVoice)
                for try await _ in stream { break }
            case .kokoroAne:
                let manager = KokoroAneManager(variant: .english, defaultVoice: engine.defaultVoice)
                try await manager.initialize()
                self.kokoro = manager
                logger.info("TTS warmup: Kokoro dummy generation…")
                _ = try await manager.synthesizeDetailed(text: "Hi", voice: selectedVoice)
            }
            logger.info("TTS warmup done (\(engine.displayName))")
        } catch {
            self.error = "TTS init failed: \(error.localizedDescription)"
            logger.error("TTS init failed: \(error.localizedDescription)")
        }
    }

    /// Synthesize text and play through the shared audio engine.
    func speak(_ text: String) async {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }

        speakTask?.cancel()

        isSpeaking = true
        playbackPhase = .synthesizing
        error = nil

        let speakTimeout: TimeInterval = 30
        let task = Task {
            do {
                guard let sharedAudio else {
                    throw TTSError.noSharedAudio
                }

                logger.info("speak start: \"\(text)\"")

                if !hasTrackedFirstInference {
                    metrics?.beginTracking("TTS (\(self.engineKind.displayName))")
                }

                let genStart = CFAbsoluteTimeGetCurrent()
                var chunkCount = 0

                if let pocket {
                    let stream = try await pocket.synthesizeStreaming(
                        text: text,
                        voice: selectedVoice,
                        temperature: 0.4
                    )
                    for try await frame in stream {
                        self.markFirstInferenceIfNeeded()
                        if Task.isCancelled { break }
                        if CFAbsoluteTimeGetCurrent() - genStart > speakTimeout {
                            logger.warning("speak timeout after \(speakTimeout)s, aborting")
                            break
                        }
                        chunkCount += 1
                        if self.playbackPhase != .playing {
                            self.playbackPhase = .playing
                        }
                        sharedAudio.scheduleTTSBuffer(frame.samples)
                    }
                } else if let kokoro {
                    let result = try await kokoro.synthesizeDetailed(text: text, voice: selectedVoice)
                    try Task.checkCancellation()
                    self.markFirstInferenceIfNeeded()
                    if !result.samples.isEmpty {
                        chunkCount = 1
                        self.playbackPhase = .playing
                        sharedAudio.scheduleTTSBuffer(result.samples)
                    }
                } else {
                    throw TTSError.engineNotLoaded
                }

                logger.info("speak generation done: \(chunkCount) chunks")

                if !Task.isCancelled && chunkCount > 0 {
                    await sharedAudio.waitForPlaybackCompletion()
                }
            } catch {
                if !Task.isCancelled {
                    logger.error("speak failed: \(error.localizedDescription)")
                    self.error = "TTS failed: \(error.localizedDescription)"
                }
            }
            if !Task.isCancelled {
                self.isSpeaking = false
                self.playbackPhase = .idle
            }
            logger.info("speak end")
        }
        speakTask = task
        await task.value
    }

    /// Stop all audio playback and cancel in-flight generation.
    func stop() {
        speakTask?.cancel()
        speakTask = nil
        sharedAudio?.stopPlayback()
        isSpeaking = false
        playbackPhase = .idle
    }

    private func markFirstInferenceIfNeeded() {
        if !hasTrackedFirstInference {
            hasTrackedFirstInference = true
            metrics?.endTracking("TTS (\(engineKind.displayName))")
        }
    }
}

enum TTSError: LocalizedError {
    case engineNotLoaded
    case noSharedAudio

    var errorDescription: String? {
        switch self {
        case .engineNotLoaded:
            return "TTS engine not loaded. Call initialize() first."
        case .noSharedAudio:
            return "No shared audio engine available."
        }
    }
}
