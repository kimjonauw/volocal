import Foundation
import AVFoundation
import CoreML
import FluidAudio
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "tts")

/// Wraps FluidAudio TTS (streaming PocketTTS or batched Supertonic-3).
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
    private var superonic: Supertonic3Manager?
    private var superonicStyles: [String: Supertonic3VoiceStyle] = [:]
    private var speakTask: Task<Void, Never>?
    private var speakGeneration = 0
    private var hasTrackedFirstInference = false
    /// True only while CoreML/Metal is producing samples — not while those
    /// samples are playing. llama.cpp waits on this so first audio can start
    /// before the reply is finished, without overlapping GPU graphs.
    private(set) var holdsInferenceGPU = false
    var metrics: SystemMetrics?
    var expressionsEnabled = true

    /// Shared audio engine — injected by VoicePipeline
    weak var sharedAudio: SharedAudioEngine?
    /// TTS CoreML and llama.cpp take turns on this lock. Playback does not.
    var gpuLock: GPUExclusive?
    /// Pause Parakeet while CoreML TTS is running so ANE graphs do not overlap.
    weak var stt: STTManager?

    var voiceNames: [String] { engineKind.voiceNames }

    init() {}

    /// Initialize the selected TTS engine. Downloads CoreML models on first use,
    /// then runs a dummy generation to warm up.
    func initialize(engine: TTSEngine, voice: String? = nil) async {
        await stopAndWait()
        pocket = nil
        superonic = nil
        superonicStyles.removeAll()
        engineKind = engine
        selectedVoice = {
            if let voice, engine.voiceNames.contains(voice) { return voice }
            return engine.defaultVoice
        }()
        error = nil

        do {
            switch engine {
            case .pocketTts:
                try await loadPocketTTS()
            case .supertonic3:
                try await loadSupertonic()
            }
            logger.info("TTS warmup done (\(self.engineKind.displayName))")
        } catch {
            self.error = "TTS init failed: \(error.localizedDescription)"
            logger.error("TTS init failed: \(error.localizedDescription)")
        }
    }

    private func loadPocketTTS() async throws {
        let manager = PocketTtsManager(
            defaultVoice: TTSEngine.pocketTts.defaultVoice,
            language: .english,
            placement: .gpu
        )
        try await manager.initialize()
        pocket = manager
        engineKind = .pocketTts
        if !TTSEngine.pocketTts.voiceNames.contains(selectedVoice) {
            selectedVoice = TTSEngine.pocketTts.defaultVoice
        }
        logger.info("TTS warmup: PocketTTS dummy generation…")
        let stream = try await manager.synthesizeStreaming(text: "Hi", voice: selectedVoice)
        for try await _ in stream { break }
    }

    private func loadSupertonic() async throws {
        _ = try await Supertonic3ResourceDownloader.ensureModels(
            veVariant: TTSEngine.superonicVariantToken
        )
        let manager = Supertonic3Manager(
            computeUnits: .cpuAndGPU,
            vectorEstimator: TTSEngine.superonicEstimator
        )
        try await manager.initialize()
        superonic = manager
        engineKind = .supertonic3
        if !TTSEngine.supertonic3.voiceNames.contains(selectedVoice) {
            selectedVoice = TTSEngine.supertonic3.defaultVoice
        }
        _ = try await superonicStyle(named: selectedVoice)
        logger.info("TTS warmup: Supertonic-3 dummy generation…")
        let style = try await superonicStyle(named: selectedVoice)
        _ = try await manager.synthesize(text: "Hi", language: "en", style: style)
    }

    private func superonicStyle(named name: String) async throws -> Supertonic3VoiceStyle {
        if let cached = superonicStyles[name] { return cached }
        let voice = Supertonic3Voice(name: name) ?? .f1
        let style = try await Supertonic3ResourceDownloader.loadVoiceStyle(voice)
        superonicStyles[voice.rawValue] = style
        return style
    }

    /// Synthesize text and play through the shared audio engine.
    func speak(_ text: String) async {
        let prepared = expressionsEnabled ? text : ExpressionParser.stripNeuralTags(text)
        let segments = ExpressionParser.parse(prepared).compactMap { segment -> SpeechSegment? in
            if case .effect = segment, !expressionsEnabled { return nil }
            return segment
        }
        guard !segments.isEmpty else { return }

        let leftover = speakTask
        leftover?.cancel()
        speakGeneration += 1
        let speakGen = speakGeneration
        if let leftover {
            await leftover.value
        }
        guard speakGen == speakGeneration else { return }

        isSpeaking = true
        playbackPhase = .synthesizing
        error = nil

        let speakTimeout: TimeInterval = 30
        let playEffects = expressionsEnabled
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

                for segment in segments {
                    if Task.isCancelled || speakGen != speakGeneration { break }
                    if CFAbsoluteTimeGetCurrent() - genStart > speakTimeout {
                        logger.warning("speak timeout after \(speakTimeout)s, aborting")
                        break
                    }
                    switch segment {
                    case .speech(let clause):
                        chunkCount += try await self.playSynthesized(
                            clause,
                            whisper: false,
                            audio: sharedAudio
                        )
                    case .whisper(let clause):
                        chunkCount += try await self.playSynthesized(
                            clause,
                            whisper: playEffects,
                            audio: sharedAudio
                        )
                    case .effect(let kind):
                        guard playEffects else { continue }
                        self.playbackPhase = .playing
                        sharedAudio.scheduleTTSBuffer(NonverbalSynth.render(kind))
                        chunkCount += 1
                    }
                }

                logger.info("speak generation done: \(chunkCount) chunks")
            } catch {
                if !Task.isCancelled {
                    logger.error("speak failed: \(error.localizedDescription)")
                    self.error = "TTS failed: \(error.localizedDescription)"
                }
            }
            if !Task.isCancelled {
                self.isSpeaking = sharedAudio?.isSpeaking == true
                if !self.isSpeaking {
                    self.playbackPhase = .idle
                }
            }
            logger.info("speak end")
        }
        speakTask = task
        await task.value
    }

    private func playSynthesized(
        _ text: String,
        whisper: Bool,
        audio: SharedAudioEngine
    ) async throws -> Int {
        stt?.pauseInference()
        defer { stt?.resumeInference() }

        var chunks = 0
        if let pocket {
            let voice = selectedVoice
            let temperature: Float = whisper ? 0.2 : 0.4
            let applyWhisper = whisper
            let generation = speakGeneration
            chunks = try await withGPU {
                let stream = try await pocket.synthesizeStreaming(
                    text: text,
                    voice: voice,
                    temperature: temperature
                )
                var count = 0
                for try await frame in stream {
                    try Task.checkCancellation()
                    let samples = applyWhisper ? Self.applyWhisper(frame.samples) : frame.samples
                    count += 1
                    let isFirst = count == 1
                    // Schedule without awaiting MainActor. Awaiting here would
                    // drop the GPU lock and let llama Metal overlap CoreML.
                    Task { @MainActor in
                        guard generation == self.speakGeneration else { return }
                        if isFirst {
                            self.markFirstInferenceIfNeeded()
                            self.playbackPhase = .playing
                        }
                        audio.scheduleTTSBuffer(samples)
                    }
                }
                return count
            }
        } else if let superonic {
            let style = try await superonicStyle(named: selectedVoice)
            let samples: [Float] = try await withGPU {
                let result = try await superonic.synthesize(
                    text: text,
                    language: "en",
                    style: style
                )
                try Task.checkCancellation()
                let pcm = whisper ? Self.applyWhisper(result.samples) : result.samples
                return Self.resample(pcm, from: 44_100, to: 24_000)
            }
            if !samples.isEmpty {
                chunks = 1
                markFirstInferenceIfNeeded()
                playbackPhase = .playing
                audio.scheduleTTSBuffer(samples)
            }
        } else {
            throw TTSError.engineNotLoaded
        }
        return chunks
    }

    private func withGPU<T>(_ body: () async throws -> T) async throws -> T {
        holdsInferenceGPU = true
        defer { holdsInferenceGPU = false }
        if let gpuLock {
            return try await gpuLock.run(body)
        }
        return try await body()
    }

    nonisolated private static func applyWhisper(_ samples: [Float]) -> [Float] {
        var prev: Float = 0
        return samples.map { x in
            let lp = prev * 0.82 + x * 0.18
            prev = lp
            return lp * 0.36
        }
    }

    nonisolated private static func resample(
        _ samples: [Float],
        from srcRate: Double,
        to dstRate: Double
    ) -> [Float] {
        guard srcRate > 0, dstRate > 0, srcRate != dstRate, !samples.isEmpty else {
            return samples
        }
        let ratio = srcRate / dstRate
        let count = max(1, Int((Double(samples.count) / ratio).rounded(.down)))
        var out = [Float](repeating: 0, count: count)
        let last = samples.count - 1
        for i in 0..<count {
            let src = Double(i) * ratio
            let i0 = min(Int(src), last)
            let i1 = min(i0 + 1, last)
            let frac = Float(src - Double(i0))
            out[i] = samples[i0] * (1 - frac) + samples[i1] * frac
        }
        return out
    }

    /// Stop all audio playback and cancel in-flight generation.
    func stop() {
        speakGeneration += 1
        speakTask?.cancel()
        holdsInferenceGPU = false
        sharedAudio?.stopPlayback()
        isSpeaking = false
        playbackPhase = .idle
    }

    /// Suspend until this engine is not running CoreML/Metal. Playback may
    /// still be in progress — that does not use the inference GPU.
    func waitUntilInferenceGPUFree() async {
        while holdsInferenceGPU {
            try? await Task.sleep(for: .milliseconds(20))
        }
    }

    /// Cancel TTS and wait until PocketTTS/CoreML actually leaves the GPU.
    /// Starting llama Metal while a cancelled synthesizer is still running is
    /// the crash after a couple of barge-ins at ~400 MB / 6 GB.
    func stopAndWait() async {
        stop()
        let leftover = speakTask
        speakTask = nil
        await leftover?.value
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
