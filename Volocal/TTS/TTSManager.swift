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
    @Published private(set) var engineKind: TTSEngine = .pocketTtsSmall
    @Published var playbackPhase: PlaybackPhase = .idle

    enum PlaybackPhase: Equatable {
        case idle
        case synthesizing
        case playing
    }

    private var pocketSmall: PocketTtsSmallManager?
    private var pocket100M: PocketTtsManager?
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
    /// Livelier PocketTTS sampling (higher temperature) for persona / phone-call style speech.
    var conversationalTone = false

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
        pocketSmall = nil
        pocket100M = nil
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
            case .pocketTtsSmall:
                try await loadPocketSmallTTS()
            case .pocketTts:
                try await loadPocket100MTTS()
            case .supertonic3:
                try await loadSupertonic()
            }
            logger.info("TTS warmup done (\(self.engineKind.displayName))")
        } catch {
            self.error = "TTS init failed: \(error.localizedDescription)"
            logger.error("TTS init failed: \(error.localizedDescription)")
        }
    }

    private func loadPocketSmallTTS() async throws {
        let manager = PocketTtsSmallManager(
            model: .english18m,
            defaultVoice: TTSEngine.pocketTtsSmall.defaultVoice,
            computeUnits: .cpuAndNeuralEngine,
            mimiPrecision: .fp32
        )
        try await manager.initialize()
        pocketSmall = manager
        engineKind = .pocketTtsSmall
        if !TTSEngine.pocketTtsSmall.voiceNames.contains(selectedVoice) {
            selectedVoice = TTSEngine.pocketTtsSmall.defaultVoice
        }
        logger.info("TTS warmup: PocketTTS small dummy generation…")
        let stream = try await manager.synthesizeStreaming(text: "Hi", voice: TTSEngine.pocketTtsSmall.defaultVoice)
        for try await _ in stream { break }
    }

    private func loadPocket100MTTS() async throws {
        _ = try await PocketTtsResourceDownloader.ensureModels(language: .english)
        let manager = PocketTtsManager(
            defaultVoice: TTSEngine.pocketTts.defaultVoice,
            language: .english,
            placement: .gpu
        )
        try await manager.initialize()
        pocket100M = manager
        engineKind = .pocketTts
        if !TTSEngine.pocketTts.voiceNames.contains(selectedVoice) {
            selectedVoice = TTSEngine.pocketTts.defaultVoice
        }
        logger.info("TTS warmup: PocketTTS 100M dummy generation…")
        let stream = try await manager.synthesizeStreaming(text: "Hi", voice: TTSEngine.pocketTts.defaultVoice)
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
        let style: Supertonic3VoiceStyle
        do {
            style = try await Supertonic3ResourceDownloader.loadVoiceStyle(voice)
        } catch {
            _ = try await Supertonic3ResourceDownloader.downloadVoiceStyle(voice)
            style = try await Supertonic3ResourceDownloader.loadVoiceStyle(voice)
        }
        superonicStyles[voice.rawValue] = style
        return style
    }

    /// Synthesize text and play through the shared audio engine.
    func speak(_ text: String, voiceOverride: String? = nil) async {
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
        stt?.bargeInWhilePaused = true
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
                            voiceOverride: voiceOverride,
                            audio: sharedAudio
                        )
                    case .whisper(let clause):
                        chunkCount += try await self.playSynthesized(
                            clause,
                            whisper: playEffects,
                            voiceOverride: voiceOverride,
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
        voiceOverride: String? = nil,
        audio: SharedAudioEngine
    ) async throws -> Int {
        stt?.pauseInference()
        defer { stt?.resumeInference() }

        var chunks = 0
        let voice = voiceOverride ?? selectedVoice
        let temperature: Float = {
            if whisper { return 0.2 }
            // 0.40 is stable and clear but flattens pitch and rhythm. Persona mode trades a
            // little stability for more natural variation. Tune 0.50-0.65 if it slurs.
            return conversationalTone ? 0.58 : 0.4
        }()
        let applyWhisper = whisper
        let generation = speakGeneration

        if let pocketSmall {
            chunks = try await withGPU {
                let stream = try await pocketSmall.synthesizeStreaming(
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
        } else if let pocket100M {
            let isCloned = Self.isClonedVoice(voice)
            chunks = try await withGPU {
                var count = 0
                if isCloned {
                    let fileURL = Self.clonedVoiceURL(named: voice)
                    let voiceData = try PocketTtsVoiceCloner.loadVoice(from: fileURL)
                    let stream = try await pocket100M.synthesizeStreaming(
                        text: text,
                        voiceData: voiceData,
                        temperature: temperature
                    )
                    for try await frame in stream {
                        try Task.checkCancellation()
                        let samples = applyWhisper ? Self.applyWhisper(frame.samples) : frame.samples
                        count += 1
                        let isFirst = count == 1
                        Task { @MainActor in
                            guard generation == self.speakGeneration else { return }
                            if isFirst {
                                self.markFirstInferenceIfNeeded()
                                self.playbackPhase = .playing
                            }
                            audio.scheduleTTSBuffer(samples)
                        }
                    }
                } else {
                    let stream = try await pocket100M.synthesizeStreaming(
                        text: text,
                        voice: voice,
                        temperature: temperature
                    )
                    for try await frame in stream {
                        try Task.checkCancellation()
                        let samples = applyWhisper ? Self.applyWhisper(frame.samples) : frame.samples
                        count += 1
                        let isFirst = count == 1
                        Task { @MainActor in
                            guard generation == self.speakGeneration else { return }
                            if isFirst {
                                self.markFirstInferenceIfNeeded()
                                self.playbackPhase = .playing
                            }
                            audio.scheduleTTSBuffer(samples)
                        }
                    }
                }
                return count
            }
        } else if let superonic {
            let style = try await superonicStyle(named: voice)
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

    /// Synthesizes and plays a short preview of the given voice.
    func speakPreview(_ text: String, voice: String) async {
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

        let task = Task {
            do {
                guard let sharedAudio else {
                    throw TTSError.noSharedAudio
                }
                _ = try await self.playSynthesized(
                    text,
                    whisper: false,
                    voiceOverride: voice,
                    audio: sharedAudio
                )
            } catch {
                if !Task.isCancelled {
                    logger.error("speakPreview failed: \(error.localizedDescription)")
                    self.error = "Preview failed: \(error.localizedDescription)"
                }
            }
            if !Task.isCancelled {
                self.isSpeaking = sharedAudio?.isSpeaking == true
                if !self.isSpeaking {
                    self.playbackPhase = .idle
                }
            }
        }
        speakTask = task
        await task.value
    }

    // MARK: - Cloned Voices

    nonisolated static var clonedVoicesDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("cloned_voices", isDirectory: true)
        if !FileManager.default.fileExists(atPath: dir.path) {
            try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    nonisolated static func listClonedVoiceNames() -> [String] {
        let dir = clonedVoicesDirectory
        guard let files = try? FileManager.default.contentsOfDirectory(atPath: dir.path) else { return [] }
        return files
            .filter { $0.hasSuffix(".bin") }
            .map { ($0 as NSString).deletingPathExtension }
            .sorted()
    }

    nonisolated static func isClonedVoice(_ name: String) -> Bool {
        listClonedVoiceNames().contains(name)
    }

    nonisolated static func clonedVoiceURL(named name: String) -> URL {
        clonedVoicesDirectory.appendingPathComponent("\(name).bin")
    }

    nonisolated static func deleteClonedVoice(named name: String) {
        let file = clonedVoiceURL(named: name)
        try? FileManager.default.removeItem(at: file)
    }

    func cloneVoice(from sourceURL: URL, as rawName: String) async throws {
        if pocket100M == nil {
            try await loadPocket100MTTS()
        }
        guard let manager = pocket100M else {
            throw TTSError.cloningRequires100M
        }

        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .filter { $0.isLetter || $0.isNumber || $0 == "_" }

        guard !name.isEmpty else {
            throw TTSError.invalidVoiceName
        }

        let preprocessed = try Self.loadAndPreprocessAudio(from: sourceURL)
        guard preprocessed.count >= 24000 else {
            throw TTSError.audioTooShort
        }

        let voiceData = try await manager.cloneVoice(from: preprocessed)
        let destination = Self.clonedVoiceURL(named: name)
        try PocketTtsVoiceCloner.saveVoice(voiceData, to: destination)
        logger.info("Cloned voice '\(name)' saved successfully (\(preprocessed.count) samples)")
    }

    func cloneVoice(from samples: [Float], as rawName: String) async throws {
        if pocket100M == nil {
            try await loadPocket100MTTS()
        }
        guard let manager = pocket100M else {
            throw TTSError.cloningRequires100M
        }

        let name = rawName.trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
            .replacingOccurrences(of: " ", with: "_")
            .filter { $0.isLetter || $0.isNumber || $0 == "_" }

        guard !name.isEmpty else {
            throw TTSError.invalidVoiceName
        }

        let preprocessed = Self.preprocessSamples(samples)
        guard preprocessed.count >= 24000 else {
            throw TTSError.audioTooShort
        }

        let voiceData = try await manager.cloneVoice(from: preprocessed)
        let destination = Self.clonedVoiceURL(named: name)
        try PocketTtsVoiceCloner.saveVoice(voiceData, to: destination)
        logger.info("Cloned voice '\(name)' saved successfully from samples (\(preprocessed.count) samples)")
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

    nonisolated private static func loadAndPreprocessAudio(from url: URL) throws -> [Float] {
        let (rawSamples, sampleRate) = try loadAudioFile(at: url)
        let resampled: [Float]
        if sampleRate != 24000 {
            resampled = resample(rawSamples, from: sampleRate, to: 24000)
        } else {
            resampled = rawSamples
        }
        return preprocessSamples(resampled)
    }

    nonisolated private static func loadAudioFile(at url: URL) throws -> (samples: [Float], sampleRate: Double) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard frameCount > 0 else {
            throw TTSError.audioTooShort
        }
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw TTSError.audioConversionFailed
        }
        try file.read(into: buffer)
        guard let channelData = buffer.floatChannelData else {
            throw TTSError.audioConversionFailed
        }
        let channels = Int(format.channelCount)
        let count = Int(buffer.frameLength)
        var mono = [Float](repeating: 0, count: count)
        if channels == 1 {
            mono.withUnsafeMutableBufferPointer { ptr in
                _ = memcpy(ptr.baseAddress!, channelData[0], count * MemoryLayout<Float>.size)
            }
        } else {
            let scale: Float = 1.0 / Float(channels)
            for ch in 0..<channels {
                let src = channelData[ch]
                for i in 0..<count {
                    mono[i] += src[i] * scale
                }
            }
        }
        return (mono, format.sampleRate)
    }

    nonisolated static func preprocessSamples(_ samples: [Float]) -> [Float] {
        guard !samples.isEmpty else { return [] }

        // 1. Peak Normalization
        var maxPeak: Float = 0
        for s in samples {
            let a = abs(s)
            if a > maxPeak { maxPeak = a }
        }

        guard maxPeak > 0.005 else {
            return samples
        }

        // Target nominal level ~ -1.0 dBFS (0.89) with a gain clamp to prevent amplifying faint noise
        let gain = min(0.89 / maxPeak, 15.0)
        let normalized = samples.map { $0 * gain }

        // 2. Windowed RMS Silence Trimming (20ms windows = 480 samples @ 24kHz)
        let windowSize = 480
        let threshold: Float = 0.015
        var startIdx = 0
        var endIdx = normalized.count

        for i in stride(from: 0, to: normalized.count - windowSize, by: windowSize) {
            var sum: Float = 0
            for j in 0..<windowSize {
                let v = normalized[i + j]
                sum += v * v
            }
            let rms = sqrt(sum / Float(windowSize))
            if rms > threshold {
                // Keep ~150ms before speech start to protect initial consonants
                startIdx = max(0, i - (24000 * 15 / 100))
                break
            }
        }

        for i in stride(from: normalized.count - windowSize, through: 0, by: -windowSize) {
            var sum: Float = 0
            for j in 0..<windowSize {
                let v = normalized[i + j]
                sum += v * v
            }
            let rms = sqrt(sum / Float(windowSize))
            if rms > threshold {
                // Keep ~200ms after speech end to preserve trailing release
                endIdx = min(normalized.count, i + windowSize + (24000 * 20 / 100))
                break
            }
        }

        if startIdx < endIdx && (endIdx - startIdx) >= 24000 {
            return Array(normalized[startIdx..<endIdx])
        }

        return normalized
    }

    /// Windowed-sinc resampler. Linear interpolation folds everything above the new
    /// Nyquist back into the audible band, which is the harsh, tinny edge on 44.1 kHz
    /// Supertonic audio. This low-passes first, so the downsample stays clean.
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
        let cutoff = min(1.0, dstRate / srcRate) * 0.94
        let halfWidth = 12.0 / cutoff
        let n = samples.count
        var out = [Float](repeating: 0, count: count)
        samples.withUnsafeBufferPointer { src in
            for i in 0..<count {
                let center = Double(i) * ratio
                let lo = max(0, Int((center - halfWidth).rounded(.up)))
                let hi = min(n - 1, Int((center + halfWidth).rounded(.down)))
                if lo > hi { continue }
                var acc = 0.0
                var norm = 0.0
                for j in lo...hi {
                    let d = Double(j) - center
                    let x = d * cutoff
                    let sinc = abs(x) < 1e-9 ? 1.0 : sin(Double.pi * x) / (Double.pi * x)
                    let window = 0.5 * (1 + cos(Double.pi * d / halfWidth))
                    let w = sinc * window
                    acc += Double(src[j]) * w
                    norm += w
                }
                if abs(norm) > 1e-9 {
                    out[i] = Float(acc / norm)
                }
            }
        }
        return out
    }

    /// Stop all audio playback and cancel in-flight generation.
    func stop() {
        speakGeneration += 1
        speakTask?.cancel()
        holdsInferenceGPU = false
        stt?.bargeInWhilePaused = false
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

    /// Playback finished and no synthesis is running. Clears a stuck
    /// `isSpeaking` so the next utterance is not thrown out as speaker echo.
    func notePlaybackIdle() {
        guard playbackPhase != .synthesizing else { return }
        isSpeaking = sharedAudio?.isSpeaking == true
        if !isSpeaking {
            playbackPhase = .idle
        }
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
    case cloningRequires100M
    case invalidVoiceName
    case audioConversionFailed
    case audioTooShort

    var errorDescription: String? {
        switch self {
        case .engineNotLoaded:
            return "TTS engine not loaded. Call initialize() first."
        case .noSharedAudio:
            return "No shared audio engine available."
        case .cloningRequires100M:
            return "Voice cloning is supported on PocketTTS 100M."
        case .invalidVoiceName:
            return "Please provide a valid voice name (letters, numbers, underscores)."
        case .audioConversionFailed:
            return "Failed to read or convert audio file."
        case .audioTooShort:
            return "Audio recording too short. Please speak clearly for at least 3 to 5 seconds."
        }
    }
}
