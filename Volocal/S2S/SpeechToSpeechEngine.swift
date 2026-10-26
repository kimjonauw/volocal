import Foundation
import AVFoundation
import os
import MLX
import MLXAudioSTS

private let logger = Logger(subsystem: "com.volocal.app", category: "s2s")

enum SpeechToSpeechError: LocalizedError {
    case notEnoughMemory(availableMB: Int, neededMB: Int)
    case notLoaded
    case micFailed(String)
    case previousCrash(stage: String?)
    case incompleteModel

    var errorDescription: String? {
        switch self {
        case .notEnoughMemory(let available, let needed):
            return "Only \(available) MB is free for Volocal and the speech model needs about \(needed) MB. Force-quit and reopen Volocal so the other models release their memory, and make sure GetMoreRAM is on for the LiveContainer host."
        case .notLoaded:
            return "The speech-to-speech model is not loaded yet."
        case .micFailed(let message):
            return "Microphone could not start: \(message)"
        case .previousCrash(let stage):
            let where_ = stage.map { " while \($0)" } ?? ""
            return "The last speech-to-speech session was killed by iOS\(where_), most likely because it ran out of memory. Tap Try again."
        case .incompleteModel:
            return "The speech model downloaded without its tokenizer or audio decoder. Delete it and try again."
        }
    }
}

/// True while the model is thinking or its reply is playing. Read from the mic thread.
private final class AgentGate: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false

    var active: Bool {
        lock.lock()
        defer { lock.unlock() }
        return value
    }

    func set(_ newValue: Bool) {
        lock.lock()
        value = newValue
        lock.unlock()
    }
}

/// Takes the first mic channel and downsamples it to 16 kHz by averaging blocks of samples.
/// This does not depend on AVAudioConverter, which can refuse the Voice Processing format.
private enum MonoResampler {
    static func mono16k(from buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channel = buffer.floatChannelData?[0] else { return [] }
        let n = Int(buffer.frameLength)
        guard n > 0 else { return [] }
        let rate = buffer.format.sampleRate
        guard rate > 0 else { return [] }
        if abs(rate - 16_000) < 1 {
            return Array(UnsafeBufferPointer(start: channel, count: n))
        }
        let ratio = rate / 16_000
        let m = Int(Double(n) / ratio)
        guard m > 0 else { return [] }
        var out = [Float](repeating: 0, count: m)
        for j in 0..<m {
            let start = min(n - 1, Int(Double(j) * ratio))
            let end = min(n, max(start + 1, Int(Double(j + 1) * ratio)))
            var sum: Float = 0
            for i in start..<end { sum += channel[i] }
            out[j] = sum / Float(end - start)
        }
        return out
    }
}

/// Real speech-to-speech with Liquid AI's LFM2.5-Audio-1.5B (6-bit, ~1.3 GB) through the
/// MLXAudioSTS library of mlx-audio-swift (our fork, which adds `generateInterleavedInline`).
/// One model hears your audio and answers in audio, so the recognizer, the language model and
/// the TTS voice are not used in this mode.
///
/// It is turn-based, like a voice-assistant call: when you stop talking (energy turn detector)
/// the model answers, streaming speech. Talking loudly over the reply for about 0.3 s cuts it off.
/// It is not full-duplex, so it will not talk over you or backchannel.
///
/// Every turn is stateless: the model gets only your latest recording. Re-feeding earlier
/// recordings made each turn slower and taught the model to repeat its own greeting.
///
/// Mic audio comes from `SharedAudioEngine` (Voice Processing echo cancellation) and is converted
/// to 16 kHz mono. The reply is decoded to 24 kHz and plays through the same engine.
///
/// All model work (generation and audio decoding) runs one step after another on one thread:
/// MLX is not thread-safe.
@MainActor
final class SpeechToSpeechEngine: ObservableObject {
    /// Same crash fence the language model uses. A jetsam kill during load must not crash-loop.
    static let fenceKey = "speech-to-speech:lfm2.5-audio-1.5b-6bit"
    /// Which stage was running. Left behind if iOS kills the app, so the next launch can say where.
    private static let stageKey = "volocal.s2s.stage"
    private static let modelId = "mlx-community/LFM2.5-Audio-1.5B-6bit"
    /// 6-bit weights are ~1.3 GB; the decoder and working memory need headroom.
    private static let neededBytes = 2_200_000_000
    /// Greedy audio sampling and no top-k sort for text. The library's default sampling sorts
    /// the vocabulary and reads a value back to the CPU for each of the 8 codebooks of every
    /// audio frame. Greedy audio skips all of that. Audio may sound a bit flatter; set this to
    /// false to compare with the library's normal sampling.
    private static let fastSampling = true
    /// Frames in a typical short reply (about 3.2 s). Used to decide how much audio to hold back
    /// before playback starts so it does not run dry. A longer reply can pause briefly.
    private static let expectedReplyFrames = 40.0

    @Published private(set) var isLoaded = false

    private var model: LFM2AudioModel?
    private var processor: LFM2AudioProcessor?
    private var detokenizer: LFM2AudioDetokenizer?
    private var micTask: Task<Void, Never>?
    private var micContinuation: AsyncStream<AVAudioPCMBuffer>.Continuation?
    private var responseTask: Task<Void, Never>?
    private let gate = AgentGate()
    private var session = 0
    private var voiceId = "NATM0"
    private weak var sharedAudioRef: SharedAudioEngine?
    private var agentSpeakingCallback: (@Sendable (Bool) -> Void)?
    private var finishedCallback: (@Sendable (String?) -> Void)?
    private var statusCallback: (@Sendable (String?) -> Void)?

    private static func setStage(_ stage: String?) {
        if let stage {
            UserDefaults.standard.set(stage, forKey: stageKey)
        } else {
            UserDefaults.standard.removeObject(forKey: stageKey)
        }
        UserDefaults.standard.synchronize()
    }

    /// "US female" or "US male" from the preset id (NATF0..VARM4).
    private static func voiceDescription(for voiceId: String) -> String {
        voiceId.contains("F") && !voiceId.hasPrefix("NATM") && !voiceId.hasPrefix("VARM")
            ? "a US female voice"
            : "a US male voice"
    }

    /// The library's required opening for speech-to-speech, the voice, then a short persona:
    /// a relaxed friend, not an assistant. No name is given so there is nothing to introduce.
    private static func systemPrompt(for voiceId: String) -> String {
        "Respond with interleaved text and speech audio. Use \(voiceDescription(for: voiceId)). "
            + "You are a relaxed, curious friend having a casual chat. "
            + "React to what the person just said the way a real friend would, in one or two short, "
            + "natural sentences with contractions, and sometimes ask a quick question back. "
            + "Do not introduce yourself and do not offer help."
    }

    // MARK: - Loading

    func load(progress: @escaping @Sendable (String) -> Void) async throws {
        if model != nil { return }

        let leftoverStage = UserDefaults.standard.string(forKey: Self.stageKey)

        if LLMLoadFence.shouldSkipLoad(path: Self.fenceKey) {
            throw SpeechToSpeechError.previousCrash(stage: leftoverStage)
        }

        var killNote: String?
        if let leftoverStage {
            killNote = "The last session was killed by iOS while \(leftoverStage)."
            Self.setStage(nil)
        }

        let available = Int(os_proc_available_memory())
        if available > 0, available < Self.neededBytes {
            throw SpeechToSpeechError.notEnoughMemory(
                availableMB: available / 1_000_000,
                neededMB: Self.neededBytes / 1_000_000
            )
        }

        // MLX keeps temporary Metal buffers unless capped. There is plenty of room for a
        // larger cache with this small model, and a tiny cap makes every step re-allocate.
        Memory.cacheLimit = 256 * 1024 * 1024

        LLMLoadFence.markStarting(path: Self.fenceKey, contextSize: 0)
        Self.setStage("loading the model weights")
        do {
            progress("Loading LFM2.5-Audio (about 1.3 GB, downloads the first time)...")
            let loaded = try await LFM2AudioModel.fromPretrained(Self.modelId)
            guard let loadedProcessor = loaded.processor, let directory = loaded.modelDirectory else {
                throw SpeechToSpeechError.incompleteModel
            }
            progress("Loading the audio decoder...")
            let loadedDetokenizer = try LFM2AudioDetokenizer.fromPretrained(modelPath: directory)

            model = loaded
            processor = loadedProcessor
            detokenizer = loadedDetokenizer
            isLoaded = true
            Self.setStage(nil)
            LLMLoadFence.clear()
            logger.info("LFM2.5-Audio loaded")

            let freeMB = Int(os_proc_available_memory()) / 1_000_000
            var message = "Loaded. \(freeMB) MB still free."
            if let killNote { message += " " + killNote }
            progress(message)
            try? await Task.sleep(for: .seconds(killNote == nil ? 2 : 4))
        } catch {
            Self.setStage(nil)
            LLMLoadFence.clear()
            throw error
        }
    }

    func unload() {
        responseTask?.cancel()
        responseTask = nil
        model = nil
        processor = nil
        detokenizer = nil
        isLoaded = false
        Memory.clearCache()
    }

    // MARK: - Conversation

    /// Starts listening. Callbacks may be called from any thread. `onStepTime` is unused here
    /// (kept so the pipeline code is the same for any speech-to-speech engine). `onStatus` is a
    /// live one-line status for the screen.
    func start(
        sharedAudio: SharedAudioEngine,
        voiceId: String,
        onAgentSpeaking: @escaping @Sendable (Bool) -> Void,
        onStepTime: @escaping @Sendable (Double) -> Void,
        onFinished: @escaping @Sendable (String?) -> Void,
        onStatus: (@Sendable (String?) -> Void)? = nil
    ) async throws {
        _ = onStepTime
        guard model != nil, processor != nil, detokenizer != nil else {
            throw SpeechToSpeechError.notLoaded
        }

        // The model is not thread-safe. Let any previous reply finish before starting again.
        micTask?.cancel()
        micContinuation?.finish()
        responseTask?.cancel()
        let previous = responseTask
        await previous?.value

        session += 1
        let currentSession = session
        self.voiceId = voiceId
        sharedAudioRef = sharedAudio
        agentSpeakingCallback = onAgentSpeaking
        finishedCallback = onFinished
        statusCallback = onStatus
        gate.set(false)

        let (stream, continuation) = AsyncStream.makeStream(
            of: AVAudioPCMBuffer.self,
            bufferingPolicy: .bufferingNewest(16)
        )
        micContinuation = continuation
        sharedAudio.bridge.inputContinuation = continuation
        await sharedAudio.beginInputCapture()
        if let captureError = sharedAudio.error {
            continuation.finish()
            sharedAudio.bridge.inputContinuation = nil
            micContinuation = nil
            throw SpeechToSpeechError.micFailed(captureError)
        }

        // Left set until a clean stop. If iOS kills the app mid-conversation, the next launch
        // reports it.
        Self.setStage("running the conversation")
        onStatus?("Listening...")

        let gate = self.gate
        micTask = Task.detached(priority: .userInitiated) { [weak self] in
            let detector = MicTurnDetector()
            var inTurn = false
            var turnAudio: [Float] = []
            var preroll: [[Float]] = []
            var loudSeconds = 0.0
            var wasActive = false
            var lastStatusAt = CFAbsoluteTimeGetCurrent()

            for await buffer in stream {
                if Task.isCancelled { break }

                // The detector only needs the energy, so it runs before any resampling.
                let rms = SharedAudioEngine.rmsEnergy(buffer)
                let bufferSeconds = Double(buffer.frameLength) / max(buffer.format.sampleRate, 1)
                let samples = MonoResampler.mono16k(from: buffer)
                if samples.isEmpty { continue }

                // While the model is thinking or talking, only watch for the user cutting in.
                if gate.active {
                    if !wasActive {
                        wasActive = true
                        detector.resetUtterance()
                        inTurn = false
                        turnAudio = []
                        loudSeconds = 0
                    }
                    preroll.append(samples)
                    if preroll.count > 6 { preroll.removeFirst() }
                    loudSeconds = rms >= 0.05 ? loudSeconds + bufferSeconds : 0
                    if loudSeconds >= 0.3 {
                        loudSeconds = 0
                        await self?.interruptResponse(session: currentSession)
                        wasActive = false
                        inTurn = true
                        turnAudio = preroll.flatMap { $0 }
                        preroll.removeAll()
                        _ = detector.observe(rms: rms, eager: false, quietEnd: 0.55)
                    }
                    continue
                }
                wasActive = false

                // No transcript to judge completeness from, so a fixed 0.55 s of quiet ends the turn.
                let decision = detector.observe(rms: rms, eager: false, quietEnd: 0.55)
                if decision.feedAsr {
                    if !inTurn {
                        inTurn = true
                        turnAudio = preroll.flatMap { $0 }
                        preroll.removeAll()
                    }
                    turnAudio.append(contentsOf: samples)
                } else if !inTurn {
                    preroll.append(samples)
                    if preroll.count > 6 { preroll.removeFirst() }
                }

                // A live line so you can see what the app hears (about 4 times a second).
                let now = CFAbsoluteTimeGetCurrent()
                if now - lastStatusAt > 0.25 {
                    lastStatusAt = now
                    if inTurn {
                        let seconds = Double(turnAudio.count) / 16_000
                        onStatus?(String(format: "Hearing you... %.1f s (mic %.3f)", seconds, rms))
                    } else {
                        onStatus?(String(format: "Listening... mic level %.3f, needs more than 0.010 to count as speech", rms))
                    }
                }

                let tooLong = turnAudio.count > 16_000 * 30
                if inTurn, decision.endTurn || tooLong {
                    inTurn = false
                    let audio = turnAudio
                    turnAudio = []
                    // Ignore coughs and clicks shorter than 0.6 s.
                    if audio.count >= 16_000 * 6 / 10 {
                        await self?.submit(audio: audio, session: currentSession)
                    } else {
                        onStatus?("That was too short to count (under 0.6 s). Listening...")
                    }
                }
            }
        }
    }

    func stop(sharedAudio: SharedAudioEngine) {
        session += 1
        micTask?.cancel()
        micTask = nil
        micContinuation?.finish()
        micContinuation = nil
        responseTask?.cancel()
        gate.set(false)
        sharedAudio.endInputCapture()
        sharedAudio.stopPlayback()
        // A clean stop: nothing was killed.
        Self.setStage(nil)
    }

    // MARK: - Replies

    private func interruptResponse(session currentSession: Int) {
        guard currentSession == session else { return }
        responseTask?.cancel()
        gate.set(false)
        sharedAudioRef?.stopPlayback()
        agentSpeakingCallback?(false)
        statusCallback?("Interrupted. Listening...")
    }

    private func submit(audio: [Float], session currentSession: Int) {
        guard currentSession == session, let sharedAudio = sharedAudioRef else { return }
        statusCallback?(String(format: "Got %.1f s of speech. Thinking...", Double(audio.count) / 16_000))
        beginResponse(userAudio: audio, sharedAudio: sharedAudio, session: currentSession)
    }

    private func beginResponse(userAudio: [Float], sharedAudio: SharedAudioEngine, session currentSession: Int) {
        guard let model, let processor, let detokenizer else { return }
        gate.set(true)
        responseTask?.cancel()
        let previous = responseTask
        let systemText = Self.systemPrompt(for: voiceId)
        let gate = self.gate
        let speaking = agentSpeakingCallback
        let status = statusCallback
        let firstChunkFrames = 6
        let chunkFrames = 12
        let overlapFrames = 3
        let expectedFrames = Self.expectedReplyFrames
        let startedAt = CFAbsoluteTimeGetCurrent()
        let fast = Self.fastSampling
        let config: LFMGenerationConfig = fast
            ? LFMGenerationConfig(
                maxNewTokens: 400, temperature: 0.7, topK: 0, topP: 1.0,
                audioTemperature: 0, audioTopK: 1
            )
            : LFMGenerationConfig(maxNewTokens: 400)

        responseTask = Task.detached(priority: .userInitiated) {
            await previous?.value
            if Task.isCancelled { return }

            // Stateless: system line plus only this turn's recording.
            let chat = ChatState(processor: processor)
            chat.newTurn(role: "system")
            chat.addText(systemText)
            chat.endTurn()
            chat.newTurn(role: "user")
            chat.addAudio(MLXArray(userAudio), sampleRate: 16_000)
            chat.endTurn()
            chat.newTurn(role: "assistant")

            var codes: [MLXArray] = []
            var replyTokens: [Int] = []
            var decodedFrames = 0
            var pendingPCM: [[Float]] = []
            var playbackStarted = false
            var firstAudioAt: Double?
            var firstFrameAt: Double?
            var prefillDoneAt: Double?

            // Decode frames [decodedFrames ..< end). A few earlier frames are decoded again for
            // context and their audio is dropped, so chunk seams do not click.
            func decodeChunk(upTo end: Int) -> [Float] {
                let overlap = min(overlapFrames, decodedFrames)
                let slice = Array(codes[(decodedFrames - overlap)..<end])
                let stacked = MLX.stacked(slice, axis: 0)
                let input = stacked.transposed(1, 0).expandedDimensions(axis: 0)
                let wave = detokenizer(input)
                let all = wave.asType(.float32).asArray(Float.self)
                let perFrame = all.count / max(slice.count, 1)
                decodedFrames = end
                return Array(all.dropFirst(overlap * perFrame))
            }

            // Hands audio to the player on the main thread, in order.
            func schedule(_ samples: [Float]) {
                guard !samples.isEmpty else { return }
                DispatchQueue.main.async {
                    MainActor.assumeIsolated {
                        sharedAudio.scheduleTTSBuffer(samples)
                    }
                }
            }

            func startPlayback() {
                guard !playbackStarted else { return }
                playbackStarted = true
                firstAudioAt = CFAbsoluteTimeGetCurrent()
                for chunk in pendingPCM { schedule(chunk) }
                pendingPCM.removeAll()
                speaking?(true)
            }

            // Playback starts once enough audio is ready that it will not run dry: the model
            // makes `rate` frames per second and speech needs 12.5, so a slower model needs a
            // bigger head start. A model that keeps up starts after the first chunk.
            func considerStart() {
                guard !playbackStarted, let t0 = firstFrameAt else { return }
                let produced = codes.count
                let elapsed = CFAbsoluteTimeGetCurrent() - t0
                guard produced >= 4, elapsed > 0.05 else { return }
                let rate = Double(produced - 1) / elapsed
                let needed = max(Double(firstChunkFrames), expectedFrames * max(0, 1 - rate / 12.5))
                status?(String(
                    format: "Generating audio at %.1f frames/s (speech needs 12.5). Sound starts once %d frames are ready (%d so far).",
                    rate, Int(needed.rounded(.up)), decodedFrames
                ))
                if Double(decodedFrames) >= needed { startPlayback() }
            }

            func deliver(_ samples: [Float]) {
                if playbackStarted {
                    schedule(samples)
                } else {
                    pendingPCM.append(samples)
                    considerStart()
                }
            }

            // The model and the audio decoder both run right here, one step after another.
            model.generateInterleavedInline(
                textTokens: chat.getTextTokens(),
                audioFeatures: chat.getAudioFeatures(),
                modalities: chat.getModalities(),
                config: config,
                onPrefillDone: {
                    let now = CFAbsoluteTimeGetCurrent()
                    prefillDoneAt = now
                    status?(String(format: "Your recording was processed in %.1f s. Generating the reply...", now - startedAt))
                },
                onToken: { token, modality in
                    if Task.isCancelled { return false }
                    eval(token)
                    if modality == .text {
                        replyTokens.append(token.item(Int.self))
                    } else if modality == .audioOut {
                        // The end-of-audio frame is skipped; only the end of generation ends
                        // the reply (same as the library's own example).
                        if token[0].item(Int.self) == lfmAudioEOSToken { return true }
                        if firstFrameAt == nil { firstFrameAt = CFAbsoluteTimeGetCurrent() }
                        codes.append(token)
                        // A short first chunk, then bigger ones.
                        let target = decodedFrames == 0 ? firstChunkFrames : chunkFrames
                        if codes.count - decodedFrames >= target {
                            deliver(decodeChunk(upTo: codes.count))
                        }
                    }
                    return true
                }
            )

            if Task.isCancelled { return }

            // Decode what is left and make sure everything that is ready gets played.
            if codes.count > decodedFrames {
                let tail = decodeChunk(upTo: codes.count)
                if playbackStarted {
                    schedule(tail)
                } else {
                    pendingPCM.append(tail)
                }
            }
            if !playbackStarted, !pendingPCM.isEmpty { startPlayback() }

            let replyText = processor.decodeText(replyTokens)
            let doneAt = CFAbsoluteTimeGetCurrent()
            let total = doneAt - startedAt
            let mode = fast ? "fast" : "normal"
            if codes.isEmpty {
                status?(String(format: "No audio came back after %.1f s (%d text tokens, %@ sampling). Text: ", total, replyTokens.count, mode) + replyText)
            } else {
                // 8-code frames are 0.08 s of speech each.
                let audioSeconds = Double(codes.count) * 0.08
                let prepared = (prefillDoneAt ?? doneAt) - startedAt
                let generating = max(total - prepared, 0.01)
                let fps = Double(codes.count) / generating
                let first = firstAudioAt.map { $0 - startedAt } ?? total
                status?(String(
                    format: "Replied in %.1f s for %.1f s of audio. Recording processed in %.1f s, then %.1f frames/s (speech needs 12.5). First sound after %.1f s, %@ sampling: ",
                    total, audioSeconds, prepared, fps, first, mode
                ) + replyText)
            }

            // Wait for the queued main-thread hand-offs, then for the audio to finish playing.
            await MainActor.run {}
            await sharedAudio.waitForPlaybackCompletion()
            if playbackStarted { speaking?(false) }
            try? await Task.sleep(for: .milliseconds(300))
            gate.set(false)
            Memory.clearCache()
        }
    }
}
