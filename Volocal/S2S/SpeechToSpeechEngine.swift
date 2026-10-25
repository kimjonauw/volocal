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

private struct Exchange {
    var audio: [Float]
    var reply: String
}

/// Real speech-to-speech with Liquid AI's LFM2.5-Audio-1.5B (6-bit, ~1.3 GB) through the
/// MLXAudioSTS library of mlx-audio-swift. One model hears your audio and answers in audio, so
/// the recognizer, the language model and the TTS voice are not used in this mode.
///
/// It is turn-based, like a voice-assistant call: when you stop talking (energy turn detector)
/// the model answers, streaming speech. Talking loudly over the reply for about 0.3 s cuts it off.
/// It is not full-duplex, so it will not talk over you or backchannel.
///
/// Mic audio comes from `SharedAudioEngine` (Voice Processing echo cancellation) and is converted
/// to 16 kHz mono. The reply is decoded to 24 kHz and plays through the same engine.
@MainActor
final class SpeechToSpeechEngine: ObservableObject {
    /// Same crash fence the language model uses. A jetsam kill during load must not crash-loop.
    static let fenceKey = "speech-to-speech:lfm2.5-audio-1.5b-6bit"
    /// Which stage was running. Left behind if iOS kills the app, so the next launch can say where.
    private static let stageKey = "volocal.s2s.stage"
    private static let modelId = "mlx-community/LFM2.5-Audio-1.5B-6bit"
    /// 6-bit weights are ~1.3 GB; the decoder and working memory need headroom.
    private static let neededBytes = 2_200_000_000

    @Published private(set) var isLoaded = false

    private var model: LFM2AudioModel?
    private var processor: LFM2AudioProcessor?
    private var detokenizer: LFM2AudioDetokenizer?
    private var micTask: Task<Void, Never>?
    private var micContinuation: AsyncStream<AVAudioPCMBuffer>.Continuation?
    private var responseTask: Task<Void, Never>?
    private var history: [Exchange] = []
    private let gate = AgentGate()
    private var session = 0
    private var voiceId = "NATM0"
    private weak var sharedAudioRef: SharedAudioEngine?
    private var agentSpeakingCallback: (@Sendable (Bool) -> Void)?
    private var finishedCallback: (@Sendable (String?) -> Void)?

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

    private static func systemPrompt(for voiceId: String) -> String {
        "Respond with interleaved text and speech audio. Use \(voiceDescription(for: voiceId)). "
            + "Talk like a warm, natural friend on a phone call. Keep every reply to one or two short sentences."
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

        // MLX keeps temporary Metal buffers unless capped.
        Memory.cacheLimit = 64 * 1024 * 1024

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
        history.removeAll()
        isLoaded = false
        Memory.clearCache()
    }

    // MARK: - Conversation

    /// Starts listening. Callbacks may be called from any thread. `onStepTime` is unused here
    /// (kept so the pipeline code is the same for any speech-to-speech engine).
    func start(
        sharedAudio: SharedAudioEngine,
        voiceId: String,
        onAgentSpeaking: @escaping @Sendable (Bool) -> Void,
        onStepTime: @escaping @Sendable (Double) -> Void,
        onFinished: @escaping @Sendable (String?) -> Void
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
        history.removeAll()
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

        let gate = self.gate
        micTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let target = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 16_000,
                channels: 1,
                interleaved: false
            ) else { return }

            let detector = MicTurnDetector()
            var converter: AVAudioConverter?
            var converterInput: AVAudioFormat?
            var inTurn = false
            var turnAudio: [Float] = []
            var preroll: [[Float]] = []
            var loudSeconds = 0.0
            var wasActive = false

            for await buffer in stream {
                if Task.isCancelled { break }

                // Mic buffer (device rate) -> 16 kHz mono.
                if converter == nil || converterInput != buffer.format {
                    converter = AVAudioConverter(from: buffer.format, to: target)
                    converterInput = buffer.format
                }
                guard let converter else { continue }
                let ratio = 16_000 / buffer.format.sampleRate
                let capacity = AVAudioFrameCount(Double(buffer.frameLength) * ratio) + 64
                guard let out = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: capacity) else { continue }
                var consumed = false
                var convertError: NSError?
                _ = converter.convert(to: out, error: &convertError) { _, inputStatus in
                    if consumed {
                        inputStatus.pointee = .noDataNow
                        return nil
                    }
                    consumed = true
                    inputStatus.pointee = .haveData
                    return buffer
                }
                if convertError != nil { continue }
                guard let data = out.floatChannelData?[0], out.frameLength > 0 else { continue }
                let samples = Array(UnsafeBufferPointer(start: data, count: Int(out.frameLength)))

                let rms = SharedAudioEngine.rmsEnergy(buffer)
                let bufferSeconds = Double(buffer.frameLength) / buffer.format.sampleRate

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
                        _ = detector.observe(rms: rms, eager: false)
                    }
                    continue
                }
                wasActive = false

                let decision = detector.observe(rms: rms, eager: false)
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

                let tooLong = turnAudio.count > 16_000 * 30
                if inTurn, decision.endTurn || tooLong {
                    inTurn = false
                    let audio = turnAudio
                    turnAudio = []
                    // Ignore coughs and clicks shorter than 0.6 s.
                    if audio.count >= 16_000 * 6 / 10 {
                        await self?.submit(audio: audio, session: currentSession)
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
    }

    private func submit(audio: [Float], session currentSession: Int) {
        guard currentSession == session, let sharedAudio = sharedAudioRef else { return }
        beginResponse(userAudio: audio, sharedAudio: sharedAudio, session: currentSession)
    }

    private func beginResponse(userAudio: [Float], sharedAudio: SharedAudioEngine, session currentSession: Int) {
        guard let model, let processor, let detokenizer else { return }
        gate.set(true)
        responseTask?.cancel()
        let previous = responseTask
        let exchanges = history
        let systemText = Self.systemPrompt(for: voiceId)
        let gate = self.gate
        let speaking = agentSpeakingCallback
        let finished = finishedCallback
        let chunkFrames = 12
        let overlapFrames = 3

        responseTask = Task.detached(priority: .userInitiated) { [weak self] in
            await previous?.value
            if Task.isCancelled { return }

            // Build the conversation: system line, the last few exchanges, then this turn.
            let chat = ChatState(processor: processor)
            chat.newTurn(role: "system")
            chat.addText(systemText)
            chat.endTurn()
            for exchange in exchanges {
                chat.newTurn(role: "user")
                chat.addAudio(MLXArray(exchange.audio), sampleRate: 16_000)
                chat.endTurn()
                chat.newTurn(role: "assistant")
                chat.addText(exchange.reply)
                chat.endTurn()
            }
            chat.newTurn(role: "user")
            chat.addAudio(MLXArray(userAudio), sampleRate: 16_000)
            chat.endTurn()
            chat.newTurn(role: "assistant")

            var codes: [MLXArray] = []
            var replyTokens: [Int] = []
            var decodedFrames = 0
            var announcedSpeaking = false
            var failure: String?

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

            func play(_ samples: [Float]) async {
                guard !samples.isEmpty, !Task.isCancelled else { return }
                await MainActor.run {
                    sharedAudio.scheduleTTSBuffer(samples)
                }
                if !announcedSpeaking {
                    announcedSpeaking = true
                    speaking?(true)
                }
            }

            do {
                let config = LFMGenerationConfig(maxNewTokens: 640)
                for try await (token, modality) in model.generateInterleaved(
                    textTokens: chat.getTextTokens(),
                    audioFeatures: chat.getAudioFeatures(),
                    modalities: chat.getModalities(),
                    config: config
                ) {
                    if Task.isCancelled { break }
                    eval(token)
                    if modality == .text {
                        replyTokens.append(token.item(Int.self))
                    } else if modality == .audioOut {
                        if token[0].item(Int.self) == lfmAudioEOSToken { break }
                        codes.append(token)
                        if codes.count - decodedFrames >= chunkFrames {
                            await play(decodeChunk(upTo: codes.count))
                        }
                    }
                }
                if !Task.isCancelled, codes.count > decodedFrames {
                    await play(decodeChunk(upTo: codes.count))
                }
            } catch {
                failure = "Speech-to-speech stopped: \(error.localizedDescription)"
            }

            if Task.isCancelled { return }

            // Let the reply finish playing, then listen again.
            await sharedAudio.waitForPlaybackCompletion()
            if announcedSpeaking { speaking?(false) }
            try? await Task.sleep(for: .milliseconds(300))
            gate.set(false)

            let wasCancelled = Task.isCancelled
            await MainActor.run {
                guard let self, self.session == currentSession, !wasCancelled else { return }
                if let failure {
                    finished?(failure)
                    return
                }
                let reply = processor.decodeText(replyTokens)
                self.history.append(Exchange(audio: userAudio, reply: reply))
                while self.history.count > 3 { self.history.removeFirst() }
                var total = self.history.reduce(0) { $0 + $1.audio.count }
                while total > 16_000 * 45, self.history.count > 1 {
                    total -= self.history.removeFirst().audio.count
                }
                Memory.clearCache()
            }
        }
    }
}
