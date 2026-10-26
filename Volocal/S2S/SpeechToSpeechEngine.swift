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

/// Turn-based LFM2.5-Audio with inline generation callbacks. Decode and enqueue each chunk
/// immediately: no expected-reply-length estimate and no adaptive playback prebuffer.
/// Generation can under-run playback on slower devices; low onset latency takes priority.
@MainActor
final class SpeechToSpeechEngine: ObservableObject {
    static let fenceKey = "speech-to-speech:lfm2.5-audio-1.5b-6bit"
    private static let stageKey = "volocal.s2s.stage"
    private static let modelId = "mlx-community/LFM2.5-Audio-1.5B-6bit"
    private static let neededBytes = 2_200_000_000
    private static let fastSampling = true

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

    private static func voiceDescription(for voiceId: String) -> String {
        voiceId.contains("F") && !voiceId.hasPrefix("NATM") && !voiceId.hasPrefix("VARM")
            ? "a US female voice" : "a US male voice"
    }

    private static func systemPrompt(for voiceId: String) -> String {
        "Respond with interleaved text and speech audio. Use \(voiceDescription(for: voiceId)). "
            + "You are a relaxed, curious friend having a casual chat. "
            + "React to what the person just said the way a real friend would, in one or two short, "
            + "natural sentences with contractions, and sometimes ask a quick question back. "
            + "Do not introduce yourself and do not offer help."
    }

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
                availableMB: available / 1_000_000, neededMB: Self.neededBytes / 1_000_000
            )
        }
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
            of: AVAudioPCMBuffer.self, bufferingPolicy: .bufferingNewest(16)
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
                let rms = SharedAudioEngine.rmsEnergy(buffer)
                let bufferSeconds = Double(buffer.frameLength) / max(buffer.format.sampleRate, 1)
                let samples = MonoResampler.mono16k(from: buffer)
                if samples.isEmpty { continue }
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
        Self.setStage(nil)
    }

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
        let startedAt = CFAbsoluteTimeGetCurrent()
        let fast = Self.fastSampling
        let config: LFMGenerationConfig = fast
            ? LFMGenerationConfig(
                maxNewTokens: 400, temperature: 0.7, topK: 0, topP: 1.0,
                audioTemperature: 0, audioTopK: 1
            )
            : LFMGenerationConfig(maxNewTokens: 400)

        responseTask = Task.detached(priority: .userInitiated) { [weak self] in
            await previous?.value
            if Task.isCancelled { return }
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
            var playbackStarted = false
            var firstQueuedAt: Double?
            var prefillDoneAt: Double?
            var outputSamples = 0

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

            // Enqueue every decoded chunk immediately, in FIFO order. The task and session
            // checks discard hand-offs that were queued before an interruption or restart.
            let responseTaskHandle = withUnsafeCurrentTask { $0 }
            func schedule(_ samples: [Float]) {
                guard !samples.isEmpty, !Task.isCancelled else { return }
                let first = !playbackStarted
                playbackStarted = true
                outputSamples += samples.count
                if first { firstQueuedAt = CFAbsoluteTimeGetCurrent() }
                DispatchQueue.main.async { [weak self] in
                    MainActor.assumeIsolated {
                        guard let self, self.session == currentSession,
                              responseTaskHandle?.isCancelled != true else { return }
                        sharedAudio.scheduleTTSBuffer(samples)
                        if first { speaking?(true) }
                    }
                }
            }

            model.generateInterleavedInline(
                textTokens: chat.getTextTokens(),
                audioFeatures: chat.getAudioFeatures(),
                modalities: chat.getModalities(),
                config: config,
                onPrefillDone: {
                    let now = CFAbsoluteTimeGetCurrent()
                    prefillDoneAt = now
                    status?(String(format: "Recording processed in %.1f s. Playing each chunk as soon as decoded.", now - startedAt))
                },
                onToken: { token, modality in
                    if Task.isCancelled { return false }
                    eval(token)
                    if modality == .text {
                        replyTokens.append(token.item(Int.self))
                    } else if modality == .audioOut {
                        if token[0].item(Int.self) == lfmAudioEOSToken { return true }
                        codes.append(token)
                        let target = decodedFrames == 0 ? firstChunkFrames : chunkFrames
                        if codes.count - decodedFrames >= target {
                            schedule(decodeChunk(upTo: codes.count))
                        }
                    }
                    return true
                }
            )
            if Task.isCancelled { return }
            if codes.count > decodedFrames {
                schedule(decodeChunk(upTo: codes.count))
            }

            let replyText = processor.decodeText(replyTokens)
            let doneAt = CFAbsoluteTimeGetCurrent()
            let total = doneAt - startedAt
            let mode = fast ? "fast" : "normal"
            if outputSamples == 0 {
                status?(String(format: "No audio came back after %.1f s (%d text tokens, %@ sampling). Text: ", total, replyTokens.count, mode) + replyText)
            } else {
                let audioSeconds = Double(outputSamples) / 24_000
                let prepared = (prefillDoneAt ?? doneAt) - startedAt
                let fps = Double(codes.count) / max(total - prepared, 0.01)
                let first = firstQueuedAt.map { $0 - startedAt } ?? total
                status?(String(
                    format: "Generated in %.1f s for %.1f s of PCM. Input processing %.1f s, then %.1f frames/s. First chunk queued after %.1f s, no prebuffer, %@ sampling: ",
                    total, audioSeconds, prepared, fps, first, mode
                ) + replyText)
            }

            // Drain earlier FIFO audio hand-offs before checking playback completion.
            await withCheckedContinuation { (continuation: CheckedContinuation<Void, Never>) in
                DispatchQueue.main.async { continuation.resume() }
            }
            if Task.isCancelled { return }
            await sharedAudio.waitForPlaybackCompletion()
            if Task.isCancelled { return }
            if playbackStarted { speaking?(false) }
            try? await Task.sleep(for: .milliseconds(300))
            if Task.isCancelled { return }
            gate.set(false)
            Memory.clearCache()
        }
    }
}
