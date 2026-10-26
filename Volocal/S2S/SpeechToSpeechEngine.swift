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
        case .notEnoughMemory(let a, let n): return "Only \(a) MB free; speech model needs about \(n) MB. Reopen Volocal with GetMoreRAM enabled on the host."
        case .notLoaded: return "Speech-to-speech model is not loaded."
        case .micFailed(let s): return "Microphone could not start: \(s)"
        case .previousCrash(let s): return "Previous session ended unexpectedly while \(s ?? "loading"). Tap Try again."
        case .incompleteModel: return "Speech model is missing its processor or decoder."
        }
    }
}

private final class SpeechRun: @unchecked Sendable {
    private let lock = NSLock()
    private var cancelled = false
    var active: Bool { lock.lock(); defer { lock.unlock() }; return !cancelled }
    func cancel() { lock.lock(); cancelled = true; lock.unlock() }
}

private final class AgentGate: @unchecked Sendable {
    private let lock = NSLock()
    private var value = false
    var active: Bool { lock.lock(); defer { lock.unlock() }; return value }
    func set(_ v: Bool) { lock.lock(); value = v; lock.unlock() }
}

private enum MonoResampler {
    static func mono16k(from buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channel = buffer.floatChannelData?[0], buffer.frameLength > 0,
              buffer.format.sampleRate > 0 else { return [] }
        let n = Int(buffer.frameLength)
        let ratio = buffer.format.sampleRate / 16_000
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

/// Immediate chunk playback with bounded generation. Prefix decoding retains context; it is
/// an interim, more expensive alternative to a stateful decoder, not a full-reply prebuffer.
@MainActor
final class SpeechToSpeechEngine: ObservableObject {
    static let fenceKey = "speech-to-speech:lfm2.5-audio-1.5b-6bit"
    private static let stageKey = "volocal.s2s.stage"
    private static let modelId = "mlx-community/LFM2.5-Audio-1.5B-6bit"
    @Published private(set) var isLoaded = false
    private var model: LFM2AudioModel?
    private var processor: LFM2AudioProcessor?
    private var detokenizer: LFM2AudioDetokenizer?
    private var decoderTailSamples = 1280
    private var micTask: Task<Void, Never>?
    private var micContinuation: AsyncStream<AVAudioPCMBuffer>.Continuation?
    private var responseTask: Task<Void, Never>?
    private var watchdog: Task<Void, Never>?
    private var run: SpeechRun?
    private let gate = AgentGate()
    private var session = 0
    private var responseRevision = 0
    private var voiceId = "NATM0"
    private var lastReplyStatus = ""
    private weak var sharedAudioRef: SharedAudioEngine?
    private var speakingCallback: (@Sendable (Bool) -> Void)?
    private var statusCallback: (@Sendable (String?) -> Void)?

    private static func setStage(_ s: String?) {
        if let s { UserDefaults.standard.set(s, forKey: stageKey) }
        else { UserDefaults.standard.removeObject(forKey: stageKey) }
        UserDefaults.standard.synchronize()
    }

    private func recordStop(_ message: String) {
        lastReplyStatus = message
        let stamp = ISO8601DateFormatter().string(from: Date())
        UserDefaults.standard.set(stamp + " — " + message, forKey: PersonaConversationSettings.stopKey)
        statusCallback?(message)
    }

    func load(progress: @escaping @Sendable (String) -> Void) async throws {
        if model != nil { return }
        let old = UserDefaults.standard.string(forKey: Self.stageKey)
        if LLMLoadFence.shouldSkipLoad(path: Self.fenceKey) {
            throw SpeechToSpeechError.previousCrash(stage: old)
        }
        let available = Int(os_proc_available_memory())
        if available > 0, available < 2_200_000_000 {
            throw SpeechToSpeechError.notEnoughMemory(availableMB: available / 1_000_000, neededMB: 2200)
        }
        Memory.cacheLimit = 256 * 1024 * 1024
        LLMLoadFence.markStarting(path: Self.fenceKey, contextSize: 0)
        Self.setStage("loading model")
        do {
            progress("Loading LFM2.5-Audio...")
            let loaded = try await LFM2AudioModel.fromPretrained(Self.modelId)
            guard let p = loaded.processor, let dir = loaded.modelDirectory else {
                throw SpeechToSpeechError.incompleteModel
            }
            progress("Loading audio decoder...")
            let decoder = try LFM2AudioDetokenizer.fromPretrained(modelPath: dir)
            if let data = try? Data(contentsOf: dir.appendingPathComponent("audio_detokenizer/config.json")),
               let json = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any],
               let n = json["n_fft"] as? Int, n > 0 { decoderTailSamples = n }
            model = loaded; processor = p; detokenizer = decoder; isLoaded = true
            LLMLoadFence.clear(); Self.setStage(nil)
            progress("Loaded. \(Int(os_proc_available_memory()) / 1_000_000) MB free.")
            try? await Task.sleep(for: .seconds(2))
        } catch { Self.setStage(nil); LLMLoadFence.clear(); throw error }
    }

    func unload() {
        if let audio = sharedAudioRef { stop(sharedAudio: audio) }
        else { run?.cancel(); responseTask?.cancel(); watchdog?.cancel(); micTask?.cancel() }
        model = nil; processor = nil; detokenizer = nil; isLoaded = false
    }

    func start(
        sharedAudio: SharedAudioEngine, voiceId: String,
        onAgentSpeaking: @escaping @Sendable (Bool) -> Void,
        onStepTime: @escaping @Sendable (Double) -> Void,
        onFinished: @escaping @Sendable (String?) -> Void,
        onStatus: (@Sendable (String?) -> Void)? = nil
    ) async throws {
        _ = onStepTime; _ = onFinished
        guard model != nil, processor != nil, detokenizer != nil else { throw SpeechToSpeechError.notLoaded }
        stop(sharedAudio: sharedAudio)
        let ticket = session
        await responseTask?.value
        guard ticket == session else { return }
        self.voiceId = voiceId; sharedAudioRef = sharedAudio
        speakingCallback = onAgentSpeaking; statusCallback = onStatus
        lastReplyStatus = ""
        let (stream, continuation) = AsyncStream.makeStream(
            of: AVAudioPCMBuffer.self, bufferingPolicy: .bufferingNewest(16)
        )
        micContinuation = continuation
        sharedAudio.bridge.inputContinuation = continuation
        await sharedAudio.beginInputCapture()
        guard ticket == session else { continuation.finish(); return }
        if let e = sharedAudio.error {
            sharedAudio.endInputCapture(); continuation.finish(); throw SpeechToSpeechError.micFailed(e)
        }
        Self.setStage("listening")
        onStatus?("Listening. Persona prompt & status opens the editor and last stop reason.")
        let gate = self.gate
        micTask = Task.detached(priority: .userInitiated) { [weak self] in
            let detector = MicTurnDetector()
            var audio: [Float] = []
            var preroll: [[Float]] = []
            var inTurn = false
            var wasBusy = false
            var lastStatus = CFAbsoluteTimeGetCurrent()
            for await buffer in stream {
                if Task.isCancelled { break }
                // No automatic sound-triggered interruption in this diagnostic build.
                if gate.active {
                    wasBusy = true; audio.removeAll(); preroll.removeAll(); inTurn = false
                    detector.resetUtterance(); continue
                }
                if wasBusy { detector.resetUtterance(); wasBusy = false }
                let samples = MonoResampler.mono16k(from: buffer)
                guard !samples.isEmpty else { continue }
                let rms = SharedAudioEngine.rmsEnergy(buffer)
                let decision = detector.observe(rms: rms, eager: false, quietEnd: 0.55)
                if !inTurn {
                    preroll.append(samples)
                    if preroll.count > 6 { preroll.removeFirst() }
                    if decision.feedAsr {
                        inTurn = true; audio = preroll.flatMap { $0 }; preroll.removeAll()
                    }
                } else { audio.append(contentsOf: samples) }
                if inTurn, decision.endTurn || audio.count >= 16_000 * 20 {
                    let turn = audio; audio = []; inTurn = false; detector.resetUtterance()
                    if turn.count >= 9600 { await self?.submit(turn, session: ticket) }
                }
                let now = CFAbsoluteTimeGetCurrent()
                if !gate.active, now - lastStatus > 0.3 {
                    lastStatus = now
                    await self?.showMicStatus(inTurn: inTurn, rms: rms, session: ticket)
                }
            }
        }
    }

    private func showMicStatus(inTurn: Bool, rms: Float, session ticket: Int) {
        guard ticket == session, !gate.active else { return }
        // A stop reason stays on screen until the next captured turn is submitted.
        guard lastReplyStatus.isEmpty else { return }
        statusCallback?(String(format: inTurn ? "Hearing you (mic %.3f)..." : "Listening (mic %.3f)...", rms))
    }

    func stop(sharedAudio: SharedAudioEngine) {
        if run?.active == true { recordStop("Session stopped externally (button, mode change, or pipeline callback); not an S2S mic-triggered interrupt.") }
        session += 1; responseRevision += 1
        run?.cancel(); watchdog?.cancel(); watchdog = nil
        responseTask?.cancel(); micTask?.cancel(); micTask = nil
        micContinuation?.finish(); micContinuation = nil
        sharedAudio.endInputCapture(); sharedAudio.stopPlayback()
        gate.set(false); Self.setStage(nil)
    }

    private func submit(_ audio: [Float], session ticket: Int) {
        guard session == ticket, !gate.active,
              let model, let processor, let detokenizer, let sharedAudio = sharedAudioRef else { return }
        run?.cancel(); responseTask?.cancel(); watchdog?.cancel()
        let previous = responseTask
        responseRevision += 1
        let revision = responseRevision
        let token = SpeechRun(); run = token; gate.set(true)
        lastReplyStatus = ""
        let female = voiceId.contains("F")
        let saved = (UserDefaults.standard.string(forKey: PersonaConversationSettings.promptKey) ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let personality = saved.isEmpty ? PersonaConversationSettings.defaultPrompt : saved
        let system = "Respond with interleaved text and speech audio. Use a US \(female ? "female" : "male") voice. " + personality
        let tailGuard = decoderTailSamples
        statusCallback?("Processing your speech...")
        Self.setStage("generating reply")
        watchdog = Task { [weak self] in
            try? await Task.sleep(for: .seconds(30))
            guard !Task.isCancelled else { return }
            self?.finishResponse(token, revision: revision, session: ticket, forced: true,
                                 message: "Safety cutoff: 30-second response/playback deadline.")
        }
        responseTask = Task.detached(priority: .userInitiated) { [weak self] in
            await previous?.value
            guard token.active, !Task.isCancelled else { return }
            let started = CFAbsoluteTimeGetCurrent()
            let chat = ChatState(processor: processor)
            chat.newTurn(role: "system"); chat.addText(system); chat.endTurn()
            chat.newTurn(role: "user"); chat.addAudio(MLXArray(audio), sampleRate: 16_000); chat.endTurn()
            chat.newTurn(role: "assistant")
            var codes: [MLXArray] = []
            var text: [Int] = []
            var recent: [[Int32]] = []
            var emitted = 0
            var lastDecoded = 0
            var firstQueue: Double?
            var reason: String?
            var deliveredTokens = 0
            let config = LFMGenerationConfig(maxNewTokens: 180, temperature: 0.7, topK: 50,
                                             topP: 1, audioTemperature: 0.8, audioTopK: 4)

            func emit(final: Bool) {
                guard token.active, !Task.isCancelled, !codes.isEmpty else { return }
                let input = MLX.stacked(codes, axis: 0).transposed(1, 0).expandedDimensions(axis: 0)
                let waveform = detokenizer(input).asType(.float32).asArray(Float.self)
                lastDecoded = codes.count
                let end = final ? waveform.count : max(0, waveform.count - tailGuard)
                guard end > emitted else { return }
                let pcm = Array(waveform[emitted..<end])
                guard pcm.allSatisfy({ $0.isFinite }) else { reason = "Invalid decoder output"; return }
                let safe = pcm.map { max(Float(-1), min(Float(1), $0)) }
                emitted = end
                let first = firstQueue == nil
                if first { firstQueue = CFAbsoluteTimeGetCurrent() - started }
                DispatchQueue.main.async { [weak self] in
                    MainActor.assumeIsolated {
                        guard let self, self.session == ticket, self.responseRevision == revision, token.active else { return }
                        sharedAudio.scheduleTTSBuffer(safe)
                        if first { self.speakingCallback?(true) }
                    }
                }
            }

            model.generateInterleavedInline(
                textTokens: chat.getTextTokens(), audioFeatures: chat.getAudioFeatures(),
                modalities: chat.getModalities(), config: config,
                onToken: { value, modality in
                    guard token.active, !Task.isCancelled else { return false }
                    deliveredTokens += 1
                    if CFAbsoluteTimeGetCurrent() - started > 20 { reason = "Generation time limit (20 seconds)"; return false }
                    eval(value)
                    if modality == .text {
                        text.append(value.item(Int.self))
                        if text.count >= 80 { reason = "Text length limit (80 tokens)"; return false }
                    } else if modality == .audioOut {
                        let frame = value.asArray(Int32.self)
                        guard frame.count == 8 else { reason = "Unexpected audio frame shape"; return false }
                        if frame[0] == Int32(lfmAudioEOSToken) { return true }
                        guard frame.allSatisfy({ $0 >= 0 && $0 < Int32(lfmAudioEOSToken) }) else { reason = "Invalid audio codes"; return false }
                        recent.append(frame)
                        if recent.count > 20 { recent.removeFirst() }
                        if recent.count >= 10, recent.suffix(10).allSatisfy({ $0 == frame }) { reason = "Repeated identical audio frames"; return false }
                        if recent.count == 20, Array(recent.prefix(10)) == Array(recent.suffix(10)) { reason = "Repeated audio-code sequence"; return false }
                        codes.append(value)
                        if codes.count - lastDecoded >= (lastDecoded == 0 ? 6 : 12) { emit(final: false) }
                        if reason != nil { return false }
                        if codes.count >= 100 { reason = "Audio length limit (100 frames)"; return false }
                    }
                    return true
                }
            )
            guard token.active, !Task.isCancelled else { return }
            if reason == nil, deliveredTokens >= 180 { reason = "Generation token budget (180 tokens)" }
            if reason == nil { emit(final: true) }
            let summary: String
            if let reason { summary = "Safety cutoff: " + reason + "." }
            else if emitted == 0 { summary = "Model generation ended without playable audio." }
            else {
                summary = String(format: "Model generation ended; %.1f s, first chunk queued %.1f s, %d audio frames.",
                                 CFAbsoluteTimeGetCurrent() - started, firstQueue ?? -1, codes.count)
            }
            await withCheckedContinuation { (c: CheckedContinuation<Void, Never>) in DispatchQueue.main.async { c.resume() } }
            guard token.active, !Task.isCancelled else { return }
            if reason != nil {
                await self?.finishResponse(token, revision: revision, session: ticket, forced: true, message: summary)
                return
            }
            await sharedAudio.waitForPlaybackCompletion()
            guard token.active, !Task.isCancelled else { return }
            try? await Task.sleep(for: .milliseconds(350))
            await self?.finishResponse(token, revision: revision, session: ticket, forced: false, message: summary)
        }
    }

    private func finishResponse(_ token: SpeechRun, revision: Int, session ticket: Int, forced: Bool, message: String) {
        guard session == ticket, responseRevision == revision, token.active else { return }
        token.cancel(); watchdog?.cancel(); watchdog = nil
        if forced { responseTask?.cancel(); sharedAudioRef?.stopPlayback() }
        speakingCallback?(false); gate.set(false); Self.setStage("listening")
        recordStop(message)
    }
}
