import Foundation
import AVFoundation
import os
import PersonaPlex
import AudioCommon

private let logger = Logger(subsystem: "com.volocal.app", category: "s2s")

enum SpeechToSpeechError: LocalizedError {
    case notEnoughMemory(availableMB: Int, neededMB: Int)
    case notLoaded
    case micFailed(String)
    case previousCrash

    var errorDescription: String? {
        switch self {
        case .notEnoughMemory(let available, let needed):
            return "Only \(available) MB is free for Volocal and PersonaPlex needs about \(needed) MB. Turn on GetMoreRAM for the LiveContainer host, then force-quit and reopen Volocal so the speech, language and voice models release their memory."
        case .notLoaded:
            return "PersonaPlex is not loaded yet."
        case .micFailed(let message):
            return "Microphone could not start: \(message)"
        case .previousCrash:
            return "The last speech-to-speech load was killed by iOS, most likely because it ran out of memory. Make sure GetMoreRAM is on for the LiveContainer host, then tap Try again."
        }
    }
}

/// Real speech-to-speech: NVIDIA PersonaPlex 7B (4-bit) through the `PersonaPlex` library of
/// soniqo/speech-swift. It listens and speaks at the same time (`respondRealtime`), one audio
/// frame per 80 ms step. It replaces the recognizer, the language model and the TTS voice.
///
/// Mic audio comes from `SharedAudioEngine` (Voice Processing echo cancellation), is converted to
/// 24 kHz mono, and goes into a ring buffer the model reads every step. Agent audio comes back as
/// 24 kHz frames and plays through the same engine.
@MainActor
final class SpeechToSpeechEngine: ObservableObject {
    /// Same crash fence the language model uses. A jetsam kill during load must not crash-loop.
    static let fenceKey = "speech-to-speech:personaplex-7b-4bit"
    /// The 4-bit weights alone are ~4.9 to 5.5 GB.
    private static let neededBytes = 5_000_000_000

    /// Pre-tokenized SentencePiece ids (the library takes ids, not text):
    /// "You are a helpful assistant. Answer questions clearly and concisely. Listen carefully to
    /// what the user says, then respond directly to their question or request. Stay on topic.
    /// Be concise." Taken from the speech-swift PersonaPlexDemo.
    private static let systemPromptTokens: [Int32] = [
        607, 4831, 578, 493, 298, 272, 3850, 5019, 263,
        506, 1292, 2366, 267, 22876, 362, 263,
        17453, 6716, 269, 419, 262, 819, 1182, 261, 409,
        4816, 1312, 269, 347, 560, 307, 2498, 263, 17308,
        291, 3398, 263, 1451, 22876, 263, 607, 4831, 578,
    ]

    @Published private(set) var isLoaded = false

    private var model: PersonaPlexModel?
    private var inferenceTask: Task<Void, Never>?
    private var micTask: Task<Void, Never>?
    private var micContinuation: AsyncStream<AVAudioPCMBuffer>.Continuation?
    private var session = 0

    // MARK: - Loading

    func load(progress: @escaping @Sendable (String) -> Void) async throws {
        if model != nil { return }

        if LLMLoadFence.shouldSkipLoad(path: Self.fenceKey) {
            throw SpeechToSpeechError.previousCrash
        }

        let available = Int(os_proc_available_memory())
        if available > 0, available < Self.neededBytes {
            throw SpeechToSpeechError.notEnoughMemory(
                availableMB: available / 1_000_000,
                neededMB: Self.neededBytes / 1_000_000
            )
        }

        LLMLoadFence.markStarting(path: Self.fenceKey, contextSize: 0)
        do {
            let loaded = try await PersonaPlexModel.fromPretrained(
                modelId: PersonaPlexModel.defaultModelId,
                offlineMode: false
            ) { fraction, status in
                progress("\(status) (\(Int(fraction * 100))%)")
            }
            progress("Warming up (compiling Metal kernels)...")
            await Task.detached(priority: .userInitiated) { loaded.warmUp() }.value
            model = loaded
            isLoaded = true
            LLMLoadFence.clear()
            logger.info("PersonaPlex loaded")
        } catch {
            LLMLoadFence.clear()
            throw error
        }
    }

    func unload() {
        inferenceTask?.cancel()
        inferenceTask = nil
        model = nil
        isLoaded = false
    }

    // MARK: - Conversation

    /// Starts the full-duplex loop. Callbacks may be called from any thread.
    func start(
        sharedAudio: SharedAudioEngine,
        voiceId: String,
        onAgentSpeaking: @escaping @Sendable (Bool) -> Void,
        onStepTime: @escaping @Sendable (Double) -> Void,
        onFinished: @escaping @Sendable (String?) -> Void
    ) async throws {
        guard let model else { throw SpeechToSpeechError.notLoaded }

        // The model is not thread-safe. Let any previous loop finish before starting another.
        micTask?.cancel()
        inferenceTask?.cancel()
        micContinuation?.finish()
        let previous = inferenceTask
        await previous?.value

        session += 1
        let currentSession = session
        let voice = PersonaPlexVoice(rawValue: voiceId) ?? .NATM0
        let promptTokens = Self.systemPromptTokens
        let ring = AudioRingBuffer(capacity: 24_000 * 5)

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

        // Mic buffers (device rate, from Voice Processing) -> 24 kHz mono -> ring buffer.
        micTask = Task.detached(priority: .userInitiated) {
            guard let target = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: 24_000,
                channels: 1,
                interleaved: false
            ) else { return }
            var converter: AVAudioConverter?
            var converterInput: AVAudioFormat?
            for await buffer in stream {
                if Task.isCancelled { break }
                if converter == nil || converterInput != buffer.format {
                    converter = AVAudioConverter(from: buffer.format, to: target)
                    converterInput = buffer.format
                }
                guard let converter else { continue }
                let ratio = 24_000 / buffer.format.sampleRate
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
                if let data = out.floatChannelData?[0], out.frameLength > 0 {
                    ring.write(from: UnsafePointer(data), count: Int(out.frameLength))
                }
            }
        }

        // All model work stays on this one task (MLX is not thread-safe).
        inferenceTask = Task.detached(priority: .userInitiated) { [weak self] in
            let frames = model.respondRealtime(
                voice: voice,
                systemPromptTokens: promptTokens,
                userAudioBuffer: ring,
                verbose: false
            )

            let prebufferCount = 3
            var prebuffer: [[Float]] = []
            var started = false
            var lastFrameAt = CFAbsoluteTimeGetCurrent()
            var intervals: [Double] = []
            var quietRun = 0
            var agentSpeaking = false
            var failure: String?

            do {
                for try await frame in frames {
                    if Task.isCancelled { break }

                    let now = CFAbsoluteTimeGetCurrent()
                    intervals.append(now - lastFrameAt)
                    lastFrameAt = now
                    if intervals.count >= 25 {
                        let average = intervals.reduce(0, +) / Double(intervals.count) * 1000
                        intervals.removeAll(keepingCapacity: true)
                        onStepTime(average)
                    }

                    var sum: Float = 0
                    for sample in frame { sum += sample * sample }
                    let rms = sqrtf(sum / Float(max(frame.count, 1)))
                    if rms > 0.01 {
                        quietRun = 0
                        if !agentSpeaking {
                            agentSpeaking = true
                            onAgentSpeaking(true)
                        }
                    } else {
                        quietRun += 1
                        if quietRun >= 6, agentSpeaking {
                            agentSpeaking = false
                            onAgentSpeaking(false)
                        }
                    }

                    if !started {
                        prebuffer.append(frame)
                        if prebuffer.count >= prebufferCount {
                            started = true
                            let batch = prebuffer
                            prebuffer = []
                            await MainActor.run {
                                for item in batch { sharedAudio.scheduleTTSBuffer(item) }
                                // The model must hear the mic even while it talks. The quiet-mic
                                // gate used for the TTS pipeline would deafen it.
                                sharedAudio.bridge.isSpeaking = false
                            }
                        }
                    } else {
                        await MainActor.run {
                            sharedAudio.scheduleTTSBuffer(frame)
                            sharedAudio.bridge.isSpeaking = false
                        }
                    }
                }
            } catch {
                failure = "Speech-to-speech stopped: \(error.localizedDescription)"
            }

            let wasCancelled = Task.isCancelled
            await MainActor.run {
                guard let self, self.session == currentSession else { return }
                if !wasCancelled { onFinished(failure) }
            }
        }
    }

    func stop(sharedAudio: SharedAudioEngine) {
        session += 1
        inferenceTask?.cancel()
        micTask?.cancel()
        micTask = nil
        micContinuation?.finish()
        micContinuation = nil
        sharedAudio.endInputCapture()
        sharedAudio.stopPlayback()
    }
}
