import Foundation
import AVFoundation
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "audio")

/// Shared state between MainActor and the real-time audio thread.
/// @unchecked Sendable because Bool load/store is atomic on ARM64
/// and AsyncStream.Continuation.yield() is thread-safe.
final class AudioBridge: @unchecked Sendable {
    /// Speaking gate — true while TTS is playing. Audio thread reads, MainActor writes.
    var isSpeaking: Bool = false

    /// STT buffer continuation — set by STTManager on MainActor, yielded from audio thread.
    var inputContinuation: AsyncStream<AVAudioPCMBuffer>.Continuation?
}

/// Single shared AVAudioEngine for both STT input and TTS output.
/// VP (Voice Processing) is enabled when input capture starts, providing hardware AEC.
@MainActor
final class SharedAudioEngine: ObservableObject {
    @Published var isRunning: Bool = false
    @Published var error: String?

    private var engine: AVAudioEngine?
    private var playerNode: AVAudioPlayerNode?
    private var queuedBuffers = 0
    private var pendingChunks: [[Float]] = []
    private var playbackEpoch = 0
    /// In-flight `scheduleBuffer` calls. Extra PCM sits in `pendingChunks`
    /// so TTS never waits on the speaker — the next clause can synthesize
    /// while this one plays.
    private static let maxInFlightBuffers = 8
    /// True while STT wants buffers. The hardware tap stays installed for the session.
    private var inputCaptureActive = false
    private var micTapInstalled = false

    /// Thread-safe bridge between MainActor and the real-time audio thread.
    let bridge = AudioBridge()

    let ttsFormat: AVAudioFormat

    init() {
        ttsFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: 24000, // PocketTTS sample rate
            channels: 1,
            interleaved: false
        )!
    }

    /// Whether TTS is currently playing or still has PCM to pump.
    var isSpeaking: Bool { bridge.isSpeaking }

    // MARK: - Lifecycle

    /// Start the audio engine for TTS playback. Call once during app init.
    func start() {
        guard engine == nil else { return }

        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playAndRecord, mode: .voiceChat, options: [.defaultToSpeaker])
            try session.setActive(true)

            let eng = AVAudioEngine()
            // Touch input now so the graph includes the mic before the first start().
            _ = eng.inputNode

            let player = AVAudioPlayerNode()
            eng.attach(player)
            eng.connect(player, to: eng.mainMixerNode, format: ttsFormat)

            eng.prepare()
            try eng.start()
            player.play()

            self.engine = eng
            self.playerNode = player
            self.isRunning = true

            logger.info("SharedAudioEngine started (TTS ready)")
        } catch {
            self.error = "Audio engine start failed: \(error.localizedDescription)"
            logger.error("SharedAudioEngine start failed: \(error.localizedDescription)")
        }
    }

    func stop() {
        endInputCapture()
        if micTapInstalled, let eng = engine {
            var nsError: NSError?
            _ = VolocalCatchException({
                eng.inputNode.removeTap(onBus: 0)
            }, &nsError)
            micTapInstalled = false
        }
        playerNode?.stop()
        engine?.stop()
        engine = nil
        playerNode = nil
        isRunning = false
        queuedBuffers = 0
        bridge.isSpeaking = false
        bridge.inputContinuation = nil
        logger.info("SharedAudioEngine stopped")
    }

    // MARK: - STT (Input Capture)

    /// Start capturing mic input with Voice Processing AEC.
    /// The tap is installed once per engine lifetime — reinstalling crashes with
    /// `required condition is false: nullptr == Tap()` on LiveContainer / iOS 26.
    func beginInputCapture() async {
        error = nil
        let granted = await AVAudioApplication.requestRecordPermission()
        guard granted else {
            error = "Microphone permission denied"
            logger.error("Microphone permission denied")
            return
        }

        guard let eng = engine else {
            error = "Audio engine is not running"
            return
        }

        if micTapInstalled {
            inputCaptureActive = true
            return
        }

        do {
            try installMicTap(on: eng)
            inputCaptureActive = true
            logger.info("Input capture started with VP AEC")
        } catch {
            logger.error("Failed to start input capture: \(error.localizedDescription)")
            self.error = "Mic capture failed: \(error.localizedDescription)"
            restartEngineForTTS(eng)
        }
    }

    /// Stop capturing mic input. Engine and tap keep running for TTS / next listen.
    func endInputCapture() {
        inputCaptureActive = false
        bridge.inputContinuation = nil
        logger.info("Input capture paused")
    }

    // MARK: - TTS (Playback)

    /// Copy samples into the player. Never waits for playback — leftover PCM
    /// sits in `pendingChunks` so CoreML can start the next clause immediately.
    func scheduleTTSBuffer(_ samples: [Float]) {
        if Task.isCancelled || samples.isEmpty { return }
        pendingChunks.append(samples)
        bridge.isSpeaking = true
        pumpPending()
    }

    private func pumpPending() {
        let epoch = playbackEpoch
        guard let node = playerNode else { return }
        while queuedBuffers < Self.maxInFlightBuffers, !pendingChunks.isEmpty {
            if epoch != playbackEpoch { return }
            let samples = pendingChunks.removeFirst()
            guard !samples.isEmpty else { continue }

            guard let buffer = AVAudioPCMBuffer(
                pcmFormat: ttsFormat,
                frameCapacity: AVAudioFrameCount(samples.count)
            ) else {
                continue
            }
            buffer.frameLength = AVAudioFrameCount(samples.count)
            if let channelData = buffer.floatChannelData {
                samples.withUnsafeBufferPointer { src in
                    channelData[0].update(from: src.baseAddress!, count: samples.count)
                }
            }

            queuedBuffers += 1
            bridge.isSpeaking = true

            var nsError: NSError?
            let scheduled = VolocalCatchException({
                node.scheduleBuffer(buffer, completionCallbackType: .dataConsumed) { [weak self] _ in
                    Task { @MainActor in
                        guard let self, epoch == self.playbackEpoch else { return }
                        self.queuedBuffers = max(self.queuedBuffers - 1, 0)
                        self.pumpPending()
                        if self.queuedBuffers == 0 && self.pendingChunks.isEmpty {
                            self.bridge.isSpeaking = false
                        }
                    }
                }
            }, &nsError)
            if !scheduled {
                queuedBuffers = max(queuedBuffers - 1, 0)
                logger.error("scheduleTTSBuffer exception: \(nsError?.localizedDescription ?? "unknown")")
            }
        }
        if queuedBuffers == 0 && pendingChunks.isEmpty {
            bridge.isSpeaking = false
        }
    }

    /// Stop all TTS playback immediately.
    func stopPlayback() {
        playbackEpoch += 1
        queuedBuffers = 0
        pendingChunks.removeAll()
        bridge.isSpeaking = false
        guard let node = playerNode else { return }
        var nsError: NSError?
        _ = VolocalCatchException({
            node.stop()
            if node.engine?.isRunning == true {
                node.play()
            }
        }, &nsError)
        if let nsError {
            logger.error("stopPlayback exception: \(nsError.localizedDescription)")
        }
    }

    /// Wait for all queued TTS buffers to finish playing.
    func waitForPlaybackCompletion() async {
        let epoch = playbackEpoch
        while queuedBuffers > 0 || !pendingChunks.isEmpty {
            if Task.isCancelled || epoch != playbackEpoch { break }
            try? await Task.sleep(for: .milliseconds(50))
        }
    }

    nonisolated static func rmsEnergy(_ buffer: AVAudioPCMBuffer) -> Float {
        guard let data = buffer.floatChannelData?[0] else { return 0 }
        let n = Int(buffer.frameLength)
        guard n > 0 else { return 0 }
        var sum: Float = 0
        for i in 0..<n {
            let x = data[i]
            sum += x * x
        }
        return sqrtf(sum / Float(n))
    }

    // MARK: - Mic tap

    private func installMicTap(on eng: AVAudioEngine) throws {
        playerNode?.stop()
        if eng.isRunning {
            eng.stop()
        }

        let inputNode = eng.inputNode
        stripTap(inputNode)

        var voiceProcessingOn = false
        do {
            try inputNode.setVoiceProcessingEnabled(true)
            try eng.outputNode.setVoiceProcessingEnabled(true)
            voiceProcessingOn = true
            // VP I/O can install its own tap on LiveContainer / iOS 26.
            stripTap(inputNode)
            logger.info("Voice processing AEC enabled")
        } catch {
            logger.warning("Voice processing not available: \(error.localizedDescription)")
        }

        let format = inputNode.outputFormat(forBus: 0)
        let tapFormat: AVAudioFormat? = format.sampleRate >= 8000 ? format : nil
        let audioBridge = self.bridge

        if !installTap(on: inputNode, format: tapFormat, bridge: audioBridge) {
            if voiceProcessingOn {
                try? inputNode.setVoiceProcessingEnabled(false)
                try? eng.outputNode.setVoiceProcessingEnabled(false)
                stripTap(inputNode)
                logger.warning("Retrying mic tap without voice processing")
            }
            if !installTap(on: inputNode, format: tapFormat, bridge: audioBridge) {
                throw NSError(
                    domain: "com.localiosllm.volocal.audio",
                    code: 1,
                    userInfo: [NSLocalizedDescriptionKey: "Could not attach microphone (tap already present)."]
                )
            }
        }

        micTapInstalled = true
        eng.prepare()
        try eng.start()
        playerNode?.play()
    }

    private func stripTap(_ node: AVAudioInputNode) {
        var nsError: NSError?
        _ = VolocalCatchException({
            node.removeTap(onBus: 0)
        }, &nsError)
    }

    private func installTap(
        on node: AVAudioInputNode,
        format: AVAudioFormat?,
        bridge audioBridge: AudioBridge
    ) -> Bool {
        var nsError: NSError?
        let ok = VolocalCatchException({
            node.installTap(onBus: 0, bufferSize: 4096, format: format) { buffer, _ in
                SharedAudioEngine.forwardMicBuffer(buffer, bridge: audioBridge)
            }
        }, &nsError)
        if let nsError {
            logger.error("installTap: \(nsError.localizedDescription)")
        }
        return ok
    }

    private func restartEngineForTTS(_ eng: AVAudioEngine) {
        if eng.isRunning { return }
        do {
            eng.prepare()
            try eng.start()
            playerNode?.play()
        } catch {
            logger.error("TTS engine restart failed: \(error.localizedDescription)")
        }
    }

    nonisolated private static func forwardMicBuffer(_ buffer: AVAudioPCMBuffer, bridge: AudioBridge) {
        guard bridge.inputContinuation != nil else { return }

        // RMS on the tap buffer first so silence during TTS does not allocate.
        let rms = rmsEnergy(buffer)
        if bridge.isSpeaking, rms < 0.012 { return }

        guard let copy = AVAudioPCMBuffer(
            pcmFormat: buffer.format,
            frameCapacity: buffer.frameLength
        ) else { return }
        copy.frameLength = buffer.frameLength
        if let src = buffer.floatChannelData, let dst = copy.floatChannelData {
            for ch in 0..<Int(buffer.format.channelCount) {
                dst[ch].update(from: src[ch], count: Int(buffer.frameLength))
            }
        }

        bridge.inputContinuation?.yield(copy)
    }
}
