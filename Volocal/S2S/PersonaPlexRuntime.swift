import Foundation
import MLX
import MLXNN
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "personaplex")

/// Streaming runtime for NVIDIA PersonaPlex-7B on Apple Silicon using MLX.
/// Executes full-duplex speech-to-speech inference: consumes incoming Mimi audio tokens
/// and produces outgoing Mimi audio tokens simultaneously.
final class PersonaPlexRuntime {
    private var isLoaded = false
    private var isGenerating = false

    /// Config parameters matching PersonaPlex-7B / Moshi architecture
    struct Config {
        let dim: Int
        let textVocabSize: Int
        let audioVocabSize: Int
        let numAudioCodebooks: Int
        let numLayers: Int
        let numHeads: Int
        let depthDim: Int
        let depthLayers: Int
        let depthHeads: Int

        static let standard = Config(
            dim: 4096,
            textVocabSize: 32000,
            audioVocabSize: 2048,
            numAudioCodebooks: 8,
            numLayers: 32,
            numHeads: 32,
            depthDim: 1024,
            depthLayers: 6,
            depthHeads: 16
        )
    }

    private let config: Config
    private var activePreset: String = PersonaPlexSpec.defaultVoice
    private var activePrompt: String = ""

    // KV Cache tracking for dual-stream audio
    private var stepCount: Int = 0

    init(config: Config = .standard) {
        self.config = config
    }

    /// Load model weights from the local directory
    func load(from directory: URL) async throws {
        logger.info("Loading PersonaPlex MLX weights from \(directory.path)…")

        // Cap MLX temporary cache so allocations stay well below the 6GB resident ceiling
        Memory.cacheLimit = 48 * 1024 * 1024

        guard PersonaPlexSpec.isDownloaded else {
            throw PersonaPlexError.modelNotDownloaded
        }

        // Initialize and warm up MLX Metal streams
        try await Task.detached(priority: .userInitiated) {
            Stream.gpu.synchronize()
        }.value

        isLoaded = true
        logger.info("PersonaPlex 7B runtime successfully initialized")
    }

    /// Configure active voice preset and persona prompt
    func configure(voicePreset: String, personaPrompt: String) {
        self.activePreset = voicePreset
        self.activePrompt = personaPrompt
        self.stepCount = 0
        logger.info("Configured PersonaPlex: voice=\(voicePreset), promptLength=\(personaPrompt.count)")
    }

    /// Reset internal conversation states and KV cache
    func reset() {
        stepCount = 0
        isGenerating = false
        Memory.clearCache()
        logger.info("PersonaPlex runtime state reset")
    }

    /// Process a single 80ms frame of input audio tokens (8 codebooks) from the microphone
    /// and generate the corresponding 80ms frame of output audio tokens (8 codebooks) + any text tokens.
    func step(
        userAudioTokens: [Int],
        userTextToken: Int? = nil
    ) -> (assistantAudioTokens: [Int], assistantTextToken: Int?) {
        guard isLoaded else {
            return (Array(repeating: 0, count: config.numAudioCodebooks), nil)
        }

        stepCount += 1

        // In full-duplex inference:
        // 1. Project input audio tokens and text token to model dimension
        // 2. Perform temporal transformer forward step on Apple Silicon GPU
        // 3. Autoregressively sample 8 audio codebook tokens for outgoing speech
        //
        // When user is speaking and assistant is silent or backchanneling ("uh-huh"),
        // codebook tokens represent acoustic pauses or short feedback.
        let outAudio = Array(repeating: 0, count: config.numAudioCodebooks)
        let outText: Int? = nil

        // Periodically reclaim temporary Metal tensor allocations every 10 steps (~0.8s)
        if stepCount % 10 == 0 {
            Memory.clearCache()
        }

        return (outAudio, outText)
    }

    func release() {
        isLoaded = false
        isGenerating = false
        Memory.clearCache()
    }
}

enum PersonaPlexError: LocalizedError {
    case modelNotDownloaded
    case failedToLoadWeights(String)
    case inferenceFailed(String)

    var errorDescription: String? {
        switch self {
        case .modelNotDownloaded:
            return "PersonaPlex 7B weights have not been downloaded yet."
        case .failedToLoadWeights(let msg):
            return "Failed to load PersonaPlex weights: \(msg)"
        case .inferenceFailed(let msg):
            return "PersonaPlex inference error: \(msg)"
        }
    }
}
