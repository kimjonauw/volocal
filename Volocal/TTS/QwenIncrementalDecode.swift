import Foundation
@preconcurrency import MLX
import MLXAudioTTS
import MLXAudioCore

/// Caller must retain GPUExclusive until this method returns.
enum QwenIncrementalDecode {
    static func samples(model: Qwen3TTSModel, text: String, voice: String) async throws -> [Float] {
        try Task.checkCancellation()
        // Keep the consumer uncancelled: stream cancellation can return before
        // its producer exits. Drain the bounded request before releasing the GPU.
        let worker = Task.detached(priority: .userInitiated) { () throws -> [Float] in
            GenerationTrace.record("qwen.incremental.begin")
            defer { GenerationTrace.record("qwen.incremental.worker.exit") }
            var result: [Float] = []
            var tokens = 0
            let stream = model.generateStream(
                text: text, voice: voice, refAudio: nil, refText: nil,
                language: "English",
                generationParameters: GenerateParameters(maxTokens: 192, temperature: 0.8, topP: 1, repetitionPenalty: 1.1),
                streamingInterval: 0.5
            )
            for try await event in stream {
                switch event {
                case .audio(let audio):
                    GenerationTrace.record("qwen.pcm.convert", count: result.count)
                    result.append(contentsOf: audio.asType(.float32).asArray(Float.self))
                    GenerationTrace.record("qwen.pcm.collected", count: result.count)
                case .token:
                    tokens += 1
                    if tokens == 1 || tokens % 32 == 0 {
                        GenerationTrace.record("qwen.tokens", count: tokens)
                    }
                default:
                    break
                }
            }
            GenerationTrace.record("qwen.stream.finished", count: result.count)
            return result
        }
        let result = try await worker.value
        try Task.checkCancellation()
        return result
    }
}
