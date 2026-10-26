import Foundation
@preconcurrency import MLX
import MLXAudioTTS
import MLXAudioCore

/// Incremental waveform decoding without abandoning the producer on cancellation.
/// The caller must retain GPUExclusive until this method returns.
enum QwenIncrementalDecode {
    static func samples(model: Qwen3TTSModel, text: String, voice: String) async throws -> [Float] {
        try Task.checkCancellation()
        // Do not cancel this consumer: AsyncThrowingStream cancellation can return
        // before its producer exits. Drain the bounded request before releasing GPUExclusive.
        let worker = Task.detached(priority: .userInitiated) { () throws -> [Float] in
            var result: [Float] = []
            let stream = model.generateStream(
                text: text, voice: voice, refAudio: nil, refText: nil,
                language: "English",
                generationParameters: GenerateParameters(maxTokens: 192, temperature: 0.8, topP: 1, repetitionPenalty: 1.1),
                streamingInterval: 0.5
            )
            for try await event in stream {
                if case .audio(let audio) = event {
                    result.append(contentsOf: audio.asType(.float32).asArray(Float.self))
                }
            }
            return result
        }
        let result = try await worker.value
        try Task.checkCancellation()
        return result
    }
}
