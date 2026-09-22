import Foundation

/// One-at-a-time gate for llama.cpp Metal and CoreML TTS.
/// Playback must not take this lock — only `llama_decode` and synthesizers.
actor GPUExclusive {
    func run<T>(_ body: () async throws -> T) async throws -> T {
        try await body()
    }

    func run<T>(_ body: () async -> T) async -> T {
        await body()
    }
}
