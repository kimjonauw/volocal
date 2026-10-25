import Foundation

/// One-at-a-time gate for llama.cpp Metal and CoreML TTS.
/// Playback must not take this lock — only `llama_decode` and synthesizers.
///
/// A plain actor method is re-entrant: while `body` is suspended at an `await`,
/// another caller can enter. This is a real FIFO mutex, so a second caller waits
/// until the first body has fully finished.
actor GPUExclusive {
    private var locked = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    private func acquire() async {
        if !locked {
            locked = true
            return
        }
        await withCheckedContinuation { continuation in
            waiters.append(continuation)
        }
    }

    private func release() {
        if waiters.isEmpty {
            locked = false
        } else {
            // Ownership passes straight to the next waiter; `locked` stays true.
            waiters.removeFirst().resume()
        }
    }

    func run<T>(_ body: () async throws -> T) async throws -> T {
        await acquire()
        defer { release() }
        return try await body()
    }

    func run<T>(_ body: () async -> T) async -> T {
        await acquire()
        defer { release() }
        return await body()
    }
}
