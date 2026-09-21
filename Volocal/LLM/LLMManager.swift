import Foundation
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "llm")

/// Manages LLM inference using llama.cpp via the LlamaContext actor.
/// Prompt formatting comes from the loaded GGUF, not a hardcoded model family.
@MainActor
final class LLMManager: ObservableObject {
    @Published var response: String = ""
    @Published var isGenerating: Bool = false
    @Published var error: String?
    @Published var tokensPerSecond: Double = 0
    @Published var loadedModelName: String?

    private var llamaContext: LlamaContext?
    private var generationTask: Task<Void, Never>?

    private let systemPrompt = """
    You are Volocal, a helpful voice assistant running entirely on-device. \
    Keep responses concise and conversational — typically 1-3 sentences. \
    You're speaking out loud, so avoid markdown, code blocks, or lists. \
    Be friendly, direct, and natural.
    """

    init() {}

    func loadModel(path: String, displayName: String? = nil) async throws {
        unload()
        llamaContext = try LlamaContext.create(path: path, contextSize: 4096)
        loadedModelName = displayName ?? URL(fileURLWithPath: path).lastPathComponent
    }

    func unload() {
        stopGeneration()
        llamaContext = nil
        loadedModelName = nil
    }

    /// Generate response from conversation history.
    /// History should already contain the latest user message.
    func generate(history: [ConversationMessage] = []) -> AsyncStream<String> {
        generationTask?.cancel()
        generationTask = nil

        return AsyncStream { continuation in
            generationTask = Task {
                guard let ctx = llamaContext else {
                    continuation.finish()
                    return
                }

                await MainActor.run {
                    self.isGenerating = true
                    self.response = ""
                    self.tokensPerSecond = 0
                }

                let turns: [(role: String, content: String)] = history.map {
                    ($0.role == .user ? "user" : "assistant", $0.text)
                }

                let startTime = CFAbsoluteTimeGetCurrent()
                var tokenCount = 0
                var thinkFilter = ThinkTagFilter()

                do {
                    let fullPrompt = try await ctx.formatChat(system: systemPrompt, history: turns)
                    await ctx.clear()
                    try await ctx.completionInit(text: fullPrompt)

                    while !Task.isCancelled {
                        guard let token = await ctx.completionLoop() else { break }

                        tokenCount += 1
                        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                        let tps = elapsed > 0 ? Double(tokenCount) / elapsed : 0

                        let spoken = thinkFilter.push(token)
                        guard !spoken.isEmpty else {
                            await MainActor.run { self.tokensPerSecond = tps }
                            continue
                        }

                        continuation.yield(spoken)

                        await MainActor.run {
                            self.response += spoken
                            self.tokensPerSecond = tps
                        }
                    }

                    let tail = thinkFilter.flush()
                    if !tail.isEmpty {
                        continuation.yield(tail)
                        await MainActor.run { self.response += tail }
                    }
                } catch {
                    await MainActor.run {
                        self.error = error.localizedDescription
                    }
                }

                await MainActor.run {
                    self.isGenerating = false
                }
                continuation.finish()
            }
        }
    }

    func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
    }

    var isModelLoaded: Bool {
        llamaContext != nil
    }
}
