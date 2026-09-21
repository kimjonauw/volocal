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
    @Published var generatePhase: GeneratePhase = .idle
    @Published var hiddenTokenCount: Int = 0
    @Published var spokenCharCount: Int = 0

    enum GeneratePhase: Equatable {
        case idle
        case readingPrompt
        case hiddenReasoning
        case writingSpeech
    }

    private var llamaContext: LlamaContext?
    private var generationTask: Task<Void, Never>?

    static let defaultInstructions = """
    You are Volocal, a helpful voice assistant running entirely on-device. \
    Keep replies to 1-2 short spoken sentences. No markdown, lists, or inner monologue. \
    Answer immediately.
    """

    var systemPrompt: String = LLMManager.defaultInstructions
    var contextSize: UInt32 = 2048

    init() {}

    func loadModel(path: String, displayName: String? = nil) async throws {
        unload()
        await Task.yield()
        llamaContext = try LlamaContext.create(path: path, contextSize: contextSize)
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
                    await MainActor.run {
                        self.error = "No GGUF loaded."
                    }
                    continuation.finish()
                    return
                }

                await MainActor.run {
                    self.isGenerating = true
                    self.response = ""
                    self.tokensPerSecond = 0
                    self.hiddenTokenCount = 0
                    self.spokenCharCount = 0
                    self.generatePhase = .readingPrompt
                }

                let turns: [(role: String, content: String)] = history.map {
                    ($0.role == .user ? "user" : "assistant", $0.text)
                }

                let startTime = CFAbsoluteTimeGetCurrent()
                var tokenCount = 0
                var hiddenTokens = 0
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
                        if spoken.isEmpty {
                            hiddenTokens += 1
                            await MainActor.run {
                                self.tokensPerSecond = tps
                                self.hiddenTokenCount = hiddenTokens
                                self.generatePhase = thinkFilter.isInsideThink ? .hiddenReasoning : .writingSpeech
                            }
                            continue
                        }

                        continuation.yield(spoken)

                        await MainActor.run {
                            self.response += spoken
                            self.tokensPerSecond = tps
                            self.spokenCharCount = self.response.count
                            self.generatePhase = .writingSpeech
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
                    self.generatePhase = .idle
                }
                continuation.finish()
            }
        }
    }

    func stopGeneration() {
        generationTask?.cancel()
        generationTask = nil
        isGenerating = false
        generatePhase = .idle
    }

    var isModelLoaded: Bool {
        llamaContext != nil
    }
}
