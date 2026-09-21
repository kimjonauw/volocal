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
    var contextSize: UInt32 = LLMContextWindow.default

    init() {}

    func loadModel(path: String, displayName: String? = nil) async throws {
        unload()
        await Task.yield()
        let ctxSize = contextSize
        llamaContext = try await Task.detached(priority: .userInitiated) {
            try LlamaContext.create(path: path, contextSize: ctxSize)
        }.value
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

/// llama.cpp `n_ctx`. KV cache grows with this; the GGUF's trained window is the other ceiling.
enum LLMContextWindow {
    static let min: UInt32 = 2048
    static let max: UInt32 = 131_072
    static let step: UInt32 = 512
    static let `default`: UInt32 = 2048

    static var logRange: ClosedRange<Double> {
        log2(Double(min))...log2(Double(max))
    }

    static func clamp(_ size: UInt32) -> UInt32 {
        let bounded = Swift.min(max, Swift.max(min, size))
        let stepped = ((bounded + step / 2) / step) * step
        return Swift.min(max, Swift.max(min, stepped))
    }

    static func fromLog(_ logValue: Double) -> UInt32 {
        let raw = pow(2.0, logValue)
        guard raw.isFinite, raw > 0 else { return `default` }
        return clamp(UInt32(raw.rounded()))
    }

    static func logValue(_ size: UInt32) -> Double {
        log2(Double(clamp(size)))
    }

    /// User+assistant messages kept in the voice transcript (system prompt is separate).
    static func historyEntries(for size: UInt32) -> Int {
        min(80, max(8, Int(clamp(size) / 256)))
    }

    static func label(_ size: UInt32) -> String {
        let n = clamp(size)
        let tokens = formatted(n)
        if n <= min { return "\(tokens) · light" }
        if n >= 65_536 { return "\(tokens) · more RAM" }
        return "\(tokens) tokens"
    }

    static func formatted(_ size: UInt32) -> String {
        let formatter = NumberFormatter()
        formatter.numberStyle = .decimal
        return formatter.string(from: NSNumber(value: clamp(size))) ?? "\(clamp(size))"
    }
}

/// Survives a LiveContainer/jetsam kill during `llama_model_load_from_file` so relaunch does not crash-loop.
enum LLMLoadFence {
    private static let key = "volocal.pendingLLMLoad"
    private static var retryAllowed = false

    private struct Record: Codable {
        var path: String
        var contextSize: UInt32
    }

    static func shouldSkipLoad(path: String) -> Bool {
        guard let rec = read() else { return false }
        if retryAllowed { return false }
        return rec.path == path
    }

    static func allowRetry() {
        retryAllowed = true
    }

    static func markStarting(path: String, contextSize: UInt32) {
        retryAllowed = false
        let rec = Record(path: path, contextSize: contextSize)
        if let data = try? JSONEncoder().encode(rec) {
            UserDefaults.standard.set(data, forKey: key)
            UserDefaults.standard.synchronize()
        }
    }

    static func clear() {
        retryAllowed = false
        UserDefaults.standard.removeObject(forKey: key)
        UserDefaults.standard.synchronize()
    }

    private static func read() -> Record? {
        guard let data = UserDefaults.standard.data(forKey: key) else { return nil }
        return try? JSONDecoder().decode(Record.self, from: data)
    }
}
