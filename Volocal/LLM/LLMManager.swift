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
    private var inferenceEpoch = 0
    /// llama.cpp Metal and CoreML TTS take turns on this lock.
    var gpuLock: GPUExclusive?
    /// Superonic/Pocket only hold this while CoreML is synthesizing.
    weak var inferenceHold: TTSManager?

    static let defaultInstructions = """
    You are Volocal, a helpful voice assistant running entirely on-device. \
    Keep replies to 1-2 short spoken sentences. No markdown, lists, or inner monologue. \
    Never recap the user, mention tone, or plan the reply. Answer with the spoken words only.
    """

    static let expressionCue = """
     When a reaction fits, insert [laugh], [chuckle], [sigh], [gasp], [cough], or [whisper]this part[/whisper] in the spoken line. Never speak the brackets. Use them rarely.
    """

    static func systemPrompt(from base: String, expressions: Bool) -> String {
        let trimmed = base.trimmingCharacters(in: .whitespacesAndNewlines)
        let core = trimmed.isEmpty ? defaultInstructions : base
        guard expressions else { return core }
        if core.contains("[laugh]") { return core }
        return core + expressionCue
    }

    static func promptWithMemory(_ system: String, memory: String) -> String {
        let mem = memory.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !mem.isEmpty else { return system }
        return system + "\n\nEarlier conversation (facts only; do not read this block aloud):\n" + mem
    }

    var systemPrompt: String = LLMManager.defaultInstructions
    var contextSize: UInt32 = LLMContextWindow.default
    /// Close Qwen `<think>` / Gemma 4 thought channels in the chat template.
    /// llama.cpp b10549 cannot pass `enable_thinking:false` as kwargs.
    var suppressThinking: Bool = true

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
    func generate(history: [ConversationMessage] = [], memory: String = "") -> AsyncStream<String> {
        let previous = generationTask
        previous?.cancel()
        inferenceEpoch += 1
        let epoch = inferenceEpoch
        let memorySnapshot = memory
        let promptSystem = systemPrompt
        let skipThink = suppressThinking
        let historySnapshot = history
        let gpu = gpuLock

        return AsyncStream { continuation in
            generationTask = Task.detached { [weak self] in
                _ = await previous?.value
                guard let self else {
                    continuation.finish()
                    return
                }
                let stillCurrent = await MainActor.run { self.inferenceEpoch == epoch }
                guard stillCurrent else {
                    continuation.finish()
                    return
                }
                let ctx = await MainActor.run { self.llamaContext }
                guard let ctx else {
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

                let turns: [(role: String, content: String)] = historySnapshot.map {
                    ($0.role == .user ? "user" : "assistant", $0.text)
                }

                let startTime = CFAbsoluteTimeGetCurrent()
                var tokenCount = 0
                var channelFilter = ThoughtChannelFilter(
                    hideUntilThinkClose: skipThink && ctx.hidesThinkUntilClose
                )
                var preambleFilter = ReasoningPreambleFilter()

                do {
                    let fullPrompt = try await ctx.formatChat(
                        system: Self.promptWithMemory(promptSystem, memory: memorySnapshot),
                        history: turns,
                        suppressThinking: skipThink
                    )
                    await ctx.clear()
                    if let gpu {
                        try await gpu.run { try await ctx.completionInit(text: fullPrompt) }
                    } else {
                        try await ctx.completionInit(text: fullPrompt)
                    }

                    while !Task.isCancelled {
                        while await MainActor.run(body: { self.inferenceHold?.holdsInferenceGPU == true }) {
                            try? await Task.sleep(for: .milliseconds(20))
                        }
                        let token: String?
                        if let gpu {
                            token = await gpu.run { await ctx.completionLoop() }
                        } else {
                            token = await ctx.completionLoop()
                        }
                        guard let token else { break }

                        tokenCount += 1
                        let elapsed = CFAbsoluteTimeGetCurrent() - startTime
                        let tps = elapsed > 0 ? Double(tokenCount) / elapsed : 0
                        let spoken = preambleFilter.push(channelFilter.push(token))
                        if spoken.isEmpty {
                            await MainActor.run {
                                self.tokensPerSecond = tps
                                self.generatePhase = channelFilter.isInside ? .hiddenReasoning : .writingSpeech
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
                        // Let the pipeline start TTS on this clause before the
                        // next llama_decode so CoreML can take the GPU lock.
                        await Task.yield()
                    }

                    let spokenTail = preambleFilter.push(channelFilter.flush()) + preambleFilter.flush()
                    if !spokenTail.isEmpty {
                        continuation.yield(spokenTail)
                        await MainActor.run { self.response += spokenTail }
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
        inferenceEpoch += 1
        generationTask?.cancel()
        isGenerating = false
        generatePhase = .idle
    }

    var isModelLoaded: Bool {
        llamaContext != nil
    }

    func tokenCount(for text: String) async -> Int? {
        guard let ctx = llamaContext else { return nil }
        return await ctx.countTokens(text)
    }

    /// Quiet one-shot completion used to compress dropped chat into memory.
    func compressNotes(_ notes: String) async -> String? {
        guard let ctx = llamaContext, !isGenerating else { return nil }
        let epoch = inferenceEpoch
        let clipped = String(notes.suffix(2500))
        do {
            let prompt = try await ctx.formatChat(
                system: "Compress these conversation notes into at most 70 words. Third person. Keep names, decisions, and facts. No lists, no preamble.",
                history: [("user", clipped)],
                suppressThinking: true
            )
            guard epoch == inferenceEpoch, !Task.isCancelled else { return nil }
            await ctx.clear()
            guard epoch == inferenceEpoch, !Task.isCancelled else { return nil }
            if let gpu = gpuLock {
                try await gpu.run { try await ctx.completionInit(text: prompt) }
            } else {
                try await ctx.completionInit(text: prompt)
            }
            var out = ""
            var n = 0
            let gpu = gpuLock
            while n < 120, epoch == inferenceEpoch, !Task.isCancelled {
                let token: String?
                if let gpu {
                    token = await gpu.run { await ctx.completionLoop() }
                } else {
                    token = await ctx.completionLoop()
                }
                guard let token else { break }
                out += token
                n += 1
            }
            guard epoch == inferenceEpoch, !Task.isCancelled else { return nil }
            let trimmed = out.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        } catch {
            logger.error("memory compress failed: \(error.localizedDescription)")
            return nil
        }
    }
}

enum PromptTokenCounter {
    /// Fast on-device estimate when the GGUF tokenizer is not loaded yet.
    static func estimate(_ text: String) -> Int {
        guard !text.isEmpty else { return 0 }
        var cjk = 0
        var other = 0
        for scalar in text.unicodeScalars {
            if (0x4E00...0x9FFF).contains(scalar.value)
                || (0x3040...0x30FF).contains(scalar.value)
                || (0xAC00...0xD7AF).contains(scalar.value) {
                cjk += 1
            } else {
                other += 1
            }
        }
        return max(1, cjk + (other + 3) / 4)
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
        Swift.min(80, Swift.max(8, Int(clamp(size) / 256)))
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
