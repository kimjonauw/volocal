import Foundation
import Darwin
import LlamaSwift
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "llama")

private func volocalLlamaLog(
    _ level: ggml_log_level,
    _ text: UnsafePointer<CChar>?,
    _ userData: UnsafeMutableRawPointer?
) {
    guard let text else { return }
    LlamaLogSink.append(String(cString: text))
}

private enum LlamaLogSink {
    private static let lock = NSLock()
    private static var lines: [String] = []

    static func install() {
        llama_log_set(volocalLlamaLog, nil)
    }

    static func clear() {
        lock.lock()
        lines.removeAll()
        lock.unlock()
    }

    static func append(_ raw: String) {
        let text = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        let lower = text.lowercased()
        let keep = lower.contains("error") || lower.contains("fail") || lower.contains("unable")
            || lower.contains("invalid") || lower.contains("unknown") || lower.contains("not support")
            || lower.contains("oom") || lower.contains("failed")
        guard keep else { return }
        lock.lock()
        lines.append(text)
        if lines.count > 10 {
            lines.removeFirst(lines.count - 10)
        }
        lock.unlock()
    }

    static func hint() -> String {
        lock.lock()
        let joined = lines.suffix(4).joined(separator: " · ")
        lock.unlock()
        return joined
    }
}

// MARK: - Batch Helpers (from official llama.cpp SwiftUI example)

private func llama_batch_clear(_ batch: inout llama_batch) {
    batch.n_tokens = 0
}

private func llama_batch_add(
    _ batch: inout llama_batch,
    _ id: llama_token,
    _ pos: llama_pos,
    _ seq_ids: [llama_seq_id],
    _ logits: Bool
) {
    batch.token[Int(batch.n_tokens)] = id
    batch.pos[Int(batch.n_tokens)] = pos
    batch.n_seq_id[Int(batch.n_tokens)] = Int32(seq_ids.count)
    for i in 0..<seq_ids.count {
        batch.seq_id[Int(batch.n_tokens)]![i] = seq_ids[i]
    }
    batch.logits[Int(batch.n_tokens)] = logits ? 1 : 0
    batch.n_tokens += 1
}

// MARK: - LlamaContext Actor

/// Thread-safe actor wrapping the llama.cpp C API for on-device LLM inference.
/// Based on the official llama.cpp SwiftUI example (LibLlama.swift).
actor LlamaContext {
    private var model: OpaquePointer
    private var context: OpaquePointer
    private var vocab: OpaquePointer
    private var sampling: UnsafeMutablePointer<llama_sampler>
    private var batch: llama_batch
    private var tokensList: [llama_token]
    private var nCur: Int32 = 0
    private var nDecode: Int32 = 0
    private var isDone: Bool = false

    /// Create a new LlamaContext by loading a GGUF model file.
    /// Blocking mmap/Metal work must not run on the main thread.
    nonisolated static func create(path: String, contextSize: UInt32 = 2048) throws -> LlamaContext {
        let url = URL(fileURLWithPath: path)
        try validateLoadable(at: url)
        let bytes = (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? UInt64) ?? 0
        let primaryLayers = gpuLayers(forFileBytes: bytes)

        do {
            return try loadBlocking(path: path, gpuLayers: primaryLayers, contextSize: contextSize)
        } catch {
            if primaryLayers != 0 {
                logger.error("LLM load retry CPU-only after: \(error.localizedDescription)")
                return try loadBlocking(path: path, gpuLayers: 0, contextSize: contextSize)
            }
            throw error
        }
    }

    nonisolated private static func validateLoadable(at url: URL) throws {
        let name = url.lastPathComponent.lowercased()
        if name.contains("iq1_") || name.contains("iq2_") || name.contains("iq3_") {
            throw LlamaContextError.unsupportedQuant
        }
        if name.contains("embed") || name.contains("rerank") || name.contains("bge-")
            || name.contains("-e5-") || name.contains("minilm") || name.contains("mmproj") {
            throw LlamaContextError.modelLoadFailed("This looks like an embedding, rerank, or vision GGUF — not a chat model. Qwen 3.5 2B Q4_K is the known-good pick.")
        }
        guard GGUFFile.looksLikeGGUF(at: url) else {
            throw LlamaContextError.modelLoadFailed("File is not a GGUF (wrong magic).")
        }
        let bytes = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size] as? UInt64) ?? 0
        if bytes > 6_500_000_000 {
            throw LlamaContextError.tooLarge(bytes)
        }
        if bytes < 1_048_576 {
            throw LlamaContextError.modelLoadFailed("File is smaller than 1 MB — truncated download.")
        }
    }

    nonisolated private static func gpuLayers(forFileBytes bytes: UInt64) -> Int32 {
        #if targetEnvironment(simulator)
        return 0
        #else
        if bytes > 3_500_000_000 { return 0 }
        if bytes > 2_200_000_000 { return 16 }
        if bytes > 1_400_000_000 { return 32 }
        return 99
        #endif
    }

    nonisolated private static func cappedContext(fileBytes: UInt64, requested: UInt32) -> UInt32 {
        let ramCap: UInt32
        if fileBytes > 3_500_000_000 { ramCap = 2048 }
        else if fileBytes > 2_200_000_000 { ramCap = 4096 }
        else if fileBytes > 1_400_000_000 { ramCap = 8192 }
        else { ramCap = requested }
        return Swift.min(requested, ramCap)
    }

    nonisolated private static func loadBlocking(
        path: String,
        gpuLayers: Int32,
        contextSize: UInt32
    ) throws -> LlamaContext {
        LlamaBackend.retain()
        LlamaLogSink.clear()

        var modelParams = llama_model_default_params()
        modelParams.n_gpu_layers = gpuLayers

        guard let model = llama_model_load_from_file(path, modelParams) else {
            LlamaBackend.release()
            throw LlamaContextError.modelLoadFailed(LlamaLogSink.hint())
        }

        let trained = UInt32(max(0, llama_model_n_ctx_train(model)))
        var nctx = contextSize
        if trained > 0 {
            nctx = Swift.min(nctx, trained)
        }
        nctx = cappedContext(
            fileBytes: (try? FileManager.default.attributesOfItem(atPath: path)[.size] as? UInt64) ?? 0,
            requested: nctx
        )
        nctx = max(UInt32(256), nctx)
        nctx = (nctx / 32) * 32

        var descBuf = [CChar](repeating: 0, count: 256)
        _ = llama_model_desc(model, &descBuf, descBuf.count)
        let modelDesc = String(cString: descBuf)

        let threadCount = Int32(max(1, min(8, ProcessInfo.processInfo.processorCount - 2)))
        let nBatchBase = Swift.min(UInt32(256), nctx)

        let attempts: [(flash: llama_flash_attn_type, offloadKQV: Bool, ctx: UInt32)] = [
            (LLAMA_FLASH_ATTN_TYPE_AUTO, true, nctx),
            (LLAMA_FLASH_ATTN_TYPE_DISABLED, true, nctx),
            (LLAMA_FLASH_ATTN_TYPE_DISABLED, false, nctx),
            (LLAMA_FLASH_ATTN_TYPE_DISABLED, false, Swift.min(nctx, 2048)),
            (LLAMA_FLASH_ATTN_TYPE_DISABLED, false, Swift.min(nctx, 512)),
        ]

        for attempt in attempts {
            var ctxParams = llama_context_default_params()
            ctxParams.n_ctx = max(UInt32(256), (attempt.ctx / 32) * 32)
            ctxParams.n_batch = Swift.min(nBatchBase, ctxParams.n_ctx)
            ctxParams.n_ubatch = ctxParams.n_batch
            ctxParams.n_seq_max = 1
            ctxParams.n_threads = threadCount
            ctxParams.n_threads_batch = threadCount
            ctxParams.flash_attn_type = attempt.flash
            ctxParams.offload_kqv = attempt.offloadKQV
            ctxParams.embeddings = false

            if let context = llama_init_from_model(model, ctxParams) {
                logger.info("llama context ready desc=\(modelDesc, privacy: .public) n_ctx=\(ctxParams.n_ctx) kqv=\(attempt.offloadKQV)")
                return LlamaContext(model: model, context: context, batchSize: ctxParams.n_batch)
            }
        }

        llama_model_free(model)
        LlamaBackend.release()
        throw LlamaContextError.contextCreationFailed(model: modelDesc, log: LlamaLogSink.hint())
    }

    private init(model: OpaquePointer, context: OpaquePointer, batchSize: UInt32) {
        self.model = model
        self.context = context
        self.tokensList = []
        self.batch = llama_batch_init(Int32(max(UInt32(32), batchSize)), 0, 1)
        self.vocab = llama_model_get_vocab(model)

        // Generic voice-assistant sampling. Not tied to a single model family.
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)!
        llama_sampler_chain_add(
            self.sampling,
            llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, 1.05, 0.0, 0.0)
        )
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_k(40))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_p(0.9, 1))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_min_p(0.05, 1))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.7))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
    }

    /// Format a voice conversation using the GGUF's built-in chat template when
    /// llama.cpp recognizes it. Falls back to ChatML so Qwen/OpenChat still work.
    func formatChat(system: String, history: [(role: String, content: String)]) throws -> String {
        var boxes: [CStringBox] = []
        boxes.append(CStringBox("system"))
        boxes.append(CStringBox(system))
        for turn in history {
            boxes.append(CStringBox(turn.role))
            boxes.append(CStringBox(turn.content))
        }

        var messages: [llama_chat_message] = []
        messages.reserveCapacity(boxes.count / 2)
        for i in stride(from: 0, to: boxes.count, by: 2) {
            messages.append(llama_chat_message(role: boxes[i].ptr, content: boxes[i + 1].ptr))
        }

        let tmpl = llama_model_chat_template(model, nil)
        let prompt: String
        if let formatted = applyChatTemplate(tmpl: tmpl, messages: messages) {
            prompt = formatted
        } else {
            logger.warning("GGUF chat template was not applied; using ChatML. Llama/Gemma GGUFs may need a llama.cpp template update.")
            prompt = ChatMLFallback.format(system: system, history: history)
        }
        return ThinkingPrefix.suppress(prompt)
    }

    deinit {
        llama_sampler_free(sampling)
        llama_batch_free(batch)
        llama_free(context)
        llama_model_free(model)
        LlamaBackend.release()
    }

    /// Tokenize and evaluate the prompt, preparing for token generation.
    func completionInit(text: String) throws {
        let utf8 = text.utf8CString

        // First call with nil buffer returns negative required count
        let requiredTokens = utf8.withUnsafeBufferPointer { buffer in
            llama_tokenize(vocab, buffer.baseAddress, Int32(buffer.count - 1), nil, 0, true, true)
        }
        let nTokens = abs(requiredTokens)
        guard nTokens > 0 else {
            throw LlamaContextError.decodeFailed
        }

        tokensList = Array(repeating: llama_token(), count: Int(nTokens))
        let actualCount = utf8.withUnsafeBufferPointer { buffer in
            llama_tokenize(vocab, buffer.baseAddress, Int32(buffer.count - 1),
                          &tokensList, nTokens, true, true)
        }
        tokensList = Array(tokensList.prefix(Int(actualCount)))

        let nCtx = llama_n_ctx(context)
        guard tokensList.count <= nCtx else {
            throw LlamaContextError.promptTooLong
        }

        llama_batch_clear(&batch)

        for (i, token) in tokensList.enumerated() {
            llama_batch_add(&batch, token, Int32(i), [0], i == tokensList.count - 1)
        }

        guard llama_decode(context, batch) >= 0 else {
            throw LlamaContextError.decodeFailed
        }

        nCur = Int32(tokensList.count)
        nDecode = 0
        isDone = false
    }

    /// Generate the next token. Returns the decoded text, or nil if generation is complete.
    func completionLoop() -> String? {
        guard !isDone else { return nil }

        let newTokenId = llama_sampler_sample(sampling, context, batch.n_tokens - 1)

        // Check for end of generation
        if llama_vocab_is_eog(vocab, newTokenId) {
            isDone = true
            return nil
        }

        // Convert token to text
        let bufSize = 128
        var buf = [CChar](repeating: 0, count: bufSize)
        let nChars = llama_token_to_piece(vocab, newTokenId, &buf, Int32(bufSize), 0, false)

        guard nChars >= 0 else {
            isDone = true
            return nil
        }

        let text = String(cString: buf)

        // Prepare next batch
        llama_batch_clear(&batch)
        llama_batch_add(&batch, newTokenId, nCur, [0], true)

        nDecode += 1
        nCur += 1

        if llama_decode(context, batch) != 0 {
            isDone = true
            return nil
        }

        return text
    }

    /// Check if generation is complete.
    var generationDone: Bool {
        isDone
    }

    /// Clear the context for a new conversation turn.
    func clear() {
        if let memory = llama_get_memory(context) {
            llama_memory_clear(memory, false)
        }
        nCur = 0
        nDecode = 0
        isDone = false
    }

    private func applyChatTemplate(tmpl: UnsafePointer<CChar>?, messages: [llama_chat_message]) -> String? {
        messages.withUnsafeBufferPointer { buffer in
            guard let base = buffer.baseAddress else { return nil }
            var storage = [CChar](repeating: 0, count: 4096)
            var needed = llama_chat_apply_template(
                tmpl,
                base,
                buffer.count,
                true,
                &storage,
                Int32(storage.count)
            )
            if needed < 0 { return nil }
            if needed >= storage.count {
                storage = [CChar](repeating: 0, count: Int(needed) + 1)
                needed = llama_chat_apply_template(
                    tmpl,
                    base,
                    buffer.count,
                    true,
                    &storage,
                    Int32(storage.count)
                )
            }
            guard needed >= 0 else { return nil }
            return String(cString: storage)
        }
    }
}

private enum LlamaBackend {
    private static var refCount = 0
    private static let lock = NSLock()

    static func retain() {
        lock.lock()
        defer { lock.unlock() }
        if refCount == 0 {
            llama_backend_init()
            LlamaLogSink.install()
        }
        refCount += 1
    }

    static func release() {
        lock.lock()
        defer { lock.unlock() }
        refCount -= 1
        if refCount == 0 {
            llama_backend_free()
        }
    }
}

private final class CStringBox {
    let ptr: UnsafeMutablePointer<CChar>

    init(_ string: String) {
        ptr = strdup(string) ?? {
            let empty = strdup("")!
            return empty
        }()
    }

    deinit {
        free(ptr)
    }
}

private enum ChatMLFallback {
    static func format(system: String, history: [(role: String, content: String)]) -> String {
        var prompt = "<|im_start|>system\n\(system)<|im_end|>\n"
        for turn in history {
            prompt += "<|im_start|>\(turn.role)\n\(turn.content)<|im_end|>\n"
        }
        prompt += "<|im_start|>assistant\n<think>\n</think>\n"
        return prompt
    }
}

/// Qwen-class GGUFs leave an open `<think>` so the model reasons silently and TTS waits.
/// Close that prefix so the first tokens are spoken words.
private enum ThinkingPrefix {
    static func suppress(_ prompt: String) -> String {
        let lower = prompt.lowercased()
        if let open = lower.range(of: "<think>", options: .backwards) {
            let after = lower[open.upperBound...]
            if after.range(of: "</think>") == nil {
                var closed = prompt
                if !closed.hasSuffix("\n") { closed += "\n" }
                return closed + "</think>\n"
            }
            return prompt
        }

        let trimmed = prompt.trimmingCharacters(in: .newlines)
        if trimmed.hasSuffix("<|im_start|>assistant") {
            var out = prompt
            if !out.hasSuffix("\n") { out += "\n" }
            return out + "<think>\n</think>\n"
        }
        return prompt
    }
}

// MARK: - Errors

enum LlamaContextError: LocalizedError {
    case modelLoadFailed(String)
    case contextCreationFailed(model: String, log: String)
    case promptTooLong
    case decodeFailed
    case tooLarge(UInt64)
    case unsupportedQuant
    case crashedLastLaunch

    var errorDescription: String? {
        switch self {
        case .modelLoadFailed(let log):
            if log.isEmpty {
                return "llama.cpp could not load this GGUF (unknown architecture, bad quant, or truncated file). Qwen 3.5 2B Q4_K is the known-good pick."
            }
            return "llama.cpp could not load this GGUF. \(log)"
        case .contextCreationFailed(let model, let log):
            var parts = ["This GGUF loaded but llama.cpp could not create a context."]
            if !model.isEmpty { parts.append(model) }
            parts.append("Qwen 3.5 2B at 2048 working means RAM is fine — this file is a different architecture, an embedding/rerank GGUF, or needs flash-attn/KV settings iOS Metal rejected.")
            if !log.isEmpty { parts.append(log) }
            return parts.joined(separator: " ")
        case .promptTooLong:
            return "Prompt exceeds context window"
        case .decodeFailed:
            return "Token decode failed"
        case .tooLarge(let bytes):
            let size = ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
            return "This GGUF is \(size). With speech models resident that will jetsam LiveContainer. Pick a Q4 2B–4B."
        case .unsupportedQuant:
            return "IQ1/IQ2/IQ3 GGUFs are not safe in this llama.cpp build. Pick a Q4_K / Q5_K file."
        case .crashedLastLaunch:
            return "The last language-model load crashed (the app was killed). That GGUF is probably too big, an unsupported quant, or the context slider is too high. Pick a Q4 2B–4B, or Reset context to 2,048, then Try again."
        }
    }
}
