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
    private let nBatch: Int32
    private let chatFamily: ChatFamily
    /// 12B+ Gemma 4 gets a closed empty thought primer. E2B/E4B must not — that primer opens CoT.
    private let gemmaThoughtPrimer: Bool
    /// Gemma 3 R1 writes the thought as the reply unless the prompt already ends with `</think>`.
    nonisolated let hidesThinkUntilClose: Bool
    private var stopTracker = StopSequenceTracker()
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
        if name.contains("draft") || name.contains("mtp") || name.contains("nextn") {
            throw LlamaContextError.modelLoadFailed("This is a draft/MTP speculative-decoding companion, not the chat weights. A Qwen 3.8 27B Q4_0 under 4 GB is that draft file (~1.7 GB). The real 27B Q4_K_M is ~17 GB and will not fit next to speech models. Keep Qwen 3.5 2B.")
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
                return LlamaContext(
                    model: model,
                    context: context,
                    batchSize: ctxParams.n_batch,
                    fileName: URL(fileURLWithPath: path).lastPathComponent
                )
            }
        }

        llama_model_free(model)
        LlamaBackend.release()
        throw LlamaContextError.contextCreationFailed(model: modelDesc, log: LlamaLogSink.hint())
    }

    private init(model: OpaquePointer, context: OpaquePointer, batchSize: UInt32, fileName: String) {
        self.model = model
        self.context = context
        self.tokensList = []
        self.nBatch = Int32(max(UInt32(32), batchSize))
        self.batch = llama_batch_init(self.nBatch, 0, 1)
        self.vocab = llama_model_get_vocab(model)
        self.chatFamily = LlamaContext.detectFamily(model: model, vocab: self.vocab, fileName: fileName)
        let params = llama_model_n_params(model)
        self.gemmaThoughtPrimer = chatFamily == .gemma4 && params >= UInt64(6_000_000_000)
        let nameBlob = fileName + " " + LlamaContext.metaString(model, "general.name")
            + " " + LlamaContext.metaString(model, "general.basename")
        self.hidesThinkUntilClose = chatFamily == .gemma3 && ChatPrompt.isGemmaReasoningTune(nameBlob)

        // Generic voice-assistant sampling. Not tied to a single model family.
        let sparams = llama_sampler_chain_default_params()
        self.sampling = llama_sampler_chain_init(sparams)!
        LlamaContext.installLogitBias(sampling: self.sampling, vocab: self.vocab)
        llama_sampler_chain_add(
            self.sampling,
            llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), 64, 1.05, 0.0, 0.0)
        )
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_k(40))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_top_p(0.9, 1))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_min_p(0.05, 1))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_temp(0.7))
        llama_sampler_chain_add(self.sampling, llama_sampler_init_dist(1234))
        logger.info("chat family=\(self.chatFamily.rawValue, privacy: .public) gemmaPrimer=\(self.gemmaThoughtPrimer ? "1" : "0", privacy: .public) hideThink=\(self.hidesThinkUntilClose ? "1" : "0", privacy: .public)")
    }

    /// Format a voice conversation. Known families use native templates so thinking
    /// is off in the prompt (Gemma 4 is not ChatML). Builtin `llama_chat_apply_template`
    /// is only a last resort — it does not run Jinja.
    func formatChat(
        system: String,
        history: [(role: String, content: String)],
        suppressThinking: Bool = true
    ) throws -> String {
        if chatFamily != .unknown {
            return ChatPrompt.format(
                family: chatFamily,
                system: system,
                history: history,
                thinking: !suppressThinking,
                gemmaThoughtPrimer: gemmaThoughtPrimer,
                gemmaReasoningTune: hidesThinkUntilClose
            )
        }

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
        if let formatted = applyChatTemplate(tmpl: tmpl, messages: messages) {
            return formatted
        }
        logger.warning("GGUF chat template was not applied; using ChatML without a think block.")
        return ChatPrompt.format(
            family: .chatml,
            system: system,
            history: history,
            thinking: false
        )
    }

    /// Count tokens in an arbitrary string with this GGUF's tokenizer.
    func countTokens(_ text: String) -> Int {
        let utf8 = text.utf8CString
        let required = utf8.withUnsafeBufferPointer { buffer in
            llama_tokenize(vocab, buffer.baseAddress, Int32(buffer.count - 1), nil, 0, false, true)
        }
        return Int(abs(required))
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
        stopTracker.reset()
        let utf8 = text.utf8CString
        let addBos = llama_vocab_get_add_bos(vocab) && !ChatPrompt.startsWithBos(text)

        // First call with nil buffer returns negative required count
        let requiredTokens = utf8.withUnsafeBufferPointer { buffer in
            llama_tokenize(vocab, buffer.baseAddress, Int32(buffer.count - 1), nil, 0, addBos, true)
        }
        let nTokens = abs(requiredTokens)
        guard nTokens > 0 else {
            throw LlamaContextError.decodeFailed
        }

        tokensList = Array(repeating: llama_token(), count: Int(nTokens))
        let actualCount = utf8.withUnsafeBufferPointer { buffer in
            llama_tokenize(vocab, buffer.baseAddress, Int32(buffer.count - 1),
                          &tokensList, nTokens, addBos, true)
        }
        tokensList = Array(tokensList.prefix(Int(actualCount)))

        let nCtx = llama_n_ctx(context)
        guard tokensList.count <= nCtx else {
            throw LlamaContextError.promptTooLong
        }

        // n_batch is 256. Writing the whole prompt in one llama_batch_add
        // overruns the C buffer after a couple of turns and kills the process
        // at ~400 MB of a 6 GB jetsam cap.
        var offset = 0
        while offset < tokensList.count {
            llama_batch_clear(&batch)
            let chunk = min(Int(nBatch), tokensList.count - offset)
            for j in 0..<chunk {
                let index = offset + j
                llama_batch_add(
                    &batch,
                    tokensList[index],
                    Int32(index),
                    [0],
                    index == tokensList.count - 1
                )
            }
            guard llama_decode(context, batch) >= 0 else {
                throw LlamaContextError.decodeFailed
            }
            offset += chunk
        }

        nCur = Int32(tokensList.count)
        nDecode = 0
        isDone = false
    }

    /// Generate the next token. Returns the decoded text, or nil if generation is complete.
    func completionLoop() -> String? {
        guard !isDone else { return nil }

        let idx = max(Int32(0), batch.n_tokens - 1)
        let newTokenId = llama_sampler_sample(sampling, context, idx)
        llama_sampler_accept(sampling, newTokenId)

        // Check for end of generation
        if llama_vocab_is_eog(vocab, newTokenId) {
            isDone = true
            return nil
        }

        guard let text = decodeTokenPiece(newTokenId) else {
            isDone = true
            return nil
        }

        let nCtx = Int32(llama_n_ctx(context))
        if nCur >= nCtx {
            isDone = true
            return nil
        }

        // Prepare next batch
        llama_batch_clear(&batch)
        llama_batch_add(&batch, newTokenId, nCur, [0], true)

        nDecode += 1
        nCur += 1

        if llama_decode(context, batch) != 0 {
            isDone = true
            return nil
        }

        switch stopTracker.push(text) {
        case .hold:
            return ""
        case .text(let spoken):
            return spoken
        case .stop(let spoken):
            isDone = true
            return spoken.isEmpty ? nil : spoken
        }
    }

    /// `llama_token_to_piece` does not NUL-terminate. `String(cString:)` walks off
    /// the stack when a piece fills the buffer — crash after a few turns, not OOM.
    private func decodeTokenPiece(_ token: llama_token) -> String? {
        var bufSize = 128
        for _ in 0..<4 {
            var buf = [CChar](repeating: 0, count: bufSize)
            let nChars = llama_token_to_piece(vocab, token, &buf, Int32(bufSize), 0, true)
            if nChars < 0 {
                bufSize = Int(-nChars) + 1
                continue
            }
            if nChars == 0 { return "" }
            let bytes = buf.prefix(Int(nChars)).map { UInt8(bitPattern: $0) }
            return String(decoding: bytes, as: UTF8.self)
        }
        return nil
    }

    /// Check if generation is complete.
    var generationDone: Bool {
        isDone
    }

    /// Clear the context for a new conversation turn.
    /// `data: true` wipes Metal KV, not just sequence metadata. Metadata-only
    /// clear after barge-in leaves GPU cache dirty and crashes on the next decode
    /// while RAM still looks fine (~400 MB of a 6 GB jetsam cap).
    func clear() {
        if let memory = llama_get_memory(context) {
            llama_memory_clear(memory, true)
        }
        llama_sampler_reset(sampling)
        llama_batch_clear(&batch)
        tokensList = []
        nCur = 0
        nDecode = 0
        isDone = false
        stopTracker.reset()
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

    private static func detectFamily(model: OpaquePointer, vocab: OpaquePointer, fileName: String) -> ChatFamily {
        ChatPrompt.detect(
            architecture: metaString(model, "general.architecture"),
            template: {
                if let ptr = llama_model_chat_template(model, nil) {
                    return String(cString: ptr)
                }
                return ""
            }(),
            desc: {
                var buf = [CChar](repeating: 0, count: 256)
                _ = llama_model_desc(model, &buf, buf.count)
                return String(cString: buf)
            }(),
            fileName: fileName,
            hasTurnToken: specialTokenID("<|turn>", vocab: vocab) != nil
        )
    }

    private static func metaString(_ model: OpaquePointer, _ key: String) -> String {
        var buf = [CChar](repeating: 0, count: 128)
        let n = llama_model_meta_val_str(model, key, &buf, buf.count)
        guard n > 0 else { return "" }
        return String(cString: buf)
    }

    /// Ban think-openers so the sampler does not start a reasoning channel.
    /// Do not ban EOS / `<|im_end|>` — those must remain available to end the turn.
    private static func installLogitBias(
        sampling: UnsafeMutablePointer<llama_sampler>,
        vocab: OpaquePointer
    ) {
        let nVocab = llama_vocab_n_tokens(vocab)
        var biases: [llama_logit_bias] = []
        let banned = [
            "<think>",
            "<|think|>",
            "<|channel>",
            "<|im_start|>",
            "<|im_start>",
        ]
        for piece in banned {
            guard let id = specialTokenID(piece, vocab: vocab) else { continue }
            if llama_vocab_is_eog(vocab, id) { continue }
            biases.append(llama_logit_bias(token: id, bias: -1.0e10))
        }
        guard !biases.isEmpty else { return }
        let biasSampler = biases.withUnsafeBufferPointer { ptr in
            llama_sampler_init_logit_bias(nVocab, Int32(biases.count), ptr.baseAddress)
        }
        llama_sampler_chain_add(sampling, biasSampler)
    }

    private static func specialTokenID(_ text: String, vocab: OpaquePointer) -> llama_token? {
        var tokens = [llama_token](repeating: 0, count: 16)
        let n = text.utf8CString.withUnsafeBufferPointer { buffer in
            llama_tokenize(vocab, buffer.baseAddress, Int32(buffer.count - 1), &tokens, 16, false, true)
        }
        guard n == 1 else { return nil }
        return tokens[0]
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
