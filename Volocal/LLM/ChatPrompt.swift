import Foundation

/// GGUF chat families we format ourselves.
///
/// `llama_chat_apply_template` in llama.cpp b10549 is **not** a Jinja engine — it only
/// matches a short builtin list. Gemma 4's template is Jinja, so apply fails and the
/// old ChatML fallback made Gemma emit `<|im_start|>` / thinking tokens.
enum ChatFamily: String {
    case gemma4
    case gemma3
    case qwen3
    case chatml
    case llama3
    case unknown
}

enum ChatPrompt {
    static func detect(
        architecture: String,
        template: String,
        desc: String,
        fileName: String = "",
        hasTurnToken: Bool = false
    ) -> ChatFamily {
        let arch = architecture.lowercased()
        let descLower = desc.lowercased()
        let name = fileName.lowercased()
        if hasTurnToken || template.contains("<|turn>") || arch.contains("gemma4")
            || descLower.contains("gemma4") || name.contains("gemma-4") || name.contains("gemma4") {
            return .gemma4
        }
        if template.contains("<start_of_turn>") || arch.hasPrefix("gemma") || descLower.contains("gemma")
            || name.contains("gemma-3") || name.contains("gemma3") {
            return .gemma3
        }
        if arch.contains("qwen3") || descLower.contains("qwen3")
            || name.contains("qwen3") || name.contains("qwen-3") {
            return .qwen3
        }
        if template.contains("<|start_header_id|>") || arch.hasPrefix("llama3") {
            return .llama3
        }
        if template.contains("<|im_start|>") || arch.contains("qwen") {
            return .chatml
        }
        return .unknown
    }

    static func startsWithBos(_ prompt: String) -> Bool {
        prompt.hasPrefix("<bos>") || prompt.hasPrefix("<|begin_of_text|>")
    }

    static func format(
        family: ChatFamily,
        system: String,
        history: [(role: String, content: String)],
        thinking: Bool,
        gemmaThoughtPrimer: Bool = false,
        gemmaReasoningTune: Bool = false
    ) -> String {
        switch family {
        case .gemma4:
            return gemma4(
                system: system,
                history: history,
                thinking: thinking,
                thoughtPrimer: gemmaThoughtPrimer
            )
        case .gemma3:
            return gemma3(
                system: system,
                history: history,
                thinking: thinking,
                reasoningTune: gemmaReasoningTune
            )
        case .qwen3:
            return chatml(system: system, history: history, qwenThinkSwitch: true, thinking: thinking)
        case .chatml, .unknown:
            return chatml(system: system, history: history, qwenThinkSwitch: false, thinking: false)
        case .llama3:
            return llama3(system: system, history: history)
        }
    }

    /// Gemma 4. E2B/E4B must **not** get an empty thought primer — that primer is
    /// only for 12B+ and on the small models it *starts* a thought channel
    /// ("the user just said…"). 12B+ get a closed empty thought so they skip CoT.
    private static func gemma4(
        system: String,
        history: [(role: String, content: String)],
        thinking: Bool,
        thoughtPrimer: Bool
    ) -> String {
        var systemText = system.trimmingCharacters(in: .whitespacesAndNewlines)
        if !thinking {
            let lock = " Speak only the words the user should hear. Do not recap them, describe tone, or plan the reply."
            if !systemText.lowercased().contains("speak only the words") {
                systemText += lock
            }
        }
        let primer = (!thinking && thoughtPrimer) ? "<|channel>thought\n<channel|>" : ""
        var prompt = "<bos>"
        prompt += "<|turn>system\n"
        if thinking { prompt += "<|think|>\n" }
        prompt += systemText
        prompt += "<turn|>\n"
        for turn in history {
            if turn.role == "assistant" {
                prompt += "<|turn>model\n"
                prompt += primer
                prompt += turn.content.trimmingCharacters(in: .whitespacesAndNewlines)
                prompt += "<turn|>\n"
            } else {
                prompt += "<|turn>user\n"
                prompt += turn.content.trimmingCharacters(in: .whitespacesAndNewlines)
                prompt += "<turn|>\n"
            }
        }
        prompt += "<|turn>model\n"
        prompt += primer
        return prompt
    }

    /// Gemma 3 R1 (TheDrummer and similar) is not a special-token thinker.
    /// Prefilling `<think>` *starts* the reasoning and the reply becomes the thought.
    /// A closing tag alone tells that tune the thought is already over.
    static func isGemmaReasoningTune(_ name: String) -> Bool {
        let n = name.lowercased()
        if n.contains("-r1") || n.contains("_r1") || n.contains("r1-") || n.contains("r1_") { return true }
        if n.contains("reasoning") || n.contains("deep-reason") || n.contains("deep_reason") { return true }
        return false
    }

    private static func gemma3(
        system: String,
        history: [(role: String, content: String)],
        thinking: Bool,
        reasoningTune: Bool
    ) -> String {
        // Gemma 3 has no system role: instructions are prefixed to the first user
        // turn. Do not append more rules here, and do not prefill `</think>`.
        // That close tag makes everything above it (the instructions, and any
        // saved context) look like a finished thought, so the model answers
        // those lines as if they were the user's turns and often says the tag.
        _ = thinking
        _ = reasoningTune
        let systemText = system.trimmingCharacters(in: .whitespacesAndNewlines)
        var prompt = "<bos>"
        var firstUser = true
        for turn in history {
            if turn.role == "assistant" {
                let spoken = Self.stripThinkMarkers(turn.content)
                if Self.isPromptEcho(spoken, system: systemText) { continue }
                prompt += "<start_of_turn>model\n"
                prompt += spoken
                prompt += "<end_of_turn>\n"
            } else {
                prompt += "<start_of_turn>user\n"
                if firstUser {
                    if !systemText.isEmpty {
                        prompt += systemText + "\n\n"
                    }
                    firstUser = false
                }
                prompt += turn.content.trimmingCharacters(in: .whitespacesAndNewlines)
                prompt += "<end_of_turn>\n"
            }
        }
        prompt += "<start_of_turn>model\n"
        return prompt
    }

    /// Think tags are ordinary text on Gemma 3 R1. Drop them from saved turns
    /// so a leaked tag is not replayed as part of the conversation.
    static func stripThinkMarkers(_ text: String) -> String {
        var cleaned = text
        let tags = ["</think>", "<think>", "</|think|>", "<|think|>", "<speak>", "</speak>", "<|channel>thought", "<|channel|>thought", "<channel|>", "<|channel|>"]
        for tag in tags {
            cleaned = cleaned.replacingOccurrences(of: tag, with: "", options: .caseInsensitive)
        }
        return cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// True when a saved reply is the instructions being read back, not something
    /// the user should hear again on the next turn.
    private static func isPromptEcho(_ text: String, system: String) -> Bool {
        let lower = text.lowercased()
        if lower.contains("you are volocal") || lower.contains("speak only the words") { return true }
        if lower.contains("helpful voice assistant") || lower.contains("do not write a think") { return true }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        return trimmed.count >= 24 && system.lowercased().contains(trimmed)
    }

    /// Qwen 3.x: `/no_think` on the last user turn plus a closed think block so the
    /// model never opens a reasoning channel. Other ChatML models get no think tags.
    private static func chatml(
        system: String,
        history: [(role: String, content: String)],
        qwenThinkSwitch: Bool,
        thinking: Bool
    ) -> String {
        var prompt = "<|im_start|>system\n\(system)<|im_end|>\n"
        let lastUser = history.lastIndex(where: { $0.role == "user" })
        for (index, turn) in history.enumerated() {
            var content = turn.content
            if qwenThinkSwitch, !thinking, index == lastUser, !content.contains("/no_think") {
                content = content.trimmingCharacters(in: .whitespacesAndNewlines) + " /no_think"
            }
            prompt += "<|im_start|>\(turn.role)\n\(content)<|im_end|>\n"
        }
        prompt += "<|im_start|>assistant\n"
        if qwenThinkSwitch, !thinking {
            prompt += "<think>\n</think>\n"
        } else if qwenThinkSwitch, thinking {
            prompt += "<think>\n"
        }
        return prompt
    }

    private static func llama3(system: String, history: [(role: String, content: String)]) -> String {
        var prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        prompt += system
        prompt += "<|eot_id|>"
        for turn in history {
            let role = turn.role == "assistant" ? "assistant" : "user"
            prompt += "<|start_header_id|>\(role)<|end_header_id|>\n\n"
            prompt += turn.content
            prompt += "<|eot_id|>"
        }
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt
    }
}

/// Ends generation when the model emits a chat-delimiter token as text.
/// This is an antiprompt (stop), not a think-content strip: the turn is over.
struct StopSequenceTracker {
    private var buffer = ""
    private(set) var stopped = false

    static let sequences = [
        "<|im_end|>",
        "<|im_start|>",
        "<|im_end>",
        "<|im_start>",
        "<end_of_turn>",
        "<start_of_turn>",
        "<turn|>",
        "<|turn>",
        "<|eot_id|>",
        "<|eom_id|>",
        "<|endoftext|>",
        "<|end_of_text|>",
    ]

    enum Result {
        case hold
        case text(String)
        case stop(String)
    }

    mutating func reset() {
        buffer = ""
        stopped = false
    }

    mutating func push(_ piece: String) -> Result {
        if stopped { return .stop("") }
        buffer += piece
        let lower = buffer.lowercased()
        var hit: Range<String.Index>?
        for stop in Self.sequences {
            guard let range = lower.range(of: stop.lowercased()) else { continue }
            if hit == nil || range.lowerBound < hit!.lowerBound {
                hit = range
            }
        }
        if let hit {
            let prefix = String(buffer[buffer.startIndex..<hit.lowerBound])
            buffer = ""
            stopped = true
            return .stop(prefix)
        }
        if let hold = incompletePrefixCount() {
            if hold < buffer.count {
                let keepFrom = buffer.index(buffer.endIndex, offsetBy: -hold)
                let emit = String(buffer[buffer.startIndex..<keepFrom])
                buffer = String(buffer[keepFrom...])
                return emit.isEmpty ? .hold : .text(emit)
            }
            return .hold
        }
        let emit = buffer
        buffer = ""
        return .text(emit)
    }

    mutating func flush() -> String {
        if stopped {
            buffer = ""
            return ""
        }
        let leftover = buffer
        buffer = ""
        return leftover
    }

    private func incompletePrefixCount() -> Int? {
        let lower = buffer.lowercased()
        var hold: Int?
        for stop in Self.sequences {
            let maxN = min(stop.count - 1, lower.count)
            guard maxN >= 1 else { continue }
            for n in stride(from: maxN, through: 1, by: -1) {
                if lower.hasSuffix(String(stop.lowercased().prefix(n))) {
                    hold = max(hold ?? 0, n)
                    break
                }
            }
        }
        return hold
    }
}

/// Drop a hidden reasoning span so TTS only gets the answer.
/// Gemma 4 uses `<|channel>thought` … `<channel|>`. Gemma 3 thinking finetunes
/// and Qwen use `<think>` … `</think>`. The prompt already closes these; this
/// only catches a model that opens one anyway.
struct ThoughtChannelFilter {
    private static let spans: [(open: String, closes: [String])] = [
        ("<|channel>thought", ["<channel|>", "<|channel|>"]),
        ("<|channel|>thought", ["<channel|>", "<|channel|>"]),
        ("<think>", ["</think>"]),
        ("<|think|>", ["</think>", "</|think|>"]),
    ]

    private var buffer = ""
    private var inside = false
    private var closes: [String] = []
    /// R1-style tunes may emit the thought and only then `</think>`, with the
    /// opening tag left in the prompt. Hold the start of the reply until we
    /// know it is not that thought.
    private var waitingForThinkClose: Bool
    var isInside: Bool { inside }

    init(hideUntilThinkClose: Bool = false) {
        waitingForThinkClose = hideUntilThinkClose
    }

    mutating func push(_ chunk: String) -> String {
        if waitingForThinkClose && !inside {
            buffer += chunk
            if let released = releaseIfThinkFinished() {
                return released
            }
            return ""
        }
        buffer += chunk
        var output = ""
        while !buffer.isEmpty {
            if inside {
                if let end = earliestRange(among: closes) {
                    buffer.removeSubrange(buffer.startIndex..<end.upperBound)
                    if buffer.first == "\n" { buffer.removeFirst() }
                    inside = false
                    closes = []
                    continue
                }
                let keep = closes.map(\.count).max() ?? 12
                if buffer.count > keep {
                    buffer = String(buffer.suffix(keep))
                }
                break
            }
            if let start = earliestOpen() {
                output += buffer[buffer.startIndex..<start.range.lowerBound]
                buffer.removeSubrange(buffer.startIndex..<start.range.upperBound)
                inside = true
                closes = start.closes
                continue
            }
            if let end = earliestRange(among: ["</think>", "</|think|>"]) {
                let before = String(buffer[buffer.startIndex..<end.lowerBound])
                if !Self.looksLikeUnspokenThought(before) {
                    output += before
                }
                buffer.removeSubrange(buffer.startIndex..<end.upperBound)
                if buffer.first == "\n" { buffer.removeFirst() }
                continue
            }
            if let end = earliestRange(among: ["<speak>", "</speak>"]) {
                output += buffer[buffer.startIndex..<end.lowerBound]
                buffer.removeSubrange(buffer.startIndex..<end.upperBound)
                continue
            }
            if let hold = incompletePrefixCount() {
                if hold < buffer.count {
                    let keepFrom = buffer.index(buffer.endIndex, offsetBy: -hold)
                    output += buffer[buffer.startIndex..<keepFrom]
                    buffer = String(buffer[keepFrom...])
                }
                break
            }
            output += buffer
            buffer = ""
        }
        return output
    }

    mutating func flush() -> String {
        if waitingForThinkClose {
            if let spoken = releaseIfThinkFinished() {
                return spoken
            }
            let leftover = buffer
            buffer = ""
            waitingForThinkClose = false
            if leftover.lowercased().contains("<think") { return "" }
            return leftover
        }
        if inside {
            buffer = ""
            inside = false
            closes = []
            return ""
        }
        let leftover = buffer
        buffer = ""
        return leftover
    }

    /// Nil while this is still the hidden thought. A direct answer with no
    /// think tag stays buffered until `flush`, so a late `</think>` can still
    /// cut the draft out.
    private mutating func releaseIfThinkFinished() -> String? {
        if let end = earliestRange(among: ["</think>", "</|think|>"]) {
            buffer.removeSubrange(buffer.startIndex..<end.upperBound)
            if buffer.first == "\n" { buffer.removeFirst() }
            waitingForThinkClose = false
            let rest = buffer
            buffer = ""
            return push(rest)
        }
        if earliestOpen() != nil {
            waitingForThinkClose = false
            let rest = buffer
            buffer = ""
            return push(rest)
        }
        return nil
    }

    private static func looksLikeUnspokenThought(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return true }
        let lower = trimmed.lowercased()
        if lower.contains("you are volocal") || lower.contains("speak only the words") { return true }
        if lower.hasPrefix("the user") || lower.hasPrefix("i need to") || lower.hasPrefix("i should") { return true }
        return false
    }

    private func earliestOpen() -> (range: Range<String.Index>, closes: [String])? {
        var best: (range: Range<String.Index>, closes: [String])?
        for span in Self.spans {
            guard let range = range(of: span.open) else { continue }
            if best == nil || range.lowerBound < best!.range.lowerBound {
                best = (range, span.closes)
            }
        }
        return best
    }

    private func earliestRange(among needles: [String]) -> Range<String.Index>? {
        var best: Range<String.Index>?
        for needle in needles {
            guard let range = range(of: needle) else { continue }
            if best == nil || range.lowerBound < best!.lowerBound {
                best = range
            }
        }
        return best
    }

    private func range(of needle: String) -> Range<String.Index>? {
        buffer.range(of: needle, options: .caseInsensitive)
    }

    private func incompletePrefixCount() -> Int? {
        var tags = Self.spans.flatMap { [$0.open] + $0.closes }
        tags.append("<channel|>")
        tags.append("<|channel|>")
        tags.append("<speak>")
        tags.append("</speak>")
        let lower = buffer.lowercased()
        var hold: Int?
        for tag in tags {
            let maxN = min(tag.count - 1, lower.count)
            guard maxN >= 1 else { continue }
            for n in stride(from: maxN, through: 1, by: -1) {
                if lower.hasSuffix(String(tag.prefix(n))) {
                    hold = max(hold ?? 0, n)
                    break
                }
            }
        }
        return hold
    }
}

/// Drops leading inner-monologue sentences so TTS only gets the spoken reply.
/// Gemma 4 often emits several of these ("Okay." then "I need to respond naturally…")
/// before the greeting — skip all of them, not just a first-line prefix.
struct ReasoningPreambleFilter {
    /// Instructions and saved notes live in the Gemma 3 user turn. If the model
    /// reads them back, drop those sentences instead of speaking them.
    var systemEcho: String = ""
    private var buffer = ""
    private var passed = false

    init(systemEcho: String = "") {
        self.systemEcho = systemEcho
    }

    private static let cotPrefixes = [
        "the user",
        "the person",
        "they said",
        "the human",
        "the assistant",
        "the speaker",
        "this is a greeting",
        "user just",
        "user said",
        "okay i need",
        "ok i need",
        "alright i need",
    ]

    private static let planningPhrases = [
        "the user",
        "respond naturally",
        "keeping the tone",
        "keep the tone",
        "the tone",
        "casual and conversational",
        "conversational tone",
        "inner monologue",
        "my response",
        "the reply should",
        "stay casual",
        "be conversational",
        "i need to respond",
        "i should respond",
        "i need to greet",
        "i should greet",
        "i will greet",
        "let me think",
        "let me greet",
        "keep it casual",
        "naturally keeping",
        "they greeted",
        "user just said",
    ]

    private static let fillerSentences: Set<String> = [
        "okay", "ok", "alright", "so", "well", "right", "hmm", "uh", "um", "yeah", "yes",
    ]

    mutating func push(_ chunk: String) -> String {
        if passed { return chunk }
        buffer += chunk
        var output = ""
        while let split = nextSentenceRange() {
            let sentence = String(buffer[split])
            buffer.removeSubrange(split)
            if isChainOfThought(sentence) { continue }
            passed = true
            output += sentence
            output += buffer
            buffer = ""
            break
        }
        if !passed, buffer.count > 160, isChainOfThought(buffer) {
            buffer = ""
        }
        return output
    }

    mutating func flush() -> String {
        if passed {
            let leftover = buffer
            buffer = ""
            return leftover
        }
        if isChainOfThought(buffer) {
            buffer = ""
            return ""
        }
        let leftover = buffer
        buffer = ""
        return leftover
    }

    private func nextSentenceRange() -> Range<String.Index>? {
        guard let idx = buffer.firstIndex(where: { ".!?\n".contains($0) }) else { return nil }
        return buffer.startIndex..<buffer.index(after: idx)
    }

    private func isChainOfThought(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        var lower = trimmed.lowercased()
            .trimmingCharacters(in: .punctuationCharacters.union(.whitespacesAndNewlines))
        guard !lower.isEmpty else { return true }
        if isSystemEcho(lower) { return true }
        if Self.fillerSentences.contains(lower) { return true }
        for filler in ["okay, ", "ok, ", "alright, ", "so, ", "well, ", "right, ", "hmm, "] {
            if lower.hasPrefix(filler) {
                lower = String(lower.dropFirst(filler.count))
            }
        }
        if Self.cotPrefixes.contains(where: { lower.hasPrefix($0) }) { return true }
        let haystack = trimmed.lowercased()
        if Self.planningPhrases.contains(where: { haystack.contains($0) || lower.contains($0) }) {
            return true
        }
        return false
    }

    private func isSystemEcho(_ lower: String) -> Bool {
        let sys = systemEcho.lowercased()
        if lower.count >= 16, sys.contains(lower) { return true }
        let needles = [
            "you are volocal",
            "speak only the words",
            "do not write a think",
            "do not read this block",
            "earlier conversation",
            "helpful voice assistant",
            "running entirely on-device",
        ]
        return needles.contains { lower.contains($0) }
    }
}
