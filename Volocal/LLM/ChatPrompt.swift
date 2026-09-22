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
        gemmaThoughtPrimer: Bool = false
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
            return gemma3(system: system, history: history)
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

    private static func gemma3(system: String, history: [(role: String, content: String)]) -> String {
        var prompt = "<bos>"
        var firstUser = true
        for turn in history {
            if turn.role == "assistant" {
                prompt += "<start_of_turn>model\n"
                prompt += turn.content.trimmingCharacters(in: .whitespacesAndNewlines)
                prompt += "<end_of_turn>\n"
            } else {
                prompt += "<start_of_turn>user\n"
                if firstUser {
                    let sys = system.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !sys.isEmpty {
                        prompt += sys + "\n\n"
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

/// Skip Gemma 4 `<|channel>thought` … `<channel|>` so TTS only gets the answer channel.
struct ThoughtChannelFilter {
    private var buffer = ""
    private var inside = false
    var isInside: Bool { inside }

    mutating func push(_ chunk: String) -> String {
        buffer += chunk
        var output = ""
        while !buffer.isEmpty {
            if inside {
                if let end = range(of: "<channel|>") ?? range(of: "<|channel|>") {
                    buffer.removeSubrange(buffer.startIndex..<end.upperBound)
                    if buffer.first == "\n" { buffer.removeFirst() }
                    inside = false
                    continue
                }
                if buffer.count > 12 {
                    buffer = String(buffer.suffix(12))
                }
                break
            }
            if let start = range(of: "<|channel>thought") ?? range(of: "<|channel|>thought") {
                output += buffer[buffer.startIndex..<start.lowerBound]
                buffer.removeSubrange(buffer.startIndex..<start.upperBound)
                inside = true
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
        if inside {
            buffer = ""
            inside = false
            return ""
        }
        let leftover = buffer
        buffer = ""
        return leftover
    }

    private func range(of needle: String) -> Range<String.Index>? {
        buffer.range(of: needle, options: .caseInsensitive)
    }

    private func incompletePrefixCount() -> Int? {
        let tags = ["<|channel>thought", "<|channel|>thought", "<channel|>", "<|channel|>"]
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
    private var buffer = ""
    private var passed = false

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
}
