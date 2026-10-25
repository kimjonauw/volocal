import Foundation

/// Drops leading inner-monologue sentences so TTS only gets the spoken reply.
/// Replaces `ReasoningPreambleFilter`, which also deleted legitimate openers.
///
/// - A reply that opens with a lone "Yeah." / "Okay." / "Right." keeps it. Those are held
///   and only dropped if a planning sentence follows ("Okay. I need to respond naturally.").
/// - Only specific planning phrases count as reasoning. "The user guide is on page three."
///   and "the tone of that email" are ordinary speech.
/// - The verbatim prompt-echo check needs a long match, so a short example line the model
///   copies from the prompt is spoken.
struct SpokenPreambleFilter {
    var systemEcho: String = ""
    private var buffer = ""
    private var held = ""
    private var passed = false

    init(systemEcho: String = "") {
        self.systemEcho = systemEcho
    }

    /// Sentence openers that are planning, not speech.
    private static let cotPrefixes = [
        "the user said", "the user is", "the user wants", "the user asked", "the user just",
        "the user seems", "the user greeted", "the user has", "the user asks",
        "the user mentioned", "the user says", "the user wrote",
        "the person said", "the person is", "the human said", "the assistant should",
        "the speaker said", "the speaker is",
        "they said", "this is a greeting", "user just", "user said",
        "okay i need", "ok i need", "alright i need",
    ]

    private static let planningPhrases = [
        "respond naturally",
        "keeping the tone",
        "keep the tone",
        "casual and conversational",
        "conversational tone",
        "inner monologue",
        "the reply should",
        "i need to respond",
        "i should respond",
        "i need to greet",
        "i should greet",
        "i will greet",
        "let me think",
        "let me greet",
        "keep it casual",
        "naturally keeping",
        "user just said",
    ]

    private static let fillerSentences: Set<String> = [
        "okay", "ok", "alright", "so", "well", "right", "hmm", "uh", "um", "yeah", "yes",
    ]

    mutating func push(_ chunk: String) -> String {
        if passed { return chunk }
        buffer += chunk
        while let split = nextSentenceRange() {
            let sentence = String(buffer[split])
            buffer.removeSubrange(split)
            if isFillerOnly(sentence) {
                held += sentence
                continue
            }
            if isChainOfThought(sentence) {
                held = ""
                continue
            }
            passed = true
            let output = held + sentence + buffer
            held = ""
            buffer = ""
            return output
        }
        if buffer.count > 160 {
            if isChainOfThought(buffer) {
                buffer = ""
                held = ""
            } else {
                passed = true
                let output = held + buffer
                held = ""
                buffer = ""
                return output
            }
        }
        return ""
    }

    mutating func flush() -> String {
        if passed {
            let leftover = buffer
            buffer = ""
            return leftover
        }
        let leftover = buffer
        let keptHeld = held
        buffer = ""
        held = ""
        let trimmed = leftover.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return keptHeld }
        if isChainOfThought(leftover) { return "" }
        return keptHeld + leftover
    }

    private func nextSentenceRange() -> Range<String.Index>? {
        guard let idx = buffer.firstIndex(where: { ".!?\n".contains($0) }) else { return nil }
        return buffer.startIndex..<buffer.index(after: idx)
    }

    private func isFillerOnly(_ text: String) -> Bool {
        let lower = text.lowercased()
            .trimmingCharacters(in: .punctuationCharacters.union(.whitespacesAndNewlines))
        return Self.fillerSentences.contains(lower)
    }

    private func isChainOfThought(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return true }
        var lower = trimmed.lowercased()
            .trimmingCharacters(in: .punctuationCharacters.union(.whitespacesAndNewlines))
        guard !lower.isEmpty else { return true }
        if isSystemEcho(lower) { return true }
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
        if lower.count >= 40, sys.contains(lower) { return true }
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
