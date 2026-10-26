import Foundation

/// Delivery is metadata, never text sent to the waveform generator.
enum DynamicDelivery {
    struct Clause {
        let text: String
        let style: String
    }

    static let prompt = """


    Voice delivery protocol: Prefix EVERY spoken sentence with exactly one tag:
    [delivery=neutral], [delivery=warm], [delivery=playful], [delivery=reassuring],
    [delivery=excited], [delivery=curious], [delivery=serious], [delivery=sad],
    or [delivery=whisper]. Choose a delivery that fits the sentence and context.
    Prefer neutral or warm for ordinary replies. Use emotion sparingly and naturally.
    These tags are silent metadata and are the only exception to spoken-words-only output.
    Do not explain the tags. Do not use closing tags, other stage directions,
    [whisper] blocks, or effect tags in this mode. Example:
    [delivery=reassuring]Take your time. [delivery=curious]What happened next?
    """

    private static let instructions: [String: String] = [
        "neutral": "Speak naturally and conversationally with a neutral, clear delivery. Do not shout.",
        "warm": "Speak warmly and kindly, with a relaxed, friendly conversational tone. Do not shout.",
        "playful": "Speak with gentle playfulness and a light smile in your voice. Stay natural; do not shout.",
        "reassuring": "Speak gently and reassuringly, with calm confidence and a caring tone. Do not shout.",
        "excited": "Speak with upbeat enthusiasm and lively intonation, without shouting or rushing.",
        "curious": "Speak with genuine curiosity and an interested, inviting tone. Do not shout.",
        "serious": "Speak thoughtfully and seriously, with steady pacing and a composed tone. Do not shout.",
        "sad": "Speak softly with subdued, sympathetic sadness. Remain clear and natural; do not exaggerate.",
        "whisper": "Speak softly in a gentle whisper without shouting."
    ]

    // Consume unknown delivery values too, so unsupported metadata is not spoken.
    private static let tags = try! NSRegularExpression(pattern: #"\[\s*/?delivery\b[^\]]*\]"#, options: [.caseInsensitive])
    private static let unfinished = try! NSRegularExpression(pattern: #"\[\s*/?delivery\b[^\]]*$"#, options: [.caseInsensitive])

    private static func matches(_ text: String) -> [NSTextCheckingResult] {
        tags.matches(in: text, range: NSRange(text.startIndex..., in: text))
    }

    private static func style(_ tag: String) -> String {
        guard let equal = tag.firstIndex(of: "=") else { return "neutral" }
        let raw = tag[tag.index(after: equal)...]
            .trimmingCharacters(in: CharacterSet.whitespacesAndNewlines.union(CharacterSet(charactersIn: "]\"'")))
            .lowercased()
        return instructions[raw] == nil ? "neutral" : raw
    }

    static func instruction(for value: String) -> String {
        instructions[value] ?? instructions["neutral"]!
    }

    static func strip(_ text: String) -> String {
        let complete = tags.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
        return unfinished.stringByReplacingMatches(in: complete, range: NSRange(complete.startIndex..., in: complete), withTemplate: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Carry a sentence's tag over early-clause and hard-length splits, never across turns.
    static func carry(_ text: String, style previous: inout String?) -> String {
        let found = matches(text)
        let inherited = previous
        for match in found {
            if let range = Range(match.range, in: text) {
                previous = style(String(text[range]))
            }
        }
        if let inherited {
            if let first = found.first, first.range.location == 0 { return text }
            return "[delivery=\(inherited)]" + text
        }
        return text
    }

    static func clauses(_ text: String) -> [Clause] {
        var result: [Clause] = []
        var cursor = text.startIndex
        var current = "neutral"
        func append(_ value: String) {
            let clean = strip(value)
            for clause in QwenConversationBackend.clauses(clean) {
                result.append(Clause(text: clause, style: current))
            }
        }
        for match in matches(text) {
            guard let range = Range(match.range, in: text) else { continue }
            append(String(text[cursor..<range.lowerBound]))
            current = style(String(text[range]))
            cursor = range.upperBound
        }
        append(String(text[cursor...]))
        return result
    }
}
