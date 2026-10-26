import Foundation

/// Delivery metadata is kept separate from the waveform generator's spoken text.
enum DynamicDelivery {
    struct Clause {
        let text: String
        // Carries a validated label plus an optional bounded direction.
        let style: String
    }

    static let prompt = """


    Voice delivery protocol: Prefix EVERY spoken sentence with one silent tag:
    [delivery=LABEL;direction=SHORT ACTING DIRECTION]Sentence to speak.
    LABEL must be neutral, warm, playful, reassuring, excited, curious, serious,
    sad, or whisper. Choose how YOU should sound, not a diagnosis of the user's emotion.
    Prefer neutral or warm for ordinary replies; keep emotion subtle and contextual.
    Write a concise sentence-specific direction, usually 8-20 words, at most 160 characters.
    Describe tone, emotional intensity, pacing, or emphasis only when useful.
    Do not add factual content, dialogue, speaker changes, or instructions to change words.
    Do not use brackets, semicolons, or newlines inside a direction.
    A direction is optional: [delivery=warm] is valid when the preset is enough.
    Tags are silent metadata and the only exception to spoken-words-only output.
    Do not explain tags, use closing tags, [whisper] blocks, or other stage directions.
    Examples:
    [delivery=reassuring;direction=Soft and steady, emphasizing take your time without sounding pitying]Take your time.
    [delivery=curious;direction=Gentle interest with a slight upward inflection, not an interrogation]What happened next?
    [delivery=excited;direction=Light upbeat enthusiasm, conversational volume, no rushing]That's wonderful news!
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

    private static let maxDirectionCharacters = 160
    private static let tags = try! NSRegularExpression(pattern: #"\[\s*/?delivery\b[^\]]*\]"#, options: [.caseInsensitive])
    private static let unfinished = try! NSRegularExpression(pattern: #"\[\s*/?delivery\b[^\]]*$"#, options: [.caseInsensitive])

    private static func matches(_ text: String) -> [NSTextCheckingResult] {
        tags.matches(in: text, range: NSRange(text.startIndex..., in: text))
    }

    private static func parseValue(_ value: String) -> (label: String, direction: String?) {
        let parts = value.split(separator: ";", maxSplits: 1, omittingEmptySubsequences: false)
        let label = String(parts[0]).trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard instructions[label] != nil else { return ("neutral", nil) }
        guard parts.count == 2 else { return (label, nil) }
        let field = String(parts[1]).trimmingCharacters(in: .whitespacesAndNewlines)
        guard field.lowercased().hasPrefix("direction=") else { return (label, nil) }
        let raw = String(field.dropFirst("direction=".count))
        guard !raw.contains("["), !raw.contains("]"), !raw.contains(";") else { return (label, nil) }
        let clean = raw.split(whereSeparator: { $0.isWhitespace }).joined(separator: " ")
        let bounded = String(clean.prefix(maxDirectionCharacters)).trimmingCharacters(in: .whitespacesAndNewlines)
        return (label, bounded.isEmpty ? nil : bounded)
    }

    private static func style(_ tag: String) -> String {
        guard let equal = tag.firstIndex(of: "=") else { return "neutral" }
        var value = String(tag[tag.index(after: equal)...])
        if value.hasSuffix("]") { value.removeLast() }
        let parsed = parseValue(value)
        if let direction = parsed.direction {
            return parsed.label + ";direction=" + direction
        }
        return parsed.label
    }

    static func instruction(for value: String) -> String {
        let parsed = parseValue(value)
        let base = instructions[parsed.label] ?? instructions["neutral"]!
        guard let direction = parsed.direction else { return base }
        return base + " Sentence-specific delivery: " + direction
            + ". Keep the supplied spoken words unchanged."
    }

    static func strip(_ text: String) -> String {
        let complete = tags.stringByReplacingMatches(in: text, range: NSRange(text.startIndex..., in: text), withTemplate: "")
        return unfinished.stringByReplacingMatches(in: complete, range: NSRange(complete.startIndex..., in: complete), withTemplate: "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Preserve both label and direction over clause splits, never across turns.
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
