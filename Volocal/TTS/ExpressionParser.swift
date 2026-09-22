import Foundation

enum NonverbalKind: String, Equatable {
    case laugh
    case gasp
    case sigh
}

enum SpeechSegment: Equatable {
    case speech(String)
    case whisper(String)
    case effect(NonverbalKind)
}

/// Splits model text into spoken clauses and expression cues.
/// PocketTTS/Supertonic play [laugh]/[gasp]/[sigh] as short cues.
enum ExpressionParser {
    static func parse(_ text: String, passthroughEffects: Bool = false) -> [SpeechSegment] {
        var remaining = passthroughEffects ? normalizeNeuralTags(text) : text
        var segments: [SpeechSegment] = []

        while !remaining.isEmpty {
            if let match = firstTag(in: remaining, passthroughEffects: passthroughEffects) {
                let before = String(remaining[..<match.range.lowerBound])
                pushSpeech(before, onto: &segments)
                switch match {
                case .effect(_, let kind):
                    segments.append(.effect(kind))
                case .whisper(_, let inner):
                    let whispered = inner.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !whispered.isEmpty {
                        segments.append(.whisper(whispered))
                    }
                }
                remaining = String(remaining[match.range.upperBound...])
                continue
            }
            pushSpeech(remaining, onto: &segments)
            break
        }

        return segments.filter { segment in
            switch segment {
            case .speech(let t), .whisper(let t):
                return !t.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            case .effect:
                return true
            }
        }
    }

    static func spokenPlain(_ text: String) -> String {
        parse(text).compactMap { segment in
            switch segment {
            case .speech(let t), .whisper(let t): return t
            case .effect: return nil
            }
        }
        .joined(separator: " ")
        .replacingOccurrences(of: "  ", with: " ")
        .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private enum TagMatch {
        case effect(Range<String.Index>, NonverbalKind)
        case whisper(Range<String.Index>, String)

        var range: Range<String.Index> {
            switch self {
            case .effect(let range, _), .whisper(let range, _): return range
            }
        }
    }

    private static func firstTag(in text: String, passthroughEffects: Bool = false) -> TagMatch? {
        var best: TagMatch?

        func consider(_ candidate: TagMatch) {
            if let current = best {
                if candidate.range.lowerBound < current.range.lowerBound {
                    best = candidate
                } else if candidate.range.lowerBound == current.range.lowerBound,
                          candidate.range.upperBound > current.range.upperBound {
                    best = candidate
                }
            } else {
                best = candidate
            }
        }

        // Nano speaks laugh/chuckle/sigh/cough. Gasp is still a short cue.
        let pairs: [(String, NonverbalKind)] = passthroughEffects
            ? [
                ("[gasp]", .gasp), ("[gasps]", .gasp), ("(gasp)", .gasp), ("(gasps)", .gasp),
            ]
            : [
                ("[laugh]", .laugh), ("[laughs]", .laugh), ("[laughter]", .laugh), ("[chuckle]", .laugh),
                ("(laughs)", .laugh), ("(laugh)", .laugh),
                ("[gasp]", .gasp), ("[gasps]", .gasp), ("(gasp)", .gasp), ("(gasps)", .gasp),
                ("[sigh]", .sigh), ("[sighs]", .sigh), ("(sigh)", .sigh), ("(sighs)", .sigh),
            ]
        for (tag, kind) in pairs {
            if let range = text.range(of: tag, options: .caseInsensitive) {
                consider(.effect(range, kind))
            }
        }

        if let open = text.range(of: "[whisper]", options: .caseInsensitive) {
            if let close = text.range(
                of: "[/whisper]",
                options: .caseInsensitive,
                range: open.upperBound..<text.endIndex
            ) {
                consider(.whisper(open.lowerBound..<close.upperBound, String(text[open.upperBound..<close.lowerBound])))
            }
        }

        return best
    }

    static func stripNeuralTags(_ text: String) -> String {
        var t = text
        let tags = [
            "[laugh]", "[laughs]", "[laughter]", "[chuckle]", "[cough]",
            "[gasp]", "[gasps]", "[sigh]", "[sighs]",
            "(laughs)", "(laugh)", "(gasp)", "(gasps)", "(sigh)", "(sighs)",
            "[whisper]", "[/whisper]",
        ]
        for tag in tags {
            t = t.replacingOccurrences(of: tag, with: "", options: .caseInsensitive)
        }
        return t.replacingOccurrences(of: "  ", with: " ")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func normalizeNeuralTags(_ text: String) -> String {
        var t = text
        let replacements = [
            "(laughs)": "[laugh]", "(laugh)": "[laugh]", "[laughs]": "[laugh]", "[laughter]": "[laugh]",
            "(gasp)": "[gasp]", "(gasps)": "[gasp]", "[gasps]": "[gasp]",
            "(sigh)": "[sigh]", "(sighs)": "[sigh]", "[sighs]": "[sigh]",
        ]
        for (from, to) in replacements {
            t = t.replacingOccurrences(of: from, with: to, options: .caseInsensitive)
        }
        return t
    }

    private static func pushSpeech(_ raw: String, onto segments: inout [SpeechSegment]) {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        segments.append(.speech(trimmed))
    }
}
