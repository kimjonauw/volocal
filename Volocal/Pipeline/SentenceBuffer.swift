import Foundation

/// Accumulates streaming LLM tokens and emits clauses for TTS.
/// The first chunk may be a comma clause or about 10 words. After that, split
/// only on real sentence punctuation so PocketTTS is not fed 40-character
/// crumbs that pause in the middle of a phrase.
final class SentenceBuffer {
    enum ClauseMode {
        /// PocketTTS / Supertonic — whole sentences; one synth job per sentence.
        case streaming
    }

    var mode: ClauseMode = .streaming
    var onSentenceReady: ((String) -> Void)?

    private var buffer = ""
    private var didEmit = false

    /// Last-resort cap when a model never punctuates. Well above a spoken clause.
    private var maxChars: Int { 360 }

    func append(_ token: String) {
        // Models wrap lines; a newline is not a sentence. Splitting on `\n` used
        // a closed range through the next character, so TTS spoke the first
        // letter of the following word as its own utterance.
        let cleaned = token
            .replacingOccurrences(of: "\r\n", with: " ")
            .replacingOccurrences(of: "\n", with: " ")
            .replacingOccurrences(of: "\r", with: " ")
        buffer += cleaned

        while let range = findSentenceBoundary() {
            emit(String(buffer[buffer.startIndex..<range.upperBound]))
            buffer = String(buffer[range.upperBound...]).trimmingCharacters(in: .whitespaces)
        }

        // First audio only: a comma clause, or the first 10 words, so speech
        // starts before the model finishes a long sentence. Later chunks stay
        // on .!? — chopping every phrase is what made mid-sentence pauses.
        if !didEmit, let split = earlyClauseEnd() {
            emit(String(buffer[buffer.startIndex..<split]))
            buffer = String(buffer[split...]).trimmingCharacters(in: .whitespaces)
        }

        if !hasIncompleteExpression(buffer), buffer.count >= maxChars {
            forceSplitAtWordBoundary(limit: maxChars)
        }
    }

    func flush() {
        emit(buffer)
        buffer = ""
        didEmit = false
    }

    func reset() {
        buffer = ""
        didEmit = false
    }

    private func emit(_ raw: String) {
        let sentence = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !sentence.isEmpty else { return }
        didEmit = true
        onSentenceReady?(sentence)
    }

    // MARK: - Private

    private func findSentenceBoundary() -> Range<String.Index>? {
        for i in buffer.indices {
            let char = buffer[i]
            guard char == "." || char == "!" || char == "?" else { continue }
            if isInsideExpression(at: i) { continue }
            if char == "." && isLikelyAbbreviation(before: i) { continue }

            var boundary = buffer.index(after: i)
            // Consume trailing closing quotes or parens: ." !" ?" .) !] etc.
            while boundary < buffer.endIndex {
                let ch = buffer[boundary]
                if ch == "\"" || ch == "\u{201D}" || ch == "'" || ch == "\u{2019}" || ch == ")" || ch == "]" {
                    boundary = buffer.index(after: boundary)
                } else {
                    break
                }
            }

            // Must be followed by whitespace
            if boundary < buffer.endIndex {
                if buffer[boundary].isWhitespace {
                    return i..<boundary
                }
            }
        }
        return nil
    }

    /// Index just past the first speakable chunk.
    /// Emits on a natural major clause boundary (semicolon, colon, em-dash,
    /// or a substantive comma clause of 8+ words) so speech starts without
    /// waiting for long run-on sentences, while avoiding mid-phrase crumbs.
    private func earlyClauseEnd() -> String.Index? {
        guard !hasIncompleteExpression(buffer) else { return nil }
        var words = 0
        var inWord = false
        for i in buffer.indices {
            if isInsideExpression(at: i) { return nil }
            let ch = buffer[i]
            if ch.isWhitespace {
                inWord = false
                continue
            }
            if !inWord {
                words += 1
                inWord = true
            }

            // Semicolon or colon with at least 4 preceding words
            if (ch == ";" || ch == ":") && words >= 4 {
                let next = buffer.index(after: i)
                if next < buffer.endIndex, buffer[next].isWhitespace {
                    return next
                }
            }

            // Em-dash (—) with at least 4 preceding words
            if ch == "—" && words >= 4 {
                let next = buffer.index(after: i)
                return next
            }

            // Substantive introductory clause with a comma (at least 8 words)
            if ch == "," && words >= 8 {
                let next = buffer.index(after: i)
                if next < buffer.endIndex, buffer[next].isWhitespace {
                    return next
                }
            }
        }
        return nil
    }

    /// Force a split at the nearest word boundary when a reply never punctuates.
    private func forceSplitAtWordBoundary(limit: Int) {
        var splitIndex = buffer.startIndex
        for i in buffer.indices {
            if buffer.distance(from: buffer.startIndex, to: i) >= limit { break }
            if buffer[i] == " " {
                splitIndex = i
            }
        }

        if splitIndex > buffer.startIndex {
            emit(String(buffer[buffer.startIndex..<splitIndex]))
            let nextIndex = buffer.index(after: splitIndex)
            buffer = nextIndex < buffer.endIndex ? String(buffer[nextIndex...]) : ""
        } else if buffer.count >= limit {
            emit(buffer)
            buffer = ""
        }
    }

    private func isLikelyAbbreviation(before period: String.Index) -> Bool {
        var start = period
        var letters: [Character] = []
        while start > buffer.startIndex {
            let prev = buffer.index(before: start)
            let ch = buffer[prev]
            if ch.isLetter {
                letters.append(ch)
                start = prev
                if letters.count > 6 { break }
            } else {
                break
            }
        }
        guard !letters.isEmpty else { return false }
        let word = String(letters.reversed()).lowercased()

        // Single uppercase letter initial (e.g. "John F. Kennedy")
        if word.count == 1 {
            let originalChar = buffer[start]
            return originalChar.isUppercase
        }

        // Standard abbreviations ending in period
        let knownAbbreviations: Set<String> = [
            "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
            "vs", "etc", "eg", "ie", "st", "ave", "rd",
            "blvd", "dept", "approx", "gen", "gov", "sgt", "cpt", "col"
        ]
        return knownAbbreviations.contains(word)
    }

    private func hasIncompleteExpression(_ s: String) -> Bool {
        let lower = s.lowercased()
        if lower.contains("[whisper]"), !lower.contains("[/whisper]") { return true }
        if let open = s.lastIndex(of: "["), s[open...].firstIndex(of: "]") == nil { return true }
        return false
    }

    private func isInsideExpression(at i: String.Index) -> Bool {
        let prefix = String(buffer[buffer.startIndex...i]).lowercased()
        let opens = prefix.components(separatedBy: "[whisper]").count - 1
        let closes = prefix.components(separatedBy: "[/whisper]").count - 1
        if opens > closes { return true }
        if let open = prefix.lastIndex(of: "["), prefix[open...].firstIndex(of: "]") == nil { return true }
        return false
    }
}
