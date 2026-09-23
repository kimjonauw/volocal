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
            let nextIndex = buffer.index(after: i)
            guard nextIndex < buffer.endIndex else { continue }
            if isInsideExpression(at: i) { continue }
            let nextChar = buffer[nextIndex]

            if char == "!" || char == "?" {
                if nextChar == " " || nextChar == "\n" || nextChar == "\"" || nextChar == "\u{201D}" {
                    return i..<nextIndex
                }
            }

            if char == "." {
                if isLikelyAbbreviation(before: i) { continue }
                if nextChar == " " || nextChar == "\n" {
                    return i..<nextIndex
                }
                if nextChar == "\"" || nextChar == "\u{201D}" {
                    return i..<nextIndex
                }
            }
        }
        return nil
    }

    /// Index just past the first speakable chunk. Nil until a comma clause of
    /// 4+ words, or an 11th word proves the first 10 are complete.
    private func earlyClauseEnd() -> String.Index? {
        guard !hasIncompleteExpression(buffer) else { return nil }
        var words = 0
        var inWord = false
        var tenthWordEnd: String.Index?
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
            if (ch == "," || ch == ";") && words >= 4 {
                let next = buffer.index(after: i)
                if next < buffer.endIndex, buffer[next].isWhitespace {
                    return next
                }
            }
            if words == 10 {
                tenthWordEnd = buffer.index(after: i)
            } else if words > 10, let end = tenthWordEnd {
                return end
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
        var letters = 0
        while start > buffer.startIndex {
            let prev = buffer.index(before: start)
            let ch = buffer[prev]
            if ch.isLetter {
                letters += 1
                start = prev
                if letters > 3 { return false }
            } else {
                break
            }
        }
        return letters == 1 || letters == 2
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
