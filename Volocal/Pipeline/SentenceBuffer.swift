import Foundation

/// Accumulates streaming LLM tokens and emits clauses for TTS as soon as possible.
/// Prefers `.!?` / `:;` boundaries; the first chunk is forced out around 40 characters
/// so voice can start before a full sentence exists.
final class SentenceBuffer {
    private var buffer = ""
    private var didEmit = false

    var onSentenceReady: ((String) -> Void)?

    private let maxChars = 160
    private let firstChunkChars = 40

    func append(_ token: String) {
        buffer += token

        while let range = findSentenceBoundary() {
            emit(String(buffer[buffer.startIndex...range.upperBound]))
            let nextIndex = buffer.index(after: range.upperBound)
            buffer = nextIndex < buffer.endIndex
                ? String(buffer[nextIndex...]).trimmingCharacters(in: .whitespaces)
                : ""
        }

        let limit = didEmit ? maxChars : firstChunkChars
        if buffer.count >= limit {
            forceSplitAtWordBoundary(limit: limit)
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
            let nextChar = buffer[nextIndex]

            // Primary boundaries: .!?
            if char == "!" || char == "?" {
                if nextChar == " " || nextChar == "\n" || nextChar == "\"" || nextChar == "\u{201D}" {
                    return i..<nextIndex
                }
            }

            // Period: only split if next char is uppercase (avoids Dr., 3.14, U.S.)
            if char == "." {
                if nextChar == " " || nextChar == "\n" {
                    // Check if char after space is uppercase
                    let afterSpace = buffer.index(after: nextIndex)
                    if afterSpace < buffer.endIndex {
                        let charAfter = buffer[afterSpace]
                        if charAfter.isUppercase {
                            return i..<nextIndex
                        }
                    }
                    // Also split on period + quote
                } else if nextChar == "\"" || nextChar == "\u{201D}" {
                    return i..<nextIndex
                }
            }

            // Secondary boundaries: : and ; (clause boundaries)
            if char == ":" || char == ";" {
                if nextChar == " " || nextChar == "\n" {
                    return i..<nextIndex
                }
            }
        }
        return nil
    }

    /// Force a split at the nearest word boundary when buffer is too long.
    private func forceSplitAtWordBoundary(limit: Int) {
        var splitIndex = buffer.startIndex
        for i in buffer.indices {
            if buffer.distance(from: buffer.startIndex, to: i) >= limit { break }
            if buffer[i] == " " || buffer[i] == "," {
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
}
