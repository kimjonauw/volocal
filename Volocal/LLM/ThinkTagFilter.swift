import Foundation

/// Strips `<think>…</think>` (and leftover tags) from a streaming token feed
/// so reasoning models do not get spoken by TTS.
struct ThinkTagFilter {
    private var buffer = ""
    private var insideThink = false

    private let open = "<think>"
    private let close = "</think>"

    var isInsideThink: Bool { insideThink }

    mutating func push(_ chunk: String) -> String {
        buffer += chunk
        var output = ""

        while !buffer.isEmpty {
            if insideThink {
                if let end = buffer.range(of: close, options: .caseInsensitive) {
                    buffer.removeSubrange(buffer.startIndex..<end.upperBound)
                    if buffer.first == "\n" { buffer.removeFirst() }
                    insideThink = false
                    continue
                }
                if buffer.count > close.count {
                    buffer = String(buffer.suffix(close.count))
                }
                break
            }

            if let start = buffer.range(of: open, options: .caseInsensitive) {
                output += buffer[buffer.startIndex..<start.lowerBound]
                buffer.removeSubrange(buffer.startIndex..<start.upperBound)
                insideThink = true
                continue
            }

            if let hold = incompleteTagSuffixCount() {
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
        if insideThink {
            buffer = ""
            insideThink = false
            return ""
        }
        let leftover = buffer
        buffer = ""
        return leftover
    }

    /// Number of trailing characters that might still complete `<think>` / `</think>`.
    private func incompleteTagSuffixCount() -> Int? {
        let lower = buffer.lowercased()
        for tag in [open, close] {
            let maxN = min(tag.count - 1, lower.count)
            guard maxN >= 1 else { continue }
            for n in stride(from: maxN, through: 1, by: -1) {
                if lower.hasSuffix(String(tag.prefix(n))) {
                    return n
                }
            }
        }
        return nil
    }
}
