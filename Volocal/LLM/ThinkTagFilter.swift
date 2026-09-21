import Foundation

/// Strips `<think>…</think>` (and leftover tags) from a streaming token feed
/// so reasoning models do not get spoken by TTS.
struct ThinkTagFilter {
    private var buffer = ""
    private var insideThink = false

    mutating func push(_ chunk: String) -> String {
        buffer += chunk
        var output = ""

        while !buffer.isEmpty {
            if insideThink {
                if let end = buffer.range(of: "</think>", options: .caseInsensitive) {
                    buffer.removeSubrange(buffer.startIndex..<end.upperBound)
                    if buffer.first == "\n" { buffer.removeFirst() }
                    insideThink = false
                    continue
                }
                if buffer.count > 16 {
                    buffer = String(buffer.suffix(16))
                }
                break
            }

            if let start = buffer.range(of: "<think>", options: .caseInsensitive) {
                output += buffer[buffer.startIndex..<start.lowerBound]
                buffer.removeSubrange(buffer.startIndex..<start.upperBound)
                insideThink = true
                continue
            }

            if let partial = partialTagPrefix() {
                if partial == buffer {
                    break
                }
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

    private func partialTagPrefix() -> String? {
        let open = "<think>"
        let close = "</think>"
        for tag in [open, close] {
            for n in 1..<tag.count {
                if buffer.lowercased().hasSuffix(String(tag.prefix(n)).lowercased()) {
                    return String(buffer.suffix(n))
                }
            }
        }
        return nil
    }
}
