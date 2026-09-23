import Foundation

/// Energy-based turn taking. Parakeet EOU will not fire if HVAC / rustle keeps
/// the decoder "busy"; after real speech, a stretch of quiet ends the utterance.
/// A finished phrase in a quiet room ends after ~0.28s. An unfinished phrase,
/// or a noisy room, still waits ~0.8s.
final class MicTurnDetector: @unchecked Sendable {
    private let lock = NSLock()
    private var noiseFloor: Float = 0.008
    private var lastSpeechAt: CFAbsoluteTime = 0
    private var inUtterance = false

    /// `eager` is true when the partial transcript already looks like a finished thought.
    func observe(rms: Float, eager: Bool) -> (feedAsr: Bool, endTurn: Bool) {
        lock.lock()
        defer { lock.unlock() }
        let now = CFAbsoluteTimeGetCurrent()
        let speechThresh = max(Float(0.010), noiseFloor * 2.2)

        if rms >= speechThresh {
            lastSpeechAt = now
            inUtterance = true
            return (true, false)
        }

        noiseFloor = min(Float(0.028), max(Float(0.0035), noiseFloor * 0.96 + rms * 0.04))
        let quietFor = now - lastSpeechAt
        let noisy = noiseFloor > 0.018
        let endAfter: CFAbsoluteTime = (eager && !noisy) ? 0.28 : 0.80
        let feedUntil: CFAbsoluteTime = (eager && !noisy) ? endAfter : 0.40
        if inUtterance && lastSpeechAt > 0 && quietFor < feedUntil {
            return (true, false)
        }
        if inUtterance && lastSpeechAt > 0 && quietFor >= endAfter {
            inUtterance = false
            return (false, true)
        }
        return (false, false)
    }

    func resetUtterance() {
        lock.lock()
        inUtterance = false
        lastSpeechAt = 0
        lock.unlock()
    }
}

/// Whether a partial transcript is a finished thought. Dangling words ("and", "um", "the")
/// keep the longer quiet wait so a mid-sentence pause is not answered.
enum UtteranceReadiness {
    private static let dangling: Set<String> = [
        "and", "or", "but", "so", "to", "the", "a", "an", "of", "in", "on", "at",
        "for", "with", "from", "um", "uh", "uhh", "hmm", "like", "if", "when",
        "because", "about", "into", "that", "this", "my", "your",
    ]

    static func looksComplete(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        if trimmed.hasSuffix(",") || trimmed.hasSuffix("...") || trimmed.hasSuffix("…") {
            return false
        }
        let words = trimmed.split(whereSeparator: \.isWhitespace)
        guard let lastRaw = words.last else { return false }
        let last = String(lastRaw).lowercased().trimmingCharacters(in: .punctuationCharacters)
        if dangling.contains(last) { return false }
        if let mark = trimmed.last, ".?!".contains(mark) {
            return words.count >= 1 && last.count >= 2
        }
        if words.count >= 3 { return true }
        if words.count == 2 { return true }
        return last.count >= 4
    }
}
