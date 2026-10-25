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
    /// Start of an unbroken run of above-threshold frames. Speech dips between words;
    /// a fan or music does not. A long run is treated as ambient and the floor rises.
    private var loudRunStart: CFAbsoluteTime = 0

    /// `eager` is true when the partial transcript already looks like a finished thought.
    /// `quietEnd` overrides how long the quiet must last before the turn ends (used by
    /// speech-to-speech, which has no transcript to judge completeness from).
    func observe(
        rms: Float,
        eager: Bool,
        quietEnd: CFAbsoluteTime? = nil
    ) -> (feedAsr: Bool, endTurn: Bool) {
        lock.lock()
        defer { lock.unlock() }
        let now = CFAbsoluteTimeGetCurrent()
        let speechThresh = max(Float(0.010), noiseFloor * 2.2)

        if rms >= speechThresh {
            if loudRunStart == 0 { loudRunStart = now }
            if now - loudRunStart > 10.0 {
                let risen = min(Float(0.020), noiseFloor * 0.99 + rms * 0.01)
                noiseFloor = max(noiseFloor, risen)
            }
            lastSpeechAt = now
            inUtterance = true
            return (true, false)
        }

        loudRunStart = 0
        noiseFloor = min(Float(0.028), max(Float(0.0035), noiseFloor * 0.96 + rms * 0.04))
        let quietFor = now - lastSpeechAt
        let noisy = noiseFloor > 0.018
        let endAfter: CFAbsoluteTime = quietEnd ?? ((eager && !noisy) ? 0.28 : 0.80)
        let feedUntil: CFAbsoluteTime = min(endAfter, (eager && !noisy) ? endAfter : 0.40)
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
        loudRunStart = 0
        lock.unlock()
    }
}

/// Whether a partial transcript is a finished thought. Dangling words ("and", "um", "the",
/// "is", "want") keep the longer quiet wait so a mid-sentence pause is not answered.
enum UtteranceReadiness {
    /// Never a finished thought, even with a period after them.
    private static let fillers: Set<String> = ["um", "uh", "uhh", "hmm", "like"]

    /// Words a sentence cannot sensibly end on without more to come.
    private static let dangling: Set<String> = fillers.union(Set<String>([
        "and", "or", "but", "so", "to", "the", "a", "an", "of", "in", "on", "at",
        "for", "with", "from", "if", "when", "because", "about", "into", "that",
        "this", "my", "your",
        "i", "i'm", "i'll", "i've", "i'd", "we", "we're", "he", "she", "they", "they're",
        "he's", "she's", "it's", "that's", "there's", "what's", "you're",
        "is", "are", "was", "were", "am", "be", "been", "have", "has", "had",
        "can", "could", "will", "would", "should", "might", "may", "must",
        "want", "need", "wanna", "gonna", "going", "think", "know", "mean", "trying",
        "not", "just", "maybe", "also", "then", "than", "as", "by",
        "what", "how", "why", "who", "where", "which",
        "their", "his", "its", "our", "some", "any", "very", "really", "more", "most",
    ]))

    static func looksComplete(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        if trimmed.hasSuffix(",") || trimmed.hasSuffix("...") || trimmed.hasSuffix("…") {
            return false
        }
        let words = trimmed.split(whereSeparator: \.isWhitespace)
        guard let lastRaw = words.last else { return false }
        let last = String(lastRaw).lowercased().trimmingCharacters(in: .punctuationCharacters)
        if fillers.contains(last) { return false }
        // A real full stop, question mark, or bang ends the thought ("I don't know.").
        if let mark = trimmed.last, ".?!".contains(mark) {
            return last.count >= 2
        }
        if dangling.contains(last) { return false }
        if words.count >= 3 { return true }
        if words.count == 2 { return true }
        return last.count >= 4
    }
}
