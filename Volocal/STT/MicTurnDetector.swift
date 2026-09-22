import Foundation

/// Energy-based turn taking. Parakeet EOU will not fire if HVAC / rustle keeps
/// the decoder "busy"; after real speech, a stretch of quiet ends the utterance.
final class MicTurnDetector: @unchecked Sendable {
    private let lock = NSLock()
    private var noiseFloor: Float = 0.008
    private var lastSpeechAt: CFAbsoluteTime = 0
    private var inUtterance = false

    func observe(rms: Float) -> (feedAsr: Bool, endTurn: Bool) {
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
        if inUtterance && lastSpeechAt > 0 && quietFor < 0.40 {
            return (true, false)
        }
        if inUtterance && lastSpeechAt > 0 && quietFor >= 0.80 {
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
