import Foundation

/// Short 24 kHz mono cues for [laugh] / [gasp] / [sigh]. Not a neural laugh —
/// PocketTTS cannot render those, and a Dia/Orpheus GGUF will not fit next to the LLM.
enum NonverbalSynth {
    static let sampleRate: Double = 24_000

    static func render(_ kind: NonverbalKind) -> [Float] {
        switch kind {
        case .laugh: return laugh()
        case .gasp: return gasp()
        case .sigh: return sigh()
        }
    }

    private static func laugh() -> [Float] {
        // Soft breathy chuckle pulses instead of synthetic sine tones
        var out: [Float] = []
        var seed: UInt64 = 0xC0FFEE
        let pulses = 3
        for i in 0..<pulses {
            let n = Int(sampleRate * 0.08)
            var lp: Double = 0
            for t in 0..<n {
                let env = sin(Double.pi * Double(t) / Double(max(n, 1)))
                let white = nextNoise(&seed)
                lp = lp * 0.85 + white * 0.15
                out.append(Float(lp * env * 0.18))
            }
            if i < pulses - 1 {
                out.append(contentsOf: [Float](repeating: 0, count: Int(sampleRate * 0.04)))
            }
        }
        return out
    }

    private static func gasp() -> [Float] {
        let n = Int(sampleRate * 0.22)
        var seed: UInt64 = 0xA11
        var prev: Float = 0
        var out = [Float](repeating: 0, count: n)
        for t in 0..<n {
            let x = Double(t) / Double(max(n, 1))
            let env = pow(x, 0.45) * (1 - x)
            let white = nextNoise(&seed)
            let hp = Float(white) - prev
            prev = Float(white)
            out[t] = hp * Float(env) * 1.8
        }
        return out
    }

    private static func sigh() -> [Float] {
        let n = Int(sampleRate * 0.55)
        var seed: UInt64 = 0x51C4
        var lp: Double = 0
        var out = [Float](repeating: 0, count: n)
        for t in 0..<n {
            let x = Double(t) / Double(max(n, 1))
            let env = (1 - x) * (1 - x) * (0.35 + 0.65 * (1 - x))
            lp = lp * 0.92 + nextNoise(&seed) * 0.08
            out[t] = Float(lp * env * 0.9)
        }
        return out
    }

    private static func nextNoise(_ seed: inout UInt64) -> Double {
        seed = seed &* 6_364_136_223_846_793_005 &+ 1
        let v = Double((seed >> 33) & 0xFFFFFFFF) / Double(UInt32.max)
        return v * 2 - 1
    }
}
