import Foundation

/// Specification, presets, and personas for NVIDIA PersonaPlex-7B (MLX).
struct PersonaPlexSpec {
    static let repoId = "aufklarer/PersonaPlex-7B-MLX-4bit"
    static let displayName = "NVIDIA PersonaPlex 7B"
    static let shortName = "PersonaPlex 7B"
    static let sizeDescription = "~3.8 GB (4-bit)"
    static let detail = "NVIDIA's full-duplex Speech-to-Speech foundation model (January 2026). Hears vocal emotion and responds simultaneously."

    static var directory: URL {
        let dir = ModelRegistry.modelsDirectory.appendingPathComponent("personaplex", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static var isDownloaded: Bool {
        let fm = FileManager.default
        let dir = directory
        guard fm.fileExists(atPath: dir.path) else { return false }

        if let contents = try? fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: [.fileSizeKey]) {
            let hasWeights = contents.contains { url in
                url.pathExtension.lowercased() == "safetensors" &&
                !url.lastPathComponent.contains(".index.") &&
                ((try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) > 1_500_000_000
            }
            let hasConfig = fm.fileExists(atPath: dir.appendingPathComponent("config.json").path)
            return hasWeights && hasConfig
        }
        return false
    }

    struct VoicePreset: Identifiable, Hashable {
        let id: String
        let displayName: String
        let description: String
        let gender: String
    }

    static let voicePresets: [VoicePreset] = [
        VoicePreset(id: "NATF0", displayName: "Natural Female 1", description: "Warm, balanced tone", gender: "Female"),
        VoicePreset(id: "NATF1", displayName: "Natural Female 2", description: "Clear, conversational", gender: "Female"),
        VoicePreset(id: "NATF2", displayName: "Natural Female 3", description: "Soft, gentle cadence", gender: "Female"),
        VoicePreset(id: "NATF3", displayName: "Natural Female 4", description: "Expressive, animated", gender: "Female"),
        VoicePreset(id: "NATM0", displayName: "Natural Male 1", description: "Deep, resonant tone", gender: "Male"),
        VoicePreset(id: "NATM1", displayName: "Natural Male 2", description: "Friendly, casual", gender: "Male"),
        VoicePreset(id: "NATM2", displayName: "Natural Male 3", description: "Calm, steady presence", gender: "Male"),
        VoicePreset(id: "NATM3", displayName: "Natural Male 4", description: "Crisp, dynamic pace", gender: "Male"),
        VoicePreset(id: "VARF0", displayName: "Variant Female", description: "Rich prosodic range", gender: "Female"),
        VoicePreset(id: "VARM0", displayName: "Variant Male", description: "Characterful expression", gender: "Male"),
    ]

    static let defaultVoice = "NATF0"

    /// Maps PersonaPlex voice preset to matching Mimi acoustic profile
    static func mimiVoice(for presetId: String) -> String {
        switch presetId {
        case "NATF0": return "alba"
        case "NATF1": return "fantine"
        case "NATF2": return "cosette"
        case "NATF3": return "eponine"
        case "NATM0": return "marius"
        case "NATM1": return "jean"
        case "NATM2": return "javert"
        case "NATM3": return "enjolras"
        case "VARF0": return "alba"
        case "VARM0": return "gavroche"
        default: return "alba"
        }
    }

    struct PersonaProfile: Identifiable, Hashable {
        let id: String
        let title: String
        let summary: String
        let systemPrompt: String
    }

    static let defaultPersonas: [PersonaProfile] = [
        PersonaProfile(
            id: "natural",
            title: "Natural Companion",
            summary: "Warm, empathetic partner that converses naturally like a close friend.",
            systemPrompt: "You are a thoughtful, warm conversational partner. You speak naturally with spontaneous inflections, slight laughter when amused, and genuine curiosity. You match the user's emotional state, listening closely and answering concisely."
        ),
        PersonaProfile(
            id: "direct",
            title: "Direct Assistant",
            summary: "Sharp, concise, and focused purely on practical problem-solving.",
            systemPrompt: "You are an intelligent, concise personal assistant. You respond crisply and accurately without unnecessary pleasantries or conversational filler."
        ),
        PersonaProfile(
            id: "witty",
            title: "Witty & Playful",
            summary: "Lively, lighthearted, and humorous with clever commentary.",
            systemPrompt: "You are a witty, playful conversationalist. You enjoy banter, subtle humor, and clever observations, while still remaining helpful and engaging."
        ),
        PersonaProfile(
            id: "calm",
            title: "Calm & Grounded",
            summary: "Soothing, relaxed voice for unwinding or meditative reflection.",
            systemPrompt: "You speak in a calm, soothing, and reassuring manner. Your pacing is relaxed and steady, helping the listener feel centered and at ease."
        ),
    ]

    static let defaultPersonaId = "natural"
}
