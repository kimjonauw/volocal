import Foundation

/// Persona conversation mode: personality profiles plus PocketTTS 100M voices.
/// Real NVIDIA PersonaPlex is audio-to-audio Moshi and is not available on this
/// iPhone path — these presets are the natural spoken stand-in.
struct PersonaPlexSpec {
    static let displayName = "Persona conversation"
    static let shortName = "Persona"
    static let detail = "Personality profile plus a natural PocketTTS 100M voice over your language model. True PersonaPlex Moshi speech-to-speech is not available on this iPhone."

    /// Kept so old downloads can still be deleted from disk.
    static let repoId = "aufklarer/PersonaPlex-7B-MLX-4bit"

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
            return contents.contains {
                $0.pathExtension.lowercased() == "safetensors"
                    && ((try? $0.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0) > 100_000_000
            }
        }
        return false
    }

    struct VoicePreset: Identifiable, Hashable {
        let id: String
        let displayName: String
        let description: String
        let gender: String
        /// PocketTTS 100M speaker id. Must exist in TTSEngine.pocketTts.voiceNames.
        let pocketVoice: String
    }

    static let voicePresets: [VoicePreset] = [
        VoicePreset(id: "NATF0", displayName: "Warm Woman", description: "Soft, conversational", gender: "Female", pocketVoice: "fantine"),
        VoicePreset(id: "NATF1", displayName: "Clear Woman", description: "Bright and direct", gender: "Female", pocketVoice: "cosette"),
        VoicePreset(id: "NATF2", displayName: "Gentle Woman", description: "Quiet, unhurried", gender: "Female", pocketVoice: "alba"),
        VoicePreset(id: "NATF3", displayName: "Expressive Woman", description: "Animated and lively", gender: "Female", pocketVoice: "eponine"),
        VoicePreset(id: "NATM0", displayName: "Warm Man", description: "Friendly, mid tone", gender: "Male", pocketVoice: "marius"),
        VoicePreset(id: "NATM1", displayName: "Casual Man", description: "Easy and open", gender: "Male", pocketVoice: "jean"),
        VoicePreset(id: "NATM2", displayName: "Steady Man", description: "Calm, grounded", gender: "Male", pocketVoice: "javert"),
        VoicePreset(id: "NATM3", displayName: "Bright Man", description: "Crisp pace", gender: "Male", pocketVoice: "jean"),
        VoicePreset(id: "VARF0", displayName: "Soft Woman", description: "Light and airy", gender: "Female", pocketVoice: "azelma"),
        VoicePreset(id: "VARM0", displayName: "Deep Man", description: "Lower, deliberate", gender: "Male", pocketVoice: "javert"),
    ]

    static let defaultVoice = "NATF0"

    /// PocketTTS speaker for a persona preset. Always a real 100M voice name.
    static func pocketVoice(for presetId: String) -> String {
        voicePresets.first { $0.id == presetId }?.pocketVoice
            ?? TTSEngine.pocketTts.defaultVoice
    }

    /// Older call sites used this name for the PocketTTS mapping.
    static func mimiVoice(for presetId: String) -> String {
        pocketVoice(for: presetId)
    }

    struct PersonaProfile: Identifiable, Hashable {
        let id: String
        let title: String
        let summary: String
        let systemPrompt: String
    }

    /// Spoken-first prompts: short turns, natural cadence, no essay answers.
    private static let spokenRules = """
    Reply as spoken conversation, not as written text. Keep answers to one or two short sentences unless the user asks for more. Use everyday words. Do not list steps, use markdown, or narrate your reasoning. Prefer contractions. Sound like a person on a phone call.
    """

    static let defaultPersonas: [PersonaProfile] = [
        PersonaProfile(
            id: "natural",
            title: "Natural Companion",
            summary: "Warm friend on a call — curious, brief, human.",
            systemPrompt: """
            You are a warm conversational partner on a voice call. Match the user's energy. Ask a short follow-up when it feels natural.
            \(spokenRules)
            """
        ),
        PersonaProfile(
            id: "direct",
            title: "Direct Assistant",
            summary: "Crisp answers with no filler.",
            systemPrompt: """
            You are a sharp personal assistant on a voice call. Lead with the answer. Skip pleasantries.
            \(spokenRules)
            """
        ),
        PersonaProfile(
            id: "witty",
            title: "Witty & Playful",
            summary: "Light banter without overdoing it.",
            systemPrompt: """
            You are a playful conversationalist on a voice call. One light joke or aside is enough — then be helpful.
            \(spokenRules)
            """
        ),
        PersonaProfile(
            id: "calm",
            title: "Calm & Grounded",
            summary: "Slow, steady, reassuring tone.",
            systemPrompt: """
            You are a calm presence on a voice call. Speak slowly and reassuringly. Leave space; do not rush.
            \(spokenRules)
            """
        ),
    ]

    static let defaultPersonaId = "natural"
}
