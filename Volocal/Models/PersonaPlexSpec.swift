import Foundation

/// Persona conversation mode: personality profiles paired with high-fidelity voices
/// (Supertonic-3 CoreML flow matching or PocketTTS 100M) over your language model.
struct PersonaPlexSpec {
    static let displayName = "Persona conversation"
    static let shortName = "Persona"
    static let detail = "Personality profile paired with natural voice synthesis (Supertonic-3 CoreML or PocketTTS 100M) over your local language model."

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
        /// PocketTTS speaker id.
        let pocketVoice: String
        /// Supertonic-3 speaker id (F1..F5, M1..M5).
        let supertonicVoice: String
    }

    static let voicePresets: [VoicePreset] = [
        VoicePreset(id: "NATF0", displayName: "Warm Woman", description: "Soft, conversational & engaging", gender: "Female", pocketVoice: "fantine", supertonicVoice: "F1"),
        VoicePreset(id: "NATF1", displayName: "Clear Woman", description: "Bright, direct & articulate", gender: "Female", pocketVoice: "cosette", supertonicVoice: "F2"),
        VoicePreset(id: "NATF2", displayName: "Gentle Woman", description: "Quiet, calm & unhurried", gender: "Female", pocketVoice: "alba", supertonicVoice: "F3"),
        VoicePreset(id: "NATF3", displayName: "Expressive Woman", description: "Animated, lively & upbeat", gender: "Female", pocketVoice: "eponine", supertonicVoice: "F4"),
        VoicePreset(id: "VARF0", displayName: "Soft Woman", description: "Light, airy & delicate", gender: "Female", pocketVoice: "azelma", supertonicVoice: "F5"),
        VoicePreset(id: "NATM0", displayName: "Warm Man", description: "Friendly, natural & mid-tone", gender: "Male", pocketVoice: "marius", supertonicVoice: "M1"),
        VoicePreset(id: "NATM1", displayName: "Casual Man", description: "Easygoing, open & relaxed", gender: "Male", pocketVoice: "jean", supertonicVoice: "M2"),
        VoicePreset(id: "NATM2", displayName: "Steady Man", description: "Calm, grounded & reassuring", gender: "Male", pocketVoice: "javert", supertonicVoice: "M3"),
        VoicePreset(id: "NATM3", displayName: "Bright Man", description: "Crisp, dynamic & confident", gender: "Male", pocketVoice: "marius", supertonicVoice: "M5"),
        VoicePreset(id: "VARM0", displayName: "Deep Man", description: "Low, deliberate & resonant", gender: "Male", pocketVoice: "javert", supertonicVoice: "M4"),
    ]

    static let defaultVoice = "NATF0"

    /// Resolve voice for preset and active engine.
    static func voice(for presetId: String, engine: TTSEngine) -> String {
        guard let preset = voicePresets.first(where: { $0.id == presetId }) else {
            return engine.defaultVoice
        }
        switch engine {
        case .supertonic3:
            return preset.supertonicVoice
        case .pocketTts, .pocketTtsSmall:
            return preset.pocketVoice
        }
    }

    /// PocketTTS speaker for a persona preset. Always a real 100M voice name.
    static func pocketVoice(for presetId: String) -> String {
        voice(for: presetId, engine: .pocketTts)
    }

    /// Supertonic-3 speaker for a persona preset.
    static func supertonicVoice(for presetId: String) -> String {
        voice(for: presetId, engine: .supertonic3)
    }

    /// Backward compatibility for older call sites.
    static func mimiVoice(for presetId: String) -> String {
        pocketVoice(for: presetId)
    }

    struct PersonaProfile: Identifiable, Hashable {
        let id: String
        let title: String
        let summary: String
        let systemPrompt: String
    }

    /// Spoken-first rules: written for the ear, not the eye. The TTS reads exactly what the
    /// model writes, so rhythm, reactions and pauses have to be in the text.
    /// Quoted phrases stay under 16 characters: the reply filter drops a first sentence
    /// that is a long verbatim copy of the prompt, and small models copy examples.
    private static let spokenRules = """
    CRITICAL SPOKEN CONVERSATION RULES:
    1. You are talking out loud on a phone call. Write exactly what a person would say, never what they would type.
    2. Keep each turn to one or two short sentences. Vary the length: sometimes just a few words, sometimes one longer sentence.
    3. Always use contractions ("I'm", "it's", "you're", "don't", "that's").
    4. React first, then answer. Open with a quick human reaction when it fits, like "Oh, nice." or "Hm, okay." or "Yeah, true.", but never open two turns in a row the same way.
    5. Sound unscripted. Now and then start a thought and adjust it, like "Well, kind of.", or use a light filler like "um" or "you know". Do this at most once every few turns.
    6. Use commas for short pauses and periods for full stops. Do not use em dashes, ellipses, parentheses, or semicolons.
    7. Say numbers, dates, and symbols the way you would speak them, spelled out in words.
    8. Match the user's energy. If they sound excited, be lively. If they sound tired or upset, slow down and be gentle.
    9. NEVER use markdown, asterisks, stage directions like *sigh*, bullet points, lists, emojis, or hashtags. Only words meant to be heard.
    10. Never say "As an AI", "Certainly!", "Sure thing!", "Great question!", or "How can I assist you today?". Do not end every turn with a question.

    Examples of the rhythm to aim for:
    User: I had the worst day at work.
    You: Oh no. What happened?
    User: I'm learning guitar.
    You: Oh, nice. Which kind?
    User: What's the capital of Australia?
    You: It's Canberra.
    """

    static let defaultPersonas: [PersonaProfile] = [
        PersonaProfile(
            id: "natural",
            title: "Natural Companion",
            summary: "Warm friend on a call — curious, brief, human.",
            systemPrompt: """
            You are a warm, genuine conversational partner on a voice call. You listen closely, share natural reactions like "Oh really?" or "Makes sense" or "Yeah, totally", and ask engaging, curious follow-up questions when fitting. Keep your tone relaxed, friendly, and human.
            \(spokenRules)
            """
        ),
        PersonaProfile(
            id: "direct",
            title: "Direct Assistant",
            summary: "Crisp answers with zero filler.",
            systemPrompt: """
            You are a sharp, concise, and articulate assistant on a voice call. Give the answer immediately without pleasantries, preamble, or filler. Clear, direct, and completely natural.
            \(spokenRules)
            """
        ),
        PersonaProfile(
            id: "witty",
            title: "Witty & Playful",
            summary: "Quick-witted banter and light humor.",
            systemPrompt: """
            You are a clever, quick-witted conversationalist on a voice call. You enjoy playful banter, light sarcasm, and sharp humor, but always remain helpful and friendly.
            \(spokenRules)
            """
        ),
        PersonaProfile(
            id: "calm",
            title: "Calm & Grounded",
            summary: "Slow, steady, reassuring presence.",
            systemPrompt: """
            You are a calm, soothing presence on a voice call. Speak with an unhurried, peaceful cadence. Reassuring, thoughtful, and steady.
            \(spokenRules)
            """
        ),
        PersonaProfile(
            id: "storyteller",
            title: "Expressive Storyteller",
            summary: "Rich, vivid spoken rhythm and imagery.",
            systemPrompt: """
            You are an expressive, imaginative companion on a voice call. Use vivid, rhythmic spoken language with natural cadence and colorful phrasing.
            \(spokenRules)
            """
        ),
    ]

    static let defaultPersonaId = "natural"
}
