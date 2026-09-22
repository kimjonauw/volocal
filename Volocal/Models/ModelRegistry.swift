import Foundation

/// Paths and labels for the on-device model stack.
/// The LLM itself is not hardcoded here — see `LLMModelSpec` / `UnifiedModelManager.selectedLLM`.
enum ModelRegistry {
    enum ModelType: String, CaseIterable, Identifiable {
        case llm
        case stt
        case tts

        var id: String { rawValue }

        var displayName: String {
            switch self {
            case .llm: return "Language Model"
            case .stt: return "Speech Recognition"
            case .tts: return "Text-to-Speech"
            }
        }

        var icon: String {
            switch self {
            case .llm: return "brain"
            case .stt: return "mic.fill"
            case .tts: return "speaker.wave.3.fill"
            }
        }

        var sizeDescription: String {
            switch self {
            case .llm: return "GGUF from Hugging Face"
            case .stt: return "On-device ASR"
            case .tts: return "On-device voice"
            }
        }

        var detail: String {
            switch self {
            case .llm: return "Any llama.cpp GGUF"
            case .stt: return "Parakeet EOU or Nemotron"
            case .tts: return "PocketTTS or Supertonic-3"
            }
        }
    }

    static var modelsDirectory: URL {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first!
        let dir = docs.appendingPathComponent("models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    static var llmDirectory: URL {
        let dir = modelsDirectory.appendingPathComponent("llm", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
