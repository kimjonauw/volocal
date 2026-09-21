import Foundation
import FluidAudio

/// Speech-to-text backends that work with live barge-in.
enum STTEngine: String, CaseIterable, Codable, Identifiable {
    /// Default. Built-in end-of-utterance, ~320 ms chunks, ~5% WER.
    case parakeetEou320
    /// Lower latency, slightly worse accuracy.
    case parakeetEou160
    /// More accurate (~2% WER), ~560 ms chunks, pause-based turn taking.
    case nemotron560

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .parakeetEou320: return "Parakeet EOU 320"
        case .parakeetEou160: return "Parakeet EOU 160"
        case .nemotron560: return "Nemotron Streaming 0.6B"
        }
    }

    var detail: String {
        switch self {
        case .parakeetEou320:
            return "Best for live interrupt. Native end-of-utterance on ANE."
        case .parakeetEou160:
            return "Same model, 160 ms chunks — snappier, a bit less accurate."
        case .nemotron560:
            return "Clearer transcripts, heavier (~600 MB), slower turns."
        }
    }

    var sizeDescription: String {
        switch self {
        case .parakeetEou320, .parakeetEou160: return "~230 MB"
        case .nemotron560: return "~600 MB"
        }
    }

    var repo: Repo {
        switch self {
        case .parakeetEou320: return .parakeetEou320
        case .parakeetEou160: return .parakeetEou160
        case .nemotron560: return .nemotronStreaming560
        }
    }

    var eouChunkSize: StreamingChunkSize? {
        switch self {
        case .parakeetEou320: return .ms320
        case .parakeetEou160: return .ms160
        case .nemotron560: return nil
        }
    }

    func isDownloaded(in asrRoot: URL) -> Bool {
        let dir = asrRoot.appendingPathComponent(repo.folderName, isDirectory: true)
        switch self {
        case .parakeetEou320, .parakeetEou160:
            return FileManager.default.fileExists(
                atPath: dir.appendingPathComponent(ModelNames.ParakeetEOU.encoderFile).path
            )
        case .nemotron560:
            return FileManager.default.fileExists(
                atPath: dir.appendingPathComponent(ModelNames.NemotronStreaming.encoderInt8File).path
            )
        }
    }
}

/// Text-to-speech backends. Both output 24 kHz mono, matching SharedAudioEngine.
enum TTSEngine: String, CaseIterable, Codable, Identifiable {
    /// Default. Streaming (~26 ms to first audio). Leaves ANE for STT.
    case pocketTts
    /// Prettier, sentence-batched (~240 ms). Shares ANE with Parakeet.
    case kokoroAne

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .pocketTts: return "PocketTTS v2.1"
        case .kokoroAne: return "Kokoro ANE"
        }
    }

    var detail: String {
        switch self {
        case .pocketTts:
            return "Streaming voice. Best with barge-in. GPU + CPU."
        case .kokoroAne:
            return "Nicer timbre, waits for a full sentence. Can fight STT for ANE."
        }
    }

    var sizeDescription: String {
        switch self {
        case .pocketTts: return "~550 MB"
        case .kokoroAne: return "~350 MB"
        }
    }

    var defaultVoice: String {
        switch self {
        case .pocketTts: return "fantine"
        case .kokoroAne: return "af_heart"
        }
    }

    var voiceChoices: [TTSVoiceChoice] {
        switch self {
        case .pocketTts:
            return [
                TTSVoiceChoice(id: "fantine", displayName: "Fantine", detail: "Woman"),
                TTSVoiceChoice(id: "cosette", displayName: "Cosette", detail: "Woman"),
                TTSVoiceChoice(id: "eponine", displayName: "Éponine", detail: "Woman"),
                TTSVoiceChoice(id: "azelma", displayName: "Azelma", detail: "Woman"),
                TTSVoiceChoice(id: "alba", displayName: "Alba", detail: "Woman"),
                TTSVoiceChoice(id: "marius", displayName: "Marius", detail: "Man"),
                TTSVoiceChoice(id: "jean", displayName: "Jean", detail: "Man"),
                TTSVoiceChoice(id: "javert", displayName: "Javert", detail: "Man"),
            ]
        case .kokoroAne:
            return [
                TTSVoiceChoice(id: "af_heart", displayName: "Heart", detail: "Woman (US)"),
                TTSVoiceChoice(id: "af_bella", displayName: "Bella", detail: "Woman (US)"),
                TTSVoiceChoice(id: "af_nicole", displayName: "Nicole", detail: "Woman (US)"),
                TTSVoiceChoice(id: "bf_emma", displayName: "Emma", detail: "Woman (UK)"),
                TTSVoiceChoice(id: "am_michael", displayName: "Michael", detail: "Man (US)"),
                TTSVoiceChoice(id: "am_fenrir", displayName: "Fenrir", detail: "Man (US)"),
            ]
        }
    }

    var voiceNames: [String] { voiceChoices.map(\.id) }

    func isDownloaded() -> Bool {
        guard let cache = try? TtsCacheDirectory.ensure() else { return false }
        let models = cache.appendingPathComponent(PocketTtsConstants.defaultModelsSubdirectory)
        switch self {
        case .pocketTts:
            let languageRoot = models
                .appendingPathComponent(Repo.pocketTts.folderName)
                .appendingPathComponent(PocketTtsLanguage.english.repoSubdirectory)
            return ModelNames.PocketTTS.requiredModels.allSatisfy { name in
                FileManager.default.fileExists(atPath: languageRoot.appendingPathComponent(name).path)
            }
        case .kokoroAne:
            let repoDir = models.appendingPathComponent(Repo.kokoroAne.folderName)
            return FileManager.default.fileExists(
                atPath: repoDir.appendingPathComponent(ModelNames.KokoroAne.albert).path
            )
        }
    }
}

struct TTSVoiceChoice: Identifiable, Equatable {
    let id: String
    let displayName: String
    let detail: String
}

enum FluidAudioCache {
    /// Root passed to FluidAudio `loadModels(to:)` / `ModelHub.download`.
    static var asrModelsRoot: URL {
        let base = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let dir = base
            .appendingPathComponent("FluidAudio", isDirectory: true)
            .appendingPathComponent("Models", isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
