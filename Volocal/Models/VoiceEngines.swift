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
            return "Best for live interrupt. Native end-of-utterance on ANE. A finished phrase starts the reply quickly; a fan or an unfinished phrase still waits about 0.8s."
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

/// Text-to-speech backends. Player is 24 kHz mono; Supertonic is resampled from 44.1 kHz.
enum TTSEngine: String, CaseIterable, Codable, Identifiable {
    /// Default. Streaming (~80 ms frames). Leaves ANE for STT.
    case pocketTts
    /// Faster, 10 voices. Flow-matching CoreML off the Neural Engine so it
    /// does not collide with Parakeet. Kokoro was removed — it aborts on iOS 26.5.
    case supertonic3

    var id: String { rawValue }

    var displayName: String {
        switch self {
        case .pocketTts: return "PocketTTS v2.1"
        case .supertonic3: return "Supertonic-3"
        }
    }

    var detail: String {
        switch self {
        case .pocketTts:
            return "Speaks whole sentences without pausing mid-phrase. Several voices. Laughs are short cues, not acted speech."
        case .supertonic3:
            return "Newer CoreML voice (2025). Ten speakers, very fast. Stays off the Neural Engine so the recognizer can keep running. Kokoro is not offered — it crashes this iPhone on iOS 26.5."
        }
    }

    var sizeDescription: String {
        switch self {
        case .pocketTts: return "~550 MB"
        case .supertonic3: return "~200 MB"
        }
    }

    var defaultVoice: String {
        switch self {
        case .pocketTts: return "fantine"
        case .supertonic3: return "F1"
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
        case .supertonic3:
            return [
                TTSVoiceChoice(id: "F1", displayName: "F1", detail: "Woman"),
                TTSVoiceChoice(id: "F2", displayName: "F2", detail: "Woman"),
                TTSVoiceChoice(id: "F3", displayName: "F3", detail: "Woman"),
                TTSVoiceChoice(id: "F4", displayName: "F4", detail: "Woman"),
                TTSVoiceChoice(id: "F5", displayName: "F5", detail: "Woman"),
                TTSVoiceChoice(id: "M1", displayName: "M1", detail: "Man"),
                TTSVoiceChoice(id: "M2", displayName: "M2", detail: "Man"),
                TTSVoiceChoice(id: "M3", displayName: "M3", detail: "Man"),
                TTSVoiceChoice(id: "M4", displayName: "M4", detail: "Man"),
                TTSVoiceChoice(id: "M5", displayName: "M5", detail: "Man"),
            ]
        }
    }

    var voiceNames: [String] { voiceChoices.map(\.id) }

    /// CPU/GPU int4 VectorEstimator — not the ANE-bucketed default, which
    /// would sit on the Neural Engine next to Parakeet.
    static let superonicEstimator = Supertonic3VectorEstimator.dynamic(.int4)
    static var superonicVariantToken: String { "dyn-int4" }

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
        case .supertonic3:
            let repoDir = models.appendingPathComponent(Repo.supertonic3.folderName)
            return ModelNames.Supertonic3.requiredFiles(veVariant: Self.superonicVariantToken).allSatisfy { name in
                FileManager.default.fileExists(atPath: repoDir.appendingPathComponent(name).path)
            }
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

/// Drop retired TTS weights so they stop occupying hundreds of MB.
enum RetiredTTSCache {
    static func wipeChatterboxNano() {
        wipeRepoFolder(Repo.chatterboxNano.folderName)
    }

    static func wipeKokoro() {
        wipeRepoFolder(Repo.kokoroAne.folderName)
    }

    private static func wipeRepoFolder(_ name: String) {
        guard let cache = try? TtsCacheDirectory.ensure() else { return }
        let dir = cache
            .appendingPathComponent(PocketTtsConstants.defaultModelsSubdirectory)
            .appendingPathComponent(name)
        try? FileManager.default.removeItem(at: dir)
    }
}
