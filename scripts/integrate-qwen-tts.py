"""Guarded build-time source overlay. Run before XcodeGen; never edits dependencies.

Keeps the existing long pipeline/model-manager implementations intact. Every edit
requires a unique inspected anchor, and all files are checked before any write.
Qwen is optional, 4-bit only, and uses fixed delivery in this first integration.
"""
from pathlib import Path

changes = {}
def read(path):
    return changes.get(path, Path(path).read_text())
def replace(path, old, new):
    text = read(path)
    if text.count(old) != 1:
        raise RuntimeError(f'Qwen overlay anchor changed in {path}: {old[:90]!r}')
    changes[path] = text.replace(old, new, 1)

engines = 'Volocal/Models/VoiceEngines.swift'
replace(engines, '    case supertonic3\n', '    case supertonic3\n    case qwen3Tts\n')
replace(engines, 'case .supertonic3: return "Supertonic-3"', 'case .supertonic3: return "Supertonic-3"\n        case .qwen3Tts: return "Qwen3-TTS 1.7B (4-bit)"')
replace(engines, '        case .supertonic3:\n            return "Newer CoreML', '        case .qwen3Tts:\n            return "Experimental expressive local voice. Uses your selected LLM with fixed warm/playful delivery, shared playback, and GPU serialization. More memory than PocketTTS. Audio is buffered per short clause; LLM-generated delivery is not wired yet."\n        case .supertonic3:\n            return "Newer CoreML')
replace(engines, 'case .supertonic3: return "~200 MB"', 'case .supertonic3: return "~200 MB"\n        case .qwen3Tts: return "Multi-GB download; high RAM use"')
replace(engines, 'case .supertonic3: return "F1"', 'case .supertonic3: return "F1"\n        case .qwen3Tts: return "Serena"')
replace(engines, '    var voiceChoices: [TTSVoiceChoice] {\n        switch self {', '    var voiceChoices: [TTSVoiceChoice] {\n        switch self {\n        case .qwen3Tts:\n            return ["Serena", "Vivian", "Ryan", "Aiden"].map { TTSVoiceChoice(id: $0, displayName: $0, detail: "Qwen preset") }')
replace(engines, '        case .supertonic3:\n            let repoDir', '        case .qwen3Tts:\n            return QwenConversationBackend.isDownloaded\n        case .supertonic3:\n            let repoDir')
replace(engines, '    func isDownloaded() -> Bool {\n', '    func isDownloaded() -> Bool {\n        if self == .qwen3Tts { return QwenConversationBackend.isDownloaded }\n')

manager = 'Volocal/TTS/TTSManager.swift'
replace(manager, '    private var superonic: Supertonic3Manager?\n', '    private var superonic: Supertonic3Manager?\n    private var qwen: QwenConversationBackend?\n')
replace(manager, '        await stopAndWait()\n        pocketSmall = nil', '        await stopAndWait()\n        await releaseQwen()\n        pocketSmall = nil')
replace(manager, '            case .supertonic3:\n                try await loadSupertonic()', '            case .qwen3Tts:\n                let backend = QwenConversationBackend()\n                try await withGPU { try await backend.load() }\n                qwen = backend\n            case .supertonic3:\n                try await loadSupertonic()')
replace(manager, '        if let pocketSmall {\n            chunks = try await withGPU {', '''        if let qwen {
            for clause in QwenConversationBackend.clauses(text) {
                try Task.checkCancellation()
                guard generation == speakGeneration else { throw CancellationError() }
                let samples = try await withGPU {
                    try await qwen.synthesize(clause, speaker: voice, whisper: whisper)
                }
                try Task.checkCancellation()
                guard generation == speakGeneration else { throw CancellationError() }
                markFirstInferenceIfNeeded()
                playbackPhase = .playing
                audio.scheduleTTSBuffer(samples)
                chunks += 1
            }
        } else if let pocketSmall {
            chunks = try await withGPU {''')
replace(manager, '        speakTask?.cancel()\n        holdsInferenceGPU = false\n', '        speakTask?.cancel()\n        // withGPU clears this only after the worker actually drains.\n')
replace(manager, '    // MARK: - Cloned Voices\n', '''    /// Called after stopAndWait, including when switching to speech-to-speech.
    func releaseQwen() async {
        guard let backend = qwen else { return }
        _ = try? await withGPU { await backend.unload() }
        qwen = nil
    }

    // MARK: - Cloned Voices
''')

models = 'Volocal/Models/UnifiedModelManager.swift'
replace(models, '            case .supertonic3:\n                _ = try await Supertonic3ResourceDownloader.ensureModels(', '            case .qwen3Tts:\n                try await QwenConversationBackend.prepareFiles()\n            case .supertonic3:\n                _ = try await Supertonic3ResourceDownloader.ensureModels(')

persona = 'Volocal/Models/PersonaPlexSpec.swift'
replace(persona, '        switch engine {\n        case .supertonic3:', '        switch engine {\n        case .qwen3Tts:\n            return preset.gender == "Female" ? "Serena" : "Ryan"\n        case .supertonic3:')

pipeline = 'Volocal/Pipeline/VoicePipeline.swift'
replace(pipeline, '            ttsManager.stop()\n            llmManager.unload()', '            await ttsManager.stopAndWait()\n            await ttsManager.releaseQwen()\n            llmManager.unload()')

for path, text in changes.items():
    Path(path).write_text(text)
    print(f'Applied guarded Qwen integration: {path}')
