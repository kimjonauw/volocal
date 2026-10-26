"""Checked build-time source overlay. Run before XcodeGen.

Qwen uses incremental waveform decoding and playback pacing. No RAM refusal
thresholds. Existing Persona removal and engine-selection changes are preserved.
Every source edit must match exactly once before any file is written.
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
replace(engines, 'case .supertonic3: return "Supertonic-3"', 'case .supertonic3: return "Supertonic-3"\n        case .qwen3Tts: return "Qwen3-TTS 0.6B (4-bit)"')
replace(engines, '        case .supertonic3:\n            return "Newer CoreML', '        case .qwen3Tts:\n            return "Experimental expressive local voice (Qwen3-TTS 0.6B, 4-bit). Uses incremental waveform decoding, clause-buffered playback, and GPU serialization. Needs more memory than PocketTTS."\n        case .supertonic3:\n            return "Newer CoreML')
replace(engines, 'case .supertonic3: return "~200 MB"', 'case .supertonic3: return "~200 MB"\n        case .qwen3Tts: return "Large download; extra RAM"')
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
                await audio.waitForPlaybackCompletion()
                try Task.checkCancellation()
                guard generation == speakGeneration else { throw CancellationError() }
                GenerationTrace.record("qwen.clause.begin")
                do {
                    let samples = try await withGPU {
                        try await qwen.synthesize(clause, speaker: voice, whisper: whisper)
                    }
                    try Task.checkCancellation()
                    guard generation == speakGeneration else { throw CancellationError() }
                    markFirstInferenceIfNeeded()
                    playbackPhase = .playing
                    GenerationTrace.record("qwen.playback.schedule", count: samples.count)
                    audio.scheduleTTSBuffer(samples)
                }
                chunks += 1
                await audio.waitForPlaybackCompletion()
                try Task.checkCancellation()
                guard generation == speakGeneration else { throw CancellationError() }
                GenerationTrace.record("qwen.playback.complete")
            }
        } else if let pocketSmall {
            chunks = try await withGPU {''')
replace(manager, '        speakTask?.cancel()\n        holdsInferenceGPU = false\n', '        speakTask?.cancel()\n        // withGPU clears this only after the worker actually drains.\n')
replace(manager, '    // MARK: - Cloned Voices\n', '''    /// Called after stopAndWait when the TTS engine changes.
    func releaseQwen() async {
        guard let backend = qwen else { return }
        _ = try? await withGPU { await backend.unload() }
        qwen = nil
    }

    // MARK: - Cloned Voices
''')

backend = 'Volocal/TTS/QwenConversationBackend.swift'
replace(backend, '''            let audio = try await model.generate(text: text, voice: voice + ", " + delivery,
                refAudio: nil, refText: nil, language: "English",
                generationParameters: GenerateParameters(maxTokens: 192, temperature: 0.8, topP: 1, repetitionPenalty: 1.1))
            try Task.checkCancellation()
            let samples = audio.asType(.float32).asArray(Float.self)''', '''            let samples = try await QwenIncrementalDecode.samples(
                model: model, text: text, voice: voice + ", " + delivery
            )
            try Task.checkCancellation()''')
replace(backend, '            return samples.map { $0 * gain }', '            GenerationTrace.record("qwen.normalize", count: samples.count)\n            return samples.map { $0 * gain }')

llm = 'Volocal/LLM/LLMManager.swift'
replace(llm, '        let epoch = inferenceEpoch\n        let memorySnapshot = memory', '        let epoch = inferenceEpoch\n        GenerationTrace.record("llm.request", count: epoch)\n        let memorySnapshot = memory')
replace(llm, '                let mlx = await MainActor.run { self.mlxContainer }', '                GenerationTrace.record("llm.worker.begin", count: epoch)\n                defer { GenerationTrace.record("llm.worker.end", count: epoch) }\n                let mlx = await MainActor.run { self.mlxContainer }')
replace(llm, '                let stream = try await MLXRuntime.openStream(', '                GenerationTrace.record("llm.mlx.begin")\n                let stream = try await MLXRuntime.openStream(')
replace(llm, '                    pieces.append(piece)', '                    pieces.append(piece)\n                    if pieces.count == 1 || pieces.count % 32 == 0 {\n                        GenerationTrace.record("llm.mlx.decode", count: pieces.count)\n                    }')
replace(llm, '                return pieces\n', '                GenerationTrace.record("llm.mlx.collected", count: pieces.count)\n                return pieces\n')
replace(llm, '                    let startDecode: (String) async throws -> Void = { prompt in\n', '                    let startDecode: (String) async throws -> Void = { prompt in\n                        GenerationTrace.record("llm.gguf.prefill.begin")\n                        defer { GenerationTrace.record("llm.gguf.prefill.exit") }\n')
replace(llm, '                        tokenCount += 1\n                        let elapsed', '                        tokenCount += 1\n                        if tokenCount == 1 || tokenCount % 32 == 0 {\n                            GenerationTrace.record("llm.gguf.decode", count: tokenCount)\n                        }\n                        let elapsed')

models = 'Volocal/Models/UnifiedModelManager.swift'
replace(models, '            case .supertonic3:\n                _ = try await Supertonic3ResourceDownloader.ensureModels(', '            case .qwen3Tts:\n                try await QwenConversationBackend.prepareFiles()\n            case .supertonic3:\n                _ = try await Supertonic3ResourceDownloader.ensureModels(')
replace(models, '            pipelineMode = mode\n        } else {', '            _ = mode\n            pipelineMode = .modular\n        } else {')

persona = 'Volocal/Models/PersonaPlexSpec.swift'
replace(persona, '        switch engine {\n        case .supertonic3:', '        switch engine {\n        case .qwen3Tts:\n            return preset.gender == "Female" ? "Serena" : "Ryan"\n        case .supertonic3:')

picker = 'Volocal/Models/VoiceEnginePickerView.swift'
replace(picker, '                        Text("Persona").tag(PipelineMode.personaplex)\n', '')
replace(picker, '                if modelManager.pipelineMode == .personaplex {\n                    personaplexSections\n                } else {\n                    modularSections\n                }', '                modularSections')
replace(picker, '.navigationTitle(modelManager.pipelineMode == .personaplex ? "Persona conversation" : "Voice engines")', '.navigationTitle("Voice engines")')

for path, text in changes.items():
    Path(path).write_text(text)
    print(f'Applied checked Qwen integration: {path}')
