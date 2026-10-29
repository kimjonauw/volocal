"""Checked build-time Qwen overlay. Run before XcodeGen.

Both CustomVoice sizes use incremental decoding and bounded clause lookahead.
Dynamic delivery tags are passed to both; 0.6B instruction following is experimental.
The 1.7B backend reuses the older QwenConversation4bit cache folder.
All nine CustomVoice speakers are offered in the picker.
No RAM refusal thresholds. All edits must match before files are written.
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
replace(engines, '    case supertonic3\n', '    case supertonic3\n    case qwen3Tts\n    case qwen3Tts17B\n')
replace(engines, 'case .supertonic3: return "Supertonic-3"', 'case .supertonic3: return "Supertonic-3"\n        case .qwen3Tts: return "Qwen3-TTS 0.6B (4-bit)"\n        case .qwen3Tts17B: return "Qwen3-TTS 1.7B (4-bit, experimental)"')
replace(engines, '        case .supertonic3:\n            return "Newer CoreML', '        case .qwen3Tts:\n            return "Smaller local CustomVoice model. Incremental decoding and bounded clause lookahead. Delivery tags are wired, but 0.6B instruction following is experimental."\n        case .qwen3Tts17B:\n            return "Larger CustomVoice model with delivery-instruction support. Uses the same incremental decoding and bounded lookahead. Experimental: more memory, not verified stable in LiveContainer."\n        case .supertonic3:\n            return "Newer CoreML')
replace(engines, 'case .supertonic3: return "~200 MB"', 'case .supertonic3: return "~200 MB"\n        case .qwen3Tts: return "Large download; extra RAM"\n        case .qwen3Tts17B: return "Larger download; higher RAM"')
replace(engines, 'case .supertonic3: return "F1"', 'case .supertonic3: return "F1"\n        case .qwen3Tts, .qwen3Tts17B: return "Serena"')
replace(engines, '    var voiceChoices: [TTSVoiceChoice] {\n        switch self {', '    var voiceChoices: [TTSVoiceChoice] {\n        switch self {\n        case .qwen3Tts, .qwen3Tts17B:\n            return [\n                TTSVoiceChoice(id: "Aiden", displayName: "Aiden", detail: "Sunny American man, clear midrange. English-native."),\n                TTSVoiceChoice(id: "Ryan", displayName: "Ryan", detail: "Dynamic man with a strong rhythmic drive. English-native."),\n                TTSVoiceChoice(id: "Serena", displayName: "Serena", detail: "Warm, gentle young woman. Chinese-native; English may sound accented."),\n                TTSVoiceChoice(id: "Vivian", displayName: "Vivian", detail: "Bright, slightly edgy young woman. Chinese-native; English may sound accented."),\n                TTSVoiceChoice(id: "Uncle_Fu", displayName: "Uncle Fu", detail: "Seasoned man with a low, mellow voice. Chinese-native; English may sound accented."),\n                TTSVoiceChoice(id: "Dylan", displayName: "Dylan", detail: "Youthful Beijing man. Chinese-native; English may sound accented."),\n                TTSVoiceChoice(id: "Eric", displayName: "Eric", detail: "Lively Chengdu man, slightly husky. Chinese-native; English may sound accented."),\n                TTSVoiceChoice(id: "Ono_Anna", displayName: "Ono Anna", detail: "Playful Japanese woman. Japanese-native; English may sound accented."),\n                TTSVoiceChoice(id: "Sohee", displayName: "Sohee", detail: "Warm Korean woman. Korean-native; English may sound accented."),\n            ]')
replace(engines, '        case .supertonic3:\n            let repoDir', '        case .qwen3Tts:\n            return QwenConversationBackend.isDownloaded\n        case .qwen3Tts17B:\n            return QwenConversationBackend17B.isDownloaded\n        case .supertonic3:\n            let repoDir')
replace(engines, '    func isDownloaded() -> Bool {\n', '    func isDownloaded() -> Bool {\n        if self == .qwen3Tts { return QwenConversationBackend.isDownloaded }\n        if self == .qwen3Tts17B { return QwenConversationBackend17B.isDownloaded }\n')

manager = 'Volocal/TTS/TTSManager.swift'
replace(manager, '    private var superonic: Supertonic3Manager?\n', '    private var superonic: Supertonic3Manager?\n    private var qwen: QwenConversationBackend?\n    private var qwen17: QwenConversationBackend17B?\n')
replace(manager, '        await stopAndWait()\n        pocketSmall = nil', '        await stopAndWait()\n        await releaseQwen()\n        pocketSmall = nil')
replace(manager, '            case .supertonic3:\n                try await loadSupertonic()', '            case .qwen3Tts:\n                let backend = QwenConversationBackend()\n                try await withGPU { try await backend.load() }\n                qwen = backend\n            case .qwen3Tts17B:\n                let backend = QwenConversationBackend17B()\n                try await withGPU { try await backend.load() }\n                qwen17 = backend\n            case .supertonic3:\n                try await loadSupertonic()')
replace(manager, '        let prepared = expressionsEnabled ? text : ExpressionParser.stripNeuralTags(text)', '        let isQwen = engineKind == .qwen3Tts || engineKind == .qwen3Tts17B\n        let deliveryText = isQwen && expressionsEnabled ? text : DynamicDelivery.strip(text)\n        let prepared = expressionsEnabled ? deliveryText : ExpressionParser.stripNeuralTags(deliveryText)')
replace(manager, '        if let pocketSmall {\n            chunks = try await withGPU {', '''        if qwen != nil || qwen17 != nil {
            let small = qwen
            let large = qwen17
            for clause in DynamicDelivery.clauses(text) {
                try Task.checkCancellation()
                guard generation == speakGeneration else { throw CancellationError() }
                GenerationTrace.record(large != nil ? "qwen17.clause.begin" : "qwen06.clause.begin")
                do {
                    let samples: [Float] = try await withGPU {
                        if let small {
                            return try await small.synthesize(clause.text, speaker: voice, whisper: whisper, style: clause.style)
                        }
                        guard let large else { throw TTSError.engineNotLoaded }
                        return try await large.synthesize(clause.text, speaker: voice, whisper: whisper, style: clause.style)
                    }
                    try Task.checkCancellation()
                    guard generation == speakGeneration else { throw CancellationError() }
                    await audio.waitForPlaybackCompletion()
                    try Task.checkCancellation()
                    guard generation == speakGeneration else { throw CancellationError() }
                    GenerationTrace.record("qwen.prior.playback.complete")
                    markFirstInferenceIfNeeded()
                    playbackPhase = .playing
                    GenerationTrace.record("qwen.playback.schedule", count: samples.count)
                    audio.scheduleTTSBuffer(samples)
                }
                chunks += 1
            }
        } else if let pocketSmall {
            chunks = try await withGPU {''')
replace(manager, '        speakTask?.cancel()\n        holdsInferenceGPU = false\n', '        speakTask?.cancel()\n        // withGPU clears this only after the worker actually drains.\n')
replace(manager, '    // MARK: - Cloned Voices\n', '''    /// Called after stopAndWait; only one Qwen size is loaded at a time.
    func releaseQwen() async {
        if let backend = qwen {
            _ = try? await withGPU { await backend.unload() }
        }
        if let backend = qwen17 {
            _ = try? await withGPU { await backend.unload() }
        }
        qwen = nil
        qwen17 = nil
    }

    // MARK: - Cloned Voices
''')

backend = 'Volocal/TTS/QwenConversationBackend.swift'
replace(backend, '        removeLegacyDownload()\n', '        // Keep both model downloads; selecting a size must not delete the other.\n')
replace(backend, '    func synthesize(_ text: String, speaker: String, whisper: Bool) async throws -> [Float] {', '    func synthesize(_ text: String, speaker: String, whisper: Bool, style: String = "neutral") async throws -> [Float] {')
replace(backend, '        let delivery = whisper ? "Speak softly in a gentle whisper without shouting." : "Warm and gently playful. Speak naturally and conversationally without shouting."', '        let delivery = DynamicDelivery.instruction(for: whisper ? "whisper" : style)')
replace(backend, '''            let audio = try await model.generate(text: text, voice: voice + ", " + delivery,
                refAudio: nil, refText: nil, language: "English",
                generationParameters: GenerateParameters(maxTokens: 192, temperature: 0.8, topP: 1, repetitionPenalty: 1.1))
            try Task.checkCancellation()
            let samples = audio.asType(.float32).asArray(Float.self)''', '''            let samples = try await QwenIncrementalDecode.samples(
                model: model, text: text, voice: voice + ", " + delivery
            )
            try Task.checkCancellation()''')
replace(backend, '            return samples.map { $0 * gain }', '            GenerationTrace.record("qwen.normalize", count: samples.count)\n            return samples.map { $0 * gain }')
large_backend = read(backend)
for old, new in [
    ('actor QwenConversationBackend {', 'actor QwenConversationBackend17B {'),
    ('mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit', 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit'),
    ('"QwenConversation06B4bit"', '"QwenConversation4bit"'),
]:
    if large_backend.count(old) != 1:
        raise RuntimeError(f'1.7B backend anchor changed: {old}')
    large_backend = large_backend.replace(old, new, 1)
changes['Volocal/TTS/QwenConversationBackend17B.swift'] = large_backend

llm = 'Volocal/LLM/LLMManager.swift'
replace(llm, '    var systemPrompt: String = LLMManager.defaultInstructions', '    var dynamicDeliveryEnabled = false\n    var systemPrompt: String = LLMManager.defaultInstructions')
replace(llm, '        let promptSystem = systemPrompt', '        let promptSystem = systemPrompt + (dynamicDeliveryEnabled ? DynamicDelivery.prompt : "")')
replace(llm, '        let epoch = inferenceEpoch\n        let memorySnapshot = memory', '        let epoch = inferenceEpoch\n        GenerationTrace.record("llm.request", count: epoch)\n        let memorySnapshot = memory')
replace(llm, '                let mlx = await MainActor.run { self.mlxContainer }', '                GenerationTrace.record("llm.worker.begin", count: epoch)\n                defer { GenerationTrace.record("llm.worker.end", count: epoch) }\n                let mlx = await MainActor.run { self.mlxContainer }')
replace(llm, '                let stream = try await MLXRuntime.openStream(', '                GenerationTrace.record("llm.mlx.begin")\n                let stream = try await MLXRuntime.openStream(')
replace(llm, '                    pieces.append(piece)', '                    pieces.append(piece)\n                    if pieces.count == 1 || pieces.count % 32 == 0 {\n                        GenerationTrace.record("llm.mlx.decode", count: pieces.count)\n                    }')
replace(llm, '                return pieces\n', '                GenerationTrace.record("llm.mlx.collected", count: pieces.count)\n                return pieces\n')
replace(llm, '                    let startDecode: (String) async throws -> Void = { prompt in\n', '                    let startDecode: (String) async throws -> Void = { prompt in\n                        GenerationTrace.record("llm.gguf.prefill.begin")\n                        defer { GenerationTrace.record("llm.gguf.prefill.exit") }\n')
replace(llm, '                        tokenCount += 1\n                        let elapsed', '                        tokenCount += 1\n                        if tokenCount == 1 || tokenCount % 32 == 0 {\n                            GenerationTrace.record("llm.gguf.decode", count: tokenCount)\n                        }\n                        let elapsed')

pipeline = 'Volocal/Pipeline/VoicePipeline.swift'
replace(pipeline, '        ttsManager.expressionsEnabled = expressions', '        llmManager.dynamicDeliveryEnabled = (tts == .qwen3Tts || tts == .qwen3Tts17B) && expressions\n        ttsManager.expressionsEnabled = expressions')
replace(pipeline, '        ttsManager.expressionsEnabled = enabled', '        llmManager.dynamicDeliveryEnabled = (ttsManager.engineKind == .qwen3Tts || ttsManager.engineKind == .qwen3Tts17B) && enabled\n        ttsManager.expressionsEnabled = enabled')
replace(pipeline, 'let assistantMessage = ConversationMessage(role: .assistant, text: currentResponse)', 'let assistantMessage = ConversationMessage(role: .assistant, text: DynamicDelivery.strip(currentResponse))')

parser = 'Volocal/TTS/ExpressionParser.swift'
replace(parser, '        parse(text).compactMap { segment in', '        parse(DynamicDelivery.strip(text)).compactMap { segment in')
replace(parser, '    static func stripNeuralTags(_ text: String) -> String {\n        var t = text', '    static func stripNeuralTags(_ text: String) -> String {\n        var t = DynamicDelivery.strip(text)')

sentence = 'Volocal/Pipeline/SentenceBuffer.swift'
replace(sentence, '    private var didEmit = false', '    private var didEmit = false\n    private var deliveryStyle: String?')
replace(sentence, '    func flush() {\n        emit(buffer)\n        buffer = ""\n        didEmit = false\n    }', '    func flush() {\n        emit(buffer)\n        buffer = ""\n        didEmit = false\n        deliveryStyle = nil\n    }')
replace(sentence, '    func reset() {\n        buffer = ""\n        didEmit = false\n    }', '    func reset() {\n        buffer = ""\n        didEmit = false\n        deliveryStyle = nil\n    }')
replace(sentence, '        onSentenceReady?(sentence)', '        onSentenceReady?(DynamicDelivery.carry(sentence, style: &deliveryStyle))')

models = 'Volocal/Models/UnifiedModelManager.swift'
replace(models, '            case .supertonic3:\n                _ = try await Supertonic3ResourceDownloader.ensureModels(', '            case .qwen3Tts:\n                try await QwenConversationBackend.prepareFiles()\n            case .qwen3Tts17B:\n                try await QwenConversationBackend17B.prepareFiles()\n            case .supertonic3:\n                _ = try await Supertonic3ResourceDownloader.ensureModels(')
replace(models, '            pipelineMode = mode\n        } else {', '            _ = mode\n            pipelineMode = .modular\n        } else {')

persona = 'Volocal/Models/PersonaPlexSpec.swift'
replace(persona, '        switch engine {\n        case .supertonic3:', '        switch engine {\n        case .qwen3Tts, .qwen3Tts17B:\n            return preset.gender == "Female" ? "Serena" : "Ryan"\n        case .supertonic3:')

picker = 'Volocal/Models/VoiceEnginePickerView.swift'
replace(picker, '                        Text("Persona").tag(PipelineMode.personaplex)\n', '')
replace(picker, '                if modelManager.pipelineMode == .personaplex {\n                    personaplexSections\n                } else {\n                    modularSections\n                }', '                modularSections')
replace(picker, '.navigationTitle(modelManager.pipelineMode == .personaplex ? "Persona conversation" : "Voice engines")', '.navigationTitle("Voice engines")')

for path, text in changes.items():
    Path(path).write_text(text)
    print(f'Applied checked Qwen integration: {path}')
