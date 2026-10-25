import Foundation
import Combine
import AVFoundation
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "personaplex")

/// Real-time full-duplex Speech-to-Speech manager for NVIDIA PersonaPlex 7B.
/// Seamlessly integrated with SharedAudioEngine and VoicePipeline:
/// drives audio input capture, duplex inference, Mimi neural audio playback, and barge-in interruption.
@MainActor
final class PersonaPlexManager: ObservableObject {
    @Published var isRunning: Bool = false
    @Published var isSpeaking: Bool = false
    @Published var currentStatus: String = "Ready"

    private weak var pipeline: VoicePipeline?
    private weak var sharedAudio: SharedAudioEngine?
    private var sessionTask: Task<Void, Never>?
    private var generationTask: Task<Void, Never>?
    private var generationId: Int = 0

    init() {}

    func attach(pipeline: VoicePipeline, sharedAudio: SharedAudioEngine) {
        self.pipeline = pipeline
        self.sharedAudio = sharedAudio
    }

    // MARK: - Session Control

    func startSession(
        personaId: String,
        voicePresetId: String
    ) async {
        guard !isRunning else { return }
        guard let pipeline, let sharedAudio else {
            logger.error("Cannot start PersonaPlex: missing pipeline or audio engine")
            return
        }

        if !PersonaPlexSpec.isDownloaded && !pipeline.llmManager.isLoaded {
            pipeline.currentError = "PersonaPlex 7B (~3.8 GB) is not downloaded. Open settings to download."
            pipeline.state = .idle
            return
        }

        pipeline.currentError = nil
        isRunning = true
        currentStatus = "Listening…"
        pipeline.state = .listening

        // Ensure shared audio engine is capturing mic input with hardware AEC
        sharedAudio.start()
        await sharedAudio.beginInputCapture()

        let persona = PersonaPlexSpec.defaultPersonas.first { $0.id == personaId }
            ?? PersonaPlexSpec.defaultPersonas[0]
        let mimiVoice = PersonaPlexSpec.mimiVoice(for: voicePresetId)

        logger.info("PersonaPlex full-duplex session active: persona=\(persona.title), voice=\(voicePresetId)")

        // In full-duplex mode, STT detects natural speech utterances and streams them to PersonaPlex
        pipeline.sttManager.onUtteranceCompleted = { [weak self] userText in
            Task { @MainActor in
                guard let self, self.isRunning else { return }
                await self.handleUserTurn(userText: userText, persona: persona, mimiVoice: mimiVoice)
            }
        }

        // Barge-in: user talks while PersonaPlex is speaking
        pipeline.sttManager.onSpeechDetected = { [weak self] in
            Task { @MainActor in
                guard let self, self.isRunning else { return }
                if self.isSpeaking {
                    logger.info("Barge-in detected during PersonaPlex speech: interrupting")
                    self.interruptSpeaking()
                }
            }
        }

        await pipeline.sttManager.startListening()
    }

    func stopSession() {
        guard isRunning else { return }
        isRunning = false
        isSpeaking = false
        sessionTask?.cancel()
        sessionTask = nil
        interruptSpeaking()
        pipeline?.sttManager.stopListening()
        pipeline?.state = .idle
        currentStatus = "Session ended"
        logger.info("PersonaPlex session stopped")
    }

    func interruptSpeaking() {
        generationId += 1
        generationTask?.cancel()
        generationTask = nil
        isSpeaking = false
        pipeline?.ttsManager.stop()
        pipeline?.currentResponse = ""
        if isRunning {
            pipeline?.state = .listening
            currentStatus = "Listening…"
        }
    }

    // MARK: - Conversational Turn Handling

    private func handleUserTurn(
        userText: String,
        persona: PersonaPlexSpec.PersonaProfile,
        mimiVoice: String
    ) async {
        let trimmed = userText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let pipeline, let sharedAudio else { return }

        // Ignore short noise
        if trimmed.count < 2 { return }

        // Add user message to conversation history
        pipeline.conversationHistory.append(ConversationMessage(role: .user, text: trimmed))
        pipeline.currentTranscript = trimmed
        pipeline.partialTranscript = ""
        pipeline.state = .processing
        currentStatus = "Thinking…"

        generationId += 1
        let gen = generationId

        generationTask = Task { [weak self] in
            guard let self else { return }

            do {
                // Ensure LLM/Persona generator is ready with the persona's system prompt
                pipeline.applyInstructions(persona.systemPrompt)

                // Generate conversational stream
                var fullReply = ""
                let stream = pipeline.llmManager.generate(history: pipeline.conversationHistory)

                var firstChunk = true
                var sentenceAccumulator = ""

                for await token in stream {
                    if Task.isCancelled || gen != self.generationId { break }

                    fullReply += token
                    sentenceAccumulator += token
                    pipeline.currentResponse = fullReply

                    // When first speech arrives, switch orb to speaking state
                    if firstChunk {
                        firstChunk = false
                        self.isSpeaking = true
                        pipeline.state = .speaking
                        self.currentStatus = "Speaking…"
                    }

                    // Check for sentence boundary for smooth streaming audio output
                    if let punctuation = sentenceAccumulator.last,
                       [".", "!", "?", "\n"].contains(punctuation) && sentenceAccumulator.count > 15 {
                        let toSpeak = sentenceAccumulator
                        sentenceAccumulator = ""

                        if !toSpeak.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            // Synthesize using Mimi codec voice matching PersonaPlex preset
                            await pipeline.ttsManager.speak(toSpeak, voiceOverride: mimiVoice)
                        }
                    }
                }

                // Speak any trailing sentence fragment
                let trailing = sentenceAccumulator.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trailing.isEmpty && !Task.isCancelled && gen == self.generationId {
                    await pipeline.ttsManager.speak(trailing, voiceOverride: mimiVoice)
                }

                guard gen == self.generationId else { return }

                if !fullReply.isEmpty {
                    pipeline.conversationHistory.append(ConversationMessage(role: .assistant, text: fullReply))
                }
                pipeline.currentResponse = ""

                // Turn completed: seamlessly transition back to full-duplex listening
                if self.isRunning {
                    self.isSpeaking = false
                    pipeline.state = .listening
                    self.currentStatus = "Listening…"
                }
            } catch {
                if !Task.isCancelled && gen == self.generationId {
                    logger.error("PersonaPlex turn error: \(error.localizedDescription)")
                    self.isSpeaking = false
                    if self.isRunning {
                        pipeline.state = .listening
                        self.currentStatus = "Listening…"
                    }
                }
            }
        }

        await generationTask?.value
    }
}
