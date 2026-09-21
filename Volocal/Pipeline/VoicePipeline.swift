import Foundation
import Combine
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "pipeline")

/// Orchestrates the full voice pipeline: STT -> LLM -> TTS
/// Listens for completed utterances from STT, generates LLM responses,
/// buffers into sentences, and sends to TTS for playback.
/// Supports barge-in: user can speak while AI is talking to interrupt.
@MainActor
final class VoicePipeline: ObservableObject {
    @Published var state: PipelineState = .idle
    @Published var conversationHistory: [ConversationMessage] = []
    @Published var currentTranscript: String = ""
    @Published var currentResponse: String = ""
    @Published var loadingStatus: String?
    @Published var isReady: Bool = false
    @Published var partialTranscript: String = ""
    @Published var currentError: String?

    let sttManager = STTManager()
    let llmManager = LLMManager()
    let ttsManager = TTSManager()
    let sharedAudio = SharedAudioEngine()
    private let sentenceBuffer = SentenceBuffer()

    private var generationTask: Task<Void, Never>?
    private var sentenceQueue: [String] = []
    private var speakingTask: Task<Void, Never>?
    private var turnRevision: Int = 0
    private var configureGeneration: Int = 0
    private var cancellables = Set<AnyCancellable>()

    /// Maximum conversation history entries (system prompt excluded).
    /// Each exchange is 2 entries (user + assistant). Keep last ~4 exchanges.
    private let maxHistoryEntries = 8

    enum PipelineState: Equatable {
        case idle
        case listening
        case processing
        case speaking

        var label: String {
            switch self {
            case .idle: return "Tap to start"
            case .listening: return "Listening..."
            case .processing: return "Thinking..."
            case .speaking: return "Speaking..."
            }
        }
    }

    init() {
        setupCallbacks()
        // Forward partial transcript from STT manager
        sttManager.$partialResult
            .receive(on: DispatchQueue.main)
            .assign(to: &$partialTranscript)
    }

    var metrics: SystemMetrics?

    func configure(
        llmModelPath: String?,
        displayName: String? = nil,
        stt: STTEngine = .parakeetEou320,
        tts: TTSEngine = .pocketTts
    ) async {
        configureGeneration += 1
        let gen = configureGeneration
        currentError = nil
        isReady = false
        // Start shared audio engine
        sharedAudio.start()

        // Inject shared audio into managers
        sttManager.sharedAudio = sharedAudio
        ttsManager.sharedAudio = sharedAudio

        loadingStatus = "Loading speech recognition..."
        metrics?.beginTracking("STT (\(stt.displayName))")
        await sttManager.initialize(engine: stt)
        guard gen == configureGeneration else { return }
        metrics?.endTracking("STT (\(stt.displayName))")
        if let sttError = sttManager.error {
            currentError = sttError
            loadingStatus = nil
            return
        }

        loadingStatus = "Loading language model..."
        guard let path = llmModelPath else {
            currentError = "No GGUF on this iPhone. Pick a language model first."
            loadingStatus = nil
            return
        }
        metrics?.beginTracking("LLM (llama.cpp)")
        do {
            try await llmManager.loadModel(path: path, displayName: displayName ?? URL(fileURLWithPath: path).lastPathComponent)
        } catch {
            guard gen == configureGeneration else { return }
            logger.error("LLM load failed: \(error.localizedDescription)")
            currentError = "LLM failed to load: \(error.localizedDescription)"
            loadingStatus = nil
            return
        }
        guard gen == configureGeneration else { return }
        metrics?.endTracking("LLM (llama.cpp)")

        loadingStatus = "Loading text-to-speech..."
        ttsManager.metrics = metrics
        await ttsManager.initialize(engine: tts)
        guard gen == configureGeneration else { return }
        if let ttsError = ttsManager.error {
            currentError = ttsError
            loadingStatus = nil
            return
        }

        loadingStatus = nil
        isReady = true
    }

    /// Drop back to the loading screen so `configure` picks up the current STT/TTS/LLM selection.
    func invalidateForReload() {
        if state == .processing || state == .speaking {
            interrupt()
        }
        if state == .listening {
            stopListening()
        }
        currentError = nil
        loadingStatus = "Loading speech recognition..."
        isReady = false
    }

    func toggleListening() {
        switch state {
        case .idle:
            startListening()
        case .listening:
            stopListening()
        case .processing, .speaking:
            interrupt()
        }
    }

    func resetChat() {
        if state == .processing || state == .speaking {
            interrupt()
        }
        if state == .listening {
            stopListening()
        }
        conversationHistory.removeAll()
        currentTranscript = ""
        currentResponse = ""
        currentError = nil
    }

    // MARK: - Pipeline Control

    private func startListening() {
        state = .listening
        currentTranscript = ""
        currentError = nil
        sttManager.startListening()
    }

    private func stopListening() {
        sttManager.stopListening()
        state = .idle
    }

    private func interrupt() {
        turnRevision += 1
        ttsManager.stop()
        llmManager.stopGeneration()
        generationTask?.cancel()
        generationTask = nil
        speakingTask?.cancel()
        speakingTask = nil
        sentenceQueue.removeAll()
        sentenceBuffer.reset()
        currentResponse = ""
        // Don't stop STT — mic stays open for barge-in
        state = .listening
    }

    // MARK: - Callbacks

    private func setupCallbacks() {
        sttManager.onUtteranceCompleted = { [weak self] text in
            Task { @MainActor in
                self?.handleUtterance(text)
            }
        }

        sttManager.onSpeechDetected = { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                // Barge-in: user started speaking while AI is active
                if self.state == .processing || self.state == .speaking {
                    self.interrupt()
                }
            }
        }

        sentenceBuffer.onSentenceReady = { [weak self] sentence in
            Task { @MainActor in
                self?.handleSentence(sentence)
            }
        }
    }

    private func handleUtterance(_ text: String) {
        // If AI is still active, interrupt first
        if state == .processing || state == .speaking {
            interrupt()
        }
        guard state == .listening else { return }

        turnRevision += 1
        let myRevision = turnRevision

        let userMessage = ConversationMessage(role: .user, text: text)
        conversationHistory.append(userMessage)
        currentTranscript = text

        // Forward partial transcript
        partialTranscript = sttManager.partialResult

        // Reset ASR for next utterance (mic stays open)
        sttManager.resetForNextUtterance()

        state = .processing
        currentResponse = ""
        sentenceBuffer.reset()
        sentenceQueue.removeAll()

        generationTask = Task {
            // History already includes the user message we just appended.
            // generate() should NOT re-append the prompt.
            for await token in llmManager.generate(history: conversationHistory) {
                guard !Task.isCancelled, myRevision == turnRevision else { break }
                currentResponse += token
                sentenceBuffer.append(token)
            }

            guard !Task.isCancelled, myRevision == turnRevision else { return }

            sentenceBuffer.flush()

            // Only append non-empty assistant messages
            if !currentResponse.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let assistantMessage = ConversationMessage(role: .assistant, text: currentResponse)
                conversationHistory.append(assistantMessage)
                trimHistory()
            }
            // Clear so the partial response bubble disappears
            // (the response is now in conversationHistory)
            currentResponse = ""

            // Wait for all queued sentences to finish speaking (with timeout)
            let waitStart = CFAbsoluteTimeGetCurrent()
            let waitTimeout: TimeInterval = 60
            while speakingTask != nil && !Task.isCancelled && myRevision == turnRevision {
                if CFAbsoluteTimeGetCurrent() - waitStart > waitTimeout {
                    logger.warning("Speaking wait timeout after \(waitTimeout)s")
                    break
                }
                try? await Task.sleep(for: .milliseconds(100))
            }

            guard !Task.isCancelled, myRevision == turnRevision else { return }
            state = .listening
        }
    }

    private func handleSentence(_ sentence: String) {
        sentenceQueue.append(sentence)
        processNextSentence()
    }

    private func processNextSentence() {
        guard speakingTask == nil, !sentenceQueue.isEmpty else { return }
        guard !Task.isCancelled else { return }

        let sentence = sentenceQueue.removeFirst()
        let myRevision = turnRevision
        state = .speaking
        speakingTask = Task {
            await ttsManager.speak(sentence)
            guard !Task.isCancelled, myRevision == turnRevision else { return }
            speakingTask = nil
            processNextSentence()
        }
    }

    /// Trim conversation history to prevent context overflow.
    /// Keeps the most recent exchanges within maxHistoryEntries.
    private func trimHistory() {
        while conversationHistory.count > maxHistoryEntries {
            conversationHistory.removeFirst()
            // Remove in pairs if possible to keep user/assistant aligned
            if !conversationHistory.isEmpty && conversationHistory.first?.role == .assistant {
                conversationHistory.removeFirst()
            }
        }
    }
}
