import Foundation
import Combine
import UIKit
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "pipeline")

/// Orchestrates the full voice pipeline: STT -> LLM -> TTS
/// Listens for completed utterances from STT, generates LLM responses,
/// buffers into sentences, and sends to TTS for playback.
/// Supports barge-in: user can speak while AI is talking to interrupt.
@MainActor
final class VoicePipeline: ObservableObject {
    @Published var state: PipelineState = .idle {
        didSet { syncIdleTimer() }
    }
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
    private let gpuLock = GPUExclusive()
    private let sentenceBuffer = SentenceBuffer()

    private var generationTask: Task<Void, Never>?
    /// Llama started on a stable partial, before the turn officially ends.
    private var draftText = ""
    private var draftTokens = ""
    private var draftTask: Task<Void, Never>?
    private var draftDebounce: Task<Void, Never>?
    private var draftRevision = 0
    private var draftCommitted = false
    private var sentenceQueue: [String] = []
    private var speakingTask: Task<Void, Never>?
    private var turnRevision: Int = 0
    private var configureGeneration: Int = 0
    private var cancellables = Set<AnyCancellable>()
    private var lastSpokenTTS = ""
    private var speakingEndedAt: Date?
    private var baseInstructions = LLMManager.defaultInstructions
    private var rollingMemory = ""
    private var memoryCompressTask: Task<Void, Never>?
    private var sceneIsActive = true

    /// Maximum conversation history entries (system prompt excluded).
    /// Each exchange is 2 entries (user + assistant). Scales with `n_ctx`.
    private var maxHistoryEntries: Int {
        LLMContextWindow.historyEntries(for: llmManager.contextSize)
    }

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
        llmManager.gpuLock = gpuLock
        ttsManager.gpuLock = gpuLock
        llmManager.inferenceHold = ttsManager
        setupCallbacks()
        sttManager.$partialResult
            .receive(on: DispatchQueue.main)
            .sink { [weak self] text in
                guard let self else { return }
                self.partialTranscript = text
                self.considerDraft(text)
            }
            .store(in: &cancellables)

        llmManager.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)
        ttsManager.objectWillChange
            .receive(on: DispatchQueue.main)
            .sink { [weak self] _ in self?.objectWillChange.send() }
            .store(in: &cancellables)
    }

    var metrics: SystemMetrics?

    func configure(
        llmModelPath: String?,
        displayName: String? = nil,
        stt: STTEngine = .parakeetEou320,
        tts: TTSEngine = .pocketTts,
        ttsVoice: String? = nil,
        instructions: String? = nil,
        contextSize: UInt32 = LLMContextWindow.default,
        expressions: Bool = true,
        suppressThinking: Bool = true
    ) async {
        configureGeneration += 1
        let gen = configureGeneration
        currentError = nil
        isReady = false
        sharedAudio.start()
        sttManager.sharedAudio = sharedAudio
        ttsManager.sharedAudio = sharedAudio
        ttsManager.stt = sttManager

        guard let path = llmModelPath else {
            currentError = "No GGUF on this iPhone. Pick a language model first."
            loadingStatus = nil
            return
        }

        if LLMLoadFence.shouldSkipLoad(path: path) {
            currentError = LlamaContextError.crashedLastLaunch.localizedDescription
            loadingStatus = nil
            return
        }

        ttsManager.expressionsEnabled = expressions
        sentenceBuffer.mode = .streaming
        applyInstructions(instructions ?? LLMManager.defaultInstructions)
        setSuppressThinking(suppressThinking)
        llmManager.contextSize = LLMContextWindow.clamp(contextSize)
        LLMLoadFence.markStarting(path: path, contextSize: llmManager.contextSize)

        loadingStatus = "Loading language model..."
        metrics?.beginTracking("LLM (llama.cpp)")
        do {
            try await llmManager.loadModel(path: path, displayName: displayName ?? URL(fileURLWithPath: path).lastPathComponent)
        } catch {
            LLMLoadFence.clear()
            guard gen == configureGeneration else { return }
            logger.error("LLM load failed: \(error.localizedDescription)")
            currentError = "LLM failed to load: \(error.localizedDescription)"
            loadingStatus = nil
            return
        }
        guard gen == configureGeneration else { return }
        metrics?.endTracking("LLM (llama.cpp)")

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

        loadingStatus = "Loading text-to-speech..."
        ttsManager.metrics = metrics
        await ttsManager.initialize(engine: tts, voice: ttsVoice)
        guard gen == configureGeneration else { return }
        sentenceBuffer.mode = .streaming
        if let ttsError = ttsManager.error {
            currentError = ttsError
            loadingStatus = nil
            return
        }

        loadingStatus = nil
        isReady = true
        LLMLoadFence.clear()
    }

    func setTTSVoice(_ name: String) {
        ttsManager.selectedVoice = name
    }

    func setExpressions(_ enabled: Bool) {
        ttsManager.expressionsEnabled = enabled
        applyInstructions(baseInstructions)
    }

    func applyInstructions(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        baseInstructions = trimmed.isEmpty ? LLMManager.defaultInstructions : text
        llmManager.systemPrompt = LLMManager.systemPrompt(
            from: baseInstructions,
            expressions: ttsManager.expressionsEnabled
        )
    }

    func setSuppressThinking(_ enabled: Bool) {
        llmManager.suppressThinking = enabled
    }

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
        cancelDraft()
        conversationHistory.removeAll()
        rollingMemory = ""
        memoryCompressTask?.cancel()
        currentTranscript = ""
        currentResponse = ""
        currentError = nil
    }

    /// Voice sessions have no taps, so iOS would otherwise lock the screen.
    func setSceneActive(_ active: Bool) {
        sceneIsActive = active
        syncIdleTimer()
    }

    private func syncIdleTimer() {
        UIApplication.shared.isIdleTimerDisabled = sceneIsActive && state != .idle
    }

    // MARK: - Pipeline Control

    private func startListening() {
        state = .listening
        currentTranscript = ""
        currentError = nil
        Task { @MainActor in
            await sttManager.startListening()
            if let err = sttManager.error {
                currentError = err
                if state == .listening {
                    state = .idle
                }
            }
        }
    }

    private func stopListening() {
        cancelDraft()
        sttManager.stopListening()
        state = .idle
    }

    private func interrupt() {
        turnRevision += 1
        cancelDraft()
        ttsManager.stop()
        llmManager.stopGeneration()
        generationTask?.cancel()
        generationTask = nil
        speakingTask?.cancel()
        speakingTask = nil
        memoryCompressTask?.cancel()
        sentenceQueue.removeAll()
        sentenceBuffer.reset()
        lastSpokenTTS = ""
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
                    if self.isLikelySpeakerEcho(self.sttManager.partialResult) { return }
                    self.interrupt()
                }
            }
        }

        sttManager.onBargeIn = { [weak self] in
            Task { @MainActor in
                guard let self else { return }
                if self.state == .processing || self.state == .speaking {
                    self.interrupt()
                }
            }
        }

        sentenceBuffer.onSentenceReady = { [weak self] sentence in
            self?.handleSentence(sentence)
        }
    }

    private func handleUtterance(_ text: String) {
        if isLikelySpeakerEcho(text) {
            logger.info("Ignoring STT echo of TTS: \(text, privacy: .public)")
            cancelDraft()
            sttManager.resetForNextUtterance()
            return
        }
        // If AI is still active, interrupt first
        if state == .processing || state == .speaking {
            interrupt()
        }
        guard state == .listening else { return }

        turnRevision += 1
        let myRevision = turnRevision
        memoryCompressTask?.cancel()
        draftDebounce?.cancel()

        let adopt = !draftText.isEmpty
            && Self.normalizedWords(text) == Self.normalizedWords(draftText)
            && (draftTask != nil || !draftTokens.isEmpty)

        let userMessage = ConversationMessage(role: .user, text: text)
        conversationHistory.append(userMessage)
        currentTranscript = text
        partialTranscript = ""
        state = .processing
        sttManager.resetForNextUtterance()

        sentenceBuffer.reset()
        sentenceQueue.removeAll()

        if adopt {
            draftCommitted = true
            let seeded = draftTokens
            draftTokens = ""
            currentResponse = seeded
            if !seeded.isEmpty {
                sentenceBuffer.append(seeded)
            }
            let pending = draftTask
            generationTask = Task {
                await pending?.value
                await finishSpeaking(myRevision)
            }
            return
        }

        cancelDraft()
        currentResponse = ""
        let history = conversationHistory
        let memory = rollingMemory
        generationTask = Task {
            // PocketTTS/llama Metal cannot overlap after barge-in. Drain the
            // cancelled synthesizer before the next decode.
            await ttsManager.stopAndWait()
            guard !Task.isCancelled, myRevision == turnRevision else { return }

            for await token in llmManager.generate(history: history, memory: memory) {
                guard !Task.isCancelled, myRevision == turnRevision else { break }
                currentResponse += token
                sentenceBuffer.append(token)
            }
            await finishSpeaking(myRevision)
        }
    }

    /// Start llama while the partial is stable, so prefill overlaps the tail of the turn.
    /// Tokens stay unspoken until the transcript commits. A changed partial throws the draft away.
    private func considerDraft(_ text: String) {
        guard state == .listening else { return }
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        // Emitting a turn clears the partial before the commit runs. An empty
        // update must not throw away the draft we are about to speak.
        guard !trimmed.isEmpty else { return }
        guard UtteranceReadiness.looksComplete(trimmed) else {
            if !draftText.isEmpty, Self.normalizedWords(trimmed) != Self.normalizedWords(draftText) {
                cancelDraft()
            }
            return
        }
        if Self.normalizedWords(trimmed) == Self.normalizedWords(draftText), draftTask != nil {
            return
        }
        draftDebounce?.cancel()
        let snapshot = trimmed
        draftDebounce = Task { @MainActor in
            try? await Task.sleep(for: .milliseconds(180))
            guard !Task.isCancelled, state == .listening else { return }
            let live = sttManager.partialResult.trimmingCharacters(in: .whitespacesAndNewlines)
            guard live == snapshot else { return }
            beginDraft(snapshot)
        }
    }

    private func beginDraft(_ text: String) {
        guard state == .listening else { return }
        if Self.normalizedWords(text) == Self.normalizedWords(draftText), draftTask != nil { return }
        cancelDraft()
        draftRevision += 1
        let revision = draftRevision
        draftText = text
        draftTokens = ""
        draftCommitted = false
        let history = conversationHistory + [ConversationMessage(role: .user, text: text)]
        let memory = rollingMemory
        draftTask = Task { @MainActor in
            for await token in llmManager.generate(history: history, memory: memory) {
                guard revision == draftRevision else { break }
                if draftCommitted {
                    guard state == .processing || state == .speaking else { break }
                    currentResponse += token
                    sentenceBuffer.append(token)
                } else {
                    guard state == .listening else { break }
                    draftTokens += token
                }
            }
        }
    }

    private func cancelDraft() {
        draftDebounce?.cancel()
        draftDebounce = nil
        let hadDraft = draftTask != nil
        draftRevision += 1
        draftTask?.cancel()
        draftTask = nil
        draftText = ""
        draftTokens = ""
        draftCommitted = false
        if hadDraft {
            llmManager.stopGeneration()
        }
    }

    private func finishSpeaking(_ myRevision: Int) async {
        guard !Task.isCancelled, myRevision == turnRevision else { return }

        sentenceBuffer.flush()
        processNextSentence()

        if !currentResponse.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            let assistantMessage = ConversationMessage(role: .assistant, text: currentResponse)
            conversationHistory.append(assistantMessage)
            trimHistory()
        }
        currentResponse = ""
        draftText = ""
        draftTask = nil
        draftCommitted = false

        let waitStart = CFAbsoluteTimeGetCurrent()
        let waitTimeout: TimeInterval = 60
        while speakingTask != nil && !Task.isCancelled && myRevision == turnRevision {
            if CFAbsoluteTimeGetCurrent() - waitStart > waitTimeout {
                logger.warning("Speaking wait timeout after \(waitTimeout)s")
                break
            }
            try? await Task.sleep(for: .milliseconds(100))
        }
        if !Task.isCancelled && myRevision == turnRevision {
            await sharedAudio.waitForPlaybackCompletion()
        }

        guard !Task.isCancelled, myRevision == turnRevision else { return }
        state = .listening
    }

    private func handleSentence(_ sentence: String) {
        guard state == .processing || state == .speaking else { return }
        sentenceQueue.append(sentence)
        processNextSentence()
    }

    private func processNextSentence() {
        guard speakingTask == nil, !sentenceQueue.isEmpty else { return }
        guard state == .processing || state == .speaking else { return }

        let sentence = sentenceQueue.removeFirst()
        let spoken = ExpressionParser.spokenPlain(sentence)
        if !spoken.isEmpty {
            lastSpokenTTS += " " + spoken
            if lastSpokenTTS.count > 400 {
                lastSpokenTTS = String(lastSpokenTTS.suffix(400))
            }
        }
        speakingEndedAt = nil
        let myRevision = turnRevision
        state = .speaking
        speakingTask = Task {
            // Returns when this clause is synthesized and queued — the next
            // clause can generate while this one is still playing.
            await ttsManager.speak(sentence)
            speakingEndedAt = Date()
            guard !Task.isCancelled, myRevision == turnRevision else { return }
            speakingTask = nil
            processNextSentence()
        }
    }

    private func isLikelySpeakerEcho(_ text: String) -> Bool {
        let playing = ttsManager.isSpeaking || sharedAudio.isSpeaking
        let justFinished = speakingEndedAt.map { Date().timeIntervalSince($0) < 0.45 } ?? false
        guard playing || justFinished || state == .speaking else { return false }

        let heard = Self.normalizedWords(text)
        guard !heard.isEmpty else { return true }
        let spoken = Self.normalizedWords(lastSpokenTTS)
        // Short barge-in ("stop", "wait", "hold on") must not be treated as echo.
        if heard.count < 12 { return false }
        guard !spoken.isEmpty else { return false }
        if spoken.contains(heard) { return true }
        let heardTokens = heard.split(separator: " ").filter { $0.count > 2 }
        let spokenTokens = Set(spoken.split(separator: " ").filter { $0.count > 2 })
        guard heardTokens.count >= 3 else { return false }
        let hits = heardTokens.filter { spokenTokens.contains($0) }.count
        return Double(hits) / Double(heardTokens.count) >= 0.75
    }

    private static func normalizedWords(_ text: String) -> String {
        text.lowercased().filter { $0.isLetter || $0.isWhitespace }
            .split(separator: " ").joined(separator: " ")
    }

    /// Sliding window of recent turns, plus a sticky recap of what was dropped.
    private func trimHistory() {
        var dropped: [ConversationMessage] = []
        while conversationHistory.count > maxHistoryEntries {
            dropped.append(conversationHistory.removeFirst())
            if !conversationHistory.isEmpty && conversationHistory.first?.role == .assistant {
                dropped.append(conversationHistory.removeFirst())
            }
        }
        if !dropped.isEmpty {
            foldDropped(dropped)
        }
    }

    private func foldDropped(_ dropped: [ConversationMessage]) {
        let lines = dropped.map { message -> String in
            let role = message.role == .user ? "User" : "Assistant"
            let clip = String(message.text.prefix(120)).replacingOccurrences(of: "\n", with: " ")
            return "\(role): \(clip)"
        }
        rollingMemory = [rollingMemory, lines.joined(separator: "\n")]
            .filter { !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
            .joined(separator: "\n")
        if rollingMemory.count > 1600 {
            rollingMemory = String(rollingMemory.suffix(1300))
            if let nl = rollingMemory.firstIndex(of: "\n") {
                rollingMemory = String(rollingMemory[rollingMemory.index(after: nl)...])
            }
        }
        scheduleMemoryCompress()
    }

    private func scheduleMemoryCompress() {
        guard rollingMemory.count > 900 else { return }
        memoryCompressTask?.cancel()
        let snapshot = rollingMemory
        let revision = turnRevision
        memoryCompressTask = Task { @MainActor in
            try? await Task.sleep(for: .seconds(2))
            guard !Task.isCancelled, revision == turnRevision, state == .listening else { return }
            guard snapshot == rollingMemory, !llmManager.isGenerating else { return }
            if let compressed = await llmManager.compressNotes(snapshot), !compressed.isEmpty {
                rollingMemory = compressed
            }
        }
    }
}
