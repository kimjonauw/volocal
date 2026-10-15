import SwiftUI

struct PipelineView: View {
    @EnvironmentObject var metrics: SystemMetrics
    @EnvironmentObject var pipeline: VoicePipeline
    @EnvironmentObject var modelManager: UnifiedModelManager
    @State private var showLLMPicker = false
    @State private var showVoicePicker = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 0) {
                // Conversation history
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(spacing: 12) {
                            ForEach(pipeline.conversationHistory) { message in
                                MessageBubble(message: message)
                                    .id(message.id)
                            }

                            // Show current partial response (visible while LLM generates and TTS speaks)
                            if !pipeline.currentResponse.isEmpty && (pipeline.state == .processing || pipeline.state == .speaking) {
                                MessageBubble(message: ConversationMessage(
                                    role: .assistant,
                                    text: pipeline.currentResponse
                                ))
                            }

                            // Scroll anchor
                            Color.clear.frame(height: 1).id("bottom")
                        }
                        .padding()
                    }
                    .onChange(of: pipeline.conversationHistory.count) {
                        withAnimation {
                            proxy.scrollTo("bottom", anchor: .bottom)
                        }
                    }
                    .onChange(of: pipeline.currentResponse) {
                        proxy.scrollTo("bottom", anchor: .bottom)
                    }
                }

                Divider()

                // Status + mic button
                VStack(spacing: 16) {
                    // Status indicator
                    Text(statusTitle)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    if let statusDetail {
                        Text(statusDetail)
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.tertiary)
                            .multilineTextAlignment(.center)
                    }

                    Text(modelManager.selectedLLM.displayName)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)

                    Text("\(modelManager.selectedSTT.displayName) · \(modelManager.selectedTTS.displayName) · \(modelManager.selectedTTS.voiceChoices.first { $0.id == modelManager.selectedTTSVoice }?.displayName ?? modelManager.selectedTTSVoice)")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)

                    // Current transcript while listening (uses pipeline.partialTranscript)
                    if pipeline.state == .listening && !pipeline.partialTranscript.isEmpty {
                        Text(pipeline.partialTranscript)
                            .font(.caption)
                            .foregroundStyle(.blue)
                    }

                    // Error display
                    if let error = pipeline.currentError {
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.red)
                            .padding(.horizontal)
                    }

                    VoicePresence(pipeline: pipeline)

                    // Mic button
                    Button {
                        pipeline.toggleListening()
                    } label: {
                        ZStack {
                            Circle()
                                .fill(buttonColor)
                                .frame(width: 72, height: 72)

                            Image(systemName: buttonIcon)
                                .font(.system(size: 28))
                                .foregroundStyle(.white)
                        }
                    }
                    .shadow(color: buttonColor.opacity(0.4), radius: pipeline.state == .listening ? 12 : 0)
                    .animation(.easeInOut(duration: 0.8).repeatForever(autoreverses: true), value: pipeline.state == .listening)
                }
                .padding(.vertical, 20)
            }
            .navigationTitle("Volocal")
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showLLMPicker = true
                    } label: {
                        Image(systemName: "cpu")
                    }
                    .accessibilityLabel("Change language model")
                }
                ToolbarItem(placement: .topBarLeading) {
                    Button {
                        showVoicePicker = true
                    } label: {
                        Image(systemName: "waveform")
                    }
                    .accessibilityLabel("Change speech and voice engines")
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button {
                        pipeline.resetChat()
                    } label: {
                        Image(systemName: "arrow.counterclockwise")
                    }
                    .disabled(pipeline.conversationHistory.isEmpty)
                }
            }
            .sheet(isPresented: $showLLMPicker) {
                LLMPickerView(
                    onModelReady: { _ in
                        LLMLoadFence.allowRetry()
                        pipeline.invalidateForReload()
                    },
                    onInstructionsChanged: { pipeline.applyInstructions($0) },
                    onNeedsReload: {
                        LLMLoadFence.allowRetry()
                        pipeline.invalidateForReload()
                    },
                    countTokens: { text in await pipeline.llmManager.tokenCount(for: text) },
                    onThinkingChanged: { pipeline.setSuppressThinking($0) }
                )
                .environmentObject(modelManager)
            }
            .sheet(isPresented: $showVoicePicker) {
                VoiceEnginePickerView(
                    onEnginesChanged: { pipeline.invalidateForReload() },
                    onVoiceChanged: { pipeline.setTTSVoice($0) },
                    onExpressionsChanged: { pipeline.setExpressions($0) }
                )
                .environmentObject(modelManager)
            }
        }
    }

    private var statusTitle: String {
        switch pipeline.state {
        case .idle:
            return "Tap to start"
        case .listening:
            return "Listening..."
        case .processing:
            switch pipeline.llmManager.generatePhase {
            case .readingPrompt:
                return "LLM: reading prompt"
            case .hiddenReasoning:
                return "LLM: reasoning (hidden, not spoken)"
            case .writingSpeech:
                return "LLM: writing reply"
            case .idle:
                return "LLM: generating"
            }
        case .speaking:
            switch pipeline.ttsManager.playbackPhase {
            case .synthesizing:
                return "TTS: generating audio"
            case .playing:
                return "TTS: playing"
            case .idle:
                return "TTS: starting"
            }
        }
    }

    private var statusDetail: String? {
        switch pipeline.state {
        case .processing, .speaking:
            let tps = pipeline.llmManager.tokensPerSecond
            let hidden = pipeline.llmManager.hiddenTokenCount
            let spoken = pipeline.llmManager.spokenCharCount
            var parts: [String] = []
            if tps > 0 {
                parts.append(String(format: "%.0f tok/s", tps))
            }
            if hidden > 0 {
                parts.append("\(hidden) hidden tokens")
            }
            if spoken > 0 {
                parts.append("\(spoken) spoken chars")
            }
            return parts.isEmpty ? "If this stays on LLM, the model is the delay. TTS only starts at TTS: generating audio." : parts.joined(separator: " · ")
        default:
            return nil
        }
    }

    private var buttonColor: Color {
        switch pipeline.state {
        case .idle: return .blue
        case .listening: return .red
        case .processing: return .orange
        case .speaking: return .green
        }
    }

    private var buttonIcon: String {
        switch pipeline.state {
        case .idle: return "mic.fill"
        case .listening: return "mic.fill"
        case .processing: return "brain"
        case .speaking: return "speaker.wave.2.fill"
        }
    }
}

/// Cheap stand-in for a face: a few bars driven by the mic and the speaker.
/// No image model, no extra GPU work.
private struct VoicePresence: View {
    let pipeline: VoicePipeline

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: pipeline.state == .idle)) { timeline in
            let levels = pipeline.sharedAudio.bridge.levels()
            let time = timeline.date.timeIntervalSinceReferenceDate
            let drive = level(for: pipeline.state, mic: levels.mic, playback: levels.playback, time: time)
            HStack(alignment: .center, spacing: 5) {
                ForEach(0..<7, id: \.self) { index in
                    Capsule()
                        .fill(color.opacity(0.9))
                        .frame(width: 7, height: height(index: index, drive: drive, time: time))
                }
            }
            .frame(height: 44)
            .animation(.easeOut(duration: 0.08), value: drive)
        }
        .accessibilityLabel(accessibilityName)
    }

    private var color: Color {
        switch pipeline.state {
        case .idle: return Color(.systemGray3)
        case .listening: return .red
        case .processing: return .orange
        case .speaking: return .green
        }
    }

    private var accessibilityName: String {
        switch pipeline.state {
        case .idle: return "Idle"
        case .listening: return "Listening"
        case .processing: return "Thinking"
        case .speaking: return "Speaking"
        }
    }

    private func level(for state: VoicePipeline.PipelineState, mic: Float, playback: Float, time: TimeInterval) -> CGFloat {
        switch state {
        case .idle:
            return 0.08
        case .listening:
            return CGFloat(min(1, mic / 0.08))
        case .processing:
            return 0.22 + 0.08 * CGFloat(sin(time * 2.2))
        case .speaking:
            let spoken = CGFloat(min(1, playback / 0.12))
            return max(0.35, spoken)
        }
    }

    private func height(index: Int, drive: CGFloat, time: TimeInterval) -> CGFloat {
        let center = abs(CGFloat(index) - 3) / 3
        let wave = 0.65 + 0.35 * sin(time * (pipeline.state == .speaking ? 9 : 4) + Double(index) * 0.7)
        let shaped = (1 - center * 0.45) * (0.25 + 0.75 * drive) * wave
        return 8 + 32 * min(1, max(0.12, shaped))
    }
}

// MARK: - Message Bubble

struct MessageBubble: View {
    let message: ConversationMessage

    var body: some View {
        HStack {
            if message.role == .user { Spacer(minLength: 60) }

            Text(message.text)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(message.role == .user ? Color.blue : Color(.systemGray5))
                .foregroundStyle(message.role == .user ? .white : .primary)
                .clipShape(RoundedRectangle(cornerRadius: 16))

            if message.role == .assistant { Spacer(minLength: 60) }
        }
    }
}
