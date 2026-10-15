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

/// Glowing orb. Soft color fields drift inside a circle and swell with the
/// mic and the speaker, the same idea as the ChatGPT and Gemini live spheres.
private struct VoicePresence: View {
    let pipeline: VoicePipeline

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: false)) { timeline in
            let levels = pipeline.sharedAudio.bridge.levels()
            let time = timeline.date.timeIntervalSinceReferenceDate
            let motion = OrbMotion(
                state: pipeline.state,
                mic: levels.mic,
                playback: levels.playback,
                time: time
            )
            ZStack {
                Circle()
                    .fill(motion.glow)
                    .frame(width: 96, height: 96)
                    .blur(radius: 28)
                    .opacity(0.5 + motion.energy * 0.45)
                    .scaleEffect(1.15 + motion.energy * 0.2)

                ZStack {
                    field(Color(red: 0.2, green: 0.45, blue: 1), time * motion.spin, 26)
                    field(Color(red: 0.5, green: 0.28, blue: 1), time * motion.spin * 0.82 + 2.1, 30)
                    field(Color(red: 0.95, green: 0.38, blue: 0.72), time * motion.spin * 1.12 + 3.9, 24)
                    field(Color(red: 0.25, green: 0.86, blue: 0.95), time * motion.spin * 0.7 + 5.3, 28)
                    Circle()
                        .fill(.white.opacity(0.2 + motion.energy * 0.5))
                        .frame(width: 26 + motion.energy * 20, height: 26 + motion.energy * 20)
                        .blur(radius: 10)
                    Ellipse()
                        .fill(.white.opacity(0.5))
                        .frame(width: 34, height: 14)
                        .blur(radius: 5)
                        .offset(x: -14, y: -24)
                }
                .frame(width: 108, height: 108)
                .clipShape(Circle())
                .overlay(
                    Circle()
                        .stroke(
                            LinearGradient(
                                colors: [.white.opacity(0.75), .white.opacity(0.05), .white.opacity(0.28)],
                                startPoint: .topLeading,
                                endPoint: .bottomTrailing
                            ),
                            lineWidth: 1.5
                        )
                )
                .scaleEffect(motion.scale)
            }
            .frame(height: 136)
        }
        .accessibilityLabel(accessibilityName)
    }

    private var accessibilityName: String {
        switch pipeline.state {
        case .idle: return "Idle"
        case .listening: return "Listening"
        case .processing: return "Thinking"
        case .speaking: return "Speaking"
        }
    }

    private func field(_ color: Color, _ angle: Double, _ radius: CGFloat) -> some View {
        Circle()
            .fill(color)
            .frame(width: 74, height: 74)
            .blur(radius: 18)
            .offset(x: CGFloat(cos(angle)) * radius, y: CGFloat(sin(angle)) * radius)
    }
}

private struct OrbMotion {
    var energy: CGFloat
    var scale: CGFloat
    var spin: Double
    var glow: Color

    init(state: VoicePipeline.PipelineState, mic: Float, playback: Float, time: TimeInterval) {
        let breathe = CGFloat(0.5 + 0.5 * sin(time * 1.3))
        switch state {
        case .idle:
            energy = 0.1 + breathe * 0.08
            spin = 0.35
            glow = Color(red: 0.35, green: 0.45, blue: 1)
        case .listening:
            let heard = CGFloat(min(1, mic / 0.07))
            energy = 0.22 + heard * 0.78
            spin = 0.65 + Double(heard) * 0.45
            glow = Color(red: 0.25, green: 0.72, blue: 1)
        case .processing:
            energy = 0.32 + breathe * 0.22
            spin = 1.05
            glow = Color(red: 0.55, green: 0.38, blue: 1)
        case .speaking:
            let said = CGFloat(min(1, playback / 0.07))
            energy = 0.38 + said * 0.62
            spin = 0.85 + Double(said) * 0.55
            glow = Color(red: 0.4, green: 0.55, blue: 1)
        }
        scale = 1 + energy * 0.08
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
