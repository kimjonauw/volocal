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

/// Kawaii ball: big eyes low on the face, a tiny mouth tucked under them,
/// blush, and no brows. Baby-face proportions, shapes only.
private struct VoicePresence: View {
    let pipeline: VoicePipeline
    private let ink = Color(red: 0.33, green: 0.2, blue: 0.14)

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: false)) { timeline in
            let levels = pipeline.sharedAudio.bridge.levels()
            let time = timeline.date.timeIntervalSinceReferenceDate
            let mood = Mood(state: pipeline.state, mic: levels.mic, playback: levels.playback, time: time)
            let blink = blinkAmount(time)
            ZStack {
                Circle()
                    .fill(
                        RadialGradient(
                            colors: [
                                Color(red: 1, green: 0.93, blue: 0.62),
                                Color(red: 1, green: 0.78, blue: 0.28)
                            ],
                            center: UnitPoint(x: 0.38, y: 0.32),
                            startRadius: 2,
                            endRadius: 78
                        )
                    )
                    .frame(width: 116, height: 116)
                    .shadow(color: Color(red: 0.9, green: 0.62, blue: 0.12).opacity(0.35), radius: 10, y: 6)

                Ellipse()
                    .fill(Color.white.opacity(0.7))
                    .frame(width: 28, height: 14)
                    .offset(x: -26, y: -32)

                Ellipse()
                    .fill(Color(red: 1, green: 0.55, blue: 0.62).opacity(mood.blush))
                    .frame(width: 22, height: 12)
                    .offset(x: -34, y: 18)
                Ellipse()
                    .fill(Color(red: 1, green: 0.55, blue: 0.62).opacity(mood.blush))
                    .frame(width: 22, height: 12)
                    .offset(x: 34, y: 18)

                HStack(spacing: 22) {
                    eye(mood: mood, blink: blink)
                    eye(mood: mood, blink: blink)
                }
                .offset(y: 8 + mood.eyeShiftY)

                mouth(open: mood.mouthOpen)
                    .offset(y: 34)
            }
            .scaleEffect(x: mood.squashX, y: mood.squashY)
            .offset(y: mood.bob)
            .frame(height: 140)
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

    private func blinkAmount(_ time: TimeInterval) -> CGFloat {
        let phase = time.truncatingRemainder(dividingBy: 4.2)
        if phase < 0.09 { return 0.08 }
        return 1
    }

    private func eye(mood: Mood, blink: CGFloat) -> some View {
        let tall = 32 * mood.eyeScale * blink
        let wide = 24 * mood.eyeScale
        return ZStack {
            if mood.happy > 0.5 {
                HappyEye()
                    .stroke(ink, style: StrokeStyle(lineWidth: 3.4, lineCap: .round))
                    .frame(width: wide + 4, height: 16)
                    .scaleEffect(y: max(0.2, blink))
            } else {
                Capsule()
                    .fill(ink)
                    .frame(width: wide, height: max(3, tall))
                    .offset(x: mood.lookX, y: mood.lookY)
                Circle()
                    .fill(.white)
                    .frame(width: 8, height: 8)
                    .offset(x: mood.lookX - 4, y: mood.lookY - 7)
                    .opacity(blink > 0.4 ? 1 : 0)
                Circle()
                    .fill(.white.opacity(0.9))
                    .frame(width: 3, height: 3)
                    .offset(x: mood.lookX + 3, y: mood.lookY - 1)
                    .opacity(blink > 0.4 ? 1 : 0)
            }
        }
        .frame(width: 32, height: 34)
    }

    private func mouth(open: CGFloat) -> some View {
        ZStack {
            SmileArc()
                .stroke(ink, style: StrokeStyle(lineWidth: 3.2, lineCap: .round))
                .frame(width: 16, height: 8)
                .opacity(open < 0.22 ? 1 : 0)
            Capsule()
                .fill(ink)
                .frame(width: 9 + open * 5, height: 5 + open * 7)
                .opacity(open < 0.22 ? 0 : 1)
        }
        .frame(width: 22, height: 16)
    }
}

private struct Mood {
    var squashX: CGFloat
    var squashY: CGFloat
    var bob: CGFloat
    var eyeScale: CGFloat
    var eyeShiftY: CGFloat
    var lookX: CGFloat
    var lookY: CGFloat
    var happy: CGFloat
    var mouthOpen: CGFloat
    var blush: Double

    init(state: VoicePipeline.PipelineState, mic: Float, playback: Float, time: TimeInterval) {
        let chatter = CGFloat(abs(sin(time * 14)))
        bob = CGFloat(sin(time * 1.6)) * 2.5
        switch state {
        case .idle:
            squashX = 1
            squashY = 1
            eyeScale = 1
            eyeShiftY = 0
            lookX = 0
            lookY = 0
            happy = 0
            mouthOpen = 0
            blush = 0.38
        case .listening:
            let heard = CGFloat(min(1, mic / 0.08))
            squashX = 1 + heard * 0.02
            squashY = 1 - heard * 0.015
            eyeScale = 1.08
            eyeShiftY = 0
            lookX = 0
            lookY = 1
            happy = 0
            mouthOpen = heard * 0.35
            blush = 0.5
        case .processing:
            squashX = 1
            squashY = 1
            eyeScale = 0.94
            eyeShiftY = -7
            lookX = -1
            lookY = -3
            happy = 0
            mouthOpen = 0
            blush = 0.28
            bob = CGFloat(sin(time * 0.9)) * 1.5
        case .speaking:
            let said = CGFloat(min(1, playback / 0.08))
            let jabber = 0.25 + 0.75 * said * (0.45 + 0.55 * chatter)
            squashX = 1 + jabber * 0.045
            squashY = 1 - jabber * 0.04
            eyeScale = 1
            eyeShiftY = 0
            lookX = 0
            lookY = 0
            happy = 1
            mouthOpen = 0.28 + jabber * 0.55
            blush = 0.62
        }
    }
}

/// Upward crescent, the closed-eye smile.
private struct HappyEye: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.maxY - 1))
        path.addQuadCurve(
            to: CGPoint(x: rect.maxX, y: rect.maxY - 1),
            control: CGPoint(x: rect.midX, y: rect.minY)
        )
        return path
    }
}

/// Short smile tucked under the eyes.
private struct SmileArc: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        path.move(to: CGPoint(x: rect.minX, y: rect.minY + 1))
        path.addQuadCurve(
            to: CGPoint(x: rect.maxX, y: rect.minY + 1),
            control: CGPoint(x: rect.midX, y: rect.maxY)
        )
        return path
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
