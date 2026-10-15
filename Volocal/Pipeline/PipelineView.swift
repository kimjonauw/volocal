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

/// Flat cartoon face. Expression is a few shapes (brows, lids, pupils, mouth)
/// driven by the conversation, plus squash-and-stretch from the audio.
/// No face model.
private struct VoicePresence: View {
    let pipeline: VoicePipeline

    var body: some View {
        TimelineView(.animation(minimumInterval: 1.0 / 30.0, paused: false)) { timeline in
            let levels = pipeline.sharedAudio.bridge.levels()
            let time = timeline.date.timeIntervalSinceReferenceDate
            let mood = Mood(state: pipeline.state, mic: levels.mic, playback: levels.playback, time: time)
            ZStack {
                Circle()
                    .fill(Color(red: 0.99, green: 0.78, blue: 0.16))
                    .overlay(Circle().stroke(Color.black.opacity(0.22), lineWidth: 2.5))
                    .frame(width: 120, height: 120)
                    .scaleEffect(x: mood.squashX, y: mood.squashY)

                Ellipse()
                    .fill(.white.opacity(0.55))
                    .frame(width: 34, height: 16)
                    .offset(x: -24, y: -34)

                Circle()
                    .fill(Color.pink.opacity(mood.blush))
                    .frame(width: 16, height: 10)
                    .offset(x: -36, y: 16)
                Circle()
                    .fill(Color.pink.opacity(mood.blush))
                    .frame(width: 16, height: 10)
                    .offset(x: 36, y: 16)

                HStack(spacing: 30) {
                    brow(lift: mood.browLeft, tilt: mood.browTilt)
                    brow(lift: mood.browRight, tilt: -mood.browTilt)
                }
                .offset(y: -38)

                HStack(spacing: 18) {
                    eye(mood: mood, blink: blinkAmount(time, delay: 0))
                    eye(mood: mood, blink: blinkAmount(time, delay: 0.04))
                }
                .offset(y: -8)

                MouthShape(open: mood.mouthOpen, smile: mood.smile)
                    .fill(.black.opacity(0.9))
                    .frame(width: 40, height: 30)
                    .offset(y: 30)
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

    private func blinkAmount(_ time: TimeInterval, delay: TimeInterval) -> CGFloat {
        let phase = (time + delay).truncatingRemainder(dividingBy: 3.6)
        if phase < 0.08 { return 0.12 }
        if phase > 0.18 && phase < 0.26 { return 0.18 }
        return 1
    }

    private func brow(lift: CGFloat, tilt: Double) -> some View {
        Capsule()
            .fill(.black.opacity(0.82))
            .frame(width: 22, height: 4)
            .rotationEffect(.degrees(tilt))
            .offset(y: lift)
    }

    private func eye(mood: Mood, blink: CGFloat) -> some View {
        ZStack {
            Ellipse()
                .fill(.white)
                .frame(width: mood.eyeWidth, height: mood.eyeHeight * blink)
            Circle()
                .fill(.black)
                .frame(width: mood.pupil, height: mood.pupil)
                .offset(x: mood.lookX, y: mood.lookY)
                .opacity(blink > 0.4 ? 1 : 0)
            Circle()
                .fill(.white)
                .frame(width: 3.5, height: 3.5)
                .offset(x: mood.lookX + 2.5, y: mood.lookY - 2)
                .opacity(blink > 0.4 ? 1 : 0)
        }
        .frame(width: 32, height: 32)
        .clipped()
    }
}

private struct Mood {
    var squashX: CGFloat
    var squashY: CGFloat
    var browLeft: CGFloat
    var browRight: CGFloat
    var browTilt: Double
    var eyeWidth: CGFloat
    var eyeHeight: CGFloat
    var pupil: CGFloat
    var lookX: CGFloat
    var lookY: CGFloat
    var mouthOpen: CGFloat
    var smile: CGFloat
    var blush: Double

    init(state: VoicePipeline.PipelineState, mic: Float, playback: Float, time: TimeInterval) {
        let chatter = CGFloat(abs(sin(time * 16)))
        let slot = floor(time / 0.9)
        let glance = CGFloat(sin(slot * 12.9898)) * 3
        switch state {
        case .idle:
            squashX = 1
            squashY = 1 + 0.02 * CGFloat(sin(time * 1.5))
            browLeft = -1
            browRight = -1
            browTilt = 10
            eyeWidth = 24
            eyeHeight = 24
            pupil = 10
            lookX = glance
            lookY = CGFloat(sin(slot * 4.2)) 
            mouthOpen = 0.04
            smile = 0.9
            blush = 0.12
        case .listening:
            let heard = CGFloat(min(1, mic / 0.07))
            squashX = 1 + heard * 0.04
            squashY = 1 - heard * 0.03
            browLeft = -9
            browRight = -9
            browTilt = 16
            eyeWidth = 26
            eyeHeight = 28
            pupil = 13
            lookX = glance * 0.6
            lookY = 2
            mouthOpen = 0.1 + heard * 0.4
            smile = 0.75
            blush = 0.28
        case .processing:
            squashX = 0.98
            squashY = 1.02
            browLeft = -8
            browRight = 3
            browTilt = -8
            eyeWidth = 22
            eyeHeight = 14
            pupil = 8
            lookX = -4
            lookY = -5
            mouthOpen = 0.15 + 0.1 * CGFloat(sin(time * 2.4))
            smile = 0.1
            blush = 0
        case .speaking:
            let said = max(0.4, CGFloat(min(1, playback / 0.07)))
            let jabber = said * (0.35 + 0.65 * chatter)
            squashX = 1 + jabber * 0.14
            squashY = 1 - jabber * 0.16
            browLeft = -4 - jabber * 4
            browRight = -4 - jabber * 4
            browTilt = 12
            eyeWidth = 24
            eyeHeight = 20
            pupil = 10
            lookX = glance * 0.4
            lookY = 0
            mouthOpen = jabber
            smile = 0.45 + (1 - said) * 0.3
            blush = 0.5
        }
    }
}

/// Closed smile, or a tall open mouth while the voice is loud.
private struct MouthShape: Shape {
    var open: CGFloat
    var smile: CGFloat

    func path(in rect: CGRect) -> Path {
        var path = Path()
        let open = min(1, max(0, open))
        let smile = min(1, max(0, smile))
        let left = CGPoint(x: rect.minX + 2, y: rect.midY - open * 2)
        let right = CGPoint(x: rect.maxX - 2, y: rect.midY - open * 2)
        let dip = rect.midY + 4 + smile * 10 + open * rect.height * 0.55
        let roof = rect.midY - 2 - open * rect.height * 0.28
        path.move(to: left)
        path.addQuadCurve(to: right, control: CGPoint(x: rect.midX, y: dip))
        path.addQuadCurve(to: left, control: CGPoint(x: rect.midX, y: roof))
        path.closeSubpath()
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
