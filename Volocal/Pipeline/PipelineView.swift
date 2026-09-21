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
                    Text(pipeline.state.label)
                        .font(.subheadline)
                        .foregroundStyle(.secondary)

                    Text(modelManager.selectedLLM.displayName)
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)

                    Text("\(modelManager.selectedSTT.displayName) · \(modelManager.selectedTTS.displayName)")
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
                LLMPickerView { _ in
                    pipeline.invalidateForReload()
                }
                .environmentObject(modelManager)
            }
            .sheet(isPresented: $showVoicePicker) {
                VoiceEnginePickerView {
                    pipeline.invalidateForReload()
                }
                .environmentObject(modelManager)
            }
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
