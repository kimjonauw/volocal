import SwiftUI

enum PersonaConversationSettings {
    static let promptKey = "volocal.s2s.customConversationPrompt"
    static let stopKey = "volocal.s2s.lastResponseStop"
    static let inputTestKey = "volocal.s2s.testNextInput"
    static let inputResultKey = "volocal.s2s.lastInputTest"
    static let defaultPrompt = "Chat casually as a relaxed, curious conversational partner. Respond directly to what I say in one or two short sentences, with contractions. Ask a follow-up only when relevant. Do not introduce yourself unless asked. Avoid customer-service phrases."
}

struct PersonaPromptSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage(PersonaConversationSettings.promptKey) private var savedPrompt = ""
    @AppStorage(PersonaConversationSettings.stopKey) private var lastStop = ""
    @AppStorage(PersonaConversationSettings.inputTestKey) private var testNextInput = false
    @AppStorage(PersonaConversationSettings.inputResultKey) private var inputResult = ""
    @State private var draft = ""

    var body: some View {
        NavigationStack {
            Form {
                Section {
                    TextEditor(text: $draft)
                        .frame(minHeight: 180)
                        .accessibilityLabel("Custom conversation prompt")
                    Text("\(draft.count) characters").font(.caption).foregroundStyle(.secondary)
                    Button("Use conversational default") { draft = PersonaConversationSettings.defaultPrompt }
                } header: { Text("Custom conversation prompt") } footer: {
                    Text("Saved only on this phone. Applies to the next reply, not one already generating. Blank uses the default. The audio-generation and voice prefix is kept automatically. This does not add conversation memory.")
                }
                Section {
                    Toggle("Transcribe next utterance only", isOn: $testNextInput)
                    Text(inputResult.isEmpty ? "No input test yet." : inputResult).textSelection(.enabled)
                    Button("Clear input test result") { inputResult = "" }
                } header: { Text("Verify speech understanding") } footer: {
                    Text("Enable this, close settings, and speak one short sentence. LFM will transcribe the exact audio captured by Persona instead of answering aloud. The switch clears automatically when that turn is captured. Reopen here to read the result. No recording is saved or uploaded. The transcript stays on this phone until cleared or replaced. Normal replies do not run this extra test.")
                }
                Section("Last reply stop reason") {
                    Text(lastStop.isEmpty ? "No reply has finished with diagnostics yet." : lastStop).textSelection(.enabled)
                    Button("Clear diagnostic") { lastStop = "" }
                }
                Section("Interruption behavior") {
                    Text("Persona ignores microphone input during its reply in this diagnostic build. Use the main mic button to stop the session. Safety cutoffs are reported above. Modular mode is unchanged.")
                    Text("Use this editor, not the legacy Persona TTS controls in waveform settings, to change the speech-to-speech prompt.").font(.caption).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Persona settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Close") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save prompt") { savedPrompt = draft.trimmingCharacters(in: .whitespacesAndNewlines); dismiss() }
                }
            }
            .onAppear { draft = savedPrompt.isEmpty ? PersonaConversationSettings.defaultPrompt : savedPrompt }
        }
    }
}
