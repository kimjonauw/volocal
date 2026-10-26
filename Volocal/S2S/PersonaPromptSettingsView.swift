import SwiftUI

enum PersonaConversationSettings {
    static let promptKey = "volocal.s2s.customConversationPrompt"
    static let stopKey = "volocal.s2s.lastResponseStop"
    static let defaultPrompt = "Chat casually as a relaxed, curious conversational partner. Respond directly to what I say in one or two short sentences, with contractions. Ask a follow-up only when relevant. Do not introduce yourself unless asked. Avoid customer-service phrases."
}

struct PersonaPromptSettingsView: View {
    @Environment(\.dismiss) private var dismiss
    @AppStorage(PersonaConversationSettings.promptKey) private var savedPrompt = ""
    @AppStorage(PersonaConversationSettings.stopKey) private var lastStop = ""
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
                Section("Last reply stop reason") {
                    Text(lastStop.isEmpty ? "No reply has finished with diagnostics yet." : lastStop)
                        .textSelection(.enabled)
                    Button("Clear diagnostic") { lastStop = "" }
                }
                Section("Interruption behavior") {
                    Text("In this diagnostic build, Persona ignores microphone input during its reply. Use the main mic button to stop the session. Safety cutoffs are reported above. Modular mode is unchanged.")
                    Text("The older waveform settings still contain legacy Persona TTS controls; they do not edit this speech-to-speech prompt.")
                        .font(.caption).foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Persona settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) { Button("Cancel") { dismiss() } }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") {
                        savedPrompt = draft.trimmingCharacters(in: .whitespacesAndNewlines)
                        dismiss()
                    }
                }
            }
            .onAppear { draft = savedPrompt.isEmpty ? PersonaConversationSettings.defaultPrompt : savedPrompt }
        }
    }
}
