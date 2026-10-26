import SwiftUI

struct ContentView: View {
    @EnvironmentObject var modelManager: UnifiedModelManager
    @State private var showPersonaSettings = false

    var body: some View {
        PipelineView()
            .safeAreaInset(edge: .bottom) {
                if modelManager.pipelineMode == .personaplex {
                    Button { showPersonaSettings = true } label: {
                        Label("Persona prompt & status", systemImage: "slider.horizontal.3")
                            .frame(maxWidth: .infinity)
                            .padding(.vertical, 10)
                    }
                    .background(.regularMaterial)
                }
            }
            .sheet(isPresented: $showPersonaSettings) { PersonaPromptSettingsView() }
    }
}
