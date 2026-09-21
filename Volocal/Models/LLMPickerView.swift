import SwiftUI

struct LLMPickerView: View {
    @EnvironmentObject var modelManager: UnifiedModelManager
    @Environment(\.dismiss) private var dismiss

    var onModelReady: ((LLMModelSpec) -> Void)?
    var onInstructionsChanged: ((String) -> Void)?
    var onNeedsReload: (() -> Void)?

    @State private var searchText = ""
    @State private var repoHits: [HuggingFaceHub.RepoHit] = []
    @State private var filesByRepo: [String: [HuggingFaceHub.RemoteFile]] = [:]
    @State private var expandedRepo: String?
    @State private var isSearching = false
    @State private var isListing = false
    @State private var pasteText = ""
    @State private var localError: String?
    @State private var searchGeneration = 0
    @State private var instructionsDraft = ""

    var body: some View {
        NavigationStack {
            List {
                if let error = localError ?? modelManager.error {
                    Section {
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }

                Section {
                    Text("Only GGUF files work here — that is what llama.cpp loads on-device. Safetensors / MLX / ONNX repos will not run.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                instructionsSection
                contextSection
                installedSection
                suggestedSection
                pasteSection
                searchSection
            }
            .navigationTitle("Language Model")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") { dismiss() }
                }
            }
            .onChange(of: searchText) { _, newValue in
                scheduleSearch(newValue)
            }
            .onAppear {
                instructionsDraft = modelManager.customInstructions
                modelManager.checkExistingModels()
            }
            .onDisappear {
                saveInstructions()
            }
        }
    }

    private var instructionsSection: some View {
        Section {
            TextEditor(text: $instructionsDraft)
                .frame(minHeight: 120)
                .font(.body)
            Button("Reset to default") {
                instructionsDraft = LLMManager.defaultInstructions
            }
            .font(.caption)
        } header: {
            Text("Your context / instructions")
        } footer: {
            Text("Sent as the system prompt on every turn. Use this for who you are, how it should talk, or facts it should remember. Takes effect on the next reply — no reload.")
        }
    }

    private var contextSection: some View {
        Section {
            Picker("Tokens", selection: contextSizeBinding) {
                ForEach(LLMContextWindow.choices, id: \.self) { size in
                    Text(LLMContextWindow.label(size)).tag(size)
                }
            }
            .pickerStyle(.inline)
        } header: {
            Text("Context window")
        } footer: {
            Text("llama.cpp n_ctx: how many tokens of instructions + recent chat fit in one pass. Default 2,048 is light. 8K/16K are fine on an iPhone 17 Pro with a 2B GGUF; 8B + STT + TTS at 16K can jetsam in LiveContainer. History length scales with this. The GGUF’s trained window is a second ceiling. Changing this reloads the model.")
        }
    }

    private var contextSizeBinding: Binding<UInt32> {
        Binding(
            get: { modelManager.contextSize },
            set: { newValue in
                let previous = modelManager.contextSize
                modelManager.setContextSize(newValue)
                if modelManager.contextSize != previous {
                    onNeedsReload?()
                }
            }
        )
    }

    private var installedSection: some View {
        Section {
            let installed = modelManager.installedLLMSpecs()
            if installed.isEmpty {
                Text("None yet")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(installed) { spec in
                    modelRow(spec, trailing: spec.isDownloaded ? "Ready" : nil)
                        .buttonStyle(.plain)
                        .swipeActions(edge: .trailing, allowsFullSwipe: true) {
                            Button(role: .destructive) {
                                deleteInstalled(spec)
                            } label: {
                                Label("Delete", systemImage: "trash")
                            }
                        }
                }
                .onDelete { offsets in
                    let specs = modelManager.installedLLMSpecs()
                    for index in offsets where specs.indices.contains(index) {
                        deleteInstalled(specs[index])
                    }
                }
            }
        } header: {
            Text("On this iPhone")
        } footer: {
            Text("Swipe left to delete a GGUF and free disk. Files app → On My iPhone → Volocal → models also works if this IPA is a normal install; LiveContainer guests should use swipe-delete here. STT/TTS packs stay in the FluidAudio cache until you delete the app.")
        }
    }

    private var suggestedSection: some View {
        Section("Suggested for iPhone") {
            ForEach(LLMModelSpec.suggested) { spec in
                modelRow(spec, trailing: spec.sizeBytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) })
            }
        }
    }

    private var pasteSection: some View {
        Section("Hugging Face repo or file URL") {
            TextField("org/repo or huggingface.co/…/file.gguf", text: $pasteText)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()
            Button("Look up") {
                Task { await lookupPasted() }
            }
            .disabled(pasteText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
    }

    private var searchSection: some View {
        Section("Search Hugging Face (GGUF)") {
            TextField("qwen, llama, gemma…", text: $searchText)
                .textInputAutocapitalization(.never)
                .autocorrectionDisabled()

            if isSearching {
                ProgressView()
            }

            ForEach(repoHits) { hit in
                DisclosureGroup(
                    isExpanded: Binding(
                        get: { expandedRepo == hit.id },
                        set: { isOn in
                            expandedRepo = isOn ? hit.id : nil
                            if isOn { Task { await loadFiles(hit.id) } }
                        }
                    )
                ) {
                    if isListing && expandedRepo == hit.id {
                        ProgressView()
                    } else if let files = filesByRepo[hit.id], !files.isEmpty {
                        ForEach(files) { file in
                            let spec = LLMModelSpec(
                                repoId: hit.id,
                                filename: file.path,
                                displayName: "\(hit.id.split(separator: "/").last ?? "") · \(file.filename)",
                                sizeBytes: file.sizeBytes,
                                sha256: file.sha256
                            )
                            modelRow(spec, trailing: spec.sizeDescription)
                        }
                    } else if filesByRepo[hit.id] != nil {
                        Text("No GGUF files in this repo")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(hit.id)
                            .font(.subheadline.weight(.medium))
                        Text("\(hit.downloads) downloads")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private func modelRow(_ spec: LLMModelSpec, trailing: String?) -> some View {
        Button {
            Task { await choose(spec) }
        } label: {
            HStack {
                VStack(alignment: .leading, spacing: 4) {
                    Text(spec.displayName)
                        .foregroundStyle(.primary)
                    Text(spec.repoId)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    if spec.isLikelyTooLargeForPhone {
                        Text("Likely too large for iPhone RAM")
                            .font(.caption)
                            .foregroundStyle(.orange)
                    }
                }
                Spacer()
                if modelManager.selectedLLM.id == spec.id {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                }
                if let trailing {
                    Text(trailing)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    private func saveInstructions() {
        modelManager.setInstructions(instructionsDraft)
        onInstructionsChanged?(modelManager.customInstructions)
    }

    private func deleteInstalled(_ spec: LLMModelSpec) {
        let wasSelected = modelManager.selectedLLM.id == spec.id
            || modelManager.selectedLLM.filename == spec.filename
        modelManager.deleteLLM(spec)
        if wasSelected {
            onNeedsReload?()
        }
    }

    private func choose(_ spec: LLMModelSpec) async {
        localError = nil
        modelManager.select(spec)
        if !modelManager.selectedLLM.isDownloaded {
            await modelManager.downloadSelectedLLM()
        }
        if modelManager.selectedLLM.isDownloaded {
            onModelReady?(modelManager.selectedLLM)
            dismiss()
        } else {
            localError = modelManager.error
        }
    }

    private func lookupPasted() async {
        localError = nil
        guard let parsed = HuggingFaceHub.parseUserInput(pasteText) else {
            localError = "Could not parse that Hugging Face URL or repo id."
            return
        }
        if let filename = parsed.filename {
            let spec = LLMModelSpec(
                repoId: parsed.repoId,
                filename: filename,
                displayName: filename,
                sizeBytes: nil,
                sha256: nil
            )
            await choose(spec)
            return
        }
        expandedRepo = parsed.repoId
        if !repoHits.contains(where: { $0.id == parsed.repoId }) {
            repoHits.insert(HuggingFaceHub.RepoHit(id: parsed.repoId, downloads: 0, likes: 0), at: 0)
        }
        await loadFiles(parsed.repoId)
    }

    private func scheduleSearch(_ query: String) {
        searchGeneration += 1
        let generation = searchGeneration
        Task {
            try? await Task.sleep(for: .milliseconds(400))
            guard generation == searchGeneration else { return }
            await runSearch(query, generation: generation)
        }
    }

    private func runSearch(_ query: String, generation: Int) async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 2 else {
            repoHits = []
            return
        }
        isSearching = true
        defer { isSearching = false }
        do {
            let hits = try await HuggingFaceHub.searchRepos(query: trimmed)
            guard generation == searchGeneration else { return }
            repoHits = hits
            localError = nil
        } catch {
            guard generation == searchGeneration else { return }
            localError = error.localizedDescription
        }
    }

    private func loadFiles(_ repoId: String) async {
        if filesByRepo[repoId] != nil { return }
        isListing = true
        defer { isListing = false }
        do {
            filesByRepo[repoId] = try await HuggingFaceHub.listGGUFFiles(repoId: repoId)
            localError = nil
        } catch {
            localError = "Could not list \(repoId): \(error.localizedDescription)"
        }
    }
}
