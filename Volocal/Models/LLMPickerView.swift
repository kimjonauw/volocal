import SwiftUI

struct LLMPickerView: View {
    @EnvironmentObject var modelManager: UnifiedModelManager
    @Environment(\.dismiss) private var dismiss

    var onModelReady: ((LLMModelSpec) -> Void)?
    var onInstructionsChanged: ((String) -> Void)?
    var onNeedsReload: (() -> Void)?
    var countTokens: ((String) async -> Int?)?
    var onThinkingChanged: ((Bool) -> Void)?

    @State private var searchText = ""
    @State private var repoHits: [HuggingFaceHub.RepoHit] = []
    @State private var filesByRepo: [String: [HuggingFaceHub.RemoteFile]] = [:]
    @State private var mlxPackRepos: Set<String> = []
    @State private var mlxPackBytes: [String: Int64] = [:]
    @State private var expandedRepo: String?
    @State private var isSearching = false
    @State private var isListing = false
    @State private var pasteText = ""
    @State private var localError: String?
    @State private var searchGeneration = 0
    @State private var instructionsDraft = ""
    @State private var contextLog = LLMContextWindow.logValue(LLMContextWindow.default)
    @State private var contextAtDragStart = LLMContextWindow.default
    @State private var isDraggingContext = false
    @State private var instructionTokens: Int = 0
    @State private var instructionTokensAreExact = false
    @State private var tokenCountGeneration = 0

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
                    Text("GGUF files run in llama.cpp. MLX packs are a folder of weights and are faster on this iPhone for the same size model. Prefer a 4-bit 2B–4B. Qwen 3.5 2B is the known-good pick in either format. A GGUF fine-tune does not load as MLX unless that exact model was published as an MLX pack.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                instructionsSection
                thinkingSection
                contextSection
                installedSection
                suggestedSection
                mlxSection
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
                contextLog = LLMContextWindow.logValue(modelManager.contextSize)
                modelManager.checkExistingModels(preservingLLMDownload: true)
                refreshTokenCount()
            }
            .onChange(of: instructionsDraft) { _, _ in
                refreshTokenCount()
            }
            .onChange(of: modelManager.ttsExpressionsEnabled) { _, _ in
                refreshTokenCount()
            }
            .onDisappear {
                saveInstructions()
            }
            .safeAreaInset(edge: .bottom) {
                llmTransferBanner
            }
        }
    }

    @ViewBuilder
    private var llmTransferBanner: some View {
        if case .downloading(let progress) = modelManager.modelStates[.llm] {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    ProgressView()
                    Text("Downloading \(modelManager.selectedLLM.displayName)")
                        .font(.subheadline.weight(.medium))
                    Spacer()
                    Button("Cancel") {
                        modelManager.cancelLLMDownload()
                    }
                    .font(.caption)
                }
                ProgressView(value: progress)
                Text("\(Int(progress * 100))% · \(modelManager.selectedLLM.sizeDescription). Stay on this screen. An MLX pack is a folder: the percent moves while the weight file downloads, then the tokenizer is saved.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding()
            .background(.bar)
        } else if let message = localError ?? modelManager.error, !message.isEmpty {
            Text(message)
                .font(.caption)
                .foregroundStyle(.red)
                .padding()
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.bar)
        }
    }

    private var instructionsSection: some View {
        Section {
            TextEditor(text: $instructionsDraft)
                .frame(minHeight: 120)
                .font(.body)
            HStack {
                Text(instructionTokenCaption)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(instructionTokenColor)
                Spacer()
                Button("Reset to default") {
                    instructionsDraft = LLMManager.defaultInstructions
                }
                .font(.caption)
            }
        } header: {
            Text("Your context / instructions")
        } footer: {
            Text("Sent as the system prompt on every turn. The count is this paste plus the expression cue if that toggle is on. Chat turns are extra and use a sliding window — old turns are recapped, not replayed in full. Exact GGUF tokens appear after a language model is loaded.")
        }
    }

    private var thinkingSection: some View {
        Section {
            Toggle("Skip hidden reasoning", isOn: Binding(
                get: { modelManager.suppressThinking },
                set: { newValue in
                    modelManager.setSuppressThinking(newValue)
                    onThinkingChanged?(newValue)
                }
            ))
        } header: {
            Text("Model thinking")
        } footer: {
            Text("On (default): thinking is disabled in the prompt so Gemma 4 / Qwen 3 answer immediately. Chat markup like <|im_end|> ends the turn instead of being spoken.")
        }
    }

    private var composedInstructionPrompt: String {
        LLMManager.systemPrompt(
            from: instructionsDraft,
            expressions: modelManager.ttsExpressionsEnabled
        )
    }

    private var instructionTokenCaption: String {
        let window = Int(modelManager.contextSize)
        let percent = window == 0 ? 0 : min(100, (instructionTokens * 100) / window)
        let kind = instructionTokensAreExact ? "tokens" : "≈ tokens"
        return "\(instructionTokens) \(kind) · \(percent)% of \(LLMContextWindow.formatted(modelManager.contextSize))"
    }

    private var instructionTokenColor: Color {
        let window = Int(modelManager.contextSize)
        if instructionTokens > window - 256 { return .red }
        if instructionTokens * 2 > window { return .orange }
        return .secondary
    }

    private func refreshTokenCount() {
        let prompt = composedInstructionPrompt
        instructionTokens = PromptTokenCounter.estimate(prompt)
        instructionTokensAreExact = false
        tokenCountGeneration += 1
        let generation = tokenCountGeneration
        Task {
            try? await Task.sleep(for: .milliseconds(250))
            guard generation == tokenCountGeneration else { return }
            if let exact = await countTokens?(prompt) {
                instructionTokens = exact
                instructionTokensAreExact = true
            }
        }
    }

    private var contextSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text(LLMContextWindow.label(modelManager.contextSize))
                        .font(.body.monospacedDigit())
                    Spacer()
                    Button("Reset") {
                        let previous = modelManager.contextSize
                        modelManager.setContextSize(LLMContextWindow.default, persist: true)
                        contextLog = LLMContextWindow.logValue(LLMContextWindow.default)
                        if modelManager.contextSize != previous {
                            onNeedsReload?()
                        }
                    }
                    .font(.caption)
                    .disabled(modelManager.contextSize == LLMContextWindow.default)
                }
                Slider(value: $contextLog, in: LLMContextWindow.logRange) { editing in
                    let size = LLMContextWindow.fromLog(contextLog)
                    if editing {
                        if !isDraggingContext {
                            isDraggingContext = true
                            contextAtDragStart = modelManager.contextSize
                        }
                        modelManager.setContextSize(size, persist: false)
                    } else {
                        let started = isDraggingContext ? contextAtDragStart : modelManager.contextSize
                        isDraggingContext = false
                        modelManager.setContextSize(size, persist: true)
                        contextLog = LLMContextWindow.logValue(modelManager.contextSize)
                        if modelManager.contextSize != started {
                            onNeedsReload?()
                        }
                    }
                }
                .accessibilityLabel("Context window")
                HStack {
                    Text("2K")
                    Spacer()
                    Text("128K")
                }
                .font(.caption2)
                .foregroundStyle(.secondary)
            }
            .padding(.vertical, 4)
        } header: {
            Text("Context window")
        } footer: {
            Text("Log slider of llama.cpp n_ctx (snaps to 512). Instructions + recent chat + recap of older turns must fit. Default 2,048 is light. History is a sliding window; dropped turns are compressed into a short memory instead of being forgotten. Reloads the model when you lift your finger.")
        }
    }

    private var installedSection: some View {
        Section {
            let installed = modelManager.installedLLMSpecs()
            if installed.isEmpty {
                Text("None yet")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(installed) { spec in
                    modelRow(spec, trailing: installedTrailing(spec))
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
        Section("Suggested GGUF") {
            ForEach(LLMModelSpec.suggested) { spec in
                modelRow(spec, trailing: spec.sizeBytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) })
            }
        }
    }

    private var mlxSection: some View {
        Section("MLX Swift") {
            Text("Downloads the weight folder from Hugging Face. Faster decode than GGUF. Still shares the GPU with PocketTTS.")
                .font(.caption)
                .foregroundStyle(.secondary)
            ForEach(LLMModelSpec.suggestedMLX) { spec in
                modelRow(spec, trailing: spec.sizeDescription)
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
        Section("Search Hugging Face") {
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
                    } else if hit.offersMLX || mlxPackRepos.contains(hit.id) {
                        let leaf = hit.id.split(separator: "/").last.map(String.init) ?? hit.id
                        let bytes = mlxPackBytes[hit.id]
                        let spec = LLMModelSpec.mlx(repoId: hit.id, displayName: leaf, sizeBytes: bytes)
                        modelRow(spec, trailing: bytes == nil ? "MLX" : spec.sizeDescription)
                    } else if filesByRepo[hit.id] != nil {
                        Text("No GGUF or MLX weights in this repo")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } label: {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(hit.id)
                            .font(.subheadline.weight(.medium))
                        Text(hit.offersMLX ? "MLX pack · \(hit.downloads) downloads" : "\(hit.downloads) downloads")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }

    private func installedTrailing(_ spec: LLMModelSpec) -> String? {
        if let size = spec.sizeBytes, size > 0 {
            return ByteCountFormatter.string(fromByteCount: size, countStyle: .file)
        }
        return spec.isDownloaded ? "Ready" : nil
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
                if isDownloading(spec) {
                    ProgressView()
                } else if spec.isDownloaded && modelManager.selectedLLM.id == spec.id {
                    Image(systemName: "checkmark.circle.fill")
                        .foregroundStyle(.green)
                } else if modelManager.selectedLLM.id == spec.id {
                    Text("Selected")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                if let trailing {
                    Text(trailing)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
            }
        }
        .buttonStyle(.borderless)
    }

    private func isDownloading(_ spec: LLMModelSpec) -> Bool {
        if case .downloading = modelManager.modelStates[.llm] {
            return modelManager.selectedLLM.id == spec.id
        }
        return false
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
        modelManager.error = nil
        modelManager.select(spec)
        if modelManager.selectedLLM.isDownloaded {
            onModelReady?(modelManager.selectedLLM)
            dismiss()
            return
        }
        await modelManager.downloadSelectedLLM()
        if modelManager.selectedLLM.id != spec.id {
            return
        }
        if modelManager.selectedLLM.isDownloaded {
            onModelReady?(modelManager.selectedLLM)
            dismiss()
        } else if case .downloading = modelManager.modelStates[.llm] {
            return
        } else {
            localError = modelManager.error ?? "Download did not finish. Stay on Wi-Fi and tap the GGUF again."
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
        if parsed.repoId.lowercased().contains("mlx") {
            let leaf = parsed.repoId.split(separator: "/").last.map(String.init) ?? parsed.repoId
            await choose(LLMModelSpec.mlx(repoId: parsed.repoId, displayName: leaf))
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
            async let ggufHits = HuggingFaceHub.searchRepos(query: trimmed, filter: "gguf")
            async let mlxHits = HuggingFaceHub.searchRepos(query: trimmed, filter: "mlx")
            let merged = try await mergeSearchHits(gguf: ggufHits, mlx: mlxHits)
            guard generation == searchGeneration else { return }
            repoHits = merged
            localError = nil
        } catch {
            guard generation == searchGeneration else { return }
            localError = error.localizedDescription
        }
    }

    private func mergeSearchHits(
        gguf: [HuggingFaceHub.RepoHit],
        mlx: [HuggingFaceHub.RepoHit]
    ) -> [HuggingFaceHub.RepoHit] {
        var byID: [String: HuggingFaceHub.RepoHit] = [:]
        for hit in gguf {
            byID[hit.id] = hit
        }
        for hit in mlx {
            if var existing = byID[hit.id] {
                existing.offersMLX = true
                byID[hit.id] = existing
            } else {
                byID[hit.id] = hit
            }
        }
        return byID.values.sorted { $0.downloads > $1.downloads }
    }

    private func loadFiles(_ repoId: String) async {
        if filesByRepo[repoId] != nil { return }
        isListing = true
        defer { isListing = false }
        do {
            filesByRepo[repoId] = try await HuggingFaceHub.listGGUFFiles(repoId: repoId)
            if filesByRepo[repoId]?.isEmpty != false {
                let mlxFiles = try await HuggingFaceHub.listMLXFiles(repoId: repoId)
                let hasTokenizer = mlxFiles.contains { $0.path == "tokenizer.json" }
                let hasWeights = mlxFiles.contains {
                    $0.path.lowercased().hasSuffix(".safetensors")
                        && !$0.path.lowercased().hasSuffix(".index.json")
                }
                if hasTokenizer && hasWeights {
                    mlxPackRepos.insert(repoId)
                    let total = mlxFiles.reduce(Int64(0)) { $0 + max($1.sizeBytes ?? 0, 0) }
                    if total > 0 {
                        mlxPackBytes[repoId] = total
                    }
                }
            }
            localError = nil
        } catch {
            localError = "Could not list \(repoId): \(error.localizedDescription)"
        }
    }
}
