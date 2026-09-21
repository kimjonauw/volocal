import SwiftUI

struct LLMPickerView: View {
    @EnvironmentObject var modelManager: UnifiedModelManager
    @Environment(\.dismiss) private var dismiss

    var onModelReady: ((LLMModelSpec) -> Void)?

    @State private var searchText = ""
    @State private var repoHits: [HuggingFaceHub.RepoHit] = []
    @State private var filesByRepo: [String: [HuggingFaceHub.RemoteFile]] = [:]
    @State private var expandedRepo: String?
    @State private var isSearching = false
    @State private var isListing = false
    @State private var pasteText = ""
    @State private var localError: String?

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
        }
    }

    private var installedSection: some View {
        Section("On this iPhone") {
            let installed = modelManager.installedLLMSpecs()
            if installed.isEmpty {
                Text("None yet")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(installed) { spec in
                    modelRow(spec, trailing: spec.isDownloaded ? "Ready" : nil)
                }
            }
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
                                filename: file.filename,
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

    private func choose(_ spec: LLMModelSpec) async {
        localError = nil
        modelManager.select(spec)
        if !spec.isDownloaded {
            await modelManager.downloadSelectedLLM()
        }
        if spec.isDownloaded || modelManager.selectedLLM.isDownloaded {
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
        Task {
            try? await Task.sleep(for: .milliseconds(400))
            guard query == searchText else { return }
            await runSearch(query)
        }
    }

    private func runSearch(_ query: String) async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed.count >= 2 else {
            repoHits = []
            return
        }
        isSearching = true
        defer { isSearching = false }
        do {
            repoHits = try await HuggingFaceHub.searchRepos(query: trimmed)
            localError = nil
        } catch {
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
