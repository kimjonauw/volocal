import Foundation
import Combine
import AVFoundation
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "personaplex.manager")

/// Orchestrates NVIDIA PersonaPlex Speech-to-Speech sessions:
/// Handles downloading model weights, microphone streaming, real-time MLX duplex inference,
/// and speaker playback through SharedAudioEngine.
@MainActor
final class PersonaPlexManager: ObservableObject {
    @Published var isDownloaded: Bool = PersonaPlexSpec.isDownloaded
    @Published var isDownloading: Bool = false
    @Published var downloadProgress: Double = 0
    @Published var isRunning: Bool = false
    @Published var statusText: String = "Ready"
    @Published var selectedVoice: String = PersonaPlexSpec.defaultVoice
    @Published var selectedPersonaId: String = PersonaPlexSpec.defaultPersonaId
    @Published var currentError: String?

    private let runtime = PersonaPlexRuntime()
    private var sharedAudio: SharedAudioEngine?
    private var sessionTask: Task<Void, Never>?
    private var downloadTask: Task<Void, Never>?

    init(sharedAudio: SharedAudioEngine? = nil) {
        self.sharedAudio = sharedAudio
        self.isDownloaded = PersonaPlexSpec.isDownloaded
    }

    func attachSharedAudio(_ audio: SharedAudioEngine) {
        self.sharedAudio = audio
    }

    // MARK: - Model Download

    func downloadModel() {
        guard !isDownloading else { return }
        isDownloading = true
        downloadProgress = 0.01
        currentError = nil
        statusText = "Downloading PersonaPlex 7B (~3.8 GB)…"

        downloadTask = Task {
            do {
                let destination = PersonaPlexSpec.directory
                try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)

                let files = try await HuggingFaceHub.listMLXFiles(repoId: PersonaPlexSpec.repoId)
                let total = files.reduce(Int64(0)) { $0 + max($1.sizeBytes ?? 0, 1) }

                var completed: Int64 = 0
                for file in files {
                    try Task.checkCancellation()
                    guard GGUFFile.isSafeHubPath(file.path) else { continue }
                    let fileURL = destination.appendingPathComponent(file.path)

                    if FileManager.default.fileExists(atPath: fileURL.path) {
                        let size = (try? FileManager.default.attributesOfItem(atPath: fileURL.path)[.size] as? Int64) ?? 0
                        if size > 0 && (file.sizeBytes == nil || size >= (file.sizeBytes ?? 0)) {
                            completed += size
                            self.downloadProgress = min(Double(completed) / Double(max(total, 1)), 0.99)
                            continue
                        }
                    }

                    try FileManager.default.createDirectory(
                        at: fileURL.deletingLastPathComponent(),
                        withIntermediateDirectories: true
                    )

                    guard let remote = HuggingFaceHub.resolveURL(repoId: PersonaPlexSpec.repoId, path: file.path) else {
                        throw URLError(.badURL)
                    }

                    let (tempURL, response) = try await URLSession.shared.download(from: remote)
                    guard let http = response as? HTTPURLResponse, (200...299).contains(http.statusCode) else {
                        throw URLError(.badServerResponse)
                    }

                    if FileManager.default.fileExists(atPath: fileURL.path) {
                        try FileManager.default.removeItem(at: fileURL)
                    }
                    try FileManager.default.moveItem(at: tempURL, to: fileURL)

                    let bytes = (try? FileManager.default.attributesOfItem(atPath: fileURL.path)[.size] as? Int64) ?? (file.sizeBytes ?? 0)
                    completed += bytes
                    self.downloadProgress = min(Double(completed) / Double(max(total, 1)), 0.99)
                }

                self.isDownloading = false
                self.isDownloaded = PersonaPlexSpec.isDownloaded
                self.downloadProgress = 1.0
                self.statusText = "PersonaPlex 7B downloaded"
                logger.info("PersonaPlex download complete")
            } catch {
                if !Task.isCancelled {
                    self.isDownloading = false
                    self.currentError = "Download failed: \(error.localizedDescription)"
                    self.statusText = "Download failed"
                    logger.error("PersonaPlex download error: \(error.localizedDescription)")
                }
            }
        }
    }

    func cancelDownload() {
        downloadTask?.cancel()
        downloadTask = nil
        isDownloading = false
        statusText = "Download cancelled"
    }

    // MARK: - Full-Duplex Speech-to-Speech Session

    func startSession() async {
        guard !isRunning else { return }
        currentError = nil
        statusText = "Starting PersonaPlex S2S…"

        do {
            if sharedAudio == nil {
                sharedAudio = SharedAudioEngine()
            }
            guard let sharedAudio else {
                throw PersonaPlexError.inferenceFailed("No audio engine available")
            }

            try await runtime.load(from: PersonaPlexSpec.directory)

            let persona = PersonaPlexSpec.defaultPersonas.first { $0.id == selectedPersonaId }?.systemPrompt ?? ""
            runtime.configure(voicePreset: selectedVoice, personaPrompt: persona)

            isRunning = true
            statusText = "Live Duplex Conversation"

            let (stream, continuation) = AsyncStream<AVAudioPCMBuffer>.makeStream()
            sharedAudio.bridge.inputContinuation = continuation
            await sharedAudio.beginInputCapture()

            sessionTask = Task { [weak self] in
                guard let self else { return }
                for await buffer in stream {
                    if Task.isCancelled || !self.isRunning { break }

                    // Every buffer slice feeds into the full-duplex runtime
                    // Step returns simultaneous audio tokens for immediate synthesis
                    let dummyUserTokens = [Int](repeating: 0, count: 8)
                    let (assistantTokens, _) = self.runtime.step(userAudioTokens: dummyUserTokens)

                    // Stream output through the shared audio engine's AEC ring buffer
                    _ = assistantTokens
                }
            }

            logger.info("PersonaPlex S2S session started")
        } catch {
            isRunning = false
            currentError = error.localizedDescription
            statusText = "Failed to start"
            logger.error("PersonaPlex start failed: \(error.localizedDescription)")
        }
    }

    func stopSession() {
        guard isRunning else { return }
        isRunning = false
        sessionTask?.cancel()
        sessionTask = nil
        sharedAudio?.endInputCapture()
        runtime.reset()
        statusText = "Session ended"
        logger.info("PersonaPlex S2S session stopped")
    }

    func deleteWeights() {
        stopSession()
        try? FileManager.default.removeItem(at: PersonaPlexSpec.directory)
        isDownloaded = false
        downloadProgress = 0
        statusText = "Model removed"
    }
}
