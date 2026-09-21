import Foundation
import CryptoKit
import FluidAudio
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "models")

private let selectedLLMKey = "volocal.selectedLLM.spec"
private let selectedSTTKey = "volocal.selectedSTT.engine"
private let selectedTTSKey = "volocal.selectedTTS.engine"
private let onboardedKey = "volocal.hasCompletedOnboarding"

/// Unified model manager tracking download state for STT, TTS, and the selected GGUF.
@MainActor
final class UnifiedModelManager: ObservableObject {
    @Published var modelStates: [ModelRegistry.ModelType: ModelState] = [:]
    @Published var error: String?
    @Published var selectedLLM: LLMModelSpec
    @Published var selectedSTT: STTEngine
    @Published var selectedTTS: TTSEngine
    @Published var hasCompletedOnboarding: Bool

    enum ModelState: Equatable {
        case notDownloaded
        case downloading(progress: Double)
        case downloaded
        case error(String)

        var isReady: Bool {
            if case .downloaded = self { return true }
            return false
        }

        var progress: Double {
            if case .downloading(let p) = self { return p }
            if case .downloaded = self { return 1.0 }
            return 0
        }
    }

    var allModelsReady: Bool {
        ModelRegistry.ModelType.allCases.allSatisfy { modelStates[$0]?.isReady == true }
    }

    var llmModelPath: String? {
        guard selectedLLM.isDownloaded else { return nil }
        return selectedLLM.localURL.path
    }

    init() {
        if let data = UserDefaults.standard.data(forKey: selectedLLMKey),
           let spec = try? JSONDecoder().decode(LLMModelSpec.self, from: data) {
            selectedLLM = spec
        } else {
            selectedLLM = .default
        }
        if let raw = UserDefaults.standard.string(forKey: selectedSTTKey),
           let engine = STTEngine(rawValue: raw) {
            selectedSTT = engine
        } else {
            selectedSTT = .parakeetEou320
        }
        if let raw = UserDefaults.standard.string(forKey: selectedTTSKey),
           let engine = TTSEngine(rawValue: raw) {
            selectedTTS = engine
        } else {
            selectedTTS = .pocketTts
        }
        hasCompletedOnboarding = UserDefaults.standard.bool(forKey: onboardedKey)
        checkExistingModels()
    }

    func persistSelection() {
        if let data = try? JSONEncoder().encode(selectedLLM) {
            UserDefaults.standard.set(data, forKey: selectedLLMKey)
        }
        UserDefaults.standard.set(selectedSTT.rawValue, forKey: selectedSTTKey)
        UserDefaults.standard.set(selectedTTS.rawValue, forKey: selectedTTSKey)
        checkExistingModels()
    }

    func select(_ spec: LLMModelSpec) {
        selectedLLM = spec
        persistSelection()
    }

    /// Show onboarding again so the user can pick another engine or GGUF after a load failure.
    func reopenSetup() {
        hasCompletedOnboarding = false
        UserDefaults.standard.set(false, forKey: onboardedKey)
    }

    func selectSTT(_ engine: STTEngine) {
        guard engine != selectedSTT else { return }
        selectedSTT = engine
        persistSelection()
    }

    func selectTTS(_ engine: TTSEngine) {
        guard engine != selectedTTS else { return }
        selectedTTS = engine
        persistSelection()
    }

    func checkExistingModels() {
        if selectedLLM.isDownloaded {
            modelStates[.llm] = .downloaded
        } else {
            modelStates[.llm] = .notDownloaded
        }

        if selectedSTT.isDownloaded(in: FluidAudioCache.asrModelsRoot) {
            modelStates[.stt] = .downloaded
        } else {
            modelStates[.stt] = .notDownloaded
        }

        if selectedTTS.isDownloaded() {
            modelStates[.tts] = .downloaded
        } else {
            modelStates[.tts] = .notDownloaded
        }

        markOnboardedIfReady()
    }

    private func markOnboardedIfReady() {
        if allModelsReady {
            hasCompletedOnboarding = true
            UserDefaults.standard.set(true, forKey: onboardedKey)
        }
    }

    func downloadAllModels() async {
        await withTaskGroup(of: Void.self) { group in
            if modelStates[.llm]?.isReady != true {
                group.addTask { await self.downloadLLM() }
            }
            if modelStates[.stt]?.isReady != true {
                group.addTask { await self.downloadSTT() }
            }
            if modelStates[.tts]?.isReady != true {
                group.addTask { await self.downloadTTS() }
            }
        }
        if allModelsReady {
            hasCompletedOnboarding = true
            UserDefaults.standard.set(true, forKey: onboardedKey)
        }
    }

    func retryModel(_ type: ModelRegistry.ModelType) async {
        modelStates[type] = .notDownloaded
        error = nil

        switch type {
        case .llm: await downloadLLM()
        case .stt: await downloadSTT()
        case .tts: await downloadTTS()
        }
    }

    func downloadSelectedLLM() async {
        await downloadLLM()
    }

    func installedLLMSpecs() -> [LLMModelSpec] {
        var found: [LLMModelSpec] = []
        let fm = FileManager.default

        if selectedLLM.isDownloaded {
            found.append(selectedLLM)
        }

        let llmRoot = ModelRegistry.llmDirectory
        if let folders = try? fm.contentsOfDirectory(at: llmRoot, includingPropertiesForKeys: nil) {
            for folder in folders where folder.hasDirectoryPath {
                if let files = try? fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: [.fileSizeKey]) {
                    for file in files where LLMFileFilter.isLoadableGGUF(file.lastPathComponent) {
                        let repoId = folder.lastPathComponent.replacingOccurrences(of: "__", with: "/")
                        let size = (try? file.resourceValues(forKeys: [.fileSizeKey]).fileSize).map { Int64($0) }
                        let spec = LLMModelSpec(
                            repoId: repoId,
                            filename: file.lastPathComponent,
                            displayName: file.lastPathComponent,
                            sizeBytes: size,
                            sha256: nil
                        )
                        if !found.contains(where: { $0.id == spec.id }) {
                            found.append(spec)
                        }
                    }
                }
            }
        }

        let legacy = ModelRegistry.modelsDirectory.appendingPathComponent(LLMModelSpec.default.filename)
        if fm.fileExists(atPath: legacy.path),
           !found.contains(where: { $0.filename == LLMModelSpec.default.filename }) {
            found.insert(.default, at: 0)
        }

        return found
    }

    private func downloadLLM() async {
        let spec = selectedLLM
        modelStates[.llm] = .downloading(progress: 0)

        if spec.isDownloaded {
            modelStates[.llm] = .downloaded
            return
        }

        guard HuggingFaceHub.isValidRepoId(spec.repoId), GGUFFile.isSafeHubPath(spec.filename) else {
            modelStates[.llm] = .error("Invalid Hugging Face repo or file path")
            return
        }

        guard let url = spec.downloadURL else {
            modelStates[.llm] = .error("Invalid Hugging Face URL")
            return
        }

        let destination = spec.nestedLocalURL.standardizedFileURL
        guard GGUFFile.isInsideModelsDirectory(destination) else {
            modelStates[.llm] = .error("Refusing to write GGUF outside the models folder")
            return
        }

        try? FileManager.default.createDirectory(
            at: destination.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        let expectedBytes = spec.sizeBytes ?? 0

        let result: Result<URL, Error> = await withCheckedContinuation { continuation in
            let delegate = LLMDownloadDelegate(
                onProgress: { [weak self] bytesWritten, totalExpected in
                    let total = totalExpected > 0 ? totalExpected : max(expectedBytes, 1)
                    let fraction = Double(bytesWritten) / Double(total)
                    Task { @MainActor in
                        self?.modelStates[.llm] = .downloading(progress: min(fraction, 1.0))
                    }
                },
                onComplete: { tempURL, error in
                    if let error {
                        continuation.resume(returning: .failure(error))
                    } else if let tempURL {
                        continuation.resume(returning: .success(tempURL))
                    } else {
                        continuation.resume(returning: .failure(URLError(.badServerResponse)))
                    }
                }
            )

            let config = URLSessionConfiguration.default
            config.timeoutIntervalForResource = 3600
            config.httpAdditionalHeaders = [
                "User-Agent": "volocal-ios/1.0 (on-device; no-telemetry)"
            ]
            let session = URLSession(configuration: config, delegate: delegate, delegateQueue: nil)
            delegate.session = session

            var request = URLRequest(url: url, timeoutInterval: 3600)
            request.setValue("volocal-ios/1.0 (on-device; no-telemetry)", forHTTPHeaderField: "User-Agent")
            session.downloadTask(with: request).resume()
        }

        switch result {
        case .success(let tempURL):
            do {
                try Self.validateDownloadedGGUF(at: tempURL, expectedBytes: spec.sizeBytes)
                if FileManager.default.fileExists(atPath: destination.path) {
                    try FileManager.default.removeItem(at: destination)
                }
                try FileManager.default.moveItem(at: tempURL, to: destination)

                if let expected = spec.sha256, !expected.isEmpty {
                    let actual = try Self.sha256Hex(of: destination)
                    if actual.lowercased() != expected.lowercased() {
                        try? FileManager.default.removeItem(at: destination)
                        throw LLMDownloadError.checksumMismatch
                    }
                }

                guard spec.isDownloaded else {
                    try? FileManager.default.removeItem(at: destination)
                    throw LLMDownloadError.notGGUF
                }

                modelStates[.llm] = .downloaded
                logger.info("LLM downloaded: \(spec.id)")
                markOnboardedIfReady()
            } catch {
                try? FileManager.default.removeItem(at: tempURL)
                try? FileManager.default.removeItem(at: destination)
                modelStates[.llm] = .error(error.localizedDescription)
                self.error = "LLM download failed: \(error.localizedDescription)"
            }
        case .failure(let error):
            modelStates[.llm] = .error(error.localizedDescription)
            self.error = "LLM download failed: \(error.localizedDescription)"
            logger.error("LLM download failed: \(error.localizedDescription)")
        }
    }

    private static func validateDownloadedGGUF(at url: URL, expectedBytes: Int64?) throws {
        guard GGUFFile.looksLikeGGUF(at: url) else {
            throw LLMDownloadError.notGGUF
        }
        let attrs = try FileManager.default.attributesOfItem(atPath: url.path)
        let size = attrs[.size] as? UInt64 ?? 0
        guard size > 1_048_576 else {
            throw LLMDownloadError.truncated
        }
        if let expectedBytes, expectedBytes > 0 {
            let expected = UInt64(expectedBytes)
            if size + 65_536 < expected {
                throw LLMDownloadError.truncated
            }
        }
    }

    private static func sha256Hex(of url: URL) throws -> String {
        var hasher = SHA256()
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        while true {
            let data = handle.readData(ofLength: 1024 * 1024)
            if data.isEmpty { break }
            hasher.update(data: data)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    private func downloadSTT() async {
        modelStates[.stt] = .downloading(progress: 0)
        let engine = selectedSTT

        do {
            try await ModelHub.download(engine.repo, to: FluidAudioCache.asrModelsRoot) { [weak self] progress in
                Task { @MainActor in
                    self?.modelStates[.stt] = .downloading(progress: progress.fractionCompleted)
                }
            }
            modelStates[.stt] = .downloaded
            logger.info("STT models downloaded: \(engine.displayName)")
            markOnboardedIfReady()
        } catch {
            modelStates[.stt] = .error(error.localizedDescription)
            self.error = "STT download failed: \(error.localizedDescription)"
            logger.error("STT download failed: \(error.localizedDescription)")
        }
    }

    private func downloadTTS() async {
        modelStates[.tts] = .downloading(progress: 0)
        let engine = selectedTTS

        do {
            switch engine {
            case .pocketTts:
                _ = try await PocketTtsResourceDownloader.ensureModels(language: .english) { [weak self] progress in
                    Task { @MainActor in
                        self?.modelStates[.tts] = .downloading(progress: progress.fractionCompleted)
                    }
                }
            case .kokoroAne:
                _ = try await KokoroAneResourceDownloader.ensureModels(variant: .english) { [weak self] progress in
                    Task { @MainActor in
                        self?.modelStates[.tts] = .downloading(progress: progress.fractionCompleted)
                    }
                }
            }
            modelStates[.tts] = .downloaded
            logger.info("TTS models downloaded: \(engine.displayName)")
            markOnboardedIfReady()
        } catch {
            modelStates[.tts] = .error(error.localizedDescription)
            self.error = "TTS download failed: \(error.localizedDescription)"
            logger.error("TTS download failed: \(error.localizedDescription)")
        }
    }

    func deleteSelectedLLM() {
        try? FileManager.default.removeItem(at: selectedLLM.localURL)
        checkExistingModels()
    }
}

enum LLMDownloadError: LocalizedError {
    case checksumMismatch
    case httpStatus(Int)
    case notGGUF
    case truncated

    var errorDescription: String? {
        switch self {
        case .checksumMismatch:
            return "Downloaded GGUF failed SHA-256 check. Deleted the file — retry the download."
        case .httpStatus(let code):
            return "Hugging Face returned HTTP \(code) instead of a GGUF."
        case .notGGUF:
            return "Download was not a GGUF file (HTML error page or wrong asset)."
        case .truncated:
            return "Download was truncated. Delete and retry on Wi-Fi."
        }
    }
}

private final class LLMDownloadDelegate: NSObject, URLSessionDownloadDelegate {
    let onProgress: (Int64, Int64) -> Void
    let onComplete: (URL?, Error?) -> Void
    var session: URLSession?
    private var hasCompleted = false

    init(
        onProgress: @escaping (Int64, Int64) -> Void,
        onComplete: @escaping (URL?, Error?) -> Void
    ) {
        self.onProgress = onProgress
        self.onComplete = onComplete
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didWriteData bytesWritten: Int64,
        totalBytesWritten: Int64,
        totalBytesExpectedToWrite: Int64
    ) {
        onProgress(totalBytesWritten, totalBytesExpectedToWrite)
    }

    func urlSession(
        _ session: URLSession,
        downloadTask: URLSessionDownloadTask,
        didFinishDownloadingTo location: URL
    ) {
        if let http = downloadTask.response as? HTTPURLResponse,
           !(200..<300).contains(http.statusCode) {
            finish(tempURL: nil, error: LLMDownloadError.httpStatus(http.statusCode))
            return
        }
        if let mime = downloadTask.response?.mimeType?.lowercased(),
           mime.contains("text/html") || mime.contains("application/json") {
            finish(tempURL: nil, error: LLMDownloadError.notGGUF)
            return
        }
        let tempURL = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".gguf")
        do {
            try FileManager.default.moveItem(at: location, to: tempURL)
            finish(tempURL: tempURL, error: nil)
        } catch {
            finish(tempURL: nil, error: error)
        }
    }

    func urlSession(
        _ session: URLSession,
        task: URLSessionTask,
        didCompleteWithError error: Error?
    ) {
        if let error {
            finish(tempURL: nil, error: error)
        }
    }

    private func finish(tempURL: URL?, error: Error?) {
        guard !hasCompleted else { return }
        hasCompleted = true
        session?.finishTasksAndInvalidate()
        session = nil
        onComplete(tempURL, error)
    }
}
