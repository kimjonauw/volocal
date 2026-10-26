import Foundation
import os
@preconcurrency import MLX
import MLXAudioTTS
import MLXAudioCore

/// Used only by modular TTS under the existing GPUExclusive mutex.
actor QwenConversationBackend {
    static let repo = "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
    private var model: Qwen3TTSModel?
    private static var directory: URL {
        FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("QwenConversation06B4bit", isDirectory: true)
    }
    private static func failure(_ text: String) -> NSError {
        NSError(domain: "QwenConversation", code: 1, userInfo: [NSLocalizedDescriptionKey: text])
    }
    /// The earlier 1.7B build is no longer used by modular mode. Remove its app-owned cache.
    private static func removeLegacyDownload() {
        let legacy = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("QwenConversation4bit", isDirectory: true)
        try? FileManager.default.removeItem(at: legacy)
    }
    static var isDownloaded: Bool {
        guard let data = try? Data(contentsOf: directory.appendingPathComponent("download-complete.json")),
              let files = try? JSONDecoder().decode([String: Int64].self, from: data),
              files["config.json"] != nil, files["tokenizer_config.json"] != nil,
              files.keys.contains(where: { $0.hasPrefix("speech_tokenizer/") && $0.hasSuffix(".safetensors") }),
              files.keys.contains(where: { !$0.contains("/") && $0.hasSuffix(".safetensors") }) else { return false }
        return files.allSatisfy { path, size in
            guard GGUFFile.isSafeHubPath(path), size > 0 else { return false }
            let actual = (try? directory.appendingPathComponent(path).resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            return Int64(actual) == size
        }
    }
    static func prepareFiles() async throws {
        removeLegacyDownload()
        if isDownloaded { return }
        let root = directory
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        var excluded = root
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        try excluded.setResourceValues(values)
        let files = try await HuggingFaceHub.listMLXFiles(repoId: repo)
        guard files.contains(where: { $0.path == "config.json" }),
              files.contains(where: { $0.path.hasPrefix("speech_tokenizer/") && $0.path.hasSuffix(".safetensors") }) else {
            throw failure("Qwen download listing is incomplete.")
        }
        var completed: [String: Int64] = [:]
        for file in files {
            try Task.checkCancellation()
            guard GGUFFile.isSafeHubPath(file.path) else { throw failure("Unsafe model path.") }
            let destination = root.appendingPathComponent(file.path)
            let existing = (try? destination.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
            if existing > 0, let expected = file.sizeBytes, Int64(existing) == expected {
                completed[file.path] = expected; continue
            }
            guard let remote = HuggingFaceHub.resolveURL(repoId: repo, path: file.path) else { throw URLError(.badURL) }
            let (temp, response) = try await URLSession.shared.download(for: URLRequest(url: remote, timeoutInterval: 1800))
            defer { try? fm.removeItem(at: temp) }
            try Task.checkCancellation()
            guard let status = (response as? HTTPURLResponse)?.statusCode, (200...299).contains(status) else { throw URLError(.badServerResponse) }
            let bytes = Int64((try temp.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0)
            guard bytes > 0, file.sizeBytes == nil || bytes == file.sizeBytes else { throw failure("Incomplete Qwen file: \(file.path)") }
            try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
            try fm.moveItem(at: temp, to: destination)
            completed[file.path] = bytes
        }
        try JSONEncoder().encode(completed).write(to: root.appendingPathComponent("download-complete.json"), options: .atomic)
        guard isDownloaded else { throw failure("Qwen files did not pass completeness checks.") }
    }
    func load() async throws {
        if model != nil { return }
        try await Self.prepareFiles()
        let path = Self.directory.path
        let task = Task.detached(priority: .userInitiated) { () throws -> Qwen3TTSModel in
            let limit = Memory.cacheLimit; Memory.cacheLimit = 32 * 1024 * 1024
            defer { Memory.clearCache(); Memory.cacheLimit = limit }
            let loaded = try await TTS.loadModel(modelRepo: path, modelType: "qwen3_tts")
            try Task.checkCancellation()
            guard let model = loaded as? Qwen3TTSModel, model.sampleRate == 24_000 else {
                throw Self.failure("Qwen model or sample rate is incompatible with shared playback.")
            }
            model.train(false)
            return model
        }
        model = try await withTaskCancellationHandler(operation: { try await task.value }, onCancel: { task.cancel() })
    }
    func unload() { model = nil; Memory.clearCache() }
    func synthesize(_ text: String, speaker: String, whisper: Bool) async throws -> [Float] {
        try Task.checkCancellation()
        guard let model else { throw Self.failure("Qwen is not loaded.") }
        let supported = ["Serena", "Vivian", "Ryan", "Aiden"]
        let voice = supported.contains(speaker) ? speaker : "Serena"
        let delivery = whisper ? "Speak softly in a gentle whisper without shouting." : "Warm and gently playful. Speak naturally and conversationally without shouting."
        let task = Task.detached(priority: .userInitiated) { () throws -> [Float] in
            let limit = Memory.cacheLimit; Memory.cacheLimit = 32 * 1024 * 1024
            defer { Memory.clearCache(); Memory.cacheLimit = limit }
            try Task.checkCancellation()
            let audio = try await model.generate(text: text, voice: voice + ", " + delivery,
                refAudio: nil, refText: nil, language: "English",
                generationParameters: GenerateParameters(maxTokens: 192, temperature: 0.8, topP: 1, repetitionPenalty: 1.1))
            try Task.checkCancellation()
            let samples = audio.asType(.float32).asArray(Float.self)
            guard !samples.isEmpty, samples.count <= 24_000 * 18, samples.allSatisfy({ $0.isFinite }) else {
                throw Self.failure("Qwen returned invalid audio.")
            }
            let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
            guard peak > 0.00001 else { throw Self.failure("Qwen returned silence.") }
            let gain = min(Float(1), 0.75 / peak)
            return samples.map { $0 * gain }
        }
        // Do not release GPUExclusive until the cancelled worker has actually returned.
        return try await withTaskCancellationHandler(operation: { try await task.value }, onCancel: { task.cancel() })
    }
    nonisolated static func clauses(_ text: String) -> [String] {
        var result: [String] = []; var clause = ""
        for word in text.split(whereSeparator: { $0.isWhitespace }) {
            if !clause.isEmpty, clause.count + word.count + 1 > 120 { result.append(clause); clause = "" }
            if !clause.isEmpty { clause += " " }
            clause += word
        }
        if !clause.isEmpty { result.append(clause) }
        return result
    }
}
