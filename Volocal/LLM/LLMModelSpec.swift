import Foundation

/// A GGUF language model that can be downloaded from Hugging Face and loaded by llama.cpp.
struct LLMModelSpec: Codable, Identifiable, Equatable, Hashable {
    /// Hugging Face `org/repo`.
    var repoId: String
    /// Path inside the repo (`model.gguf` or `subdir/model.gguf`). Disk storage uses the last component only.
    var filename: String
    var displayName: String
    var sizeBytes: Int64?
    var sha256: String?

    var id: String { "\(repoId)/\(filename)" }

    var diskFileName: String {
        URL(fileURLWithPath: filename).lastPathComponent
    }

    var downloadURL: URL? {
        guard HuggingFaceHub.isValidRepoId(repoId), GGUFFile.isSafeHubPath(filename) else { return nil }
        let encodedRepo = repoId
            .split(separator: "/")
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        let encodedFile = filename
            .split(separator: "/")
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        return URL(string: "https://huggingface.co/\(encodedRepo)/resolve/main/\(encodedFile)")
    }

    var huggingFacePageURL: URL? {
        guard HuggingFaceHub.isValidRepoId(repoId) else { return nil }
        return URL(string: "https://huggingface.co/\(repoId)")
    }

    var sizeDescription: String {
        guard let sizeBytes, sizeBytes > 0 else { return "size unknown" }
        return ByteCountFormatter.string(fromByteCount: sizeBytes, countStyle: .file)
    }

    var isLikelyTooLargeForPhone: Bool {
        (sizeBytes ?? 0) > 8_000_000_000
    }

    /// Nested on-disk location: Documents/models/llm/{repo}__{file}
    var nestedLocalURL: URL {
        let folder = repoId.replacingOccurrences(of: "/", with: "__")
        return ModelRegistry.llmDirectory
            .appendingPathComponent(folder, isDirectory: true)
            .appendingPathComponent(diskFileName)
    }

    /// Original Volocal layout dumped the default GGUF in Documents/models/.
    var legacyLocalURL: URL {
        ModelRegistry.modelsDirectory.appendingPathComponent(diskFileName)
    }

    var localURL: URL {
        let nested = nestedLocalURL
        if FileManager.default.fileExists(atPath: nested.path) { return nested }
        if FileManager.default.fileExists(atPath: legacyLocalURL.path) { return legacyLocalURL }
        return nested
    }

    /// True only when the file sits under the models folder, looks like GGUF, and is not truncated.
    var isDownloaded: Bool {
        let url = localURL.standardizedFileURL
        guard GGUFFile.isInsideModelsDirectory(url) else { return false }
        guard FileManager.default.fileExists(atPath: url.path),
              let attrs = try? FileManager.default.attributesOfItem(atPath: url.path),
              let size = attrs[.size] as? UInt64,
              size > 1_048_576
        else { return false }
        guard GGUFFile.looksLikeGGUF(at: url) else { return false }
        if let sizeBytes, sizeBytes > 0 {
            let expected = UInt64(sizeBytes)
            let slack: UInt64 = 65_536
            if size + slack < expected { return false }
        }
        return true
    }

    static let `default` = LLMModelSpec(
        repoId: "bartowski/Qwen_Qwen3.5-2B-GGUF",
        filename: "Qwen_Qwen3.5-2B-Q4_K_S.gguf",
        displayName: "Qwen 3.5 2B Q4_K_S",
        sizeBytes: 1_327_696_992,
        sha256: "55b574899b75180d084238ffbf5d3d165c568a3d73e1edbd3c826682a986c8d4"
    )

    /// Phone-sized GGUF starting points. Search Hugging Face for anything else.
    static let suggested: [LLMModelSpec] = [
        .default,
        LLMModelSpec(
            repoId: "bartowski/Llama-3.2-3B-Instruct-GGUF",
            filename: "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
            displayName: "Llama 3.2 3B Instruct Q4_K_M",
            sizeBytes: nil,
            sha256: nil
        ),
        LLMModelSpec(
            repoId: "bartowski/Qwen2.5-3B-Instruct-GGUF",
            filename: "Qwen2.5-3B-Instruct-Q4_K_M.gguf",
            displayName: "Qwen 2.5 3B Instruct Q4_K_M",
            sizeBytes: nil,
            sha256: nil
        ),
        LLMModelSpec(
            repoId: "bartowski/gemma-3-4b-it-GGUF",
            filename: "gemma-3-4b-it-Q4_K_M.gguf",
            displayName: "Gemma 3 4B Instruct Q4_K_M",
            sizeBytes: nil,
            sha256: nil
        ),
        LLMModelSpec(
            repoId: "bartowski/Qwen3-4B-GGUF",
            filename: "Qwen3-4B-Q4_K_M.gguf",
            displayName: "Qwen 3 4B Q4_K_M",
            sizeBytes: nil,
            sha256: nil
        ),
        LLMModelSpec(
            repoId: "bartowski/Llama-3.1-8B-Instruct-GGUF",
            filename: "Llama-3.1-8B-Instruct-Q4_K_M.gguf",
            displayName: "Llama 3.1 8B Instruct Q4_K_M",
            sizeBytes: nil,
            sha256: nil
        )
    ]
}

enum LLMFileFilter {
    static func isLoadableGGUF(_ filename: String) -> Bool {
        let name = filename.lowercased()
        guard name.hasSuffix(".gguf") else { return false }
        if name.contains("mmproj") { return false }
        if name.contains("imatrix") { return false }
        if name.contains("gguf-split") { return false }
        // Hub shards look like `*-00001-of-00003.gguf`, not `gguf-split`.
        if name.range(of: #"-\d{5}-of-\d{5}\.gguf$"#, options: .regularExpression) != nil {
            return false
        }
        return true
    }
}

enum GGUFFile {
    static let magic = Data("GGUF".utf8)

    static func looksLikeGGUF(at url: URL) -> Bool {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return false }
        defer { try? handle.close() }
        let header = handle.readData(ofLength: 4)
        return header == magic
    }

    static func isSafeHubPath(_ path: String) -> Bool {
        let trimmed = path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !trimmed.hasPrefix("/") else { return false }
        let parts = trimmed.split(separator: "/")
        return !parts.isEmpty && parts.allSatisfy { $0 != ".." && $0 != "." }
    }

    static func isInsideModelsDirectory(_ url: URL) -> Bool {
        let path = url.standardizedFileURL.path
        let root = ModelRegistry.modelsDirectory.standardizedFileURL.path
        return path == root || path.hasPrefix(root + "/")
    }
}
