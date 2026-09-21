import Foundation

/// A GGUF language model that can be downloaded from Hugging Face and loaded by llama.cpp.
struct LLMModelSpec: Codable, Identifiable, Equatable, Hashable {
    var repoId: String
    var filename: String
    var displayName: String
    var sizeBytes: Int64?
    var sha256: String?

    var id: String { "\(repoId)/\(filename)" }

    var downloadURL: URL? {
        let encodedFile = filename.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? filename
        return URL(string: "https://huggingface.co/\(repoId)/resolve/main/\(encodedFile)")
    }

    var huggingFacePageURL: URL? {
        URL(string: "https://huggingface.co/\(repoId)")
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
            .appendingPathComponent(filename)
    }

    /// Original Volocal layout dumped the default GGUF in Documents/models/.
    var legacyLocalURL: URL {
        ModelRegistry.modelsDirectory.appendingPathComponent(filename)
    }

    var localURL: URL {
        let nested = nestedLocalURL
        if FileManager.default.fileExists(atPath: nested.path) { return nested }
        if FileManager.default.fileExists(atPath: legacyLocalURL.path) { return legacyLocalURL }
        return nested
    }

    var isDownloaded: Bool {
        guard FileManager.default.fileExists(atPath: localURL.path),
              let attrs = try? FileManager.default.attributesOfItem(atPath: localURL.path),
              let size = attrs[.size] as? UInt64,
              size > 1_048_576
        else { return false }
        if let sizeBytes, sizeBytes > 0 {
            // Allow a small mismatch so a re-quantized file still counts as present.
            let expected = UInt64(sizeBytes)
            if size + 1_048_576 < expected { return false }
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
        return true
    }
}
