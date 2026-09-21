import Foundation
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "huggingface")

/// Hugging Face Hub client used only to list and download public GGUF files.
/// Conversation audio/text never goes through here.
enum HuggingFaceHub {
    struct RepoHit: Identifiable, Equatable {
        let id: String
        let downloads: Int
        let likes: Int
        var displayName: String { id }
    }

    struct RemoteFile: Identifiable, Equatable {
        let path: String
        let sizeBytes: Int64?
        let sha256: String?

        var id: String { path }
        var filename: String { (path as NSString).lastPathComponent }

        var spec: LLMModelSpec {
            LLMModelSpec(
                repoId: "",
                filename: filename,
                displayName: filename,
                sizeBytes: sizeBytes,
                sha256: sha256
            )
        }
    }

    private static let session: URLSession = {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 3600
        config.httpAdditionalHeaders = [
            "User-Agent": "volocal-ios/1.0 (on-device; no-telemetry)",
            "Accept": "application/json"
        ]
        return URLSession(configuration: config)
    }()

    static func searchRepos(query: String) async throws -> [RepoHit] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var components = URLComponents(string: "https://huggingface.co/api/models")!
        components.queryItems = [
            URLQueryItem(name: "search", value: trimmed),
            URLQueryItem(name: "filter", value: "gguf"),
            URLQueryItem(name: "sort", value: "downloads"),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "30")
        ]
        guard let url = components.url else { throw URLError(.badURL) }

        let (data, response) = try await session.data(from: url)
        try throwIfBad(response)

        struct Hit: Decodable {
            let id: String
            let downloads: Int?
            let likes: Int?
        }
        let hits = try JSONDecoder().decode([Hit].self, from: data)
        return hits.map { RepoHit(id: $0.id, downloads: $0.downloads ?? 0, likes: $0.likes ?? 0) }
    }

    static func listGGUFFiles(repoId: String) async throws -> [RemoteFile] {
        let encoded = repoId.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? repoId
        guard let url = URL(string: "https://huggingface.co/api/models/\(encoded)/tree/main?recursive=true") else {
            throw URLError(.badURL)
        }

        let (data, response) = try await session.data(from: url)
        try throwIfBad(response)

        struct Item: Decodable {
            let path: String
            let type: String?
            let size: Int64?
            let lfs: LFS?
            struct LFS: Decodable {
                let oid: String?
                let size: Int64?
            }
        }

        let items = try JSONDecoder().decode([Item].self, from: data)
        return items.compactMap { item in
            guard item.type == "file" || item.type == nil else { return nil }
            guard LLMFileFilter.isLoadableGGUF(item.path) else { return nil }
            return RemoteFile(
                path: item.path,
                sizeBytes: item.lfs?.size ?? item.size,
                sha256: item.lfs?.oid
            )
        }
        .sorted { ($0.sizeBytes ?? 0) < ($1.sizeBytes ?? 0) }
    }

    /// Accepts `org/repo`, a Hugging Face model URL, or a direct `/resolve/` GGUF URL.
    static func parseUserInput(_ raw: String) -> (repoId: String, filename: String?)? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let url = URL(string: trimmed), let host = url.host, host.contains("huggingface.co") {
            let parts = url.path.split(separator: "/").map(String.init)
            guard parts.count >= 2 else { return nil }
            let repoId = "\(parts[0])/\(parts[1])"
            if let fileIndex = parts.firstIndex(of: "resolve") ?? parts.firstIndex(of: "blob"),
               fileIndex + 2 < parts.count {
                let filename = parts[(fileIndex + 2)...].joined(separator: "/")
                return (repoId, filename.isEmpty ? nil : filename)
            }
            return (repoId, nil)
        }

        let pieces = trimmed.split(separator: "/").map(String.init)
        guard pieces.count >= 2 else { return nil }
        if pieces.last?.lowercased().hasSuffix(".gguf") == true, pieces.count >= 3 {
            return ("\(pieces[0])/\(pieces[1])", pieces.dropFirst(2).joined(separator: "/"))
        }
        return ("\(pieces[0])/\(pieces[1])", nil)
    }

    private static func throwIfBad(_ response: URLResponse) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200..<300).contains(http.statusCode) else {
            logger.error("Hugging Face HTTP \(http.statusCode)")
            throw URLError(.badServerResponse)
        }
    }
}
