import Foundation
import os

private let logger = Logger(subsystem: "com.volocal.app", category: "huggingface")

/// Hugging Face Hub client used only to list and download public GGUF files.
/// Conversation audio/text never goes through here.
enum HuggingFaceHub {
    /// One Hub sort. Several can be on at once; their lists are merged and scored together.
    enum RankSignal: String, CaseIterable, Identifiable {
        case likes
        case newest
        case updated
        case trending
        case downloads

        var id: String { rawValue }

        var label: String {
            switch self {
            case .likes: return "Likes"
            case .newest: return "Newest"
            case .updated: return "Updated"
            case .trending: return "Trending"
            case .downloads: return "Downloads"
            }
        }

        var hubSort: String {
            switch self {
            case .likes: return "likes"
            case .newest: return "createdAt"
            case .updated: return "lastModified"
            case .trending: return "trendingScore"
            case .downloads: return "downloads"
            }
        }

        static func decoded(_ raw: String) -> Set<RankSignal> {
            let parsed = Set(raw.split(separator: ",").compactMap { RankSignal(rawValue: String($0)) })
            return parsed.isEmpty ? [.likes, .newest] : parsed
        }

        static func encoded(_ signals: Set<RankSignal>) -> String {
            allCases.filter { signals.contains($0) }.map(\.rawValue).joined(separator: ",")
        }
    }

    enum LibraryFilter: String, CaseIterable, Identifiable {
        case both
        case gguf
        case mlx

        var id: String { rawValue }

        var label: String {
            switch self {
            case .both: return "Both"
            case .gguf: return "GGUF"
            case .mlx: return "MLX"
            }
        }

        var libraries: [String] {
            switch self {
            case .both: return ["gguf", "mlx"]
            case .gguf: return ["gguf"]
            case .mlx: return ["mlx"]
            }
        }
    }

    struct RepoHit: Identifiable, Equatable {
        let id: String
        var downloads: Int
        var likes: Int
        var trendingScore: Double?
        var createdAt: Date?
        var lastModified: Date?
        /// True when the Hub tagged this repo as an MLX pack rather than a GGUF.
        var offersMLX: Bool = false
        var offersGGUF: Bool = false
        var displayName: String { id }
    }

    struct RemoteFile: Identifiable, Equatable {
        let path: String
        let sizeBytes: Int64?
        let sha256: String?

        var id: String { path }
        /// Last path component, for display only. Downloads must use `path`.
        var filename: String { (path as NSString).lastPathComponent }
    }

    static func isValidRepoId(_ repoId: String) -> Bool {
        let parts = repoId.split(separator: "/").map(String.init)
        guard parts.count == 2 else { return false }
        return parts.allSatisfy { part in
            !part.isEmpty && part != "." && part != ".." && !part.contains("\\")
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

    static func searchRepos(query: String, library: String, sort: String) async throws -> [RepoHit] {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return [] }

        var components = URLComponents(string: "https://huggingface.co/api/models")!
        components.queryItems = [
            URLQueryItem(name: "search", value: trimmed),
            URLQueryItem(name: "filter", value: library),
            URLQueryItem(name: "sort", value: sort),
            URLQueryItem(name: "direction", value: "-1"),
            URLQueryItem(name: "limit", value: "25")
        ]
        guard let url = components.url else { throw URLError(.badURL) }

        let data = try await fetchData(from: url)
        struct Hit: Decodable {
            let id: String
            let downloads: Int?
            let likes: Int?
            let trendingScore: Double?
            let createdAt: String?
            let lastModified: String?
        }
        let hits: [Hit]
        do {
            hits = try JSONDecoder().decode([Hit].self, from: data)
        } catch {
            throw HubError(status: 200, message: "Could not read the Hugging Face search results.")
        }
        let mlx = library.lowercased() == "mlx"
        return hits.map { hit in
            RepoHit(
                id: hit.id,
                downloads: hit.downloads ?? 0,
                likes: hit.likes ?? 0,
                trendingScore: hit.trendingScore,
                createdAt: parseHubDate(hit.createdAt),
                lastModified: parseHubDate(hit.lastModified),
                offersMLX: mlx,
                offersGGUF: !mlx
            )
        }
    }

    /// Merge lists from each selected rank, then score every repo on those ranks together.
    static func rank(_ hits: [RepoHit], by signals: Set<RankSignal>) -> [RepoHit] {
        let active = signals.isEmpty ? [.likes, .newest] : signals
        let likesMax = log1p(Double(hits.map(\.likes).max() ?? 0))
        let downloadsMax = log1p(Double(hits.map(\.downloads).max() ?? 0))
        let trendingMax = hits.compactMap(\.trendingScore).max() ?? 0
        let created = hits.compactMap(\.createdAt)
        let modified = hits.compactMap(\.lastModified)

        func score(_ hit: RepoHit) -> Double {
            var total = 0.0
            if active.contains(.likes), likesMax > 0 {
                total += log1p(Double(hit.likes)) / likesMax
            }
            if active.contains(.downloads), downloadsMax > 0 {
                total += log1p(Double(hit.downloads)) / downloadsMax
            }
            if active.contains(.trending), trendingMax > 0 {
                total += (hit.trendingScore ?? 0) / trendingMax
            }
            if active.contains(.newest) {
                total += recency(hit.createdAt, among: created)
            }
            if active.contains(.updated) {
                total += recency(hit.lastModified, among: modified)
            }
            return total
        }

        return hits.sorted { lhs, rhs in
            let left = score(lhs)
            let right = score(rhs)
            if left != right { return left > right }
            if lhs.likes != rhs.likes { return lhs.likes > rhs.likes }
            return lhs.id < rhs.id
        }
    }

    static func listGGUFFiles(repoId: String) async throws -> [RemoteFile] {
        guard isValidRepoId(repoId) else { throw URLError(.badURL) }
        let encoded = repoId
            .split(separator: "/")
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        guard let url = URL(string: "https://huggingface.co/api/models/\(encoded)/tree/main?recursive=true") else {
            throw URLError(.badURL)
        }

        let data = try await fetchData(from: url)

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
                sha256: GGUFFile.normalizedSHA256(item.lfs?.oid)
            )
        }
        .sorted { ($0.sizeBytes ?? 0) < ($1.sizeBytes ?? 0) }
    }

    static func resolveURL(repoId: String, path: String) -> URL? {
        guard isValidRepoId(repoId), GGUFFile.isSafeHubPath(path) else { return nil }
        let encodedRepo = repoId
            .split(separator: "/")
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        let encodedFile = path
            .split(separator: "/")
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        return URL(string: "https://huggingface.co/\(encodedRepo)/resolve/main/\(encodedFile)?download=true")
    }

    static func isMLXSnapshotFile(_ path: String) -> Bool {
        guard GGUFFile.isSafeHubPath(path) else { return false }
        let name = (path as NSString).lastPathComponent.lowercased()
        if name.hasSuffix(".safetensors") || name.hasSuffix(".safetensors.index.json") { return true }
        let allowed: Set<String> = [
            "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
            "tokenizer.model", "special_tokens_map.json", "added_tokens.json", "chat_template.jinja",
            "vocab.json", "merges.txt",
        ]
        return allowed.contains(name)
    }

    static func listMLXFiles(repoId: String) async throws -> [RemoteFile] {
        guard isValidRepoId(repoId) else { throw URLError(.badURL) }
        let encoded = repoId
            .split(separator: "/")
            .map { $0.addingPercentEncoding(withAllowedCharacters: .urlPathAllowed) ?? String($0) }
            .joined(separator: "/")
        guard let url = URL(string: "https://huggingface.co/api/models/\(encoded)/tree/main?recursive=true") else {
            throw URLError(.badURL)
        }
        let data = try await fetchData(from: url)
        struct Item: Decodable {
            let path: String
            let type: String?
            let size: Int64?
            let lfs: LFS?
            struct LFS: Decodable {
                let size: Int64?
            }
        }
        let items = try JSONDecoder().decode([Item].self, from: data)
        return items.compactMap { item in
            guard item.type == "file" || item.type == nil else { return nil }
            guard isMLXSnapshotFile(item.path) else { return nil }
            return RemoteFile(path: item.path, sizeBytes: item.lfs?.size ?? item.size, sha256: nil)
        }
    }

    /// Accepts `org/repo`, a Hugging Face model URL, or a direct `/resolve/` GGUF URL.
    static func parseUserInput(_ raw: String) -> (repoId: String, filename: String?)? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }

        if let url = URL(string: trimmed), let host = url.host?.lowercased(),
           host == "huggingface.co" || host.hasSuffix(".huggingface.co") {
            let parts = url.path.split(separator: "/").map(String.init)
            guard parts.count >= 2 else { return nil }
            let repoId = "\(parts[0])/\(parts[1])"
            guard isValidRepoId(repoId) else { return nil }
            if let fileIndex = parts.firstIndex(of: "resolve") ?? parts.firstIndex(of: "blob"),
               fileIndex + 2 < parts.count {
                let filename = parts[(fileIndex + 2)...].joined(separator: "/")
                guard GGUFFile.isSafeHubPath(filename) else { return nil }
                return (repoId, filename.isEmpty ? nil : filename)
            }
            return (repoId, nil)
        }

        let pieces = trimmed.split(separator: "/").map(String.init)
        guard pieces.count >= 2 else { return nil }
        let repoId = "\(pieces[0])/\(pieces[1])"
        guard isValidRepoId(repoId) else { return nil }
        if pieces.last?.lowercased().hasSuffix(".gguf") == true, pieces.count >= 3 {
            let filename = pieces.dropFirst(2).joined(separator: "/")
            guard GGUFFile.isSafeHubPath(filename) else { return nil }
            return (repoId, filename)
        }
        return (repoId, nil)
    }

    private static func fetchData(from url: URL) async throws -> Data {
        var attempt = 0
        while true {
            let (data, response) = try await session.data(from: url)
            guard let http = response as? HTTPURLResponse else { return data }
            if (200..<300).contains(http.statusCode) {
                return data
            }
            let message = hubMessage(from: data)
            logger.error("Hugging Face HTTP \(http.statusCode) \(message)")
            if http.statusCode == 429, attempt == 0 {
                let wait = retryDelay(http: http, data: data)
                if wait <= 8 {
                    attempt += 1
                    try await Task.sleep(for: .seconds(wait))
                    continue
                }
            }
            throw HubError(status: http.statusCode, message: message)
        }
    }

    private static func hubMessage(from data: Data) -> String {
        struct Body: Decodable { let error: String? }
        let text = (try? JSONDecoder().decode(Body.self, from: data))?.error ?? ""
        return text.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private static func retryDelay(http: HTTPURLResponse, data: Data) -> TimeInterval {
        if let header = http.value(forHTTPHeaderField: "retry-after"),
           let seconds = TimeInterval(header), seconds > 0 {
            return seconds
        }
        struct Body: Decodable { let retryAfter: Double? }
        if let seconds = (try? JSONDecoder().decode(Body.self, from: data))?.retryAfter, seconds > 0 {
            return seconds
        }
        return 5
    }

    private static func parseHubDate(_ raw: String?) -> Date? {
        guard let raw, !raw.isEmpty else { return nil }
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        if let date = formatter.date(from: raw) { return date }
        formatter.formatOptions = [.withInternetDateTime]
        return formatter.date(from: raw)
    }

    /// 1 for the newest date in the pool, 0 for the oldest. Missing dates score 0.
    private static func recency(_ date: Date?, among dates: [Date]) -> Double {
        guard let date, let oldest = dates.min(), let newest = dates.max() else { return 0 }
        let span = newest.timeIntervalSince(oldest)
        guard span > 0 else { return 1 }
        return date.timeIntervalSince(oldest) / span
    }
}

struct HubError: LocalizedError {
    let status: Int
    let message: String

    var errorDescription: String? {
        if status == 429 {
            return "Hugging Face is limiting requests from this network. Wait a minute, then search again."
        }
        if status == 401 || status == 403 {
            return "Hugging Face refused the request (HTTP \(status)). The repo may be gated."
        }
        if message.isEmpty {
            return "Hugging Face returned HTTP \(status)."
        }
        let clipped = message.count > 180 ? String(message.prefix(180)) : message
        return "Hugging Face returned HTTP \(status). \(clipped)"
    }
}
