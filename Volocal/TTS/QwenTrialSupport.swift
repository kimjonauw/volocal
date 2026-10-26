import Foundation
import CryptoKit
import Darwin

enum QwenTrialVariant: String, CaseIterable, Identifiable, Sendable {
    case fourBit, threeBit
    var id: String { rawValue }
    var label: String { self == .fourBit ? "4-bit baseline" : "3-bit experimental" }
}

enum QwenTrialPack {
    static let tag = "qwen3-tts-3bit-40287976531-1"
    static let manifestSHA = "04e3c8cdee5fbb0128283094245fff697cc919074064099f25c6acda0853fa12"
    static let base = URL(string: "https://github.com/kimjonauw/volocal/releases/download/qwen3-tts-3bit-40287976531-1/")!
    private struct Manifest: Decodable {
        let schema_version: Int
        let source_revision: String
        let bits: Int
        let group_size: Int
        let release_tag: String
        let files: [Entry]
    }
    private struct Entry: Decodable {
        let path: String
        let asset: String
        let bytes: Int64
        let sha256: String
    }
    private static func failure(_ message: String) -> NSError {
        NSError(domain: "QwenTrialPack", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
    }
    private static func hash(_ url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var digest = SHA256()
        while let data = try handle.read(upToCount: 1024 * 1024), !data.isEmpty {
            try Task.checkCancellation()
            digest.update(data: data)
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }
    private static func valid(_ url: URL, entry: Entry) throws -> Bool {
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize,
              Int64(size) == entry.bytes else { return false }
        return try hash(url) == entry.sha256
    }
    static func prepare(progress: @escaping @Sendable (String) async -> Void) async throws -> URL {
        let fm = FileManager.default
        let root = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("QwenVoiceLab/" + tag, isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        var excluded = root
        var values = URLResourceValues(); values.isExcludedFromBackup = true
        try excluded.setResourceValues(values)
        let manifestURL = root.appendingPathComponent("release-manifest.json")
        let data: Data
        if let cached = try? Data(contentsOf: manifestURL), SHA256.hash(data: cached).map({ String(format: "%02x", $0) }).joined() == manifestSHA {
            data = cached
        } else {
            await progress("Downloading pinned 3-bit manifest...")
            let (download, response) = try await URLSession.shared.data(from: base.appendingPathComponent("manifest.json"))
            guard (response as? HTTPURLResponse)?.statusCode == 200, download.count < 100_000,
                  SHA256.hash(data: download).map({ String(format: "%02x", $0) }).joined() == manifestSHA else {
                throw failure("The release manifest failed its pinned checksum.")
            }
            data = download
            try data.write(to: manifestURL, options: .atomic)
        }
        let manifest = try JSONDecoder().decode(Manifest.self, from: data)
        guard manifest.schema_version == 1, manifest.bits == 3, manifest.group_size == 64,
              manifest.release_tag == tag,
              manifest.source_revision == "52f4770fd9726457eae3d3b6aa92047a25a10776",
              !manifest.files.isEmpty, manifest.files.count <= 40 else { throw failure("Unsupported model manifest.") }
        var names = Set<String>()
        var total: Int64 = 0
        for entry in manifest.files {
            let parts = entry.path.split(separator: "/", omittingEmptySubsequences: false)
            guard !parts.isEmpty, parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }),
                  entry.path.unicodeScalars.allSatisfy({ CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_./-").contains($0) }),
                  entry.asset == entry.path.replacingOccurrences(of: "/", with: "__"),
                  entry.bytes > 0, entry.bytes < 2_147_483_648,
                  entry.sha256.count == 64, entry.sha256.allSatisfy({ "0123456789abcdef".contains($0) }),
                  names.insert(entry.path).inserted else { throw failure("Unsafe model manifest entry.") }
            total += entry.bytes
        }
        guard total < 3_000_000_000,
              Set(["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "speech_tokenizer/config.json", "speech_tokenizer/model.safetensors"]).isSubset(of: names) else {
            throw failure("Incomplete or oversized model manifest.")
        }
        for (index, entry) in manifest.files.enumerated() {
            try Task.checkCancellation()
            let destination = root.appendingPathComponent(entry.path)
            await progress("Verifying 3-bit file \(index + 1)/\(manifest.files.count): \(entry.path)")
            if try valid(destination, entry: entry) { continue }
            let capacity = try root.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]).volumeAvailableCapacityForImportantUsage
            if let capacity, capacity < entry.bytes + 500_000_000 { throw failure("Not enough free disk space for \(entry.path).") }
            await progress("Downloading 3-bit file \(index + 1)/\(manifest.files.count): \(entry.path). Keep this screen open.")
            let request = URLRequest(url: base.appendingPathComponent(entry.asset), timeoutInterval: 1800)
            let (temp, response) = try await URLSession.shared.download(for: request)
            defer { try? fm.removeItem(at: temp) }
            try Task.checkCancellation()
            guard (response as? HTTPURLResponse)?.statusCode == 200, try valid(temp, entry: entry) else {
                throw failure("Download or checksum failed for \(entry.path). Retry to reuse already verified files.")
            }
            try fm.createDirectory(at: destination.deletingLastPathComponent(), withIntermediateDirectories: true)
            if fm.fileExists(atPath: destination.path) { try fm.removeItem(at: destination) }
            try fm.moveItem(at: temp, to: destination)
        }
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("config.json"))) as? [String: Any]
        let quant = (config?["quantization"] ?? config?["quantization_config"]) as? [String: Any]
        guard quant?["bits"] as? Int == 3, config?["tts_model_type"] as? String == "custom_voice" else {
            throw failure("Downloaded model is not 3-bit CustomVoice.")
        }
        return root
    }
}

final class QwenMemoryProbe: @unchecked Sendable {
    private let lock = NSLock()
    private var peak: UInt64 = 0
    private var timer: DispatchSourceTimer?
    static func footprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return result == KERN_SUCCESS ? info.phys_footprint : 0
    }
    func sample() {
        let value = Self.footprint()
        lock.lock(); peak = max(peak, value); lock.unlock()
    }
    func start() {
        sample()
        let timer = DispatchSource.makeTimerSource(queue: DispatchQueue(label: "volocal.qwen-memory"))
        timer.schedule(deadline: .now(), repeating: .milliseconds(100))
        timer.setEventHandler { [weak self] in self?.sample() }
        self.timer = timer; timer.resume()
    }
    func stop() -> UInt64 {
        timer?.cancel(); timer = nil; sample()
        lock.lock(); defer { lock.unlock() }; return peak
    }
    deinit { timer?.cancel() }
}
