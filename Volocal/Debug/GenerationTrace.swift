import Foundation
import Darwin
@preconcurrency import MLX

/// Bounded, persistent diagnostics. Never stores prompt text or generated speech.
/// Entries are kept in memory and written on a background queue so callers
/// (including the main actor and token loops) never block on disk I/O.
enum GenerationTrace {
    private static let lock = NSLock()
    private static let writer = DispatchQueue(label: "com.volocal.generation-trace", qos: .utility)
    private static let session = UUID().uuidString
    private static let limit = 96
    private static var entries: [Entry] = []
    private static var loaded = false

    private struct Entry: Codable {
        let session: String
        let time: TimeInterval
        let stage: String
        let count: Int
        let footprintBytes: UInt64
        let availableBytes: UInt64
        let mlxActiveBytes: UInt64
        let mlxCacheBytes: UInt64
    }

    private static var fileURL: URL? {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first?
            .appendingPathComponent("generation-trace.json")
    }

    private static func footprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return status == KERN_SUCCESS ? info.phys_footprint : 0
    }

    static func record(_ stage: String, count: Int = 0) {
        let entry = Entry(
            session: session, time: Date().timeIntervalSince1970,
            stage: stage, count: count, footprintBytes: footprint(),
            availableBytes: UInt64(max(0, Int(os_proc_available_memory()))),
            mlxActiveBytes: UInt64(max(0, Int(Memory.activeMemory))),
            mlxCacheBytes: UInt64(max(0, Int(Memory.cacheMemory)))
        )
        lock.lock()
        if !loaded {
            loaded = true
            if let url = fileURL,
               let data = try? Data(contentsOf: url),
               let saved = try? JSONDecoder().decode([Entry].self, from: data) {
                entries = Array(saved.suffix(limit))
            }
        }
        entries.append(entry)
        if entries.count > limit { entries.removeFirst(entries.count - limit) }
        let snapshot = entries
        lock.unlock()
        writer.async {
            guard let url = fileURL, let data = try? JSONEncoder().encode(snapshot) else { return }
            try? data.write(to: url, options: .atomic)
        }
    }
}
