import Foundation
import Darwin
@preconcurrency import MLX

/// Bounded, persistent diagnostics. Never stores prompt text or generated speech.
enum GenerationTrace {
    private static let lock = NSLock()
    private static let session = UUID().uuidString

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
        lock.lock()
        defer { lock.unlock() }
        guard let directory = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first else { return }
        let url = directory.appendingPathComponent("generation-trace.json")
        var entries: [Entry] = []
        if let data = try? Data(contentsOf: url), let saved = try? JSONDecoder().decode([Entry].self, from: data) {
            entries = Array(saved.suffix(95))
        }
        entries.append(Entry(
            session: session, time: Date().timeIntervalSince1970,
            stage: stage, count: count, footprintBytes: footprint(),
            availableBytes: UInt64(os_proc_available_memory()),
            mlxActiveBytes: UInt64(Memory.activeMemory),
            mlxCacheBytes: UInt64(Memory.cacheMemory)
        ))
        if let data = try? JSONEncoder().encode(entries) {
            try? data.write(to: url, options: .atomic)
        }
    }
}
