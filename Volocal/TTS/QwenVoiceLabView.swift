import SwiftUI
import AVFoundation
import os
@preconcurrency import MLX
import MLXAudioTTS
import MLXAudioCore

private struct QwenAudition: Sendable {
    let samples: [Float]
    let sampleRate: Int
    let loadSeconds: Double
    let generationSeconds: Double
    let memory: String
}

private enum QwenAuditionError: LocalizedError {
    case memory(Int), model, audio
    var errorDescription: String? {
        switch self {
        case .memory(let mb): return "Only \(mb) MB available. This experimental test requires 3000 MB of headroom; close other apps and check GetMoreRAM on the host."
        case .model: return "The checkpoint did not load as Qwen3TTSModel."
        case .audio: return "Invalid or empty audio from Qwen3-TTS. Nothing was played."
        }
    }
}

@MainActor
private final class QwenVoiceLab: ObservableObject {
    @Published private(set) var busy = false
    @Published private(set) var status = "Choose a model and start at low volume. This is still a standalone audition."
    @Published private(set) var report = ""
    private var worker: Task<QwenAudition, Error>?
    private var driver: Task<Void, Never>?
    private var player: AVAudioPlayer?
    private var clip: Data?
    private var stopRequested = false

    func audition(text: String, speaker: String, delivery: String, variant: QwenTrialVariant) {
        guard !busy else { return }
        stopPlayback()
        let text = String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(160))
        let style = String(delivery.trimmingCharacters(in: .whitespacesAndNewlines).prefix(200))
        guard !text.isEmpty else { status = "Enter a short sentence first."; return }
        let voice = style.isEmpty ? speaker : speaker + ", " + style
        busy = true; stopRequested = false; report = ""
        status = "Preparing \(variant.label). Only public model files are downloaded; your text stays local."
        let work = Task.detached(priority: .userInitiated) { [weak self] () throws -> QwenAudition in
            let source: String
            if variant == .threeBit {
                source = try await QwenTrialPack.prepare { [weak self] message in await self?.updateStatus(message) }.path
            } else { source = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit" }
            try Task.checkCancellation()
            let free = Int(os_proc_available_memory())
            guard free == 0 || free >= 3_000_000_000 else { throw QwenAuditionError.memory(free / 1_000_000) }
            let previousCacheLimit = Memory.cacheLimit
            Memory.cacheLimit = 32 * 1024 * 1024
            let probe = QwenMemoryProbe(); probe.start()
            defer { _ = probe.stop(); Memory.clearCache(); Memory.cacheLimit = previousCacheLimit }
            let before = QwenMemoryProbe.footprint()
            let start = Date()
            await self?.updateStatus("Loading \(variant.label)...")
            let loaded = try await TTS.loadModel(modelRepo: source, modelType: "qwen3_tts")
            try Task.checkCancellation()
            guard let model = loaded as? Qwen3TTSModel else { throw QwenAuditionError.model }
            model.train(false)
            let afterLoad = QwenMemoryProbe.footprint(); probe.sample()
            let loadSeconds = Date().timeIntervalSince(start)
            await self?.updateStatus("Synthesizing \(variant.label), up to 128 codec frames. One short clip is buffered before playback.")
            let generationStart = Date()
            let audio = try await model.generate(
                text: text, voice: voice, refAudio: nil, refText: nil, language: "English",
                generationParameters: GenerateParameters(maxTokens: 128, temperature: 0.8, topP: 1.0, repetitionPenalty: 1.1)
            )
            try Task.checkCancellation()
            let samples = audio.asType(.float32).asArray(Float.self)
            guard model.sampleRate > 0, samples.count > model.sampleRate / 10,
                  samples.count <= model.sampleRate * 15,
                  samples.allSatisfy({ $0.isFinite }) else { throw QwenAuditionError.audio }
            let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
            guard peak > 0.00001 else { throw QwenAuditionError.audio }
            let gain = min(Float(1), 0.75 / peak)
            let afterSynthesis = QwenMemoryProbe.footprint()
            let sampledPeak = probe.stop()
            let memory = String(format: "App footprint: before load %.0f MB; after load %.0f MB; after synthesis %.0f MB\nSampled app peak (100 ms intervals): %.0f MB\nAvailable process memory after synthesis: %.0f MB\nMLX lifetime peak: %.0f MB (not per-test app RAM)",
                                Double(before) / 1e6, Double(afterLoad) / 1e6, Double(afterSynthesis) / 1e6,
                                Double(sampledPeak) / 1e6, Double(os_proc_available_memory()) / 1e6, Double(Memory.peakMemory) / 1e6)
            return QwenAudition(samples: samples.map { $0 * gain }, sampleRate: model.sampleRate,
                                loadSeconds: loadSeconds, generationSeconds: Date().timeIntervalSince(generationStart), memory: memory)
        }
        worker = work
        driver = Task { [weak self] in
            do {
                let result = try await work.value
                guard let self else { return }
                if !self.stopRequested {
                    Memory.clearCache()
                    let released = QwenMemoryProbe.footprint()
                    self.clip = Self.wav(result.samples, sampleRate: result.sampleRate)
                    let duration = Double(result.samples.count) / Double(result.sampleRate)
                    self.report = String(format: "%@ · %@\nModel load %.1f s; synthesis %.1f s; audio %.1f s\nSynthesis/audio ratio %.2f\n%@\nApp footprint after worker returned + cache clear: %.0f MB\nDelivery: %@",
                                         variant.label, speaker, result.loadSeconds, result.generationSeconds, duration,
                                         result.generationSeconds / duration, result.memory, Double(released) / 1e6, style.isEmpty ? "none" : style)
                    try self.playClip()
                    self.status = "Preview playing at 20% player volume. Compare identical text, speaker, and delivery on the other model."
                } else { self.status = "Stopped; generation has drained." }
            } catch {
                guard let self else { return }
                self.status = self.stopRequested || error is CancellationError ? "Stopped; worker has drained." : "Qwen3-TTS: \(error.localizedDescription)"
            }
            guard let self else { return }
            self.worker = nil; self.driver = nil; self.busy = false
        }
    }
    private func updateStatus(_ message: String) { if !stopRequested { status = message } }
    func stop() {
        stopRequested = true; worker?.cancel(); stopPlayback()
        status = busy ? "Stopping; waiting for download/loading or inference to return." : "Playback stopped."
    }
    func finish() async { stop(); await driver?.value; clip = nil }
    func replay() {
        guard !busy, clip != nil else { return }
        do { try playClip(); status = "Replaying cached preview; no inference." }
        catch { status = "Playback failed: \(error.localizedDescription)" }
    }
    private func playClip() throws {
        guard let clip else { return }
        player?.stop()
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default); try session.setActive(true)
        let player = try AVAudioPlayer(data: clip)
        player.volume = 0.2; player.prepareToPlay()
        guard player.play() else { throw QwenAuditionError.audio }
        self.player = player
    }
    private func stopPlayback() {
        player?.stop(); player = nil
        try? AVAudioSession.sharedInstance().setActive(false, options: .notifyOthersOnDeactivation)
    }
    private static func wav(_ samples: [Float], sampleRate: Int) -> Data {
        var data = Data()
        func ascii(_ value: String) { data.append(contentsOf: value.utf8) }
        func u16(_ value: UInt16) { var v = value.littleEndian; withUnsafeBytes(of: &v) { data.append(contentsOf: $0) } }
        func u32(_ value: UInt32) { var v = value.littleEndian; withUnsafeBytes(of: &v) { data.append(contentsOf: $0) } }
        let bytes = UInt32(samples.count * 2)
        ascii("RIFF"); u32(36 + bytes); ascii("WAVEfmt "); u32(16)
        u16(1); u16(1); u32(UInt32(sampleRate)); u32(UInt32(sampleRate * 2)); u16(2); u16(16)
        ascii("data"); u32(bytes)
        for sample in samples { u16(UInt16(bitPattern: Int16(clamping: Int((max(-1, min(1, sample)) * 32767).rounded())))) }
        return data
    }
}

struct QwenVoiceLabView: View {
    let onClose: () -> Void
    @StateObject private var lab = QwenVoiceLab()
    @Environment(\.scenePhase) private var scenePhase
    @State private var variant: QwenTrialVariant = .fourBit
    @State private var text = "Oh, you remembered that? That's actually really sweet."
    @State private var speaker = "Serena"
    @State private var delivery = "Warm, pleasantly surprised, and gently playful. Speak naturally without shouting."
    @State private var closing = false
    private let speakers = ["Serena", "Vivian", "Ryan", "Aiden"]
    var body: some View {
        NavigationStack {
            Form {
                Section("Qwen3-TTS 1.7B CustomVoice") {
                    Picker("Model", selection: $variant) { ForEach(QwenTrialVariant.allCases) { Text($0.label).tag($0) } }.disabled(lab.busy || closing)
                    Text("Standalone audition: the microphone and conversation models remain unloaded. Selecting 3-bit does not change your normal TTS engine.")
                    Text("3-bit downloads about 2.13 GB from your pinned GitHub release and verifies SHA-256 checksums. 4-bit uses Hugging Face. Text and audio stay local. Files remain cached; models reload for each audition. No cloud inference.").font(.caption)
                    Text("Compare the same text, speaker, and delivery. Copy each report before the next test; the panel shows only the latest result. Footprint readings of zero mean unavailable. Sampled peaks may miss brief spikes.").font(.caption)
                }
                Section("Words to speak (160 characters)") {
                    TextEditor(text: $text).frame(minHeight: 70)
                        .onChange(of: text) { _, value in if value.count > 160 { text = String(value.prefix(160)) } }
                }.disabled(lab.busy || closing)
                Section("Voice and delivery (200 characters)") {
                    Picker("Speaker", selection: $speaker) { ForEach(speakers, id: \.self) { Text($0).tag($0) } }
                    TextEditor(text: $delivery).frame(minHeight: 70)
                        .onChange(of: delivery) { _, value in if value.count > 200 { delivery = String(value.prefix(200)) } }
                    Button("Neutral delivery") { delivery = "Speak calmly with a neutral conversational tone." }
                    Button("Warm and playful") { delivery = "Warm, pleasantly surprised, and gently playful. Speak naturally without shouting." }
                }.disabled(lab.busy || closing)
                Section {
                    Button(lab.busy ? "Working..." : "Generate and play preview") { lab.audition(text: text, speaker: speaker, delivery: delivery, variant: variant) }.disabled(lab.busy || closing)
                    Button("Replay last preview") { lab.replay() }.disabled(lab.busy || closing)
                    Button("Stop", role: .destructive) { lab.stop() }
                }
                Section("Status and measurements") {
                    Text(lab.status).textSelection(.enabled)
                    if !lab.report.isEmpty { Text(lab.report).font(.caption.monospaced()).textSelection(.enabled) }
                    Text("Keep volume low. The 128-frame cap may truncate longer text. Stop is cooperative; loading may not stop immediately. Successful packaging does not guarantee 3-bit quality or Swift compatibility.").font(.caption)
                }
            }
            .navigationTitle("Qwen voice lab")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(closing ? "Closing..." : "Back") {
                        closing = true; Task { await lab.finish(); onClose() }
                    }.disabled(closing)
                }
            }
            .onChange(of: scenePhase) { _, phase in if phase != .active { lab.stop() } }
            .onDisappear { lab.stop() }
        }
    }
}
