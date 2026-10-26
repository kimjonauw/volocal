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
    let mlxPeakMB: Double
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
    static let checkpoint = "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit"
    @Published private(set) var busy = false
    @Published private(set) var status = "Ready. Start at low volume. This is a voice audition, not a conversation backend yet."
    @Published private(set) var report = ""
    private var worker: Task<QwenAudition, Error>?
    private var driver: Task<Void, Never>?
    private var player: AVAudioPlayer?
    private var clip: Data?
    private var stopRequested = false

    func audition(text: String, speaker: String, delivery: String) {
        guard !busy else { return }
        stopPlayback()
        let text = String(text.trimmingCharacters(in: .whitespacesAndNewlines).prefix(160))
        let style = String(delivery.trimmingCharacters(in: .whitespacesAndNewlines).prefix(200))
        guard !text.isEmpty else { status = "Enter a short sentence first."; return }
        let voice = style.isEmpty ? speaker : speaker + ", " + style
        busy = true; stopRequested = false; report = ""
        status = "Downloading/loading Qwen3-TTS. First use downloads public weights from Hugging Face; text stays on this phone."
        let work = Task.detached(priority: .userInitiated) { [weak self] () throws -> QwenAudition in
            let free = Int(os_proc_available_memory())
            guard free == 0 || free >= 3_000_000_000 else { throw QwenAuditionError.memory(free / 1_000_000) }
            let previousCacheLimit = Memory.cacheLimit
            Memory.cacheLimit = 32 * 1024 * 1024
            defer { Memory.clearCache(); Memory.cacheLimit = previousCacheLimit }
            let start = Date()
            // Explicit type avoids routing this checkpoint to the older Qwen3 implementation.
            let loaded = try await TTS.loadModel(modelRepo: Self.checkpoint, modelType: "qwen3_tts")
            try Task.checkCancellation()
            guard let model = loaded as? Qwen3TTSModel else { throw QwenAuditionError.model }
            model.train(false)
            let loadSeconds = Date().timeIntervalSince(start)
            await self?.showGenerationStatus()
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
            // Attenuate peaks; never amplify near-silence. Playback also starts at 20%.
            let gain = min(Float(1), 0.75 / peak)
            return QwenAudition(samples: samples.map { $0 * gain }, sampleRate: model.sampleRate,
                                loadSeconds: loadSeconds, generationSeconds: Date().timeIntervalSince(generationStart),
                                mlxPeakMB: Double(Memory.peakMemory) / 1_000_000)
        }
        worker = work
        driver = Task { [weak self] in
            do {
                let result = try await work.value
                guard let self else { return }
                if !self.stopRequested {
                    self.clip = Self.wav(result.samples, sampleRate: result.sampleRate)
                    let duration = Double(result.samples.count) / Double(result.sampleRate)
                    self.report = String(format: "Load/download %.1f s · synthesis %.1f s · audio %.1f s\nSynthesis/audio ratio %.2f (under 1 is faster than playback)\nMLX lifetime peak %.0f MB; this excludes some app memory.\n%@\nDelivery: %@",
                                         result.loadSeconds, result.generationSeconds, duration,
                                         result.generationSeconds / duration, result.mlxPeakMB, speaker, style.isEmpty ? "none" : style)
                    try self.playClip()
                    self.status = "Preview playing at 20% player volume. Compare the same sentence and speaker with another delivery."
                } else { self.status = "Stopped; generation has drained." }
            } catch {
                guard let self else { return }
                self.status = self.stopRequested || error is CancellationError ? "Stopped; generation has drained." : "Qwen3-TTS: \(error.localizedDescription)"
            }
            guard let self else { return }
            self.worker = nil; self.driver = nil; self.busy = false
        }
    }

    private func showGenerationStatus() { status = "Synthesizing locally, up to 128 codec frames. This first test buffers one short clip before playback." }

    func stop() {
        stopRequested = true; worker?.cancel(); stopPlayback()
        if busy { status = "Stopping; waiting for the current model operation to return. Download/loading cancellation may take longer." }
        else { status = "Playback stopped." }
    }

    func finish() async {
        stop()
        await driver?.value
        clip = nil
    }

    func replay() {
        guard !busy, clip != nil else { return }
        do { try playClip(); status = "Replaying cached preview; no model inference." }
        catch { status = "Playback failed: \(error.localizedDescription)" }
    }

    private func playClip() throws {
        guard let clip else { return }
        player?.stop()
        let session = AVAudioSession.sharedInstance()
        try session.setCategory(.playback, mode: .default)
        try session.setActive(true)
        let player = try AVAudioPlayer(data: clip)
        player.volume = 0.2
        player.prepareToPlay()
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
        for sample in samples {
            let value = Int16(clamping: Int((max(-1, min(1, sample)) * 32767).rounded()))
            u16(UInt16(bitPattern: value))
        }
        return data
    }
}

struct QwenVoiceLabView: View {
    let onClose: () -> Void
    @StateObject private var lab = QwenVoiceLab()
    @Environment(\.scenePhase) private var scenePhase
    @State private var text = "Oh, you remembered that? That's actually really sweet."
    @State private var speaker = "Serena"
    @State private var delivery = "Warm, pleasantly surprised, and gently playful. Speak naturally without shouting."
    @State private var closing = false
    private let speakers = ["Serena", "Vivian", "Ryan", "Aiden"]

    var body: some View {
        NavigationStack {
            Form {
                Section("Qwen3-TTS 1.7B CustomVoice · 4-bit") {
                    Text("Standalone on-device audition. No microphone, LLM, PocketTTS, or LFM inference is running. This does not yet establish memory or speed with the conversational LLM loaded.")
                    Text("First use downloads a large public model. Prompts and generated audio stay local. Downloads remain cached. Each new audition reloads the cached model and releases it afterward; Replay uses the clip in memory.").font(.caption)
                }
                Section("Words to speak (160 characters)") {
                    TextEditor(text: $text).frame(minHeight: 70)
                        .onChange(of: text) { _, value in if value.count > 160 { text = String(value.prefix(160)) } }
                }.disabled(lab.busy)
                Section("Voice and delivery (200 characters)") {
                    Picker("Speaker", selection: $speaker) { ForEach(speakers, id: \.self) { Text($0).tag($0) } }
                    TextEditor(text: $delivery).frame(minHeight: 70)
                        .onChange(of: delivery) { _, value in if value.count > 200 { delivery = String(value.prefix(200)) } }
                    Button("Neutral delivery") { delivery = "Speak calmly with a neutral conversational tone." }
                    Button("Warm and playful") { delivery = "Warm, pleasantly surprised, and gently playful. Speak naturally without shouting." }
                }.disabled(lab.busy)
                Section {
                    Button(lab.busy ? "Working..." : "Generate and play preview") { lab.audition(text: text, speaker: speaker, delivery: delivery) }.disabled(lab.busy || closing)
                    Button("Replay last preview") { lab.replay() }.disabled(lab.busy || closing)
                    Button("Stop", role: .destructive) { lab.stop() }
                }
                Section("Status and measurements") {
                    Text(lab.status).textSelection(.enabled)
                    if !lab.report.isEmpty { Text(lab.report).font(.caption.monospaced()).textSelection(.enabled) }
                    Text("Keep device volume low for the first test. The 128-frame cap can truncate longer text. Cancellation is cooperative; no hard wall-clock deadline is claimed.").font(.caption)
                }
            }
            .navigationTitle("Qwen voice lab")
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button(closing ? "Closing..." : "Back") {
                        closing = true
                        Task { await lab.finish(); onClose() }
                    }.disabled(closing)
                }
            }
            .onChange(of: scenePhase) { _, phase in if phase != .active { lab.stop() } }
            .onDisappear { lab.stop() }
        }
    }
}
