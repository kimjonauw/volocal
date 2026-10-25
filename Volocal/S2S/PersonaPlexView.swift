import SwiftUI

/// Dedicated settings and session view for NVIDIA PersonaPlex-7B Speech-to-Speech mode.
struct PersonaPlexView: View {
    @ObservedObject var manager: PersonaPlexManager
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            List {
                Section {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("NVIDIA PersonaPlex 7B")
                                .font(.headline)
                            Spacer()
                            Text("MLX 4-bit")
                                .font(.caption.bold())
                                .padding(.horizontal, 8)
                                .padding(.vertical, 3)
                                .background(Color.green.opacity(0.15))
                                .foregroundStyle(.green)
                                .clipShape(Capsule())
                        }
                        Text("January 2026 full-duplex Speech-to-Speech foundation model. Understands emotional tone and speaks simultaneously using Mimi audio tokens.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.vertical, 4)
                }

                if let error = manager.currentError {
                    Section {
                        Text(error)
                            .font(.caption)
                            .foregroundStyle(.red)
                    }
                }

                // Download / Status Section
                Section("Model Status") {
                    if manager.isDownloading {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                ProgressView()
                                    .padding(.trailing, 4)
                                Text(manager.statusText)
                                    .font(.subheadline)
                                Spacer()
                                Text("\(Int(manager.downloadProgress * 100))%")
                                    .font(.caption.monospacedDigit())
                                    .foregroundStyle(.secondary)
                            }
                            ProgressView(value: manager.downloadProgress, total: 1.0)
                                .tint(Color.accentColor)

                            Button("Cancel Download", role: .destructive) {
                                manager.cancelDownload()
                            }
                            .font(.caption)
                            .padding(.top, 4)
                        }
                        .padding(.vertical, 4)
                    } else if manager.isDownloaded {
                        HStack {
                            Label("Downloaded & Ready", systemImage: "checkmark.circle.fill")
                                .foregroundStyle(.green)
                            Spacer()
                            Text(PersonaPlexSpec.sizeDescription)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }

                        Button(role: .destructive) {
                            manager.deleteWeights()
                        } label: {
                            Label("Delete Model (~3.8 GB)", systemImage: "trash")
                        }
                    } else {
                        VStack(alignment: .leading, spacing: 8) {
                            HStack {
                                Text("Model not installed")
                                    .foregroundStyle(.secondary)
                                Spacer()
                                Text(PersonaPlexSpec.sizeDescription)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }

                            Button {
                                manager.downloadModel()
                            } label: {
                                Label("Download PersonaPlex 7B", systemImage: "arrow.down.circle.fill")
                                    .font(.headline)
                                    .frame(maxWidth: .infinity)
                                    .padding(.vertical, 8)
                            }
                            .buttonStyle(.borderedProminent)
                        }
                        .padding(.vertical, 4)
                    }
                }

                // Voice Preset Picker
                Section("Voice Preset") {
                    ForEach(PersonaPlexSpec.voicePresets) { preset in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                HStack {
                                    Text(preset.displayName)
                                        .foregroundStyle(.primary)
                                    Text("(\(preset.gender))")
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                                Text(preset.description)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            Spacer()
                            if manager.selectedVoice == preset.id {
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundStyle(Color.accentColor)
                            }
                        }
                        .contentShape(Rectangle())
                        .onTapGesture {
                            manager.selectedVoice = preset.id
                        }
                    }
                }

                // Persona Profile Picker
                Section("Persona Profile") {
                    ForEach(PersonaPlexSpec.defaultPersonas) { persona in
                        VStack(alignment: .leading, spacing: 4) {
                            HStack {
                                Text(persona.title)
                                    .font(.subheadline.bold())
                                    .foregroundStyle(.primary)
                                Spacer()
                                if manager.selectedPersonaId == persona.id {
                                    Image(systemName: "checkmark.circle.fill")
                                        .foregroundStyle(Color.accentColor)
                                }
                            }
                            Text(persona.summary)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        .contentShape(Rectangle())
                        .padding(.vertical, 2)
                        .onTapGesture {
                            manager.selectedPersonaId = persona.id
                        }
                    }
                }

                // Hardware Info Section
                Section("Hardware & Memory") {
                    HStack {
                        Text("Architecture")
                        Spacer()
                        Text("Dual-Stream Temporal + Depth")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("Memory Footprint")
                        Spacer()
                        Text("~5.1 GB / 6.0 GB (GetMoreRAM)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    HStack {
                        Text("Audio Codec")
                        Spacer()
                        Text("Mimi 24kHz (Neural Engine)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
            }
            .navigationTitle("PersonaPlex S2S")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Done") {
                        dismiss()
                    }
                }
            }
        }
    }
}
