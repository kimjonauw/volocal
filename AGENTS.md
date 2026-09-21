# Agent notes (this fork)

Read this before changing Volocal. Human-facing overview is in `README.md`. Trust/network review is in `docs/SECURITY.md`.

Upstream: [fikrikarim/volocal](https://github.com/fikrikarim/volocal) (MIT). This tree is a SideStore-oriented fork: modular GGUF LLM, selectable STT/TTS, unsigned IPA via GitHub Actions. The developer does **not** have a Mac.

## Product constraints (do not violate)

- **On-device voice:** mic audio, transcripts, and LLM tokens must not leave the phone. Hugging Face is allowed **only** for listing/downloading public model weights (GGUF / CoreML packs).
- **LLM = GGUF only** via llama.cpp (`llama.swift` 2.x). No MLX, safetensors, ONNX, or pirated IPAs.
- **Barge-in** is the point of this app vs Locally AI. Keep one `AVAudioEngine`, Voice Processing AEC, mic open during TTS.
- **Install:** SideStore real app slot, not a LiveContainer guest (mic/AEC is flaky). `increased-memory-limit` is in entitlements; GetMoreRAM can re-apply it. Extra RAM does not add barge-in to other apps.
- **Do not** add analytics, crash reporters, ads, accounts, or third-party SDKs that open a network path.
- **Do not** restore `com.fikrikarim.volocal` / team `FBS8R927D4`. Bundle ID is `com.localiosllm.volocal`.
- **Do not** commit `.gguf`, CoreML packs, IPAs, or secrets.

## Stack (as of 2026-09-20)

| Piece | Default | Optional | Compute |
| --- | --- | --- | --- |
| STT | Parakeet EOU 320 (FluidAudio) | EOU 160; Nemotron 0.6B 560 ms | ANE |
| LLM | any HF GGUF (Qwen 2B Q4 suggested) | user-picked GGUF | Metal |
| TTS | PocketTTS v2.1 English, streaming | Kokoro ANE (batched, prettier) | GPU+CPU (Pocket) / ANE (Kokoro) |

- FluidAudio: SPM `https://github.com/FluidInference/FluidAudio.git` **from 0.15.8** (not the old `fikrikarim/FluidAudio` `branch: main`). APIs: `ModelHub.download`, `StreamingEouAsrManager.loadModels(to:)`, `PocketTtsResourceDownloader.ensureModels(language:)`.
- llama.swift: `from: "2.0.0"`. `llama_sampler_init_penalties` is **4-arg** on this pin. Newer llama.cpp puts `n_vocab` first — update the call if you bump the package.
- PocketTTS and Kokoro both emit **24 kHz** mono; `SharedAudioEngine.ttsFormat` is 24 kHz. Do not add 44.1 kHz TTS (e.g. Supertonic-3) without resampling.
- Nemotron has **no EOU head**. `STTManager` treats ~900 ms of unchanged partial as a turn.
- Kokoro is sentence-batched and shares ANE with Parakeet; default stays PocketTTS for barge-in.

## Layout

- `project.yml` is source of truth; CI runs `xcodegen generate`. Keep `Volocal.xcodeproj` in sync.
- `Volocal/LLM/` — GGUF load, HF search (`HuggingFaceHub`), chat template + ChatML fallback, `ThinkTagFilter`.
- `Volocal/Models/` — downloads, onboarding, `LLMPickerView`, `VoiceEnginePickerView`, `VoiceEngines.swift`.
- `Volocal/STT`, `TTS`, `Audio`, `Pipeline` — live loop.
- `.github/workflows/ios-ipa.yml` + `scripts/package-unsigned-ipa.sh` — unsigned device IPA for SideStore.

## Build (no local Mac)

Linux cannot compile iOS. Push to GitHub → **Actions → Build IPA** (`macos-15`, `CODE_SIGNING_ALLOWED=NO`). User re-signs in SideStore. Do not ship random IPAs from the web.

## Hardware note

Target was iPhone 17 Pro (~12 GB). `increased-memory-limit` raises the **jetsam cap** (often ~6–8 GB), it does not add physical RAM. 8B Q4 + STT + TTS is a stretch; keep defaults light.

## If you change FluidAudio or llama.swift

Re-read public init/load/download signatures. FluidAudio 0.16+ may keep `ModelHub`; do not reintroduce `DownloadUtils`. Passing `nil` into EOU `loadModels(to:)` nests cache paths wrong — always pass `FluidAudioCache.asrModelsRoot`.
