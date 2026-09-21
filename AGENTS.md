# Agent notes (this fork)

Read this before changing Volocal. Human-facing overview is in `README.md`. Trust/network review is in `docs/SECURITY.md`.

MIT-licensed on-device iOS voice app: modular GGUF LLM, selectable STT/TTS, unsigned IPA via GitHub Actions. The developer does **not** have a Mac.

## Product constraints (do not violate)

- **On-device voice:** mic audio, transcripts, and LLM tokens must not leave the phone. Hugging Face is allowed **only** for listing/downloading public model weights (GGUF / CoreML packs).
- **LLM = GGUF only** via llama.cpp (`llama.swift` 2.x). No MLX, safetensors, ONNX, or pirated IPAs.
- **Barge-in** is the point of this app vs Locally AI. Keep one `AVAudioEngine`, Voice Processing AEC, mic open during TTS.
- **Install:** LiveContainer guest when SideStore has no free slots (this developer’s iPhone). GetMoreRAM / `increased-memory-limit` goes on the **LiveContainer host**. LC must be **3.6.65+** on iOS 26.4+; JIT-Less Diagnose must pass; Reset Symbol Offset if guests crash at launch. Mic/AEC is weaker in a guest; barge-in may degrade. Extra RAM does not add barge-in to other apps.
- **Do not** add analytics, crash reporters, ads, accounts, or third-party SDKs that open a network path.
- **Do not** restore a foreign App Store bundle ID or DEVELOPMENT_TEAM. Bundle ID is `com.localiosllm.volocal`.
- **Do not** commit `.gguf`, CoreML packs, IPAs, or secrets.

## Stack (as of 2026-09-20)

| Piece | Default | Optional | Compute |
| --- | --- | --- | --- |
| STT | Parakeet EOU 320 (FluidAudio) | EOU 160; Nemotron 0.6B 560 ms | ANE |
| LLM | any HF GGUF (Qwen 2B Q4 suggested) | user-picked GGUF | Metal |
| TTS | PocketTTS v2.1 English, streaming | Kokoro ANE (batched, prettier) | GPU+CPU (Pocket) / ANE (Kokoro) |

- llama.swift: **exact 2.10549.0**. `llama_sampler_init_penalties` is **5-arg**: `n_vocab, last_n, repeat, freq, present`. Use `llama_vocab_n_tokens(vocab)`.
- FluidAudio: SPM **exact 0.15.8**. APIs: `ModelHub.download`, `StreamingEouAsrManager.loadModels(to:)`, `PocketTtsResourceDownloader.ensureModels(language:)`. Do not float `from:` versions — CI already broke once on llama.cpp sampler arity.
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

Linux cannot compile iOS. Push to GitHub → **Actions → Build IPA** (`macos-15`). Script ad-hoc-signs and embeds `llama` / `NemoTextProcessing` so iOS 26 dyld / LiveContainer can map the binary. Publishes **one rolling pre-release** `sidestore` with `Volocal.ipa` (`--clobber`). Do **not** use Actions artifacts (they are a zip). LiveContainer re-signs the guest with the imported SideStore cert. Do not ship random IPAs from the web.

## Hardware note

Target was iPhone 17 Pro (~12 GB). `increased-memory-limit` raises the **jetsam cap** (often ~6–8 GB), it does not add physical RAM. 8B Q4 + STT + TTS is a stretch; keep defaults light.

## If you change FluidAudio or llama.swift

Re-read public init/load/download signatures. FluidAudio 0.16+ may keep `ModelHub`; do not reintroduce `DownloadUtils`. Passing `nil` into EOU `loadModels(to:)` nests cache paths wrong — always pass `FluidAudioCache.asrModelsRoot`.
