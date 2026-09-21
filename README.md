# Volocal

Fully local voice AI for iOS. No cloud, no API keys, no internet after model download.

STT → LLM → TTS, streaming on-device, with barge-in (interrupt mid-sentence).

Pick any Hugging Face GGUF for the language model, and swap STT/TTS engines. GitHub Actions builds an unsigned IPA for SideStore (no Mac required).

**Agents / future contributors:** read [AGENTS.md](AGENTS.md). Trust/network review: [docs/SECURITY.md](docs/SECURITY.md).

> **Note:** This is a work in progress. Expect bugs.

## Features

- Runs entirely on-device across Neural Engine, GPU, and CPU
- Real-time voice conversations with interrupt (barge-in)
- Hardware echo cancellation so the mic doesn't pick up its own output
- Pick any llama.cpp **GGUF** from Hugging Face
- Pick STT (Parakeet EOU 160/320 or Nemotron 560) and TTS (PocketTTS v2.1 or Kokoro ANE)
- First-launch downloads from Hugging Face with per-model progress

## Why this stack

The hard part of running three models at once on a phone is that they all fight for the same hardware. Load is spread across different compute units:

| Component | Chip | Why |
| --- | --- | --- |
| **STT** (Parakeet EOU default) | Neural Engine | CoreML — leaves GPU free for the LLM |
| **LLM** (your GGUF) | GPU | llama.cpp via Metal |
| **TTS** (PocketTTS default) | CPU + GPU | CoreML — ANE can artifact Mimi; Kokoro optional on ANE |

STT on [FluidAudio](https://github.com/FluidInference/FluidAudio) (CoreML / Neural Engine) and TTS on FluidAudio (CoreML / CPU+GPU) avoids GPU fights with llama.cpp.

### Models

| Component | Model | Download | Runtime |
| --- | --- | --- | --- |
| STT | Parakeet EOU 320 (default) or Nemotron 0.6B 560 ms | ~230–600 MB | CoreML (ANE) |
| LLM | Any llama.cpp GGUF from Hugging Face (Qwen 2B suggested) | you pick | llama.cpp (Metal) |
| TTS | PocketTTS v2.1 (default, streaming) or Kokoro ANE (prettier, batched) | ~350–550 MB | CoreML |

- **Parakeet EOU** — live barge-in with built-in end-of-utterance (~5% WER, 160/320 ms chunks). Nemotron 0.6B is optional (clearer, heavier, pause-based turns).
- **Any GGUF** — llama.cpp loads what you download. Qwen-class 2B Q4 is the suggested size for a 12 GB iPhone with STT+TTS resident.
- **PocketTTS v2.1** — streaming (~26 ms to first audio). Kokoro ANE sounds nicer but is batched and contends with Parakeet for ANE.

### Audio

One shared `AVAudioEngine` for both STT input and TTS output, with Voice Processing AEC enabled on both nodes. This is what lets barge-in work — the mic stays open during playback and the hardware cancels the echo, so there is no need to mute the mic while speaking.

Runtime memory: ~1.2 GB with the small default stack. Larger GGUFs need `increased-memory-limit` (entitlement + GetMoreRAM). That raises the process cap; it does not add physical RAM.

## Privacy / network

Voice and chat stay on the phone. The only runtime network is **Hugging Face** for public model files (and listing GGUF repos). No analytics SDKs. Details: [docs/SECURITY.md](docs/SECURITY.md).

## Getting started

This repo is **public** so GitHub Actions macOS minutes stay free (private macOS runners need paid credits).

**Install path (no SideStore slots):** LiveContainer **guest**. That is the supported path on this phone when all 3 slots are taken. Apply GetMoreRAM to **LiveContainer itself** (the guest shares that process).

1. LiveContainer **3.6.65 or newer** (required on iOS 26.4+ / 26.5.1). Older LC crashes every guest on launch.
2. In LC Settings: **Import Certificate from SideStore**, then **JIT-Less Mode Diagnose** must pass (green + Test JIT-Less Mode).
3. If guests still die instantly: tap the LC version number 5–10 times → **Reset Symbol Offset**.
4. Wait for **Actions → Build IPA** to go green, then download **Volocal.ipa** from [Releases → sidestore](https://github.com/kimjonauw/volocal/releases/tag/sidestore) (not Actions artifacts — those are always a zip).
5. In LiveContainer, **+** → import that IPA → open Volocal. If it was already imported from the old zip, delete the guest and import again.

Mic / Voice Processing AEC is weaker inside a guest than a real slot. Barge-in may be worse; the app should still launch and run on-device.

Physical iPhone, iOS 17+ (developed against iOS 26 / iPhone 17 Pro). First launch still downloads STT / TTS / GGUF from Hugging Face on-device.

If you do have a Mac: Xcode 16+, [XcodeGen](https://github.com/yonaskolb/XcodeGen) (`brew install xcodegen`).

```bash
xcodegen generate
open Volocal.xcodeproj
```

Or `./scripts/package-unsigned-ipa.sh` for a SideStore IPA.

## Architecture

```
Mic → [SharedAudioEngine] → STTManager → VoicePipeline → LLMManager → SentenceBuffer → TTSManager → Speaker
                                              ↑                                              |
                                              └──── barge-in (interrupt on speech) ──────────┘
```

- **SharedAudioEngine** — one `AVAudioEngine` shared by STT and TTS, with VP AEC on both input and output nodes.
- **VoicePipeline** — runs the loop. Turn revision guards prevent stale tasks from messing things up after a barge-in. LLM tokens stream through a sentence buffer so TTS can start before generation finishes.
- **SentenceBuffer** — splits streaming text at `.!?:;` boundaries (max 200 chars) so each TTS chunk stays short.

## Project structure

```
Volocal/
├── App/        # Entry point, content view, model loading
├── Audio/      # SharedAudioEngine (AVAudioEngine + VP AEC)
├── STT/        # Parakeet EOU or Nemotron via FluidAudio
├── LLM/        # llama.cpp via llama.swift; Hugging Face GGUF picker
├── TTS/        # PocketTTS v2.1 or Kokoro ANE via FluidAudio
├── Pipeline/   # Voice pipeline, sentence buffer, conversation UI
├── Models/     # Downloads, onboarding, LLM + voice-engine pickers
└── Debug/      # Metrics overlay (RAM, CPU, thermal)
```

## Dependencies

- [llama.swift](https://github.com/mattt/llama.swift) **2.10549.0** — Swift wrapper for llama.cpp
- [FluidAudio](https://github.com/FluidInference/FluidAudio) **0.15.8** — Parakeet EOU / Nemotron (STT), PocketTTS / Kokoro (TTS)

Both pinned exact via SPM (`project.yml`) so CI cannot silently pick a breaking llama.cpp.

## TODO

- [ ] Add [Apple Foundation Models](https://developer.apple.com/documentation/FoundationModels) as an LLM option (iOS 26+)
- [ ] Android support (might be far in the future)

## Acknowledgements

- [FluidAudio](https://github.com/FluidInference/FluidAudio) by FluidInference — CoreML implementations of Parakeet EOU and PocketTTS that make the Neural Engine strategy possible
- [llama.swift](https://github.com/mattt/llama.swift) by Mattt — clean Swift bindings for llama.cpp
- [llama.cpp](https://github.com/ggml-org/llama.cpp) by ggml — the LLM inference engine
- [Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B) by Qwen — a suggested language-model family
- [Parakeet EOU](https://huggingface.co/nvidia/parakeet-tdt_ctc-110m) by NVIDIA NeMo — speech recognition
- [PocketTTS](https://github.com/kyutai-labs/pocket-tts) by Kyutai — text-to-speech
- [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M) — optional ANE TTS via FluidAudio

## License

MIT. The `LICENSE` file must stay with the software.
