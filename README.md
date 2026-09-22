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
- Custom conversation instructions (system prompt), a 2K–128K context slider, and a toggle to skip hidden reasoning (Gemma 4 / Qwen 3.5 thinking tokens)
- Swipe-delete unused GGUFs on the phone
- Pick STT (Parakeet EOU 160/320 or Nemotron 560) and TTS (PocketTTS v2.1 or Supertonic-3). **Laughs, gasps, whispers** play as short cues; `[whisper]…[/whisper]` lowers the audio. Token count of your pasted instructions sits under the text box. Long chats keep a sliding window of recent turns plus a recap of what was dropped.
- First-launch downloads from Hugging Face with per-model progress

## Why this stack

The hard part of running three models at once on a phone is that they all fight for the same hardware. Load is spread across different compute units:

| Component | Chip | Why |
| --- | --- | --- |
| **STT** (Parakeet EOU default) | Neural Engine | CoreML — leaves GPU free for the LLM |
| **LLM** (your GGUF) | GPU | llama.cpp via Metal |
| **TTS** (PocketTTS default) | CPU + GPU | CoreML; Supertonic-3 optional (44.1 kHz, resampled to 24 kHz) |

STT on [FluidAudio](https://github.com/FluidInference/FluidAudio) (CoreML / Neural Engine) and TTS on FluidAudio (CoreML / CPU+GPU) avoids GPU fights with llama.cpp.

### Models

| Component | Model | Download | Runtime |
| --- | --- | --- | --- |
| STT | Parakeet EOU 320 (default) or Nemotron 0.6B 560 ms | ~230–600 MB | CoreML (ANE) |
| LLM | Any llama.cpp GGUF from Hugging Face (Qwen 2B suggested) | you pick | llama.cpp (Metal) |
| TTS | PocketTTS v2.1 (default, streaming) or Supertonic-3 (10 voices) | ~200–550 MB | CoreML |

- **Parakeet EOU** — live barge-in with built-in end-of-utterance (~5% WER, 160/320 ms chunks). Nemotron 0.6B is optional (clearer, heavier, pause-based turns).
- **Any GGUF chat model** — llama.cpp (b10549) loads decoder GGUFs. Prefer **Q4_K / Q5_K / Q6_K**. A 250 MB file can still fail if it is an embedding/rerank GGUF, a different architecture, or flash-attn/KV that iOS Metal rejects. Qwen 3.5 2B Q4_K at 2048 is the known-good stack.
- **PocketTTS v2.1** — streaming (~80 ms frames) on whole sentences so it does not pause mid-phrase. Laughs are short cues, not acted speech. Several built-in voices.
- **Supertonic-3** — newer on-device CoreML TTS (FluidInference, 2025). Ten speakers, ~200 MB. 44.1 kHz internally, resampled to 24 kHz. Kokoro was removed: it aborts on iOS 26.5 on this iPhone. Other recent Hugging Face names (Qwen3-TTS, Fish, CSM, Orpheus) are not CoreML iOS packs.

### Audio

One shared `AVAudioEngine` for both STT input and TTS output, with Voice Processing AEC enabled on both nodes. This is what lets barge-in work — the mic stays open during playback and the hardware cancels the echo, so there is no need to mute the mic while speaking.

Runtime memory: the top bar is **used / LiveContainer jetsam cap**. GetMoreRAM on the **LiveContainer host** often shows **~6144 MB**. **~1024 MB** means the cap was not applied to the host. A crash at a few hundred MB of a 6 GB cap (for example **446/6144**) is a bug — not out-of-memory. After a couple of turns the prompt used to overflow llama.cpp's 256-token batch. TTS and llama take turns on the GPU: speech starts on the first clause, llama pauses only while CoreML is synthesizing, then continues while audio plays.

### Your context, window size, deleting GGUFs

Open the language-model screen (CPU icon):

- **Your context / instructions** — this is the system prompt sent every turn. A live token count sits under the box (`≈ tokens` until a GGUF is loaded, then exact llama.cpp tokens), including the expression cue if that toggle is on, and as a percent of `n_ctx`. It applies on the next reply; the GGUF does not reload. Tap **Reset to default** to restore the short spoken-assistant prompt.
- **Skip hidden reasoning** — on by default. Gemma 4 E2B/E4B must not get an empty thought primer (that is what makes them narrate “the user just said…”). 12B+ get a closed empty thought. A leading inner-monologue sentence is not spoken.
- **Context window** — log slider for llama.cpp `n_ctx` (2,048–131,072, snaps to 512). Default **2048**. KV cache RAM grows with this; 64K+ plus an 8B GGUF plus STT/TTS can jetsam in LiveContainer. A 2B Q4 at 32K is usually fine on this phone. The GGUF’s trained window is a second ceiling. Reloads when you lift your finger.
- **Long chats** — recent turns stay in full. Older turns are dropped from the prompt and folded into a short recap (then compressed) so the model does not forget the whole conversation when the window fills. Reset chat clears that recap.
- **Delete GGUFs** — swipe left on **On this iPhone** (or Files → On My iPhone → Volocal → `models`, when the IPA is a normal install). LiveContainer guests should swipe-delete in-app. Speech models stay in the FluidAudio cache until you delete the whole app.
- If a GGUF **kills the app while loading**, the next launch stops and shows an error instead of crash-looping. Pick a Q4 **2B–4B**; 8B+ with STT+TTS often jetsams inside LiveContainer.

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
6. Turn **off LiveContainer Multitask** for this guest. Multitask keeps a second tap on the shared mic; Listen then crashes with `required condition is false: nullptr == Tap()`.

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
- **SentenceBuffer** — splits streaming text at `.!?:;` boundaries. First audio after ~40 characters. llama.cpp waits only while CoreML is synthesizing.

## Project structure

```
Volocal/
├── App/        # Entry point, content view, model loading
├── Audio/      # SharedAudioEngine (AVAudioEngine + VP AEC)
├── STT/        # Parakeet EOU or Nemotron via FluidAudio
├── LLM/        # llama.cpp via llama.swift; Hugging Face GGUF picker
├── TTS/        # PocketTTS or Supertonic-3 via FluidAudio
├── Pipeline/   # Voice pipeline, sentence buffer, conversation UI
├── Models/     # Downloads, onboarding, LLM + voice-engine pickers
└── Debug/      # Metrics overlay (RAM, CPU, thermal)
```

## Dependencies

- [llama.swift](https://github.com/mattt/llama.swift) **2.10549.0** — Swift wrapper for llama.cpp
- [FluidAudio](https://github.com/FluidInference/FluidAudio) **0.15.8** — Parakeet EOU / Nemotron (STT), PocketTTS / Supertonic-3 (TTS)

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
- [Supertonic-3](https://huggingface.co/FluidInference/supertonic-3-coreml) — optional CoreML TTS via FluidAudio

## License

MIT. The `LICENSE` file must stay with the software.
