# Agent notes (this fork)

Read this before changing Volocal. There is no public README yet. Trust/network review is in `docs/SECURITY.md`.

MIT-licensed on-device iOS voice app: modular GGUF LLM, selectable STT/TTS, unsigned IPA via GitHub Actions. The developer does **not** have a Mac.

## Product constraints (do not violate)

- **On-device voice:** mic audio, transcripts, and LLM tokens must not leave the phone. Hugging Face is allowed **only** for listing/downloading public model weights (GGUF / CoreML packs).
- **LLM = GGUF only** via llama.cpp (`llama.swift` 2.x). No MLX, safetensors, ONNX, or pirated IPAs.
- **Barge-in** is the point of this app vs Locally AI. Keep one `AVAudioEngine`, Voice Processing AEC, mic open during TTS. Do not treat short transcripts as speaker echo — that blocks “stop”. Drop only near-silence (≈0.012 RMS) while TTS plays. After real speech, ~0.8s of quiet must end the turn even if Parakeet EOU stays busy on HVAC. Mic `AsyncStream` must be `.bufferingNewest` (not unbounded) or barge-in jetsams LiveContainer at ~1 GB. STT process and llama decode must not sit on the main actor. After interrupt, **wait for cancelled PocketTTS to leave the GPU** before `llama_decode`. Start TTS on the first clause (GPT Live style). llama.cpp and CoreML TTS take turns on `GPUExclusive` — never overlap Metal decode with PocketTTS/Supertonic. `speak()` returns after samples are **queued**, not after they finish playing; `scheduleTTSBuffer` must not wait on the player. Wait for `waitForPlaybackCompletion()` only at end of turn. Pause Parakeet (`STTManager.pauseInference`) while CoreML TTS runs. Prompt eval must **chunk `llama_decode` to `n_batch`** (256) — dumping the whole prompt into one `llama_batch` overruns C memory after a couple of turns. `llama_memory_clear(mem, true)` — metadata-only clear (`false`) leaves Metal KV dirty. Pause ASR `process` while `reset()` runs. Overlay `used/limit` is `phys_footprint` plus `os_proc_available_memory()`; GetMoreRAM on the LC host is often **~6144 MB**. A crash at **446/6144** is not jetsam. Disable the idle timer while the pipeline is not `.idle` (`UIApplication.shared.isIdleTimerDisabled`) — voice turns are not taps, so the screen would otherwise sleep mid-conversation.
- **Install:** LiveContainer guest when SideStore has no free slots (this developer’s iPhone). GetMoreRAM / `increased-memory-limit` goes on the **LiveContainer host**. LC must be **3.6.65+** on iOS 26.4+; JIT-Less Diagnose must pass; Reset Symbol Offset if guests crash at launch. **LC Multitask off** for this guest — it holds a mic tap and Listen crashes with `nullptr == Tap()`. Mic/AEC is weaker in a guest; barge-in may degrade. Extra RAM does not add barge-in to other apps.
- **Do not** add analytics, crash reporters, ads, accounts, or third-party SDKs that open a network path.
- **Do not** restore a foreign App Store bundle ID or DEVELOPMENT_TEAM. Bundle ID is `com.localiosllm.volocal`.
- **Do not** commit `.gguf`, CoreML packs, IPAs, or secrets.

## Stack (as of 2026-09-20)

| Piece | Default | Optional | Compute |
| --- | --- | --- | --- |
| STT | Parakeet EOU 320 (FluidAudio) | EOU 160; Nemotron 0.6B 560 ms | ANE |
| LLM | any HF GGUF (Qwen 2B Q4 suggested) | user-picked GGUF | Metal |
| TTS | PocketTTS v2.1 English, streaming | Supertonic-3 (10 voices, 44.1 kHz resampled to 24 kHz) | GPU+CPU (both; ANE left for STT) |

- llama.swift: **exact 2.10549.0**. `llama_sampler_init_penalties` is **5-arg**: `n_vocab, last_n, repeat, freq, present`. Use `llama_vocab_n_tokens(vocab)`.
- **Thinking off in the prompt** (`volocal.suppressThinking`, default on). `llama_chat_apply_template` is **not Jinja**. Gemma 4 uses native `<|turn>`. **E2B/E4B must not get an empty thought primer** — that primer is 12B+ only and on the small models it *starts* CoT (“the user just said…”). Ban the `<|channel>` token. Skip a thought channel / leading “the user said…” / “I need to respond naturally…” planning sentences if they still appear. Qwen 3 gets `/no_think`. Chat delimiters are stop sequences. Toggle is on the language-model screen. `SentenceBuffer` splits only on `.!?` (or a ~360-char runaway cap). Newlines are spaces, not clause breaks — splitting on `\n` stole the next word’s first letter for TTS. Do **not** force-split on 40/64 character word crumbs — that pauses mid-phrase.
- FluidAudio: SPM **exact 0.15.8**. APIs: `ModelHub.download`, `StreamingEouAsrManager.loadModels(to:)`, `PocketTtsResourceDownloader.ensureModels(language:)`, `Supertonic3Manager`. Do not float `from:` versions — CI already broke once on llama.cpp sampler arity. Do not re-add Chatterbox Nano or Kokoro ANE (Nano: slow I/O-KV; Kokoro: uncatchable abort on iOS 26.5 / this iPhone). Saved `chatterboxNano` / `kokoroAne` selections fall back to PocketTTS. `RetiredTTSCache` deletes their caches.
- PocketTTS emits **24 kHz** mono; Supertonic-3 emits **44.1 kHz** and must be resampled to 24 kHz before `SharedAudioEngine`. Do not add Dia/Orpheus. Inflect/NeuTTS are beta and not a guest fit (NeuTTS is stateful CoreML like Nano). Hugging Face “SOTA” TTS (Qwen3-TTS, Fish, CSM) is Python/CUDA — there is no iOS CoreML pack. `[laugh]` / `[gasp]` / `[sigh]` are `NonverbalSynth` cues. `[whisper]…[/whisper]` lowers the audio.
- System-prompt token count: `PromptTokenCounter.estimate` (~4 chars / CJK 1:1) until a GGUF is loaded, then `llama_tokenize`. Long chats: `trimHistory` drops oldest turns into `rollingMemory`, then `compressNotes` (~70 words) when idle. `generate(history:memory:)` injects that recap. Reset chat clears it. Do not run compress concurrently with generate (`inferenceEpoch`).
- Nemotron has **no EOU head**. `STTManager` treats ~900 ms of unchanged partial as a turn. `MicTurnDetector` also ends a turn after ~0.8 s of quiet.
- TTS is PocketTTS (streaming default) or Supertonic-3 (`dynamic` int4 VectorEstimator on CPU+GPU so Parakeet keeps the ANE). Do not put Supertonic on `.aneBucketed` next to Parakeet on this phone.

## Layout

- `project.yml` is source of truth; CI runs `xcodegen generate`. Keep `Volocal.xcodeproj` in sync.
- `Volocal/LLM/` — GGUF load, HF search (`HuggingFaceHub`), native chat prompts (`ChatPrompt`: Gemma 4 / Gemma 3 / Qwen 3 / ChatML / Llama 3).
- `Volocal/Models/` — downloads, onboarding, `LLMPickerView`, `VoiceEnginePickerView`, `VoiceEngines.swift`.
- `Volocal/STT`, `TTS`, `Audio`, `Pipeline` — live loop.
- `.github/workflows/ios-ipa.yml` + `scripts/package-unsigned-ipa.sh` — unsigned device IPA for SideStore.

## Build (no local Mac)

Linux cannot compile iOS. Push to GitHub → **Actions → Build IPA** (`macos-15`). Script ad-hoc-signs and embeds `llama` / `NemoTextProcessing` so iOS 26 dyld / LiveContainer can map the binary. Publishes **one rolling pre-release** `sidestore` with `Volocal.ipa` (`--clobber`). Do **not** use Actions artifacts (they are a zip). LiveContainer re-signs the guest with the imported SideStore cert. Do not ship random IPAs from the web.

## Hardware note

Target was iPhone 17 Pro (~12 GB). `increased-memory-limit` raises the **jetsam cap** (often ~6–8 GB), it does not add physical RAM. Overlay **6144** means GetMoreRAM is on the host; overlay **~1024** means it is not. 8B Q4 + STT + TTS is a stretch; keep defaults light. llama.cpp `n_ctx` is a **2K–128K log slider** (default 2048, snap 512). A kill during GGUF load writes `volocal.pendingLLMLoad` so the next launch **does not crash-loop** — show the error and let them pick a smaller Q4. Custom instructions persist in `UserDefaults` (`volocal.customInstructions`) and are the system prompt; they are not extra Hub traffic. Show a token count on that paste. GGUFs live under Documents/`models/llm` and can be swipe-deleted.

## If you change FluidAudio or llama.swift

Re-read public init/load/download signatures. FluidAudio 0.16+ may keep `ModelHub`; do not reintroduce `DownloadUtils`. Passing `nil` into EOU `loadModels(to:)` nests cache paths wrong — always pass `FluidAudioCache.asrModelsRoot`.
