# Trust review (2026-09-20)

Static review of this MIT-licensed tree. **Not a pentest** and not a binary/IPA scan of store builds.

## Verdict

**No malware, no telemetry SDK, no hidden exfil of voice or chat.** The app is a local STT → LLM → TTS pipeline. Network use is model-weight download (and, in this fork, Hugging Face Hub listing for GGUFs).

## What was searched

App Swift sources, `Info.plist`, entitlements, `PrivacyInfo.xcprivacy`, Fastlane, HTML docs, and the then-pinned FluidAudio fork vs `FluidInference/FluidAudio`. Grep covered `URLSession`, `http(s)`, analytics vendors (Firebase, Sentry, Crashlytics, Amplitude, Mixpanel, PostHog, Segment), ads/IDFA, `WKWebView`, `dlopen`/`JSContext`, vendor IDs, and upload APIs.

## Runtime network (this fork)

| Path | Host | Payload |
| --- | --- | --- |
| First-run / picker STT+TTS | Hugging Face via FluidAudio `ModelHub` | CoreML packs only |
| GGUF search, file list, download | `huggingface.co` (`HuggingFaceHub`, `UnifiedModelManager`) | Repo metadata + `.gguf` bytes |
| SPM at **build** time | GitHub | `llama.swift`, FluidAudio source/binaries |

Conversation audio, transcripts, and tokens are not attached to those requests. User-Agent is `volocal-ios/1.0 (on-device; no-telemetry)`.

Hugging Face still sees IP, User-Agent, and which public repo/file you fetch. That is weight distribution, not a voice cloud. After weights are on disk, voice works offline.

## What was not found

- Analytics / crash / ads SDKs
- Background upload of recordings
- Obfuscated C2, custom crypto tunnels, or unexpected `NWConnection`
- Tracking (`NSPrivacyTracking` is false; collected data types empty)

`os.Logger` is local Console logging only.

## Upstream FluidAudio pin (historical)

This tree previously pinned an outdated FluidAudio git branch. That pin is gone; the app uses official `FluidInference/FluidAudio` `from: 0.15.8`.

## Known non-malware issues (fixed or still true)

- Incomplete GGUF downloads used to look “ready.” This fork checks size and optional SHA-256 after download.
- `fastlane/` can App Store–sign **if** someone puts Apple credentials in CI. That is a release tool, not app runtime. Unsigned IPA CI does not use it.
- GGUF/CoreML files are large untrusted blobs. Prefer known Hugging Face repos; checksum when the Hub provides `sha256`.
- This review did not execute the iOS binary (no Mac in the workspace).

## Entitlements

- Microphone (`com.apple.security.device.audio-input`) — required for STT.
- `com.apple.developer.kernel.increased-memory-limit` — jetsam headroom for SideStore + GetMoreRAM. Not extra physical RAM.
