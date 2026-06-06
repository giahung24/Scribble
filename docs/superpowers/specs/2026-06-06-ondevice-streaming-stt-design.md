# On-device Streaming Realtime STT + Translation — Design (Phase 1)

**Date:** 2026-06-06
**Status:** Approved (design phase)
**Branch:** `feat/ondevice-streaming-stt` (cut from `main`)

> This branch is cut from `main`. The experimental OpenAI-compatible "Local
> (Whisper) server" provider and its pseudo-realtime path live on a separate,
> unmerged branch (`feat/local-stt-provider`) and are **not** assumed here. On
> `main` the STT providers are **Nvidia Riva** and **Soniox**; this feature adds
> a third: an **on-device** streaming provider that needs no external server.

## Goal

Deliver an **on-device** STT engine that runs across **Windows, Linux, and
macOS**, prioritizing **Vietnamese** quality and covering **VI / EN / FR**, in
**two modes from one bundled engine + model cache**:

1. **Realtime** — low-latency **streaming** (growing-partial, near
   token-by-token) live transcription **and** translation. Replaces the slow
   LLM-per-chunk translation with a fast neural MT engine.
2. **Batch / upload** — a user can upload an audio/video file and transcribe it
   fully on-device with **no external server**, using the **best model
   (large-v3)** since latency is irrelevant for files.

The two modes share the same `faster-whisper` engine and the same downloaded
models: `faster-whisper` is a batch engine at its core, and the realtime mode is
a streaming layer built on top of it.

## Why

The existing realtime providers stream partial tokens with ~200–500 ms latency
(Riva gRPC, Soniox WS) but are cloud services. An on-device equivalent must
avoid two latency traps: (1) batch ASR that waits for an utterance to end, and
(2) an LLM round-trip per chunk for translation. This design uses **streaming
Whisper** (growing partials via a commit policy) for ASR and **CTranslate2 +
NLLB-200** for fast incremental translation.

## Reality constraints (decided during brainstorming)

- **True ~300 ms token-by-token for Vietnamese is only realistic on a GPU**
  (e.g. SeamlessStreaming). That is **Phase 2**. Off-the-shelf streaming ASR
  models for VI/FR do not exist for CPU transducer engines (sherpa-onnx ships
  streaming EN/ZH/KO only; its VI model is offline/batch).
- **Phase 1 targets the achievable CPU win**: Whisper-streaming partials
  (~1–2 s, growing) — a large improvement over batch — multilingual (VI/EN/FR),
  Whisper-grade VI accuracy; with near-instant neural translation.
- **Multi-platform is a hard requirement.** The engine must run on Win/Linux/
  macOS. GPU acceleration via CUDA (Win/Linux + NVIDIA) is an automatic upgrade;
  macOS and CPU-only machines use the same code on CPU.

## Architecture

One streaming engine, scaling by detected hardware. Lives in the Python sidecar
(consistent with how Nvidia/Soniox realtime already work) behind a new WebSocket
endpoint that mirrors the **existing** realtime JSON message shapes, so the
frontend needs no new message handling — only provider selection + routing.

```
mic / system PCM ─▶ ws /ws/ondevice-stream ─▶ OnDeviceStreamingSession
   feed_audio(pcm) ─▶ [rolling audio buffer]
      every ~250-500ms: faster-whisper decode of the buffer
         ─▶ LocalAgreement commit policy → stable prefix (committed) + tail (partial)
         ─▶ emit partial {is_final:false} as the tail grows
      on endpoint / buffer trim:
         ─▶ commit segment {is_final:true}
         ─▶ CAM++ diarize the segment PCM (reuse existing diarizer)
         ─▶ NMT translate (CTranslate2 + NLLB), re-translated/throttled
         ─▶ emit final + translation
```

### ASR: streaming Whisper

- **Backend:** `faster-whisper` (CTranslate2). Cross-platform (Win/Linux/macOS)
  on CPU; CUDA on Win/Linux + NVIDIA. Single API, `device`/`compute_type`
  selected at runtime.
- **Streaming commit policy:** **LocalAgreement-2** (UFAL `whisper_streaming`
  algorithm): repeatedly decode a growing audio window; the longest common token
  prefix between consecutive decodes is *committed* (stable); the remainder is
  the live *partial*. On a detected endpoint (silence) or max-window, commit the
  segment and trim the buffer. (AlignAtt / WhisperLiveKit is a documented
  alternative to evaluate if LocalAgreement latency is unsatisfactory; Phase 1
  ships LocalAgreement-2 for simplicity and determinism.)
- Emits **growing partials** (`is_final:false`) and **committed finals**
  (`is_final:true`) — exactly the shape the frontend already renders.

### Adaptive hardware matrix

The model column below is the **realtime** default (latency-tuned). **Batch/
upload always uses `large-v3`** regardless of hardware (see Batch mode), since a
file has no latency constraint.

| Platform | Device | compute_type | Realtime ASR model | Notes |
|---|---|---|---|---|
| Windows/Linux + NVIDIA | `cuda` | `float16` | `large-v3` | best VI; lowest latency |
| Windows/Linux CPU | `cpu` | `int8` | `small` (configurable: `medium`) | ~1–2 s partials |
| macOS Apple Silicon | `cpu` | `int8` | `small` (configurable: `medium`) | CTranslate2 has no Metal backend; Metal accel via whisper.cpp is **Phase 2** |
| macOS Intel | `cpu` | `int8` | `small` | — |

Detection: probe CUDA availability (via CTranslate2's device support /
`torch.cuda.is_available()` if torch present, else CTranslate2 capability check).
Choose `device`, `compute_type`, and default model size accordingly. The user can
override the model size in Settings (small/medium/large-v3) regardless of device,
with a warning that large-v3 on CPU will not keep up in realtime.

### Translation: CTranslate2 + NLLB-200

- **Model:** `nllb-200-distilled-600M` converted to CTranslate2 `int8` (CPU) /
  `float16` (CUDA). Supports VI/EN/FR (and ~200 langs). This is the stack OBS
  LocalVocal / Polyglot use for local realtime captions.
- **Incremental policy (re-translation):** translate `committed_text + partial`
  on a throttle (≈250–300 ms debounce; skip if text unchanged). Emit a
  `{type:"translation", chunk_id, translation, append:true}` update keyed to the
  current segment's `chunk_id`. On segment commit, do a final translation of the
  committed text. Re-translation may refine earlier words ("flicker") — accepted
  tradeoff for low latency.
- Source language = `stt_language`; target = the `translate_lang` query param /
  `TRANSLATE:` command (reusing the existing realtime translation control).

### Diarization

Reuse the existing module-level CAM++ `diarizer`: at each segment commit, run
`diarizer.identify_speaker_from_samples(samples, 16000)` over that segment's PCM
to attach `speaker` / `speaker_id`. Mirrors the Nvidia handler. No new
diarization code.

### Batch / upload mode

The same engine transcribes uploaded files — no streaming, no commit policy, no
server. Because latency is irrelevant for a file, batch uses the **best model
the user has (default `large-v3`)** for maximum VI/EN/FR accuracy, independent of
the realtime model size.

```
uploaded file ─▶ normalize → WAV 16k mono (existing pipeline step)
   ─▶ faster-whisper transcribe(wav, model=large-v3, vad_filter=True)
      → segment generator (text + start/end), emitted as progress
   ─▶ CAM++ diarize per segment (reuse existing diarizer)
   ─▶ same transcript JSON the nvidia/soniox batch paths produce
```

- **Integration:** add `ondevice` to the **existing upload pipeline's provider
  dispatch** (the same place that today routes `nvidia` vs `soniox` batch). The
  on-device branch runs faster-whisper over the normalized WAV. faster-whisper's
  built-in VAD handles long audio, and its segment generator gives natural
  progress reporting (emit a chunk event per segment).
- **Model sharing:** batch reuses the realtime model cache / download manager. If
  `large-v3` is absent it is fetched on demand (with the same progress UI).
- **Quality:** batch is **not** compromised by the realtime latency tradeoff —
  it always uses large-v3, so on-device upload matches cloud-grade VI accuracy.

This makes the on-device provider a complete local STT: **upload anytime**, and
**record live** — both without any server.

## Components

**New (Python sidecar):**
- `src-python/stt_providers/ondevice/streaming_asr.py` — `WhisperStreamingASR`:
  wraps faster-whisper + the LocalAgreement-2 commit policy. Pure-ish: feed PCM,
  poll for `(committed_delta, partial_text, endpoint?)`.
- `src-python/stt_providers/ondevice/commit_policy.py` — `LocalAgreement`
  (token-prefix agreement). **Unit-testable in isolation** (no model).
- `src-python/stt_providers/ondevice/translator.py` — `NllbTranslator` (CT2) +
  a `ThrottledRetranslator` (debounce/skip-unchanged). Translator unit-testable
  with a mocked CT2 translate fn.
- `src-python/stt_providers/ondevice/models.py` — model registry +
  **download-on-demand** manager: language/size → HF repo/URL, target dir under
  `~/.voicescribe/models/`, skip-if-present, progress callback. Unit-testable
  (path/skip logic) with mocked download.
- `src-python/stt_providers/ondevice/session.py` — `OnDeviceStreamingSession`
  (`start/feed_audio/results/stop`, same duck-typed shape as the existing
  streamers; bounded queue + worker; emits the result dicts).
- `src-python/stt_providers/ondevice/batch.py` — `transcribe_file_ondevice(wav,
  language, model_size="large-v3") -> segments`: whole-file faster-whisper
  transcription (built-in VAD), yielding segments for progress. Reuses the model
  manager. Unit-testable with a mocked faster-whisper.
- `src-python/main.py` — new `@app.websocket("/ws/ondevice-stream")` handler
  (realtime), structurally parallel to the Nvidia handler; emits the existing
  JSON shapes; validates models present (else a terminal error directing the user
  to download).
- `src-python/services/upload_pipeline.py` — add an `ondevice` branch to the
  existing batch provider dispatch (alongside nvidia/soniox) that calls
  `transcribe_file_ondevice` then layers CAM++ diarization, producing the same
  transcript JSON.

**New / modified (frontend):**
- New provider value `ondevice`; a provider tab "On-device (Realtime)" — no API
  key, no Base URL; shows model status + a download/progress control; language +
  translation reuse existing UI.
- `recording-constants.ts`: `WS_PATH_ONDEVICE = "/ws/ondevice-stream"`.
- `use-streaming-stt.ts`: route `provider === "ondevice"` → `WS_PATH_ONDEVICE`.
- `RecordingBar.tsx`: record-start readiness branch for `ondevice` (models
  present? else prompt to download / open settings).
- `UploadAudioModal.tsx`: pre-upload readiness branch for `ondevice` (batch model
  present? else prompt to download) — the same readiness pattern as recording.
- Model-management UI (Settings): list required models for the selected language,
  show downloaded/size, trigger download with progress (SSE/event), allow delete.

**Modified (Rust):**
- `src-tauri/src/lib.rs` `system_audio_ws_loop`: route `ondevice` →
  `/ws/ondevice-stream` (the same provider-routing the loop already does for
  nvidia/soniox).

## Data flow / message shapes (reused, no frontend changes)

- **Partial:** `{text, is_final:false, speaker, speaker_id, chunk_id}`
- **Final:** `{text, is_final:true, speaker, speaker_id, chunk_id}`
- **Translation:** `{type:"translation", translation, chunk_id, append:true}`
- **Terminal error / model-missing:** `{error:true, terminal:true, text, is_final:true, speaker:"System", speaker_id:-1}`
- **Model-download progress** (new, optional): a small status event the frontend
  may show; if the engine refuses to start without models, it sends the terminal
  error above instead.

## Model delivery & packaging (cross-platform — the hard part)

- **Libraries** (`faster-whisper`, `ctranslate2`, `sentencepiece`/tokenizers)
  ship as per-platform wheels with native libs. They are added to
  `requirements.txt` and bundled by PyInstaller per OS. **Risk to verify:**
  PyInstaller correctly collects the CTranslate2 / faster-whisper native
  binaries and tokenizer data on all three OSes (hidden-imports / `collect_all`
  / binary hooks may be needed). This is the #1 packaging risk and gets an
  explicit verification step in the plan.
- **Models are downloaded on demand**, NOT bundled (they are large: Whisper
  small ≈ 0.5 GB, medium ≈ 1.5 GB, large-v3 ≈ 3 GB; NLLB-600M int8 ≈ 0.6–1.2 GB).
  Bundling would bloat the installer past reason. Models are fetched on first use
  into `~/.voicescribe/models/`, with a progress UI and a Settings manager.
  Offline/air-gapped support (a bundled-models installer variant) is **out of
  scope for Phase 1** (documented as a possible later option).

## Error handling

- **Models missing / not yet downloaded** → terminal error directing the user to
  download (the readiness gate should catch this before recording starts).
- **Model load / device init failure** (e.g. CUDA OOM) → terminal error; suggest
  a smaller model / CPU.
- **Per-decode exception** → log + skip that decode; session stays alive.
- **CPU can't keep up** (decode slower than audio) → the engine drops to a longer
  decode interval and/or logs a "use a smaller model / GPU" warning; never blocks
  the audio receive loop. Documented tradeoff, not a crash.
- **Disconnect / stop** → tear down worker, free buffers (same lifecycle
  guarantees as the existing streamers).

## Testing

- **Unit — `LocalAgreement`:** feed sequences of token hypotheses; assert the
  committed prefix grows monotonically and only on agreement, and that partials
  reflect the uncommitted tail. (No model needed.)
- **Unit — `ThrottledRetranslator`:** assert debounce (no re-translate within the
  interval), skip-unchanged, and final-on-commit, with a mocked translate fn.
- **Unit — model manager:** language/size → path mapping, skip-if-present, and
  download invoked only when absent (mocked downloader).
- **Unit — `OnDeviceStreamingSession`:** with a **mocked** ASR + translator,
  assert it yields partials then a final (with PCM for diarization) and that
  `stop()` terminates `results()` (mirrors the lifecycle tests we already trust).
- **Unit — batch transcriber:** with a mocked faster-whisper, assert
  `transcribe_file_ondevice` maps the segment generator to the expected segment
  list and selects the configured model size.
- **Integration / manual E2E** (requires downloaded models): run live VI/EN/FR,
  confirm growing partials, segment commits, speaker labels, near-instant
  translation, and clean stop; verify on CPU and (if available) CUDA. **Also
  upload** a VI/EN/FR file and confirm large-v3 on-device batch transcription +
  diarization produces a correct transcript with no server running.
- **Packaging smoke:** a build on each OS that imports the engine and runs one
  decode on a fixed WAV — catches native-lib bundling failures.

## Out of scope (Phase 2)

- **SeamlessStreaming** (NVIDIA GPU) for true end-to-end token-by-token VI/EN/FR
  with no re-translation flicker — the "Riva/Soniox-instant" tier.
- **macOS Metal acceleration** (whisper.cpp / CoreML) for faster Mac ASR.
- **Hardware auto-detection that swaps to Seamless** when a capable GPU is found
  (Phase 1 only auto-selects faster-whisper device + model size).
- **Bundled-models (offline) installer variant.**

## Risks / to verify during implementation

1. **PyInstaller native-lib bundling** of CTranslate2 / faster-whisper /
   tokenizers on Windows, Linux, macOS (highest risk).
2. **CPU streaming cost**: confirm `small` (and optionally `medium`) keep up at
   the chosen decode interval on a typical CPU; `large-v3` is GPU-only in
   practice.
3. **VI/FR quality** at `small`/`medium` vs the cloud providers — set
   expectations; `medium` likely needed for good VI on CPU.
4. **Re-translation flicker** UX — confirm acceptable; throttle tuning.
5. **NLLB language codes** (FLORES-200 codes, e.g. `vie_Latn`, `eng_Latn`,
   `fra_Latn`) mapping from the app's `stt_language` / target.

## Decisions log

- **Engine:** faster-whisper (CTranslate2) + LocalAgreement-2 streaming; NLLB-200
  (CT2) for translation. Cross-platform CPU, CUDA upgrade.
- **Two modes, one engine:** realtime (streaming, small/medium) **and** batch/
  upload (whole-file, large-v3) share the engine + model cache; on-device upload
  needs no server and isn't latency-compromised (always large-v3).
- **Latency stance:** CPU = ~1–2 s growing partials (Phase 1); true ~300 ms
  token-by-token VI = GPU/SeamlessStreaming (Phase 2).
- **Provider:** new `ondevice` provider alongside Nvidia/Soniox; no server, no
  API key.
- **Models:** download-on-demand into `~/.voicescribe/models/`; not bundled.
- **Message shapes:** reuse the existing realtime JSON contract (zero frontend
  message-handling changes).
- **Multi-platform:** Win/Linux/macOS on CPU via one code path; CUDA auto-upgrade
  on Win/Linux + NVIDIA; macOS Metal deferred to Phase 2.
