# Local Realtime STT (v2) — Design

**Date:** 2026-06-06
**Status:** Approved (design phase)
**Builds on:** `2026-06-06-local-stt-provider-design.md` (v1, batch/upload only)

## Goal

Add **realtime (live) streaming transcription** to the local (self-hosted,
OpenAI-compatible Whisper) STT provider, integrating with the existing realtime
path that today supports only Nvidia Riva (gRPC streaming) and Soniox (WebSocket).
This iteration ships **STT + live translation together**.

## Context & Constraint

Whisper-family models are **batch-native** (≈30 s windows); there is no universal
realtime streaming protocol across OpenAI-compatible servers the way Riva and
Soniox expose one. We therefore implement **pseudo-realtime via VAD-chunked
batch**: buffer the live PCM stream, cut it into utterance-sized chunks on speech
pauses, and transcribe each chunk through the **same** `/v1/audio/transcriptions`
endpoint the v1 batch provider already uses.

Consequences:
- Works with **any** OpenAI-compatible server (Speaches, whisper.cpp server,
  vLLM) — the **same server** used for batch. No new server requirement.
- **Finals only** — segments appear after each utterance/pause (typically 2–6 s);
  there is no live partial-word streaming like Riva/Soniox token streaming. This
  is inherent to batch Whisper.
- **Latency tracks the user's server**: GPU + distil/large-v3 feels live; CPU
  large-v3 may lag. The user controls this via their server/model choice.

## Architecture

Local realtime mirrors the existing Nvidia realtime path structurally. We add a
**third WebSocket endpoint** `/ws/local-stream` and a streaming session class
parallel to `NvidiaStreamingSTT`. The actual transcription delegates to the v1
`LocalOpenAIProvider.transcribe_file`, so realtime and batch hit the same server
and endpoint. Diarization and live translation reuse the exact mechanisms the
Nvidia handler already uses (CAM++ + `translate_instant`).

```
mic / system PCM ─▶ ws /ws/local-stream ─▶ LocalRealtimeSession
   feed_audio(pcm) ─▶ [streaming energy endpointer]
      on pause (~0.6 s) OR 12 s cap ─▶ flush utterance ─▶ temp WAV (16 kHz mono)
         ─▶ LocalOpenAIProvider.transcribe_file  (batch /v1/audio/transcriptions)
         ─▶ CAM++ diarize (reuse nvidia)
         ─▶ translate_instant (reuse, throttled)
         ─▶ send_json FINAL segment {text, speaker, chunk_id, translation, …}
```

The capture and transport layers (Web Audio mic capture + Rust CoreAudio/WASAPI
system audio → 16 kHz int16 PCM → binary WebSocket frames) are **unchanged**; the
Rust `system_audio_ws_loop` and the frontend PCM streamer already send to a path
chosen by provider, so only the path string and the sidecar handler are new.

### Note on the upload VAD

The upload pipeline's VAD (`services/vad_splitter.py`) is **offline ffmpeg
`silencedetect` over a complete WAV file** — it cannot run on a live PCM stream.
Realtime therefore needs a **new streaming endpointer**. We reuse the upload
splitter's threshold *values* (−30 dB / 300 ms) as the basis for the streaming
endpointer's defaults so behavior stays consistent, but the detector itself is
new code.

## Components

### `src-python/stt_providers/local_realtime.py` (new)

`LocalRealtimeSession` — interface parallel to the existing streaming classes:

- `start()` — initialize the PCM buffer, the bounded chunk queue, the endpointer
  state, and a single sequential transcribe worker thread.
- `feed_audio(pcm_bytes: bytes)` — append int16 PCM from the WebSocket; run the
  streaming endpointer; on an utterance boundary or the hard cap, enqueue the
  buffered utterance for transcription.
- `results()` — iterator yielding result dicts (finals; see message shapes).
- `stop()` — cancel the worker, drain the queue, free the buffer.

**Streaming endpointer** (same module): dependency-free RMS/energy detector over
the incoming PCM. Defaults:
- `SILENCE_GAP_MS = 600` — trailing silence that closes an utterance.
- `MIN_UTTERANCE_MS = 1000` — suppress sub-second blips (don't transcribe noise).
- `MAX_UTTERANCE_MS = 12000` — hard cap; force-flush during a continuous monologue.
- `SILENCE_RMS_THRESHOLD` — derived from the −30 dB convention used by the
  upload splitter.

**Non-blocking guarantee:** `feed_audio` never blocks on transcription. Buffering
and a single sequential transcribe worker preserve utterance order. If the server
is slower than realtime, the bounded queue backs up — documented as a
"use a faster model/GPU" tradeoff, not a correctness bug.

**Session lifecycle guarantee:** the endpointer and transcribe worker are created
only *after* a WS connection AND *after* config validation passes. `stop()` (on
disconnect, recording stop, or terminal error) cancels the worker, drains the
queue, and frees the PCM buffer. No idle thread and no orphaned worker exists
between recordings. A registered FastAPI WebSocket route costs nothing while no
client is connected; all CPU-bearing work lives inside a per-connection session.

### `src-python/main.py` — `/ws/local-stream` handler (new)

Structurally parallel to `nvidia_stream_ws`:
- Read settings: `local_stt_base_url`, `local_stt_model`, `local_stt_api_key`,
  `stt_language`; read `translate_lang` from the query string.
- **Validate on connect.** If base URL or model is unset (mirroring
  `build_local_provider`), send a terminal error and close — the session is never
  started.
- Build `LocalRealtimeSession`; run the feed loop in an executor.
- `_send_results` loop emits the **same JSON shapes** the Nvidia handler emits.
- Diarization + translation reuse the Nvidia handler's logic. Where the reusable
  portion is a clean pure unit (attach speaker from PCM; throttled translate
  task), extract a small shared helper rather than copy-paste; otherwise mirror.

### Capability flag

`LocalOpenAIProvider.supports_realtime` → `True`.

### Frontend

- `src/components/recording/recording-constants.ts`: add
  `WS_PATH_LOCAL = "/ws/local-stream"`.
- `src/components/recording/use-streaming-stt.ts`: in `openStreamingWebSocket`,
  route `provider === "local"` → `WS_PATH_LOCAL` (today it is a binary
  soniox-vs-nvidia choice).
- `RecordingBar` already reads `stt_provider` and passes it through, so live
  recording with `local` works once the path routing is added.

### Record-start gate audit

We just fixed an **upload** gate that wrongly demanded an Nvidia API key for the
`local` provider (it treated any non-soniox provider as nvidia). This design
**audits the live-recording start path for the same class of bug** and fixes it
if present, so `local` can actually start a recording. (For `local`, readiness =
Base URL + Model configured; no API key required.)

## Data flow / message shapes

The handler emits the existing realtime JSON shapes so the frontend needs no new
message handling beyond path routing:

- **Final segment:**
  `{"text": str, "is_final": true, "speaker": str, "speaker_id": int,
    "chunk_id": str, "start_ms": int?, "end_ms": int?}`
- **Optional interim tick** (activity indicator while a chunk is transcribing):
  `{"text": "…", "is_final": false}` — keeps the UI's live-preview area from
  looking frozen. Approved as an optional nicety; finals carry the real text.
- **Translation:** bundled on the final message (`"translation": str`) and/or the
  existing `{"type": "translation", "translation": str, "chunk_id": str,
  "append": bool}` event, throttled like the Nvidia path (≤ once / 1.5 s or
  > 30-char change).
- **Terminal error:**
  `{"error": true, "terminal": true, "text": str, "is_final": true,
    "speaker": "System", "speaker_id": -1}` → UI stops recording + toast.

`chunk_id` is generated per finalized utterance so the frontend appends a new
transcript part per utterance (no mid-utterance replacement is needed since there
are no partials).

## Diarization

Reuse **CAM++** exactly as the Nvidia handler does: on each finalized chunk, run
`diarizer.identify_speaker_from_samples(samples, 16000)` over the chunk's PCM and
attach `speaker` / `speaker_id`. This gives speaker parity with the Nvidia path
and reuses the existing profile-history + hysteresis logic. No new diarization
code.

## Translation

Reuse `translate_instant` per finalized chunk with the same throttling the Nvidia
handler uses. Source language follows `stt_language`; target follows the
`translate_lang` query param. Translation runs as an async task so it never blocks
result delivery.

## Error handling

- **Config unset / server unreachable on connect** → terminal error, socket
  closed, session never started.
- **Per-chunk transcribe failure** → log and skip that one chunk; the session
  stays alive (one bad chunk must not kill a live meeting).
- **Server slower than realtime** → bounded queue backs up; ordering preserved;
  documented tradeoff (faster model / GPU).
- **Disconnect / stop** → `stop()` tears down the worker and frees buffers.

## Testing

- **Unit — endpointer:** feed synthetic PCM (speech/silence patterns) and assert
  flush points: utterance closes after `SILENCE_GAP_MS`, force-flush at
  `MAX_UTTERANCE_MS`, sub-`MIN_UTTERANCE_MS` blips suppressed.
- **Unit — `LocalRealtimeSession.results()`:** with a **mocked**
  `transcribe_file`, assert finals are yielded in order with correct
  `chunk_id`/`text`, and that `stop()` ends the iterator and joins the worker.
- **Diarization / translation / WS plumbing:** manual E2E, consistent with how the
  existing realtime providers are verified.

## Out of scope (this iteration)

- Live partial-word streaming (inherent Whisper limitation; not pursued).
- Bespoke realtime-whisper-server protocols (WhisperLive, whisper_streaming) and
  the OpenAI `/v1/realtime` WS protocol — explicitly rejected during brainstorming
  in favor of maximal server compatibility.

## Decisions log

- **RT mechanism:** pseudo-realtime via VAD-chunked batch (reuse v1 provider +
  CAM++ + translate_instant). Max server compatibility, finals only.
- **Translation:** included this iteration (reuses existing infra).
- **Chunk boundary:** silence-triggered (~600 ms) + 12 s hard cap + 1 s min;
  best transcription quality for VI/EN/FR.
- **Interim tick:** optional "…" activity indicator; finals carry the text.
