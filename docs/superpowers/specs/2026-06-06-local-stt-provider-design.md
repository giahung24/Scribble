# Design: Local STT Provider (OpenAI-compatible transcription)

**Date:** 2026-06-06
**Status:** Approved (design) — pending implementation plan
**Scope:** v1 = batch (Upload pipeline) only. Realtime + cabin translation deferred to v2.

---

## 1. Goal

Add a third STT provider to Scribble — a **local, self-hosted, OpenAI-compatible transcription endpoint** — alongside the existing cloud providers (Nvidia Riva, Soniox). The primary requirement is good-quality transcription for **Vietnamese, English, and French**, running fully on the user's own machine.

The provider is intentionally generic ("OpenAI-compatible transcription"), not Whisper-specific. The user supplies a **Base URL + model name** (same pattern as the existing OpenAI-compatible LLM config). Whisper (via `faster-whisper-server` / `whisper.cpp` server / Speaches) is the recommended default model, but the same adapter works for any server exposing `POST /v1/audio/transcriptions` — including vLLM-served audio models.

### Non-goals (v1)
- Realtime recording with the local provider (deferred to v2 — see §8).
- Cabin translation with the local provider (deferred to v2 — see §8).
- Bundling/spawning a Whisper server inside the app. v1 only stores a Base URL; the user runs the server themselves.
- Rewriting the existing Nvidia/Soniox internals.

---

## 2. Background — current STT architecture

STT is currently **not abstracted**; provider selection is hardcoded branching:

- **Realtime:** two separate WebSocket endpoints in `src-python/main.py`:
  - `/ws/nvidia-stream` → `NvidiaStreamingSTT`
  - `/ws/soniox-stream` → `SonioxStreamingSTT`
  - The frontend opens the endpoint matching the `stt_provider` setting. Each endpoint mixes ~280 lines of intertwined plumbing (diarization, translation, accumulation, DB flush, hallucination filter, speaker numbering) via nested functions.
- **Batch (Upload):** `src-python/services/upload_pipeline.py` branches in `_execute()`:
  - `stt_provider == "soniox"` → `_run_soniox_pipeline()` (Soniox async API).
  - else → `_run_nvidia_chunked_pipeline()` (VAD split → per-chunk Riva streaming → CAM++ clustering).
- **Settings:** `src/components/SettingsPanel.tsx` hardcodes the union `'nvidia' | 'soniox'`, with per-provider API keys and language lists, and several binary ternaries (`langOptions`, `currentApiKey`, `signupUrl`).
- **Translation (cabin):** `src-python/translate.py` uses **Nvidia Riva NMT (gRPC)**, invoked only from the realtime WS path (`main.py`). The batch pipeline does **not** translate (it sets `translation: ""` as a placeholder).

### Key insight that makes the local provider fit naturally
The Nvidia batch path is already structured as **"provider returns transcript only + CAM++ diarization layered on top"**. The local provider follows that exact shape — the *only* difference is the per-chunk transcription call (HTTP instead of gRPC). Diarization, resume/crash-safety, retry, and hallucination filtering are all reused unchanged.

Because cabin translation is realtime-only and batch never translates, **the v1 batch scope has no translation concern at all.**

---

## 3. Chosen approach

- **Engine direction:** Whisper-family, prioritizing Vietnamese quality (`large-v3` is the strongest open-source option for Vietnamese; en/fr are excellent).
- **Transport:** generic OpenAI-compatible HTTP (`POST /v1/audio/transcriptions`).
- **Server lifecycle:** user-managed. App stores Base URL + model + optional API key.
- **Refactor depth:** *interface + thin layer*. Introduce an `STTProvider` interface; implement the local provider cleanly against it; route **batch dispatch** through the interface. Riva/Soniox are **not** wrapped in v1 (v1 does not touch realtime or the shared WS plumbing), keeping blast radius minimal.
- **Modes:** batch (Upload) in v1; realtime staged to v2.

### Approaches considered and rejected
- **Full 3-layer refactor now** (rewrite Riva + Soniox as interface-conforming adapters, unify WS endpoints + upload dispatch): cleanest long-term but HIGH blast radius on working realtime code. Rejected for v1.
- **Minimal additive branching** (just add `if provider == "local"` everywhere, no interface): least code but accrues tech debt and makes future providers harder. Rejected.
- **sherpa-onnx in-process** (reuse bundled `onnxruntime`, native streaming): better for realtime, but Vietnamese model availability/quality is less proven than Whisper `large-v3`. Rejected given the Vietnamese-quality priority. May revisit for v2 realtime.
- **Vosk:** lightweight, true streaming, but Vietnamese accuracy is modest — risky for meeting minutes. Rejected.

---

## 4. Architecture

### 4.1 New package: `src-python/stt_providers/`
Kept separate from `stt.py` (already 1411 lines, overloaded).

**`base.py`**
```python
@dataclass
class STTSegment:
    text: str
    start_ms: int | None
    end_ms: int | None
    speaker_id: int | None = None   # None when provider does not diarize natively
    is_final: bool = True
    language: str | None = None

class STTProvider(ABC):
    name: str
    supports_batch: bool
    supports_realtime: bool
    native_diarization: bool        # False -> orchestration applies CAM++
    native_translation: bool        # False -> translation handled elsewhere

    async def transcribe_file(self, wav_path: Path, language: str, **opts) -> list[STTSegment]: ...

    # Declared for v2; v1 local provider raises NotImplementedError.
    def open_session(self, language: str, **opts) -> "STTSession": ...

class STTSession(ABC):   # interface only in v1, implemented in v2
    def feed_audio(self, pcm_bytes: bytes) -> None: ...
    def results(self) -> Iterator[STTSegment]: ...
    def stop(self) -> None: ...
```

**`local_openai.py` — `LocalOpenAIProvider`**
- `supports_batch=True`, `supports_realtime=False`, `native_diarization=False`, `native_translation=False`.
- `transcribe_file()`:
  - POST multipart to `{base_url}/v1/audio/transcriptions` with fields: `file` (the chunk WAV), `model` (`local_stt_model`), `language` (mapped from `stt_language`), `response_format=verbose_json` (to obtain segment timestamps), and `Authorization: Bearer {local_stt_api_key}` when a key is set.
  - Parse `verbose_json` → list of `STTSegment` (text + start/end ms). `speaker_id` left `None` (CAM++ assigns it later).
  - Robust to servers that ignore `verbose_json` / return plain `{"text": ...}`: fall back to a single segment spanning the chunk.
  - Uses `httpx` (already a dependency).

**`registry.py`**
- `get_provider(db) -> STTProvider` maps the `stt_provider` setting to a provider instance. v1 returns `LocalOpenAIProvider` for `"local"`; `"nvidia"`/`"soniox"` continue through their existing code paths (the registry need only resolve what the batch dispatch consumes).

### 4.2 Batch pipeline integration (`services/upload_pipeline.py`)
- Generalize the Nvidia chunked path so the **per-chunk transcription function is injectable** rather than hardcoded to `transcribe_nvidia_streaming`.
- `_execute()` dispatch gains a `"local"` branch that runs the **same chunked framework** as Nvidia (VAD split → per-chunk transcribe → CAM++ clustering via `batch_diarizer`), supplying the local transcribe function.
- No schema change: `upload_chunks` (text + embedding BLOB) is reused, so **per-chunk resume, retry-failed-chunks, and crash recovery work unchanged**.
- Hallucination filtering (`_filter_hallucinations`) and transcript assembly are reused.

### 4.3 Settings (`db.py` + API + frontend)
- New settings keys (string, in the existing `settings` table — no migration needed, key/value store): `local_stt_base_url`, `local_stt_model`, `local_stt_api_key`. Reuse `stt_language` for language selection.
- `src/components/SettingsPanel.tsx`:
  - Extend `stt_provider` to `'nvidia' | 'soniox' | 'local'`.
  - Replace the binary ternaries (`langOptions`, `currentApiKey`, `signupUrl`, etc.) with a provider-keyed lookup so a third option slots in cleanly.
  - Add a Local config block: **Base URL + Model + (optional) API key**, mirroring the existing OpenAI-compatible LLM block.
  - Model field: attempt `GET {base_url}/v1/models` to populate a dropdown; fall back to free-text entry when the server does not implement it.

---

## 5. Data flow (v1 batch)

```
User picks "Local" provider in Settings, enters Base URL + model
        │
Upload file ──> Tauri upload::upload_audio_to_sidecar ──> POST /meetings/upload-audio
        │
upload_pipeline.run_pipeline -> _execute()
        │  stt_provider == "local"
        ▼
  normalize -> 16kHz mono WAV
  VAD split -> chunks (persist chunk plan)
  for each pending chunk (parallel, resume-aware):
        LocalOpenAIProvider.transcribe_file(chunk.wav, language)
            -> POST {base_url}/v1/audio/transcriptions (verbose_json)
            -> [STTSegment...]
        extract CAM++ embedding (batch_diarizer)
        persist chunk text + embedding (upload_chunks)
  CAM++ cluster speakers across whole meeting
  assemble transcript -> persist
  auto-summarize (existing LLM path, best-effort)
        │
Progress streamed over SSE /jobs/{id}/events (unchanged)
```

---

## 6. Error handling
- **Server unreachable / connection refused:** surface a clear message ("Local STT server not reachable at {base_url}") and mark affected chunks failed (existing per-chunk failure tracking + `retry-failed-chunks` applies).
- **Non-2xx from server:** classify like other chunk errors (transient vs permanent) and feed into the existing retry/backoff logic in `upload_pipeline`.
- **Unexpected response shape:** fall back to single-segment text; never crash the pipeline.
- **No Base URL / model configured:** validation at job start with an actionable error pointing to Settings.

---

## 7. Testing
- **Unit:** `LocalOpenAIProvider.transcribe_file` maps `verbose_json` → `STTSegment[]` (mock `httpx`); covers the plain-`{"text":...}` fallback and the error/non-2xx path.
- **Integration:** run the batch pipeline with `provider=local` against a fake local server returning fixed `verbose_json`; assert transcript assembly, CAM++ diarization, and **resume** (kill mid-run, restart, verify only pending chunks re-POST).
- **Manual:** run a real `faster-whisper-server`, upload vi/en/fr audio, verify Vietnamese quality and timestamps.

---

## 8. v1 limitations & v2 roadmap (document for users)
**Not supported in v1:** realtime recording with the local provider; cabin translation with the local provider.

**v2 roadmap:**
1. **Realtime local** via windowed faux-streaming (buffer PCM to silence/window boundary → POST window → emit final segments; CAM++ on the same window). This requires extracting the shared realtime WS orchestrator (diarize/translate/flush) so providers plug in cleanly — a deliberate, separately-scoped refactor.
2. **Pluggable translation backend** — choose **Nvidia NMT or the configured LLM** for cabin translation. LLM-based translation (via the existing OpenAI-compatible LLM config, e.g. local Ollama) enables fully-local translation; trade-off is higher per-line latency/cost, mitigated by concise prompts and batching lines.

---

## 9. Files touched (estimate)
- **New:** `src-python/stt_providers/{__init__,base,local_openai,registry}.py`; tests under `src-python/` test layout.
- **Modified:** `src-python/services/upload_pipeline.py` (injectable transcribe + `local` dispatch); `src/components/SettingsPanel.tsx` (3rd provider + local config block); possibly `src/lib/api.ts` (model-list fetch for local) and `src-python/api/settings.py` (`/models`-style helper for local, optional).
- **Unchanged:** realtime WS endpoints, Riva/Soniox internals, DB schema, Tauri/Rust layer.

> Per repo convention (`CLAUDE.md`/`gitnexus.md`): run `gitnexus_impact` before editing `_run_nvidia_chunked_pipeline` / `_execute` / `SettingsPanel`, and `gitnexus_detect_changes` before committing.
